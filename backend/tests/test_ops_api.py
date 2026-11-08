"""告警、处置建议、工单、报告接口的集成测试（测试库 + fakeredis + 假模型，不调用外部模型）。

每个测试前清空告警/工单/报告表并扫描 48 小时，得到确定的 10 个告警事件（见 conftest 的两个场景）。
"""

import json

import psycopg
import pytest
from psycopg.errors import UniqueViolation
from pydantic import ValidationError

from app.ai.agent.prompts import ANSWER_SYSTEM
from app.ai.reporter import (
    REPORT_SYSTEM,
    Fact,
    Narrative,
    check_narrative,
    fill,
    parse_narrative,
)


@pytest.fixture
def ops(client, db_url):
    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute("TRUNCATE alarm, work_order, report RESTART IDENTITY CASCADE")
    r = client.post("/api/alarms/scan", params={"hours": 48})
    assert r.status_code == 200, r.text
    return client


def find(client, **params) -> dict:
    items = client.get("/api/alarms", params={"limit": 200, **params}).json()["items"]
    assert len(items) == 1, items
    return items[0]


def by_rule(client, rule: str) -> dict:
    """列表接口不按规则过滤（对外按告警类型）；测试里按规则 id 精确取一条。"""
    (alarm,) = [a for a in client.get("/api/alarms", params={"limit": 200}).json()["items"] if a["rule"] == rule]
    return alarm


def route(fake_llm, *, advice=None, report=None):
    """按系统提示词分派到不同的假回答。"""
    def responder(messages):
        system = messages[0]["content"]
        if system == ANSWER_SYSTEM and advice:
            return advice(messages)
        if system == REPORT_SYSTEM and report:
            return report(messages)
        raise AssertionError(f"意外的模型调用：{system[:30]}")
    fake_llm.responder = responder


def advice_ok(messages) -> str:
    """按本轮真实证据构造引用：t1 第一行第二列的单元格 + 第一条知识片段的原文前缀。"""
    obs = json.loads(messages[1]["content"])["observations"]
    t1 = next(o for o in obs if o["call_id"] == "t1")
    data = {"call_id": "t1", "row": 0, "column": 1, "value": t1["rows"][0]["values"][1]}
    rag = next(o for o in obs if o["tool"] == "rag_search")
    hit = rag["hits"][0]
    know = {"call_id": rag["call_id"], "chunk_id": hit["id"], "quote": hit["text"].strip()[:40]}
    return json.dumps({"answerable": True, "reason": "", "claims": [
        {"kind": "fact", "text": "告警前后出水总磷均值见证据表", "data_refs": [data]},
        {"kind": "hypothesis", "text": "可能与加药系统有效投加不足有关", "data_refs": [data], "knowledge_refs": [know]},
        {"kind": "recommendation", "text": "核查加药泵出口与实际投加量", "knowledge_refs": [know]},
    ]}, ensure_ascii=False)


# ---------------------------------------------------------------- 告警查询与扫描

def test_scan_is_idempotent_and_incremental_matches_full(ops):
    again = ops.post("/api/alarms/scan", params={"hours": 48}).json()
    assert (again["episodes"], again["inserted"], again["updated"]) == (10, 0, 0)
    inc = ops.post("/api/alarms/scan", params={"hours": 3}).json()
    assert (inc["inserted"], inc["updated"]) == (0, 0)
    body = ops.get("/api/alarms", params={"limit": 200}).json()
    assert body["summary"] == {"total": 10, "active": 8, "by_level": {"告警": 7, "预警": 3, "提示": 0},
                               "by_status": {"未处理": 10, "处理中": 0, "已关闭": 0}}
    assert [a["ts"] for a in body["items"]] == sorted((a["ts"] for a in body["items"]), reverse=True)


def test_alarm_chain_for_pac_pump_block(ops):
    trend = find(ops, plant="P02", kind="水质预警")
    exceed = find(ops, plant="P02", kind="水质超标")
    dosing = find(ops, plant="P02", kind="药耗异常")
    assert trend["metric"] == exceed["metric"] == "tp_out"
    assert trend["ts"] < exceed["ts"] < dosing["ts"]
    assert trend["ended_at"] == exceed["ts"]                    # 趋势预警在超标时结束
    assert trend["detail"]["eta_min"] > 0 and trend["detail"]["predicted_at"]
    assert exceed["active"] and exceed["threshold"] == 0.5 and exceed["peak"] >= exceed["value"] > 0.5
    assert dosing["metric"] == "pac_dose_kgh" and dosing["unit"] == "倍"
    assert dosing["detail"]["model"]["terms"] == ["进水量×进水总磷", "进水量×进水 COD", "进水量"]


def test_alarm_filters(ops):
    assert ops.get("/api/alarms", params={"plant": "p06", "active": "true"}).json()["total"] == 6
    assert ops.get("/api/alarms", params={"active": "false"}).json()["total"] == 2
    assert ops.get("/api/alarms", params={"level": "预警"}).json()["total"] == 3
    assert ops.get("/api/alarms", params={"status": "未关闭"}).json()["total"] == 10
    # 时间窗口取相交：只有 07:30 已恢复的 P02 趋势预警不在其中，开始更早但仍持续的告警要算进来
    assert ops.get("/api/alarms", params={"start": "2026-11-05T10:00:00", "end": "2026-11-05T12:00:00"}).json()[
        "total"] == 9
    assert ops.get("/api/alarms", params={"kind": "不存在"}).status_code == 422
    device = ops.get("/api/alarms", params={"plant": "P06", "kind": "设备异常"}).json()["items"]
    assert {a["device"]["code"] for a in device} == {"P06-BL-1"} and len(device) == 3


def test_scan_lock_and_monitor_status(ops, sync_redis):
    sync_redis.set("alarm:scan:lock", "other")
    assert ops.post("/api/alarms/scan").status_code == 409
    sync_redis.delete("alarm:scan:lock")
    status = ops.get("/api/alarms/monitor").json()
    assert status["running"] is False and status["scans"] >= 1 and status["last"]["episodes"] == 10
    assert len(status["rules"]) == 9 and status["bucket_min"] == 5
    assert next(r for r in status["rules"] if r["id"] == "inflow_load")["kind"] == "进水负荷"
    assert ops.post("/api/alarms/scan", params={"hours": 0}).status_code == 422
    assert ops.post("/api/alarms/scan", params={"hours": 169}).status_code == 422


def test_backfill_scan_publishes_one_summary(client, db_url, sync_redis):
    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute("TRUNCATE alarm, work_order, report RESTART IDENTITY CASCADE")
    ps = sync_redis.pubsub()
    ps.subscribe("alarm:events")
    ps.get_message(timeout=1)                                    # 订阅确认
    client.post("/api/alarms/scan", params={"hours": 48})
    msgs = [json.loads(m["data"]) for m in iter(lambda: ps.get_message(timeout=0.5), None)]
    assert msgs == [{"type": "alarm", "event": "backfill", "inserted": 10, "updated": 0}]
    client.post("/api/alarms/scan", params={"hours": 3})        # 没有变化：不推送
    assert ps.get_message(timeout=0.5) is None


def test_alarm_detail_includes_related(ops):
    exceed = find(ops, plant="P02", kind="水质超标")
    detail = ops.get(f"/api/alarms/{exceed['id']}").json()
    assert {a["kind"] for a in detail["related"]} == {"水质预警", "药耗异常"}
    assert detail["advice"] is None and detail["work_orders"] == []
    assert ops.get("/api/alarms/99999").status_code == 404


# ---------------------------------------------------------------- 处置建议

def test_advice_cites_fixed_evidence_and_is_cached(ops, fake_llm):
    route(fake_llm, advice=advice_ok)
    exceed = find(ops, plant="P02", kind="水质超标")
    body = ops.post(f"/api/alarms/{exceed['id']}/advice").json()
    assert body["cached"] is False and body["saved"] is True
    advice = body["advice"]
    assert advice["ok"] and advice["error"] is None
    assert [(e["id"], e["tool"]) for e in advice["evidence"]] == [
        ("t1", "sql_query"), ("t2", "sql_query"), ("t3", "sql_query"), ("t4", "rag_search"),
        ("t5", "plant_statistics")]
    stats = advice["evidence"][-1]
    assert stats["status"] == "ok" and stats["output"]["facts"]
    t1, t3 = advice["evidence"][0]["output"], advice["evidence"][2]
    assert t1["columns"][:3] == ["时间", "出水总磷均值(mg/L)", "出水总磷最高(mg/L)"]
    assert "加药泵" in t3["question"] and all(r[1] == "加药泵" for r in t3["output"]["rows"])
    assert {c["kind"] for c in advice["citations"]} == {"data", "knowledge"}
    assert "不要写未经验证的操作参数" in advice["question"]
    calls = len(fake_llm.calls)

    cached = ops.post(f"/api/alarms/{exceed['id']}/advice").json()
    assert cached["cached"] is True and cached["advice"]["generated_at"] == advice["generated_at"]
    assert len(fake_llm.calls) == calls                           # 不重复调用模型
    detail = ops.get(f"/api/alarms/{exceed['id']}").json()
    assert detail["advice_at"] == advice["generated_at"] and detail["advice"]["answer"] == advice["answer"]


def test_advice_can_cite_server_written_statistical_facts(ops, fake_llm):
    def statistical_advice(messages):
        obs = json.loads(messages[1]["content"])["observations"]
        stats = next(o for o in obs if o["tool"] == "plant_statistics")
        fact = next(f for f in stats["facts"] if f["statistic"] == "longest_exceed_hours")
        return json.dumps({"answerable": True, "claims": [{"kind": "statistic", "text": "", "statistic_refs": [
            {"call_id": stats["call_id"], "fact_id": fact["id"]}]}]})

    route(fake_llm, advice=statistical_advice)
    exceed = find(ops, plant="P02", kind="水质超标")
    response = ops.post(f"/api/alarms/{exceed['id']}/advice")
    assert response.status_code == 200, response.text
    advice = response.json()["advice"]
    assert advice["ok"] and advice["citations"][0]["kind"] == "statistic"
    assert advice["citations"][0]["text"] in advice["answer"]
    assert "最长连续" in advice["answer"]


def test_advice_failure_does_not_overwrite(ops, fake_llm):
    exceed = find(ops, plant="P02", kind="水质超标")
    route(fake_llm, advice=advice_ok)
    first = ops.post(f"/api/alarms/{exceed['id']}/advice").json()["advice"]
    route(fake_llm, advice=lambda m: "不是JSON")
    r = ops.post(f"/api/alarms/{exceed['id']}/advice", params={"refresh": "true"})
    assert r.status_code == 502 and "未通过证据校验" in r.json()["detail"]
    route(fake_llm, advice=lambda m: json.dumps({"answerable": False, "claims": [], "reason": "证据不足"}))
    refused = ops.post(f"/api/alarms/{exceed['id']}/advice", params={"refresh": "true"}).json()
    assert refused["saved"] is False and refused["advice"]["refused"] and refused["advice"]["reason"] == "证据不足"
    assert ops.get(f"/api/alarms/{exceed['id']}").json()["advice"]["answer"] == first["answer"]


def test_advice_lock_and_unavailable(ops, fake_llm, sync_redis):
    exceed = find(ops, plant="P02", kind="水质超标")
    sync_redis.set(f"alarm:advice:lock:{exceed['id']}", "other")
    assert ops.post(f"/api/alarms/{exceed['id']}/advice").status_code == 409
    fake_llm.available = False
    assert ops.post(f"/api/alarms/{exceed['id']}/advice").status_code == 503


def test_device_alarm_evidence_uses_device_series(ops, fake_llm):
    route(fake_llm, advice=advice_ok)
    stop = by_rule(ops, "device_stop")
    advice = ops.post(f"/api/alarms/{stop['id']}/advice").json()["advice"]
    t1 = advice["evidence"][0]["output"]
    assert t1["columns"] == ["时间", "运行占比(%)", "最大振动(mm/s)", "最高轴承温度(℃)", "运行平均电流(A)"]
    t3 = advice["evidence"][2]["output"]
    codes = {r[0]: r[2] for r in t3["rows"]}                   # 同厂鼓风机（含备用）的台账状态
    assert "P06-BL-1" in codes and len(codes) >= 2


# ---------------------------------------------------------------- 工单流转

def test_work_order_flow(ops, fake_llm):
    route(fake_llm, advice=advice_ok)
    exceed = find(ops, plant="P02", kind="水质超标")
    aid = exceed["id"]
    ops.post(f"/api/alarms/{aid}/advice")

    draft = ops.post(f"/api/alarms/{aid}/work-order-draft").json()
    assert draft["title"] == "协税镇污水处理站 水质超标（出水总磷）"
    assert draft["source"]["kind"] == "ai_advice"
    first = draft["steps"][0]
    assert first["source"] == "ai" and first["text"] == "核查加药泵出口与实际投加量"
    assert first["citations"][0]["kind"] == "knowledge" and first["citations"][0]["quote"]
    assert draft["steps"][-1]["source"] == "template"
    assert ops.get("/api/work-orders").json()["total"] == 0      # 草稿不写库

    wo = ops.post("/api/work-orders", json={"draft_id": draft["draft_id"], "assignee": "张工"})
    assert wo.status_code == 201, wo.text
    wo = wo.json()
    assert wo["status"] == "待处理" and wo["steps"] == draft["steps"] and wo["alarm_id"] == aid
    assert wo["source"]["edited"] == {"title": False, "steps": False} and wo["created_at"].startswith("2026-11-05T12")
    assert ops.get(f"/api/alarms/{aid}").json()["status"] == "处理中"
    assert ops.post("/api/work-orders", json={"draft_id": draft["draft_id"], "assignee": "张工"}).status_code == 404
    assert ops.post(f"/api/alarms/{aid}/work-order-draft").status_code == 409
    assert ops.post(f"/api/alarms/{aid}/close", json={"note": "误报"}).status_code == 409

    started = ops.post(f"/api/work-orders/{wo['id']}/start").json()
    assert started["status"] == "处理中" and started["started_at"]
    assert ops.post(f"/api/work-orders/{wo['id']}/start").status_code == 409

    closed = ops.post(f"/api/work-orders/{wo['id']}/close", json={"resolution": "疏通加药泵，投加恢复"}).json()
    assert closed["status"] == "已关闭" and closed["resolution"] == "疏通加药泵，投加恢复"
    assert closed["alarm_closed"] is False and "尚未恢复" in closed["notice"]   # 数据未恢复：告警保持处理中
    assert ops.get(f"/api/alarms/{aid}").json()["status"] == "处理中"
    assert ops.post(f"/api/work-orders/{wo['id']}/close", json={"resolution": "再次关闭"}).status_code == 409

    done = ops.post(f"/api/alarms/{aid}/close", json={"note": "已人工确认处置完成"}).json()
    assert done["status"] == "已关闭" and done["close_note"] == "已人工确认处置完成"
    assert ops.post(f"/api/alarms/{aid}/work-order-draft").status_code == 409
    board = ops.get("/api/work-orders").json()
    assert board["counts"] == {"待处理": 0, "处理中": 0, "已关闭": 1}


def test_work_order_edits_and_recovered_alarm_closes(ops):
    trend = find(ops, plant="P02", kind="水质预警")               # 数据已恢复的告警
    draft = ops.post(f"/api/alarms/{trend['id']}/work-order-draft").json()
    assert draft["source"]["kind"] == "template" and "不是 AI 建议" in draft["source"]["note"]
    assert all(s["source"] == "template" for s in draft["steps"])
    steps = [draft["steps"][0]["text"], "联系化验室加测总磷"]
    wo = ops.post("/api/work-orders", json={"draft_id": draft["draft_id"], "assignee": "李工", "title": "P02 总磷复核",
                                            "steps": steps}).json()
    assert wo["title"] == "P02 总磷复核" and wo["source"]["edited"] == {"title": True, "steps": True}
    assert [s["source"] for s in wo["steps"]] == ["template", "user"]
    closed = ops.post(f"/api/work-orders/{wo['id']}/close", json={"resolution": "复核正常"}).json()
    assert closed["alarm_closed"] is True and closed["notice"] is None
    alarm = ops.get(f"/api/alarms/{trend['id']}").json()
    assert alarm["status"] == "已关闭" and alarm["close_note"].startswith(f"随工单 #{wo['id']} 关闭")


def test_work_order_validation(ops):
    assert ops.post("/api/work-orders", json={"draft_id": "x" * 32, "assignee": "张工"}).status_code == 422
    assert ops.post("/api/work-orders", json={"draft_id": "a" * 32, "assignee": ""}).status_code == 422
    assert ops.post("/api/work-orders", json={"draft_id": "a" * 32, "assignee": "张工", "x": 1}).status_code == 422
    assert ops.post("/api/work-orders", json={"draft_id": "a" * 32, "assignee": "张工"}).status_code == 404
    assert ops.get("/api/work-orders/99999").status_code == 404
    assert ops.post("/api/work-orders/99999/start").status_code == 404


def test_one_open_work_order_per_alarm_is_enforced_by_db(ops, db_url):
    aid = find(ops, plant="P02", kind="水质超标")["id"]
    with psycopg.connect(db_url, autocommit=True) as conn:
        insert = "INSERT INTO work_order (alarm_id, title, status) VALUES (%s, '并发', %s)"
        conn.execute(insert, (aid, "已关闭"))
        conn.execute(insert, (aid, "待处理"))
        with pytest.raises(UniqueViolation):
            conn.execute(insert, (aid, "处理中"))


# ---------------------------------------------------------------- 报告

def report_ok(messages) -> str:
    ev = json.loads(messages[1]["content"])["events"][0]
    return json.dumps({
        "overview": ["统计时段{{period.start}}至{{period.end}}，全区处理水量{{total.water_m3}}，出水达标率{{total.compliance}}。"],
        "events": [f"{ev['plant']}发生{ev['kind']}，告警{{{{{ev['id']}.id}}}}开始于{{{{{ev['id']}.start}}}}，"
                   f"处理状态为{ev['status']}。"],
        "consumption": ["全区曝气电耗{{total.aeration_kwh}}，碳源用量{{total.carbon_kg}}，PAC用量{{total.pac_kg}}。"],
        "attention": ["P06-BL-1 相关告警仍未关闭，同一厂站的连锁告警需一起跟踪。"],
    }, ensure_ascii=False)


def report_bad(messages) -> str:
    return json.dumps({"overview": ["全区处理水量 12345 m³，共发生三起告警。"], "events": [],
                       "consumption": ["碳源用量{{total.no_such}}。"], "attention": []}, ensure_ascii=False)


def test_report_with_llm_narrative(ops, fake_llm):
    route(fake_llm, report=report_ok)
    r = ops.post("/api/reports", json={"kind": "日报"})
    assert r.status_code == 201, r.text
    rep = r.json()
    assert rep["meta"]["narrative"] == "llm" and rep["meta"]["attempts"] == [{"errors": []}]   # 通过的不存原文
    assert rep["period_start"].startswith("2026-11-05T00:00") and rep["period_end"].startswith("2026-11-05T12:00")
    md = rep["content_md"]
    assert md.startswith("# 汉水智控运行日报（2026-11-05）") and "{{" not in md and "模拟器生成" in md
    assert "口径：达标率要求每个 5 分钟桶内的原始采样点" in md and "按 5 分钟均值统计" in md   # 两种口径写进报告
    detail = ops.get(f"/api/reports/{rep['id']}").json()
    facts = detail["facts"]
    assert facts["total.water_m3"]["text"] in md and facts["alarms.total"]["value"] == 10
    lo, hi = facts["plants.kwh_per_m3.min"], facts["plants.kwh_per_m3.max"]   # 跨厂站汇总：标签写明是哪座厂站
    assert lo["value"] <= hi["value"] and "污水处理" in lo["label"] and "plants.compliance.full_n" in facts
    for section in ("## 一、运行概况", "## 二、异常事件与处置", "## 三、能耗药耗", "## 四、关注事项"):
        assert section in md
    sent = json.loads(fake_llm.calls[-1][1]["content"])
    assert all(not f["id"].startswith(("A", "W")) for f in sent["facts"])   # 告警事实只在 events 里给
    assert not any(ch.isdigit() for f in sent["facts"] for ch in f["label"])  # 标签不含数字，防止照抄


def test_report_repairs_once_then_falls_back_to_template(ops, fake_llm):
    route(fake_llm, report=report_bad)
    rep = ops.post("/api/reports", json={"kind": "日报"}).json()
    meta = rep["meta"]
    assert meta["narrative"] == "template" and len(meta["attempts"]) == 2 and len(fake_llm.calls) == 2
    errors = " ".join(meta["attempts"][0]["errors"])
    assert "数字" in errors and "三起" in errors and "total.no_such" in errors
    assert meta["attempts"][0]["output"] == report_bad([]) and "finish_reason" in meta["attempts"][0]
    md = rep["content_md"]
    assert "{{" not in md
    assert "12345" not in md and "三起" not in md
    assert "叙述：模板生成" in md

    calls = []
    def repaired(messages):
        calls.append(len(messages))
        return report_bad(messages) if len(messages) == 2 else report_ok(messages)
    route(fake_llm, report=repaired)
    rep = ops.post("/api/reports", json={"kind": "日报"}).json()
    assert rep["meta"]["narrative"] == "llm" and calls == [2, 4]
    assert "校验失败" in fake_llm.calls[-1][-1]["content"]


def test_report_without_model_uses_template(ops, fake_llm):
    fake_llm.available = False
    rep = ops.post("/api/reports", json={"kind": "周报"}).json()
    assert rep["meta"]["narrative"] == "template" and fake_llm.calls == []
    assert rep["period_start"].startswith("2026-10-30T00:00")
    assert rep["content_md"].startswith("# 汉水智控运行周报（2026-10-30 至 2026-11-05）")


def test_report_periods_and_listing(ops, fake_llm):
    fake_llm.available = False
    assert ops.post("/api/reports", json={"kind": "日报", "date": "2026-11-06"}).status_code == 422
    assert ops.post("/api/reports", json={"kind": "日报", "date": "2026-10-01"}).status_code == 422
    assert ops.post("/api/reports", json={"kind": "月报"}).status_code == 422
    rep = ops.post("/api/reports", json={"kind": "日报", "date": "2026-11-04"}).json()
    assert rep["period_end"].startswith("2026-11-05T00:00")
    listing = ops.get("/api/reports").json()
    assert listing["total"] == 1 and listing["items"][0]["narrative"] == "template"
    md = ops.get(f"/api/reports/{rep['id']}/markdown")
    assert md.headers["content-type"].startswith("text/markdown")
    assert f"shuiwu-daily-20261104-{rep['id']}.md" in md.headers["content-disposition"]
    assert md.text == rep["content_md"]
    pdf = ops.get(f"/api/reports/{rep['id']}/pdf")
    assert pdf.headers["content-type"] == "application/pdf"
    assert f"shuiwu-daily-20261104-{rep['id']}.pdf" in pdf.headers["content-disposition"]
    assert pdf.content.startswith(b"%PDF") and len(pdf.content) > 2000
    assert ops.get("/api/reports/99999").status_code == 404
    assert ops.get("/api/reports/99999/pdf").status_code == 404


def test_report_counts_work_orders_stamped_at_report_time(ops, fake_llm):
    """演示时钟冻结在 12:00：工单与报告的时间戳相同，截至此刻的报告也要算进去。"""
    fake_llm.available = False
    closed_alarm, open_alarm = find(ops, plant="P02", kind="水质预警"), by_rule(ops, "device_stop")
    for alarm, who in ((closed_alarm, "李工"), (open_alarm, "张工")):
        draft = ops.post(f"/api/alarms/{alarm['id']}/work-order-draft").json()
        wo = ops.post("/api/work-orders", json={"draft_id": draft["draft_id"], "assignee": who}).json()
        if alarm is closed_alarm:
            ops.post(f"/api/work-orders/{wo['id']}/close", json={"resolution": "复核正常"})
    rep = ops.post("/api/reports", json={"kind": "日报"}).json()
    facts = ops.get(f"/api/reports/{rep['id']}").json()["facts"]
    assert [facts[k]["value"] for k in ("orders.created", "orders.closed", "orders.open")] == [2, 1, 1]
    assert "统计结束时未关闭的工单 1 张" in rep["content_md"]
    # 历史日报（11-04）不含 11-05 才建的工单
    old = ops.post("/api/reports", json={"kind": "日报", "date": "2026-11-04"}).json()
    old_facts = ops.get(f"/api/reports/{old['id']}").json()["facts"]
    assert old_facts["orders.created"]["value"] == 0 and old_facts["orders.open"]["value"] == 0


def test_check_narrative_rules():
    facts = {"a.b": Fact("a.b", "水量", "1 m³")}
    ok = Narrative(overview=["处理水量{{a.b}}，P06-BL-1 运行正常，A2O 工艺与之一起统计。"], consumption=["无"])
    assert check_narrative(ok, facts) == []
    bad = Narrative(overview=["处理水量 {{a.c}}", "电耗 3 kWh"], consumption=["发生两起告警", "百分之九十"],
                    attention=["告警级 {{alarms.告警}}"])   # 非 ASCII 的 id 不是合法占位符，不能原样漏进报告
    errors = check_narrative(bad, facts)
    assert len(errors) == 5 and "a.c" in errors[0] and "格式不对" in errors[4]
    with pytest.raises(ValueError):
        fill("告警级 {{alarms.告警}}", facts)
    assert fill("工单{{a.b}}", {"a.b": Fact("a.b", "标题", "{{用户写的}}")}) == "工单{{用户写的}}"


def test_parse_narrative_tolerates_extra_field_and_literal_ids():
    # 线上日报 #5 两次失败的原因：顶层多了 "type"；attention 里照抄了“关联工单#2”
    facts = {"A44.id": Fact("A44.id", "告警编号", "#44", 44), "W2.id": Fact("W2.id", "工单编号", "#2", 2),
             "W2.alarm": Fact("W2.alarm", "关联告警编号", "#44", 44), "A2.id": Fact("A2.id", "告警编号", "#2", 2)}
    content = json.dumps({"type": "daily", "overview": ["告警#44已派单，关联工单#2待处理。"], "consumption": ["无"],
                          "attention": ["#44仍未关闭；工单 #9 不存在。"]}, ensure_ascii=False)
    n, notes = parse_narrative(content, facts)
    assert n.overview == ["告警{{A44.id}}已派单，关联工单{{W2.id}}待处理。"]
    # 未写种类的 #44：A44.id 与 W2.alarm 含义相同（都是告警），可以替换；#9 无对应事实，原样留给校验报错
    assert n.attention == ["{{A44.id}}仍未关闭；工单 #9 不存在。"]
    assert notes[0] == "忽略多余字段：type" and len(notes) == 4
    errors = check_narrative(n, facts)
    assert len(errors) == 1 and "#9" in errors[0]
    # #2 同时是告警和工单编号：没写种类就不猜；“2#池”是池号，不是编号
    n, _ = parse_narrative(json.dumps({"overview": ["#2未关闭", "2#池"], "consumption": ["无"]}, ensure_ascii=False), facts)
    assert n.overview == ["#2未关闭", "2#池"]
    # 告警不在本期事件里、只作为工单关联告警出现时，用 W*.alarm
    only = {"W2.alarm": Fact("W2.alarm", "关联告警编号", "#44", 44)}
    n, _ = parse_narrative(json.dumps({"overview": ["告警#44"], "consumption": ["无"]}, ensure_ascii=False), only)
    assert n.overview == ["告警{{W2.alarm}}"]
    with pytest.raises(ValidationError):
        parse_narrative(json.dumps({"overview": [], "consumption": ["无"]}), facts)


def test_parse_narrative_drops_repeated_units():
    # 线上日报 #6 第一次失败：“{{plants.compliance.full_n}}座”，替换后会成为“5 座座”
    facts = {"n": Fact("n", "厂站数", "5 座", 5), "r": Fact("r", "达标率", "99.03%", 0.99),
             "t": Fact("t", "开始时间", "11-08 10:19", "x"), "w": Fact("w", "处理水量", "60,029 m³", 60029)}
    text = "达标厂站{{n}}座，达标率{{ r }} %，{{t}}起持续，{{n}}厂站，水量{{w}}立方米。"
    n, notes = parse_narrative(json.dumps({"overview": [text], "consumption": ["无"]}, ensure_ascii=False), facts)
    # 只删与事实单位完全相同的；“起”不是时间的单位，“立方米”与“m³”写法不同，都留给校验
    assert n.overview == ["达标厂站{{n}}，达标率{{ r }}，{{t}}起持续，{{n}}厂站，水量{{w}}立方米。"]
    assert len(notes) == 2 and "“座”" in notes[0]
    assert check_narrative(n, facts) == []
    assert fill(n.overview[0], facts).startswith("达标厂站5 座，达标率99.03%，")


def test_check_relations_across_plants():
    # 线上验收时 Flash 写过的句子：数字都是真的，但范围两端取了 P04、P07，P06（风机跳停、曝气少）更低
    kwh = {"P01": 0.1006, "P04": 0.0972, "P06": 0.0929, "P07": 0.1014}
    comp = {"P01": 1.0, "P04": 1.0, "P06": 0.8924, "P07": 1.0}
    facts = {"P06.nh3n_out.max": Fact("P06.nh3n_out.max", "出水氨氮最高值", "16.73 mg/L", 16.73),
             "plants.compliance.max": Fact("plants.compliance.max", "", "100.00%", 1.0)}
    for c in kwh:
        facts[f"{c}.kwh_per_m3"] = Fact(f"{c}.kwh_per_m3", "吨水曝气电耗", f"{kwh[c]:.3f} kWh/m³", kwh[c])
        facts[f"{c}.compliance"] = Fact(f"{c}.compliance", "出水达标率", f"{comp[c] * 100:.2f}%", comp[c])
    for side, v in (("min", 0.0929), ("max", 0.1014)):
        facts[f"plants.kwh_per_m3.{side}"] = Fact(f"plants.kwh_per_m3.{side}", "", f"{v:.3f} kWh/m³", v)
    names = {"P01": "南郑中心污水处理厂", "P04": "高台镇污水处理站", "P06": "黄官镇污水处理站", "P07": "牟家坝镇污水处理站"}

    def errs(text):
        return check_narrative(Narrative(overview=[text], consumption=["无"]), facts, names)

    [e] = errs("其余各镇站吨水曝气电耗在{{P04.kwh_per_m3}}至{{P07.kwh_per_m3}}之间。")
    assert "两端不是各厂站吨水曝气电耗的最低值和最高值" in e and "{{plants.kwh_per_m3.min}}" in e
    assert errs("各厂站吨水曝气电耗在{{P06.kwh_per_m3}}至{{P07.kwh_per_m3}}之间。") == []
    assert errs("各厂站吨水曝气电耗在{{plants.kwh_per_m3.min}}至{{plants.kwh_per_m3.max}}之间。") == []
    # “均为”：值不同的厂站要写进同一句的“除……外”（编码、全名、简称都可以）
    assert errs("其余各镇站达标率除黄官镇外均为{{P04.compliance}}。") == []
    assert errs("除P06外各厂站达标率都是{{plants.compliance.max}}。") == []
    [e] = errs("黄官镇告警较多；各厂站达标率均为{{P04.compliance}}。")
    assert "黄官镇污水处理站的出水达标率不同" in e
    # 最高/最低后面紧跟的厂站须确实是最值；本厂站时段内的最高值不算跨厂站比较
    assert errs("黄官镇污水处理站吨水曝气电耗最低，为{{P06.kwh_per_m3}}。") == []
    [e] = errs("高台镇吨水曝气电耗最低，为{{P04.kwh_per_m3}}。")
    assert "最低的是黄官镇污水处理站" in e and "{{plants.kwh_per_m3.min}}" in e
    assert errs("黄官镇出水氨氮最高{{P06.nh3n_out.max}}。") == []
    assert errs("黄官镇达标率{{P06.compliance}}，低于其他厂站。") == []
    [e] = errs("高台镇吨水曝气电耗{{P04.kwh_per_m3}}，低于其余厂站。")
    assert "黄官镇污水处理站的吨水曝气电耗并不更高" in e
    # 替换后的值已带单位（“3 座”），模型再写一遍会变成“3 座座”
    facts["plants.compliance.full_n"] = Fact("plants.compliance.full_n", "", "3 座", 3)
    [e] = errs("其中{{plants.compliance.full_n}}座厂站全时段达标。")
    assert "后面又写了单位“座”" in e
    assert errs("达标率{{P06.compliance}} %。")[0].endswith("后面又写了单位“%”，替换后的值已带单位")
    assert errs("其中{{plants.compliance.full_n}}厂站全时段达标。") == []


def test_replay_saved_failed_attempt():
    from evals.report.replay import replay
    facts = {"n": Fact("n", "厂站数", "5 座", 5)}
    # #6 首次失败的形态：现在由容错删去重复单位后通过
    assert replay('{"overview":["达标厂站{{n}}座"],"consumption":["无"]}', facts, {}) == {
        "errors": [], "fixes": ["overview：删去{{n}}后重复的单位“座”"]}
    assert replay("不是JSON", facts, {})["errors"]

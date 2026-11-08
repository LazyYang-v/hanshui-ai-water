"""对话里的派单入口：范围提取（纯函数）与端到端（对话 → 草稿 → 确认派单 → 记回会话）。

全程不调用外部模型：Agent 用假模型，派单入口由服务端规则查告警表得到。
"""

import json
from datetime import datetime, timedelta

from app.ai.agent.evidence import conversation
from app.ai.agent.prompts import ANSWER_SYSTEM, PLAN_SYSTEM
from app.api.alarms import build_draft
from app.deps import LOCAL_TZ
from app.services.chat_actions import relevance, scope_from_result, turn_covers
from tests.test_ops_api import ops  # noqa: F401 —— 复用“清表 + 扫描 48 小时”的夹具

PLANTS = [{"code": f"P0{i}", "name": n} for i, n in enumerate(
    ["南郑中心污水处理厂", "协税镇污水处理站", "梁山镇污水处理站", "高台镇污水处理站", "新集镇污水处理站",
     "黄官镇污水处理站", "牟家坝镇污水处理站"], start=1)]
NOW = datetime(2026, 11, 7, 20, 0, tzinfo=LOCAL_TZ)


def stats_call(codes, start="2026-11-06T00:00:00+08:00", end="2026-11-07T00:00:00+08:00"):
    return {"id": "t1", "tool": "plant_statistics", "question": "出水氨氮超标统计", "status": "ok",
            "output": {"plants": [{"code": c} for c in codes], "period": {"start": start, "end": end}}}


# ---------------------------------------------------------------- 范围提取

def test_scope_from_statistics_output_and_resolved_question():
    result = {"question": "黄官镇昨天氨氮超标了吗", "resolved_question": "黄官镇（P06）昨天出水氨氮是否超标",
              "tool_calls": [stats_call(["P06"])]}
    s = scope_from_result(result, PLANTS, NOW)
    assert s.plants == {"P06"} and s.alarm_ids == []
    assert s.start.isoformat() == "2026-11-06T00:00:00+08:00" and s.end.isoformat() == "2026-11-07T00:00:00+08:00"
    assert "出水氨氮" in s.words


def test_scope_ignores_all_plant_rankings_and_falls_back_to_text_or_default_window():
    ranking = {"question": "各厂氨氮排名", "resolved_question": "各厂站昨天出水氨氮排名",
               "tool_calls": [stats_call([p["code"] for p in PLANTS])]}
    assert scope_from_result(ranking, PLANTS, NOW).plants == set()

    text_only = {"question": "高台镇那会儿 DO 怎么样", "tool_calls": [],
                 "resolved_question": "高台镇（P04）2026-11-07 18:00至2026-11-07 20:00好氧区DO变化"}
    s = scope_from_result(text_only, PLANTS, NOW)
    assert s.plants == {"P04"} and s.start.hour == 18 and s.end.hour == 20

    s = scope_from_result({"question": "南郑现在怎么样", "resolved_question": "南郑中心污水处理厂当前运行状态"},
                          PLANTS, NOW)
    assert s.plants == {"P01"} and s.end == NOW and s.start == NOW - timedelta(hours=24)


def test_datetime_forms_seen_in_live_planner_output():
    # Flash 实测把“最近6小时”补全成中文日期、结束时刻省略日期
    s = scope_from_result({"question": "高台镇最近6小时好氧区DO怎么样？",
                           "resolved_question": "高台镇2026年11月8日10:19:30至16:19:30最近6小时好氧区DO（1#、2#）变化情况"},
                          PLANTS, NOW)
    assert s.start.isoformat() == "2026-11-08T10:19:30+08:00" and s.end.isoformat() == "2026-11-08T16:19:30+08:00"
    # 时间只写在工具子问题里（ISO 带时区）
    s = scope_from_result({"question": "q", "resolved_question": "高台镇好氧区DO",
                           "tool_calls": [{"tool": "sql_query", "question": "查询高台镇2026-11-08T02:19:30Z至2026-11-08T08:19:30Z的DO"}]},
                          PLANTS, NOW)
    assert s.start.hour == 10 and s.end.hour == 16
    # 跨午夜；比例、池号不当成时刻
    from app.services.chat_actions import datetimes
    assert [t.day for t in datetimes("2026-11-07 23:00至01:00")] == [7, 8]
    assert datetimes("回流比 1:2，1#池 12:30") == []
    # 实测写法“18:00-20:00”：“-20:00”是结束时刻，不是时区
    assert [t.isoformat() for t in datetimes("高台镇2026-11-07 18:00-20:00好氧区DO")] == [
        "2026-11-07T18:00:00+08:00", "2026-11-07T20:00:00+08:00"]


def test_explicit_alarm_ids_only_in_clear_forms():
    def ids(q):
        return scope_from_result({"question": q, "resolved_question": ""}, PLANTS, NOW).alarm_ids
    assert ids("给 #25 派单") == [25]
    assert ids("告警25怎么处理，和告警编号 26 有关吗") == [25, 26]
    assert ids("昨天有几条告警 2 条，好氧区DO（1#）和2#鼓风机") == []


def test_relevance_prefers_alarms_on_the_asked_metric():
    words = "高台镇（P04）好氧区DO为什么低"
    assert relevance({"metric": "do_aerobic_1"}, words) == 2          # 标签“好氧区 DO（1#）”去掉池号后匹配
    assert relevance({"metric": "nh3n_out"}, words) == 0
    assert relevance({"metric": "vibration_mms", "device_id": 3}, "鼓风机振动") >= 1


def test_history_carries_user_actions_only_when_present():
    turn = {"question": "q", "now": "n", "answer": "a"}
    assert "user_actions_after_answer" not in conversation([turn])[0]
    turn["followups"] = [{"text": "用户已人工确认派单：工单 #1（告警 #2，负责人 张工）"}]
    assert conversation([turn])[0]["user_actions_after_answer"] == [turn["followups"][0]["text"]]


# ---------------------------------------------------------------- 端到端

def scoped_responder(resolved: str, seen: list):
    """规划时记下 history，便于检查派单结果是否传给了追问。"""
    def respond(messages):
        system = messages[0]["content"]
        if system == PLAN_SYSTEM:
            payload = json.loads(messages[1]["content"])
            seen.append(payload["history"])
            if payload["observations"]:
                return json.dumps({"intent": "data", "resolved_question": resolved, "action": "answer",
                                   "calls": [], "reason": ""})
            return json.dumps({"intent": "data", "resolved_question": resolved, "action": "tools",
                               "calls": [{"tool": "sql_query", "question": resolved}], "reason": ""})
        if system == ANSWER_SYSTEM:
            payload = json.loads(messages[1]["content"])
            value = payload["observations"][0]["rows"][0]["values"][0]
            return json.dumps({"answerable": True, "claims": [{"kind": "fact", "text": "共7座模拟厂站。",
                "data_refs": [{"call_id": "t1", "row": 0, "column": 0, "value": value}], "knowledge_refs": []}],
                "reason": ""})
        return '{"sql":"SELECT count(*) FROM plant","explanation":"统计模拟厂站"}'
    return respond


def test_chat_offers_related_alarm_then_dispatch_is_recorded(ops, fake_llm):  # noqa: F811
    alarms = ops.get("/api/alarms", params={"limit": 200, "status": "未关闭"}).json()["items"]
    target = next(a for a in alarms if a["metric"] and not a["device"])
    ts = datetime.fromisoformat(target["ts"]).astimezone(LOCAL_TZ)
    window = f"{(ts - timedelta(hours=2)):%Y-%m-%d %H:00}至{(ts + timedelta(hours=1)):%Y-%m-%d %H:00}"
    plant = target["plant"]
    resolved = f"{plant['name']}（{plant['code']}）{window}{target['metric_label']}变化"
    seen: list = []
    fake_llm.responder = scoped_responder(resolved, seen)

    body = ops.post("/api/ai/chat", json={"question": "那段时间怎么回事？"}).json()
    assert body["ok"]
    actions = body["actions"]
    assert 1 <= len(actions) <= 3 and all(a["plant_code"] == plant["code"] for a in actions)
    first = actions[0]
    assert first["type"] == "work_order" and first["matched_by"] == "scope" and first["open_work_order_id"] is None
    # 问到的指标排第一：首条与目标告警同指标（同指标多条时按级别、时间排序）
    assert any(a["alarm_id"] == target["id"] for a in actions)
    assert first["metric_label"] == target["metric_label"] and relevance(first, resolved) == 2
    calls_before = len(fake_llm.calls)

    # 草稿与确认派单仍走原接口；带上对话轮次后，结果记回会话。
    # 该轮回答没有核查建议，草稿退回模板（有建议时取建议，见 test_draft_prefers_chat_recommendations）
    chat = {"session_id": body["session_id"], "run_id": body["run_id"]}
    draft = ops.post(f"/api/alarms/{first['alarm_id']}/work-order-draft", json={"chat": chat}).json()
    assert draft["source"]["kind"] == "template"
    wo = ops.post("/api/work-orders", json={"draft_id": draft["draft_id"], "assignee": "张工", "chat": chat})
    assert wo.status_code == 201, wo.text
    assert wo.json()["chat_recorded"] is True
    assert len(fake_llm.calls) == calls_before                      # 派单全程不调用模型
    turn = ops.get(f"/api/ai/chat/sessions/{body['session_id']}").json()["turns"][-1]
    assert turn["actions"] == actions
    assert turn["followups"][0]["work_order_id"] == wo.json()["id"]

    # 追问时 Agent 的 history 里能看到这次人工派单；新一轮的入口显示已有工单
    second = ops.post("/api/ai/chat", json={"question": "那现在呢？", "session_id": body["session_id"]}).json()
    assert seen[-1][0]["user_actions_after_answer"][0].startswith(f"用户已人工确认派单：工单 #{wo.json()['id']}")
    again = next(a for a in second["actions"] if a["alarm_id"] == first["alarm_id"])
    assert again["open_work_order_id"] == wo.json()["id"]


def test_dispatch_with_unrelated_chat_ref_is_not_recorded(ops, fake_llm):  # noqa: F811
    fake_llm.responder = scoped_responder("全区厂站数量", [])
    body = ops.post("/api/ai/chat", json={"question": "一共有多少个厂站？"}).json()
    assert body["ok"] and body["actions"] == []                      # 没有厂站范围：不给入口
    alarm = ops.get("/api/alarms", params={"limit": 1, "status": "未关闭"}).json()["items"][0]
    chat = {"session_id": body["session_id"], "run_id": body["run_id"]}
    draft = ops.post(f"/api/alarms/{alarm['id']}/work-order-draft", json={"chat": chat}).json()
    assert draft["source"]["kind"] == "template"                     # 该轮没给过这条告警的入口，不用该轮内容
    assert ops.post(f"/api/alarms/{alarm['id']}/work-order-draft",
                    json={"chat": {"session_id": "0" * 32, "run_id": "0" * 32}}).status_code == 200   # 会话不存在也照常出草稿
    wo = ops.post("/api/work-orders", json={"draft_id": draft["draft_id"], "assignee": "李工", "chat": chat})
    # 工单照常提交，但该轮没给过这条告警的入口，不往会话里写
    assert wo.status_code == 201 and wo.json()["chat_recorded"] is False
    assert "followups" not in ops.get(f"/api/ai/chat/sessions/{body['session_id']}").json()["turns"][-1]


def test_draft_prefers_chat_recommendations():
    alarm = {"id": 44, "plant_id": 3, "plant_name": "胡家营镇污水处理站", "device_code": None, "kind": "水质超标",
             "metric": "nh3n_out", "advice": None}
    turn = {"ok": True, "session_id": "s" * 32, "run_id": "r" * 32, "question": "3 号厂氨氮为什么升高？",
            "now": "2026-11-08T10:00:00+08:00", "models": ["deepseek-flash"],
            "claims": [{"kind": "hypothesis", "text": "可能是大流量导致污泥流失（待核查）", "citation_numbers": [1, 2]},
                       {"kind": "recommendation", "text": "核查二沉池泥位与出水SS", "citation_numbers": [2, 3]}],
            "citations": [{"number": 1, "kind": "data", "call_id": "t1", "row": 0, "column_name": "进水流量", "value": 9},
                          {"number": 2, "kind": "knowledge", "title": "运行手册", "heading": "污泥流失", "quote": "泥位…"},
                          {"number": 3, "kind": "statistic", "call_id": "t2", "fact_id": "P03.nh3n_out.max",
                           "value": 6.1, "unit": "mg/L", "text": "长文本不进草稿"}]}
    now = datetime(2026, 11, 8, 10, 5, tzinfo=LOCAL_TZ)
    draft = build_draft(alarm, now, turn)
    assert draft["source"]["kind"] == "chat_answer" and draft["source"]["run_id"] == "r" * 32
    assert "问的正是本告警的指标" in draft["source"]["note"]
    first, last = draft["steps"][0], draft["steps"][-1]
    assert first["text"] == "核查二沉池泥位与出水SS" and first["source"] == "ai"     # 只取核查建议，不取原因假设
    assert [c["kind"] for c in first["citations"]] == ["knowledge", "statistic"]
    assert first["citations"][1] == {"kind": "statistic", "call_id": "t2", "fact_id": "P03.nh3n_out.max",
                                     "value": 6.1, "unit": "mg/L"}
    assert len(draft["steps"]) == 2 and last["source"] == "template"
    # 该轮拒答或没有核查建议：退回已保存的处置建议，再退回模板
    advice = {"ok": True, "generated_at": "g", "models": ["m"], "citations": [],
              "claims": [{"kind": "recommendation", "text": "核查曝气", "citation_numbers": []}]}
    assert build_draft({**alarm, "advice": advice}, now, {**turn, "ok": False})["source"]["kind"] == "ai_advice"
    assert build_draft(alarm, now, {**turn, "claims": turn["claims"][:1]})["source"]["kind"] == "template"
    # 模板里嵌入西文结尾的指标名时补空格
    ss = build_draft({**alarm, "metric": "ss_out"}, now)["steps"]
    assert ss[0]["text"].startswith("复核出水 SS 在线仪表读数") and "跟踪出水 SS 恢复情况" in ss[-1]["text"]


def test_chat_recommendations_only_for_the_asked_metric():
    # 线上走查的那一轮：问的是 3 号厂出水氨氮，工具子问题顺带查了出水 SS，入口里有 SS 告警 #44、#37 和氨氮预警 #38
    resolved = "3号厂（P03）2026年11月8日出水氨氮升高原因排查：核查同时段水质、DO、进水量、曝气风量"
    result = {"question": "3 号厂氨氮为什么升高？", "resolved_question": resolved,
              "tool_calls": [{"tool": "sql_query", "question": "P03 按小时的MLSS、出水SS、进水COD"}]}
    scope = scope_from_result(result, [], datetime(2026, 11, 8, 19, tzinfo=LOCAL_TZ))
    ss, nh3n = {"metric": "ss_out"}, {"metric": "nh3n_out"}
    assert relevance(ss, scope.words) == 2 and relevance(ss, scope.asked) == 0      # 排序时问到的氨氮排在 SS 前
    assert relevance(nh3n, scope.asked) == 2
    turn = {"resolved_question": resolved, "actions": [
        {"alarm_id": 44, "metric": "ss_out", "matched_by": "scope"},
        {"alarm_id": 38, "metric": "nh3n_out", "matched_by": "scope"},
        {"alarm_id": 25, "metric": "ss_out", "matched_by": "explicit"}]}
    assert turn_covers(turn, 38) and not turn_covers(turn, 44)
    assert turn_covers(turn, 25)                 # 用户点名的告警
    assert not turn_covers(turn, 99)             # 该轮没给过入口


def test_explicit_alarm_id_offered_even_when_agent_refuses(ops, fake_llm):  # noqa: F811
    alarm = ops.get("/api/alarms", params={"limit": 1, "status": "未关闭"}).json()["items"][0]

    def refuse(messages):
        if messages[0]["content"] == PLAN_SYSTEM:
            return json.dumps({"intent": "unsupported", "resolved_question": "派单", "action": "refuse",
                               "calls": [], "reason": "只读助手不能派单"})
        raise AssertionError("拒答不应继续调用模型")
    fake_llm.responder = refuse
    body = ops.post("/api/ai/chat", json={"question": f"帮我给告警 #{alarm['id']} 派单"}).json()
    assert body["refused"] or not body["ok"]
    assert [a["alarm_id"] for a in body["actions"]] == [alarm["id"]]
    assert body["actions"][0]["matched_by"] == "explicit"


def test_scope_actions_still_offered_when_answer_lacks_evidence(ops, fake_llm):  # noqa: F811
    """Flash 实测：诊断问题的查数超时，回答如实拒答；问到的厂站时段里的告警仍应给入口。"""
    alarm = next(a for a in ops.get("/api/alarms", params={"limit": 200, "status": "未关闭"}).json()["items"]
                 if a["metric"])
    ts = datetime.fromisoformat(alarm["ts"]).astimezone(LOCAL_TZ)
    resolved = f"{alarm['plant']['name']}{ts:%Y-%m-%d %H:00}-{(ts + timedelta(hours=1)):%H:00}{alarm['metric_label']}偏低原因诊断"

    def respond(messages):
        system = messages[0]["content"]
        if system == PLAN_SYSTEM:
            if json.loads(messages[1]["content"])["observations"]:
                return json.dumps({"intent": "diagnosis", "resolved_question": resolved, "action": "answer",
                                   "calls": [], "reason": ""})
            return json.dumps({"intent": "diagnosis", "resolved_question": resolved, "action": "tools",
                               "calls": [{"tool": "sql_query", "question": resolved}], "reason": ""})
        if system == ANSWER_SYSTEM:
            return json.dumps({"answerable": False, "claims": [], "reason": "未获得现场读数，无法判断原因"})
        return '{"sql":"SELECT 1 WHERE false","explanation":"空结果"}'
    fake_llm.responder = respond
    body = ops.post("/api/ai/chat", json={"question": "那会儿为什么低？"}).json()
    assert body["refused"] and not body["ok"]
    assert alarm["id"] in [a["alarm_id"] for a in body["actions"]]
    assert all(a["matched_by"] == "scope" for a in body["actions"])

"""新负荷规则经真实 API 扫描、展示、取证与工单草稿，模型使用假响应。"""

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg
import pytest

from app.domain.metrics import DERIVED_METRICS, LOAD_SOURCES, PLANT_METRICS

T0 = datetime(2028, 2, 8, 8, tzinfo=ZoneInfo("Asia/Shanghai"))


@pytest.fixture
def load_alarm(client, db_url, sync_redis):
    end = T0 + timedelta(minutes=30)
    sync_redis.set("sim:clock", end.isoformat())
    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute("TRUNCATE alarm, work_order, report RESTART IDENTITY CASCADE")
        # 每个历史完整小时有有效采样；负荷基线40kg/h，浓度基线40mg/L。
        for i in range(7 * 24):
            t = T0 - timedelta(days=7) + timedelta(hours=i)
            conn.execute("INSERT INTO plant_reading(ts,plant_id,inflow_m3h,cod_in) VALUES (%s,1,1000,40)", (t,))
        # 反向变化样本：负荷100kg/h；均值相乘会错成302.5kg/h。
        for i in range(6):
            t = T0 + timedelta(minutes=5 * i)
            if i < 3:
                conn.execute("INSERT INTO plant_reading(ts,plant_id,inflow_m3h,cod_in) VALUES (%s,1,100,1000),(%s,1,1000,100)",
                             (t, t + timedelta(minutes=1)))
            else:
                conn.execute("INSERT INTO plant_reading(ts,plant_id,inflow_m3h,cod_in) VALUES (%s,1,1000,40)", (t,))
    try:
        result = client.post("/api/alarms/scan", params={"hours": 3})
        assert result.status_code == 200, result.text
        [alarm] = client.get("/api/alarms", params={"kind": "进水负荷"}).json()["items"]
        yield client, alarm
    finally:
        with psycopg.connect(db_url, autocommit=True) as conn:
            conn.execute("TRUNCATE alarm, work_order, report RESTART IDENTITY CASCADE")
            conn.execute("DELETE FROM plant_reading WHERE ts >= %s AND ts < %s", (T0 - timedelta(days=7), end))


def test_load_catalog_is_separate_from_raw_columns(client):
    assert set(DERIVED_METRICS) == set(LOAD_SOURCES)
    assert not set(DERIVED_METRICS) & set(PLANT_METRICS)
    body = client.get("/api/metrics").json()
    assert {m["key"] for m in body["derived"]} == set(DERIVED_METRICS)
    assert all(m["unit"] == "kg/h" for m in body["derived"])
    assert next(m for m in body["plant"] if m["key"] == "cod_in")["unit"] == "mg/L"


def test_load_alarm_units_and_draft_need_confirmation(load_alarm):
    client, alarm = load_alarm
    assert alarm["metric"] == "cod_load_kgh" and alarm["unit"] == "kg/h"
    assert alarm["value"] == alarm["peak"] == 100 and alarm["threshold"] == 60
    assert alarm["ended_at"] == (T0 + timedelta(minutes=15)).isoformat()
    assert alarm["detail"]["source_metric"] == "cod_in"
    assert alarm["detail"]["detected_at"] == (T0 + timedelta(minutes=30)).isoformat()
    before = client.get("/api/work-orders").json()["total"]
    draft = client.post(f"/api/alarms/{alarm['id']}/work-order-draft").json()
    assert "质量负荷" in draft["title"] and draft["source"]["kind"] == "template"
    assert any("浓度" in s["text"] and "kg/h" in s["text"] for s in draft["steps"])
    assert client.get("/api/work-orders").json()["total"] == before
    again = client.post("/api/alarms/scan").json()
    assert again["inserted"] == again["updated"] == 0


def test_load_advice_has_paired_product_evidence(load_alarm, fake_llm):
    client, alarm = load_alarm

    def responder(messages):
        obs = json.loads(messages[1]["content"])["observations"]
        t1 = next(o for o in obs if o["call_id"] == "t1")
        # 使用第一个工艺列的真实单元格；完整取证 SQL 与负荷数值在下方独立检查。
        data = {"call_id": "t1", "row": 0, "column": 1, "value": t1["rows"][0]["values"][1]}
        rag = next(o for o in obs if o["tool"] == "rag_search")
        hit = rag["hits"][0]
        know = {"call_id": rag["call_id"], "chunk_id": hit["id"], "quote": hit["text"].strip()[:40]}
        return json.dumps({"answerable": True, "reason": "", "claims": [
            {"kind": "fact", "text": "进水流量见本轮证据", "data_refs": [data]},
            {"kind": "recommendation", "text": "复核进水流量和浓度并核查仪表", "knowledge_refs": [know]},
        ]}, ensure_ascii=False)

    fake_llm.responder = responder
    response = client.post(f"/api/alarms/{alarm['id']}/advice")
    assert response.status_code == 200, response.text
    advice = response.json()["advice"]
    assert advice["ok"] and advice["citations"]
    assert [(e["id"], e["tool"]) for e in advice["evidence"]] == [
        ("t1", "sql_query"), ("t2", "sql_query"), ("t3", "sql_query"), ("t4", "rag_search")]
    t1 = advice["evidence"][0]["output"]
    load_index = next(i for i, c in enumerate(t1["columns"]) if "时段均值(kg/h)" in c)
    assert [row[load_index] for row in t1["rows"]] == [40.0, 100.0, 40.0]
    assert '"cod_load_kgh"' not in t1["sql"]  # 不是数据库原始列
    assert "avg(inflow_m3h::float8" in t1["sql"]

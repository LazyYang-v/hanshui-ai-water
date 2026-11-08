"""固定统计 API、Agent 注册与 SSE 的真实测试库集成，不调用外部模型。"""

import json

import pytest
from psycopg.rows import dict_row, tuple_row
from psycopg_pool import AsyncConnectionPool

from app.ai.agent.prompts import ANSWER_SYSTEM, PLAN_SYSTEM
from app.ai.text2sql.service import execute_readonly
from app.analytics.statistics import PlantStatistics, StatisticsQuery
from tests.conftest import END

BODY = {"plant": "P06", "metric": "nh3n_out", "start": "2026-11-05T11:00:00+08:00",
        "end": END.isoformat(), "statistics": ["exceed_hours", "longest_exceed_hours", "instant_max", "bucket_mean_max"]}


@pytest.mark.parametrize("row_factory", [tuple_row, dict_row])
async def test_statistics_and_text2sql_reuse_one_connection_without_changing_rows(ro_url, row_factory):
    # 强制只有一个连接，复现真实复测中统计后的连接被Text2SQL拿到的场景。
    async with AsyncConnectionPool(ro_url, min_size=1, max_size=1,
                                   kwargs={"autocommit": True, "row_factory": row_factory}, open=False) as pool:
        result = await PlantStatistics(pool).query(StatisticsQuery.model_validate(BODY), END)
        assert result["facts"]
        async with pool.connection() as conn:
            assert conn.row_factory is row_factory
        columns, rows, truncated = await execute_readonly(pool, "SELECT 6.2::float8 AS 氨氮, 0.7::float8 AS DO")
        assert columns == ["氨氮", "do"] and rows == [[6.2, 0.7]] and not truncated
        async with pool.connection() as conn:
            assert conn.row_factory is row_factory


def test_statistics_api_works_without_model_and_exposes_coverage(client, fake_llm):
    fake_llm.available = False
    response = client.post("/api/ai/statistics", json=BODY)
    assert response.status_code == 200, response.text
    out = response.json()
    assert out["period"]["start"] == BODY["start"] and out["period"]["end"] == BODY["end"]
    assert out["metric"]["unit"] == "mg/L" and len(out["plants"]) == 1
    summary = out["plants"][0]["summary"]
    assert summary["expected_buckets"] == summary["observed_buckets"] == 12
    assert summary["coverage"] == 1 and len(out["facts"]) == 4
    assert all(f["aggregation"] and f["coverage"] for f in out["facts"])
    assert not fake_llm.calls


def test_statistics_api_rejects_invalid_fields_and_unfinished_future_period(client):
    for change in ({"metric": "nh3n_out;DROP TABLE plant"}, {"plant": "P08"}, {"sql": "SELECT 1"},
                   {"statistics": ["exceed_hours", "exceed_hours"]}, {"start": 123},
                   {"start": "2026-11-06T00:00:00+08:00", "end": "2026-11-07T00:00:00+08:00"}):
        response = client.post("/api/ai/statistics", json={**BODY, **change})
        assert response.status_code == 422, response.text
    response = client.post("/api/ai/statistics", json={**BODY, "start": "2020-01-01T00:00:00+08:00",
                                                     "end": "2020-01-01T01:00:00+08:00"})
    assert response.status_code == 200, response.text
    out = response.json()
    assert not out["facts"] and out["warnings"] and out["plants"][0]["summary"]["exceed_hours"] is None


def responder(messages):
    payload = json.loads(messages[1]["content"])
    if messages[0]["content"] == PLAN_SYSTEM:
        assert "plant_statistics" in payload["available_tools"]
        calls = [] if payload["observations"] else [
            {"tool": "plant_statistics", "question": "黄官镇指定小时氨氮统计", "parameters": BODY}]
        return json.dumps({"intent": "data", "resolved_question": "黄官镇2026年11月5日11至12时氨氮统计",
                           "action": "tools" if calls else "answer", "calls": calls, "reason": ""})
    assert messages[0]["content"] == ANSWER_SYSTEM
    obs = payload["observations"][0]
    return json.dumps({"answerable": True, "claims": [{"kind": "statistic", "text": "", "statistic_refs": [
        {"call_id": obs["call_id"], "fact_id": f["id"]} for f in obs["facts"][:2]]}], "reason": ""})


def test_agent_registered_statistics_and_sse_parameters_are_json_safe(client, fake_llm):
    fake_llm.responder = responder
    with client.stream("POST", "/api/ai/chat/stream", json={"question": "查黄官镇指定小时氨氮时长"}) as response:
        assert response.status_code == 200
        event, frames = None, []
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                frames.append((event, json.loads(line[6:])))
    assert frames[-1][0] == "done"
    result = frames[-1][1]
    assert result["ok"], result
    (call,) = result["tool_calls"]
    assert call["tool"] == "plant_statistics" and call["status"] == "ok"
    assert all(c["kind"] == "statistic" for c in result["citations"])
    # 引用里保留每条事实的完整文本；合成段落时相同的覆盖说明（“；有效N/N个5分钟桶”）只在句末写一次
    assert all(c["text"].split("；有效")[0] in result["answer"] and c["coverage"] for c in result["citations"])
    statistic_claims = [c for c in result["claims"] if c["kind"] == "statistic"]
    assert all(c["text"].count("；有效") == 1 for c in statistic_claims)
    start = next(data for name, data in frames if name == "tool_start")
    assert start["parameters"]["start"] == BODY["start"]
    assert all(message[0]["content"] in (PLAN_SYSTEM, ANSWER_SYSTEM) for message in fake_llm.calls)

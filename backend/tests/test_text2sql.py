"""Text2SQL 流程测试：假模型 + 测试库（只读账号）。验证检索、校验、执行、自修复和接口。"""

import json

import pytest
from psycopg_pool import AsyncConnectionPool

from app.ai.text2sql import ExampleIndex, Text2SQL, Text2SQLConfig
from app.ai.text2sql.schema import load_schema
from app.ai.text2sql.service import execute_readonly, nearest_sample_violation
from tests.conftest import END, FakeLLM


def answer(sql, explanation="ok"):
    return json.dumps({"sql": sql, "explanation": explanation}, ensure_ascii=False)


def scripted(*contents):
    """依次返回给定内容（最后一个重复）。"""
    items = list(contents)
    return lambda m: items.pop(0) if len(items) > 1 else items[0]


@pytest.fixture
async def ro_pool(ro_url):
    pool = AsyncConnectionPool(ro_url, min_size=1, max_size=2, open=False, kwargs={"autocommit": True})
    await pool.open(wait=True, timeout=10)
    yield pool
    await pool.close()


@pytest.fixture
async def schema(ro_pool):
    async with ro_pool.connection() as conn:
        return await load_schema(conn)


def make(llm, ro_pool, schema, **cfg):
    return Text2SQL(llm, ro_pool, schema, cfg=Text2SQLConfig(**cfg))


# ---------------------------------------------------------------- 检索与 schema

def test_example_retrieval_finds_similar_question():
    idx = ExampleIndex.load()
    top = idx.top("梁山镇昨天的碳源用了多少", 3)
    assert "碳源" in top[0]["q"]
    assert idx.top("火星探测器", 3) == []   # 毫不相关的问题不硬塞示例


def test_examples_are_valid_under_guard(schema):
    """示例库里的每条 SQL 都要能通过安全校验（否则就是在教模型写会被拒的 SQL）。"""
    from app.ai.text2sql.guard import guard
    for e in ExampleIndex.load().examples:
        guard(e["sql"], schema.names, END)


async def test_schema_from_comments(schema):
    assert set(schema.names) == {"plant", "plant_reading", "plant_reading_1h", "device", "device_reading",
                                 "weather"}
    text = schema.render(schema.select("昨天各厂的碳源投加量"))
    assert "carbon_dose_kgh" in text and "南郑中心污水处理厂" in text
    assert "device_reading" not in schema.select("昨天各厂的碳源投加量")
    assert "device" in schema.select("1号鼓风机的振动")


# ---------------------------------------------------------------- 执行

async def test_examples_execute_on_test_db(ro_pool, schema):
    """示例 SQL 在测试库上都能跑通（防止 schema 改了示例没跟着改）。"""
    from app.ai.text2sql.guard import guard
    for e in ExampleIndex.load().examples:
        await execute_readonly(ro_pool, guard(e["sql"], schema.names, END))


async def test_readonly_account_blocks_writes_even_without_guard(ro_pool):
    with pytest.raises(Exception, match="read-only|permission denied"):
        await execute_readonly(ro_pool, "DELETE FROM plant")


async def test_happy_path(ro_pool, schema):
    llm = FakeLLM(lambda m: answer(
        "SELECT p.name AS 厂站, round(sum(h.inflow_m3h)::numeric) AS 水量 FROM plant_reading_1h h "
        "JOIN plant p ON p.id = h.plant_id WHERE h.hour >= date_trunc('day', now()) GROUP BY p.id, p.name "
        "ORDER BY p.id"))
    r = await make(llm, ro_pool, schema).ask("今天各厂处理了多少水？", END)
    assert r.ok, r.error
    assert r.columns == ["厂站", "水量"] and len(r.rows) == 7
    assert r.rows[0][0] == "南郑中心污水处理厂" and r.rows[0][1] > 10000   # 中心厂半天 > 1 万方
    assert "2026-11-05T12:00:00+08:00" in r.sql and "LIMIT 200" in r.sql
    assert len(r.attempts) == 1 and r.prompt_tokens == 100


async def test_prompt_contains_schema_examples_and_clock(ro_pool, schema):
    llm = FakeLLM(lambda m: answer("SELECT 1 AS x"))
    await make(llm, ro_pool, schema).ask("昨天各厂碳源用了多少？", END)
    system = llm.calls[0][0]["content"]
    assert "2026-11-05 12:00:00+0800" in system
    assert "carbon_dose_kgh" in system and "## 示例" in system
    assert "device_reading (" not in system   # 已裁剪


async def test_ablation_switches(ro_pool, schema):
    llm = FakeLLM(lambda m: answer("SELECT 1 AS x"))
    await make(llm, ro_pool, schema, use_fewshot=False, prune_schema=False).ask("昨天各厂碳源用了多少？", END)
    system = llm.calls[0][0]["content"]
    assert "## 示例" not in system and "device_reading (" in system


async def test_self_repair_after_db_error(ro_pool, schema):
    llm = FakeLLM(scripted(answer("SELECT no_such_col FROM plant"),
                           answer("SELECT count(*) AS 厂站数 FROM plant")))
    r = await make(llm, ro_pool, schema).ask("一共有几个厂站？", END)
    assert r.ok and r.rows == [[7]]
    assert len(r.attempts) == 2 and "no_such_col" in r.attempts[0].error
    # 第二轮把报错反馈给了模型
    assert "执行报错" in llm.calls[1][-1]["content"]


async def test_self_repair_after_guard_rejection(ro_pool, schema):
    llm = FakeLLM(scripted(answer("SELECT * FROM alarm"), answer("SELECT count(*) AS n FROM device")))
    r = await make(llm, ro_pool, schema).ask("有多少设备？", END)
    assert r.ok and r.rows == [[57]]
    assert "安全校验" in r.attempts[0].error


NEAREST = ("SELECT r.ts AS 采样时间, r.nh3n_out AS 氨氮 FROM plant_reading r JOIN plant p ON p.id = r.plant_id "
           "WHERE p.code = 'P06' ORDER BY ABS(EXTRACT(EPOCH FROM (r.ts - TIMESTAMPTZ '2026-11-06 18:00+08'))) LIMIT 1")


def test_nearest_sample_violation_only_when_question_names_a_moment():
    assert nearest_sample_violation(NEAREST, "黄官镇11月6日18:00的原始出水氨氮是多少？")
    assert nearest_sample_violation(NEAREST, "最近7天黄官镇18:00的氨氮") is not None  # “最近N天”不放行
    assert nearest_sample_violation(NEAREST, "离11月6日18:00最接近的一条氨氮采样") is None
    assert nearest_sample_violation(NEAREST, "18:00前后10分钟内的采样") is None
    exact = "SELECT r.ts, r.nh3n_out FROM plant_reading r WHERE r.ts = '2026-11-06 18:00+08' ORDER BY r.ts DESC LIMIT 1"
    assert nearest_sample_violation(exact, "11月6日18:00的氨氮") is None


async def test_nearest_sample_is_sent_back_for_repair(ro_pool, schema):
    llm = FakeLLM(scripted(answer(NEAREST), answer("SELECT count(*) AS 厂站数 FROM plant")))
    r = await make(llm, ro_pool, schema).ask("黄官镇11月6日18:00的原始出水氨氮是多少？", END)
    assert r.ok and len(r.attempts) == 2
    assert "查数约定" in r.attempts[0].error
    assert "等值" in llm.calls[1][-1]["content"]


async def test_gives_up_after_max_repairs(ro_pool, schema):
    llm = FakeLLM(lambda m: "这不是 JSON")
    r = await make(llm, ro_pool, schema, max_repairs=1).ask("一共有几个厂站？", END)
    assert not r.ok and len(llm.calls) == 2 and "JSON" in r.error


async def test_unanswerable(ro_pool, schema):
    llm = FakeLLM(lambda m: answer("", "数据库里没有员工工资数据"))
    r = await make(llm, ro_pool, schema).ask("张三这个月工资多少？", END)
    assert r.refused and not r.ok and r.sql is None and "工资" in r.error and len(llm.calls) == 1


@pytest.mark.parametrize("content", ['{}', '[]', 'null', '{"sql": null}', 'not json',
                                      '{"sql": 3, "explanation": "x"}'])
async def test_malformed_output_is_an_error_not_a_refusal(content):
    from app.ai.text2sql.schema import SchemaInfo
    llm = FakeLLM(lambda m: content)
    # 不会进入数据库执行阶段；确保错误类型不被判成正确拒答。
    res = await make(llm, None, SchemaInfo({}, []), max_repairs=0).ask("下周会下雨吗？", END)
    assert not res.ok and not res.refused and "JSON" in res.error


async def test_statement_timeout(ro_pool, schema):
    llm = FakeLLM(lambda m: answer("SELECT count(*) FROM plant_reading a, plant_reading b"))
    r = await make(llm, ro_pool, schema, max_repairs=0).ask("笛卡尔积", END)
    assert not r.ok and ("timeout" in r.error.lower() or "取消" in r.error), r.error


# ---------------------------------------------------------------- 接口

def test_text2sql_endpoint(client, fake_llm):
    r = client.post("/api/ai/text2sql", json={"question": "一共有几个厂站？"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and body["rows"] == [[7]] and body["columns"] == ["n"]


def test_text2sql_endpoint_validation(client):
    assert client.post("/api/ai/text2sql", json={"question": ""}).status_code == 422
    assert client.post("/api/ai/text2sql", json={"question": "   "}).status_code == 422


def test_gateway_endpoint(client):
    r = client.get("/api/ai/gateway")
    assert r.status_code == 200 and r.json()["available"] is True


class _Truncating:
    """前 n 次像 Flash 想满 token 上限：finish_reason=length、正文为空；记录每次的 thinking 参数。"""
    def __init__(self, n, delay_s=0.0):
        self.n, self.delay_s, self.calls = n, delay_s, []

    async def chat(self, messages, **kw):
        import asyncio

        from app.ai.gateway import ChatResult
        self.calls.append({"messages": messages, "thinking": kw.get("thinking")})
        await asyncio.sleep(self.delay_s)
        if len(self.calls) <= self.n:
            return ChatResult("", "fake", "fake-model", 100, 4000, latency_ms=1.0, finish_reason="length")
        return ChatResult('{"sql": "SELECT count(*) AS n FROM plant", "explanation": "数厂站"}', "fake", "fake-model",
                          100, 20, latency_ms=1.0, finish_reason="stop")


async def test_truncated_output_is_repaired_without_thinking(ro_pool, schema):
    llm = _Truncating(1)
    r = await make(llm, ro_pool, schema).ask("一共有几个厂站？", END)
    assert r.ok and r.rows == [[7]]
    # 首次沿用厂商默认；截断后明确告诉模型原因，并关闭思考
    assert [c["thinking"] for c in llm.calls] == [None, False]
    assert "截断" in llm.calls[1]["messages"][-1]["content"]
    assert "截断" in r.attempts[0].error and r.generation_calls[0]["finish_reason"] == "length"


async def test_stops_repairing_when_time_budget_is_spent(ro_pool, schema):
    llm = _Truncating(5, delay_s=0.05)
    r = await make(llm, ro_pool, schema, time_budget_s=0.08, min_attempt_s=0.0).ask("一共有几个厂站？", END)
    assert not r.ok and len(llm.calls) == 2 and r.error.startswith("查数生成耗时过长，已停止修复")

"""Agent 图、工具白名单、证据核验、连续对话与SSE；所有LLM均为假模型。"""

import asyncio
import json
from dataclasses import replace

import pytest

from app.ai.agent import Agent, AgentConfig
from app.ai.agent.evidence import observations, validate_answer
from app.ai.agent.models import AgentResult, ToolRecord
from app.ai.agent.prompts import ANSWER_SYSTEM, PLAN_SYSTEM
from app.ai.gateway import LLMError
from app.ai.rag.corpus import Chunk
from app.ai.rag.retrieval import Hit, SearchResult
from app.ai.text2sql.service import Text2SQLResult
from tests.conftest import END, FakeLLM

QUESTION = "黄官镇昨天为什么氨氮超标？"
TEXT = "鼓风机跳停后供气下降，溶解氧不足可能影响硝化，需核查设备和水质变化的先后关系。"
CHUNK = Chunk("air:check:1", "air", "风机排查", "核查", TEXT, "demo", "自编演示资料", None, None, 3, 4)


def decision(calls=(), *, action=None, intent="diagnosis", question=QUESTION, reason=""):
    return {"intent": intent, "resolved_question": question, "action": action or ("tools" if calls else "answer"),
            "calls": [{"tool": tool, "question": q} for tool, q in calls], "reason": reason}


def draft(*, data_id="t1", knowledge_id="t2", value=6.2, kind="hypothesis"):
    return {"answerable": True, "claims": [{"kind": kind, "text": "氨氮升高可能与供气不足有关，需核查时序。",
            "data_refs": [{"call_id": data_id, "row": 0, "column": 0, "value": value}],
            "knowledge_refs": [{"call_id": knowledge_id, "chunk_id": CHUNK.id, "quote": TEXT[:20]}]}], "reason": ""}


class SQL:
    def __init__(self, *, rows=None, fail=False):
        self.calls = []
        self.rows = [[6.2, 0.7]] if rows is None else rows
        self.fail = fail

    async def ask(self, question, now):
        self.calls.append((question, now))
        return Text2SQLResult(question, ok=not self.fail, sql="SELECT 6.2, 0.7", columns=["氨氮", "DO"],
                              rows=self.rows if not self.fail else [], error="数据库超时" if self.fail else None)


class Rag:
    async def search(self, *args, **kwargs):
        return SearchResult([Hit(CHUNK, 1, 1, None, 3)], "bm25", "hybrid", 1, [], reranked=True)


def scripted(*outputs):
    pending = iter(outputs)
    return FakeLLM(lambda m: json.dumps(next(pending), ensure_ascii=False))


async def test_graph_runs_tools_with_frozen_clock_and_verified_evidence():
    llm = scripted(decision([("sql_query", "黄官镇昨天氨氮和DO"), ("rag_search", "风机与氨氮的排查")]),
                   decision(), draft())
    sql, events = SQL(), []

    async def emit(name, data):
        events.append((name, data))

    result = await Agent(llm, sql, Rag()).run(QUESTION, END, emit=emit)
    assert result.ok and not result.error and not result.refused
    assert "原因假设（待核查）" in result.answer
    assert sql.calls == [("黄官镇昨天氨氮和DO", END)]
    assert result.now == END.isoformat() and result.models == ["fake-model"]
    assert [c["kind"] for c in result.citations] == ["data", "knowledge"]
    assert result.citations[0]["value"] == 6.2 and result.citations[1]["quote"] in TEXT
    names = [e[0] for e in events]
    assert names[:2] == ["run_start", "planning"] and names[-2:] == ["planning", "answer_start"]
    for rid in ("t1", "t2"):
        start = next(i for i, (n, d) in enumerate(events) if n == "tool_start" and d["id"] == rid)
        end = next(i for i, (n, d) in enumerate(events) if n == "tool_end" and d["id"] == rid)
        assert start < end < len(events) - 2
    assert result.tool_calls[0].output["sql"] and result.planning_rounds == 2


@pytest.mark.parametrize("bad_call", [
    {"tool": "create_work_order", "question": "建一个工单"},
    {"tool": "shell", "question": "读取环境文件"},
    {"tool": "sql_query", "question": "黄官镇氨氮", "sql": "DELETE FROM plant"},
])
async def test_unregistered_tools_and_extra_parameters_never_execute(bad_call):
    bad = {**decision(), "action": "tools", "calls": [bad_call]}
    sql = SQL()
    result = await Agent(scripted(bad, bad), sql, Rag()).run(QUESTION, END)
    assert result.error and not result.ok and not result.refused
    assert not sql.calls and not result.tool_calls and len(result.validation_errors) == 2


async def test_duplicate_calls_fail_before_execution():
    plan = decision([("sql_query", "相同问题"), ("sql_query", "相同问题")])
    sql = SQL()
    result = await Agent(scripted(plan, plan), sql, Rag()).run(QUESTION, END)
    assert result.error and not sql.calls


async def test_step_budget_is_enforced_without_extra_calls():
    # 去掉知识引用，仅验证工具额度命中后直接综合已有数据。
    llm = scripted(decision([("sql_query", "昨天的氨氮")]),
                   {"answerable": True, "claims": [{**draft(kind="fact")["claims"][0], "knowledge_refs": []}], "reason": ""})
    result = await Agent(llm, SQL(), Rag(), AgentConfig(max_tools=1)).run(QUESTION, END)
    assert result.ok and len(result.tool_calls) == 1 and result.planning_rounds == 1
    assert any("上限" in w for w in result.warnings)


async def test_tool_error_may_trigger_one_bounded_followup():
    class FlakySQL(SQL):
        async def ask(self, question, now):
            self.fail = not self.calls
            return await super().ask(question, now)

    answer = draft(data_id="t2", knowledge_id="t3")
    llm = scripted(decision([("sql_query", "第一种查法")]),
                   decision([("sql_query", "缩小时间范围查数"), ("rag_search", "氨氮排查")]), answer)
    result = await Agent(llm, FlakySQL(), Rag()).run(QUESTION, END)
    assert result.ok and result.tool_calls[0].status == "error"
    assert [t.status for t in result.tool_calls[1:]] == ["ok", "ok"]
    assert result.warnings and len(llm.calls) == 3


async def test_followup_plan_cannot_rewrite_the_question():
    llm = scripted(decision([("sql_query", "昨天氨氮"), ("rag_search", "排查")]),
                   decision(question="改问另一个厂的电费", intent="knowledge"), draft())
    result = await Agent(llm, SQL(), Rag()).run(QUESTION, END)
    assert result.ok and not result.validation_errors
    assert result.intent == "diagnosis" and result.resolved_question == QUESTION


async def test_invalid_followup_plan_answers_from_existing_evidence():
    repeated = decision([("sql_query", "昨天氨氮")])  # 重复已执行的问题
    llm = scripted(decision([("sql_query", "昨天氨氮"), ("rag_search", "排查")]), repeated, repeated, draft())
    sql = SQL()
    result = await Agent(llm, sql, Rag()).run(QUESTION, END)
    assert result.ok and len(sql.calls) == 1 and len(result.tool_calls) == 2
    assert len(result.validation_errors) == 2 and any("补查规划" in w for w in result.warnings)


async def test_empty_rows_are_not_interpreted_as_zero_or_success():
    llm = scripted(decision([("sql_query", "昨天的氨氮")]), decision())
    result = await Agent(llm, SQL(rows=[]), Rag()).run(QUESTION, END)
    assert result.refused and not result.ok and not result.error and not result.citations
    assert result.tool_calls[0].status == "empty" and len(llm.calls) == 2


async def test_all_tool_failures_remain_errors_not_refusals():
    result = await Agent(scripted(decision([("sql_query", "查氨氮")]), decision()), SQL(fail=True), Rag()).run(QUESTION, END)
    assert result.error and not result.refused and not result.answer


async def test_bad_cell_is_repaired_with_specific_feedback():
    outputs = [decision([("sql_query", "昨天氨氮"), ("rag_search", "排查")]), decision(), draft(value=99), draft()]

    def respond(messages):
        if len(outputs) == 1:
            assert "单元格不一致" in messages[-1]["content"]
        return json.dumps(outputs.pop(0), ensure_ascii=False)

    result = await Agent(FakeLLM(respond), SQL(), Rag()).run(QUESTION, END)
    assert result.ok and result.citations[0]["value"] == 6.2
    assert result.validation_errors == ["answer: 数据引用值与实际单元格不一致"]


async def test_unverified_answer_is_never_emitted():
    llm = scripted(decision([("sql_query", "昨天氨氮"), ("rag_search", "排查")]), decision(), draft(value=99), draft(value=99))
    result = await Agent(llm, SQL(), Rag()).run(QUESTION, END)
    assert result.error and not result.ok and not result.answer and not result.citations and not result.claims


def evidence_result():
    return AgentResult(QUESTION, END.isoformat(), intent="diagnosis", tool_calls=[
        ToolRecord("t1", "sql_query", "昨天氨氮", "ok", {"sql": "SELECT 6.2", "columns": ["氨氮"], "rows": [[6.2]]}),
        ToolRecord("t2", "rag_search", "排查", "ok", {"hits": [Hit(CHUNK, 1, 1, None).to_dict()]}),
    ])


@pytest.mark.parametrize("mutation", ["wrong_call", "wrong_row", "wrong_column", "string_value", "wrong_chunk",
                                       "fake_quote", "hypothesis_without_data", "hypothesis_without_knowledge"])
def test_invalid_evidence_fails_closed(mutation):
    answer = draft()
    claim = answer["claims"][0]
    if mutation == "wrong_call":
        claim["data_refs"][0]["call_id"] = "old-run-t1"
    elif mutation == "wrong_row":
        claim["data_refs"][0]["row"] = 3
    elif mutation == "wrong_column":
        claim["data_refs"][0]["column"] = 2
    elif mutation == "string_value":
        claim["data_refs"][0]["value"] = "6.2"
    elif mutation == "wrong_chunk":
        claim["knowledge_refs"][0]["chunk_id"] = "not-retrieved"
    elif mutation == "fake_quote":
        claim["knowledge_refs"][0]["quote"] = "完全不存在的连续原文引用"
    elif mutation == "hypothesis_without_data":
        claim["data_refs"] = []
    else:
        claim["knowledge_refs"] = []
    with pytest.raises(ValueError):
        validate_answer(json.dumps(answer), evidence_result())


def test_only_visible_rows_can_be_cited():
    result = evidence_result()
    result.tool_calls[0].output["rows"] *= 50
    assert len(observations(result.tool_calls)[0]["rows"]) == 30
    answer = draft()
    answer["claims"][0]["data_refs"][0]["row"] = 35
    with pytest.raises(ValueError, match="行列"):
        validate_answer(json.dumps(answer), result)


async def test_history_resolves_references_but_old_values_are_not_evidence():
    history = [{"question": "黄官镇昨天的氨氮呢", "resolved_question": "黄官镇昨天的氨氮最大值", "now": END.isoformat(),
                "answer": "氨氮6.2", "ok": True, "tool_calls": [{"secret": "不能复用的旧工具原始数据"}]}]

    def respond(messages):
        payload = json.loads(messages[1]["content"])
        assert payload["history"][0]["resolved_question"].startswith("黄官镇")
        assert "tool_calls" not in payload["history"][0]
        return json.dumps(decision(action="answer", intent="data", question="黄官镇昨天的DO"))

    result = await Agent(FakeLLM(respond), SQL(), Rag()).run("那它的DO呢？", END, history=history)
    assert result.refused and not result.citations


async def test_run_deadline_cancels_inflight_tool():
    started, cancelled = asyncio.Event(), asyncio.Event()

    class WaitingSQL(SQL):
        async def ask(self, *args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    # 时限要覆盖冷启动和高负载下的规划开销；0.05秒曾在工具开始前就到期，测到的不是取消路径。
    result = await Agent(scripted(decision([("sql_query", "等待查询")])), WaitingSQL(), Rag(),
                         replace(AgentConfig(), timeout_s=1.0)).run(QUESTION, END)
    assert started.is_set() and cancelled.is_set() and "超过" in result.error and not result.answer
    assert result.tool_calls[0].status == "cancelled"


async def test_model_error_and_explicit_refusal_are_distinct():
    class BrokenLLM:
        async def chat(self, *args, **kwargs):
            raise LLMError("provider unavailable")

    result = await Agent(BrokenLLM(), SQL(), Rag()).run(QUESTION, END)
    assert result.error and not result.refused
    refusal = decision(action="refuse", intent="unsupported", reason="没有真实人员联系方式")
    result = await Agent(scripted(refusal), SQL(), Rag()).run("站长电话是多少", END)
    assert result.refused and not result.error and not result.tool_calls


def test_string_cell_citation_preserves_whitespace():
    result = evidence_result()
    result.tool_calls[0].output["rows"] = [[" 黄官镇 "]]
    answer = draft(value=" 黄官镇 ")
    _, _, citations, _ = validate_answer(json.dumps(answer), result)
    assert citations[0]["value"] == " 黄官镇 "


def test_model_written_label_is_not_duplicated_or_reclassified():
    answer = draft()
    answer["claims"][0]["text"] = "原因假设（待核查）：供氧不足可能削弱硝化。"
    text, claims, _, _ = validate_answer(json.dumps(answer, ensure_ascii=False), evidence_result())
    assert text.count("原因假设") == 1 and claims[0]["text"] == "供氧不足可能削弱硝化。"
    fact = draft(kind="fact")
    fact["claims"][0]["text"] = "原因假设：仍作为原文保留"  # 不能借去标签把事实/假设互换
    text, claims, _, _ = validate_answer(json.dumps(fact, ensure_ascii=False), evidence_result())
    assert claims[0]["text"] == "原因假设：仍作为原文保留"


def test_omitted_reference_lists_default_to_empty_without_relaxing_evidence():
    fact = draft(kind="fact")
    del fact["claims"][0]["knowledge_refs"], fact["reason"]
    _, _, citations, reason = validate_answer(json.dumps(fact, ensure_ascii=False), evidence_result())
    assert reason is None and [c["kind"] for c in citations] == ["data"]
    hypothesis = draft()
    del hypothesis["claims"][0]["knowledge_refs"]  # 省略后按空列表处理，原因假设仍缺知识出处
    with pytest.raises(ValueError, match="知识"):
        validate_answer(json.dumps(hypothesis, ensure_ascii=False), evidence_result())


def test_validation_feedback_names_the_limit_and_the_rule():
    from pydantic import ValidationError

    from app.ai.agent.models import Answer, Decision
    from app.ai.agent.service import error_detail

    many = draft(kind="fact")
    many["claims"] *= 7
    with pytest.raises(ValidationError) as exc:
        Answer.model_validate_json(json.dumps(many, ensure_ascii=False))
    assert error_detail(exc.value) == "claims: 最多6项，当前7项，请合并或删减"
    with pytest.raises(ValidationError) as exc:
        Decision.model_validate_json(json.dumps(decision(action="tools")))
    assert error_detail(exc.value) == "JSON: tools 必须带工具调用，answer/refuse 不得带调用"


async def test_too_many_claims_are_repaired_with_specific_feedback():
    many = draft(kind="fact")
    many["claims"] *= 7
    outputs = [decision([("sql_query", "昨天氨氮"), ("rag_search", "排查")]), decision(), many, draft()]

    def respond(messages):
        if len(outputs) == 1:
            assert "最多6项，当前7项" in messages[-1]["content"]
        return json.dumps(outputs.pop(0), ensure_ascii=False)

    result = await Agent(FakeLLM(respond), SQL(), Rag()).run(QUESTION, END)
    assert result.ok and len(result.claims) == 1
    assert result.validation_errors == ["answer: claims: 最多6项，当前7项，请合并或删减"]


def test_answerable_reason_is_ignored_not_displayed():
    answer = draft()
    answer["reason"] = "未附出处的补充说明"
    text, _, citations, reason = validate_answer(json.dumps(answer, ensure_ascii=False), evidence_result())
    assert reason is None and citations and "补充说明" not in text


async def test_answer_disables_thinking_and_repairs_length_stop():
    from app.ai.gateway import ChatResult

    class TruncatedLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            assert kwargs["max_tokens"] == 4000
            if messages[0]["content"] == ANSWER_SYSTEM:
                assert kwargs["thinking"] is False  # 推理不再挤占结构化回答的预算
                if len(messages) == 2:
                    return ChatResult("", "fake", "fake-model", completion_tokens=4000, finish_reason="length")
                assert "token上限" in messages[-1]["content"]
            else:
                # 首轮规划沿用厂商默认；补查和其修复关闭思考。
                payload = json.loads(messages[1]["content"])
                assert kwargs["thinking"] is (False if payload["observations"] else None)
            return await super().chat(messages, **kwargs)

    outputs = iter([decision([("sql_query", "昨天氨氮"), ("rag_search", "排查")]), decision(), draft()])
    llm = TruncatedLLM(lambda m: json.dumps(next(outputs)))
    result = await Agent(llm, SQL(), Rag()).run(QUESTION, END)
    assert result.ok and "token上限" in result.validation_errors[0]
    assert [c["finish_reason"] for c in result.generation_calls] == [None, None, "length", None]


async def test_first_plan_truncated_is_repaired_without_thinking():
    """Flash 实测首轮规划可能想满 4000 token：修复时关闭思考，其余修复仍沿用默认。"""
    from app.ai.gateway import ChatResult

    seen = []

    class TruncatedPlan(FakeLLM):
        async def chat(self, messages, **kwargs):
            if messages[0]["content"] == PLAN_SYSTEM and not json.loads(messages[1]["content"])["observations"]:
                seen.append(kwargs["thinking"])
                if len(messages) == 2:
                    return ChatResult("", "fake", "fake-model", completion_tokens=4000, finish_reason="length")
                assert "推理过长" in messages[-1]["content"]
            return await super().chat(messages, **kwargs)

    outputs = iter([decision([("sql_query", "昨天氨氮"), ("rag_search", "排查")]), decision(), draft()])
    llm = TruncatedPlan(lambda m: json.dumps(next(outputs)))
    result = await Agent(llm, SQL(), Rag()).run(QUESTION, END)
    assert result.ok and seen == [None, False] and "token上限" in result.validation_errors[0]


@pytest.mark.parametrize("disconnect", ["headers_error", "headers_disconnect", "body_disconnect"])
async def test_sse_disconnect_always_cancels_work_and_releases_lease(disconnect):
    from types import SimpleNamespace

    import fakeredis
    from starlette.requests import ClientDisconnect

    from app.ai.agent.sessions import SessionStore
    from app.api.chat import ChatRequest, stream

    started, cancelled, headers_sent, body_sent = (asyncio.Event() for _ in range(4))

    class WaitingAgent:
        gateway = SimpleNamespace(available=True)

        async def run(self, question, now, *, emit, **kwargs):
            started.set()
            try:
                await emit("run_start", {"now": now.isoformat()})
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    async with fakeredis.FakeAsyncRedis(decode_responses=True) as redis:
        store = SessionStore(redis)
        lease = await store.acquire()
        await store.release(lease)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(agent=WaitingAgent(), agent_sessions=store,
                                                                            pool=None)))
        response = await stream(ChatRequest(question="查询氨氮", session_id=lease.session_id), request, END)

        async def send(message):
            if message["type"] == "http.response.start":
                headers_sent.set()
                if disconnect == "headers_error":
                    raise OSError("connection closed")
                if disconnect == "headers_disconnect":
                    await asyncio.Event().wait()
            elif message["type"] == "http.response.body":
                body_sent.set()

        async def receive():
            await (body_sent if disconnect == "body_disconnect" else headers_sent).wait()
            return {"type": "http.disconnect"}

        scope = {"type": "http", "asgi": {"spec_version": "2.4" if disconnect == "headers_error" else "2.3"}}
        async with asyncio.timeout(2):
            if disconnect == "headers_error":
                with pytest.raises(ClientDisconnect):
                    await response(scope, receive, send)
            else:
                await response(scope, receive, send)
        assert not await redis.exists(f"ai:session:{lease.session_id}:lock")
        assert (await store.get(lease.session_id))["turns"] == []
        assert started.is_set() == (disconnect == "body_disconnect")
        assert cancelled.is_set() == started.is_set()


def api_responder(messages):
    system = messages[0]["content"]
    if system == PLAN_SYSTEM:
        payload = json.loads(messages[1]["content"])
        resolved = "各厂站总数量" if not payload["history"] else "同一范围的厂站数量"
        if payload["observations"]:
            return json.dumps(decision(intent="data", question=resolved))
        return json.dumps(decision([("sql_query", resolved)], intent="data", question=resolved))
    if system == ANSWER_SYSTEM:
        payload = json.loads(messages[1]["content"])
        row = payload["observations"][0]["rows"][0]["values"]
        return json.dumps({"answerable": True, "claims": [{"kind": "fact", "text": "共7座模拟厂站。",
            "data_refs": [{"call_id": "t1", "row": 0, "column": 0, "value": row[0]}], "knowledge_refs": []}], "reason": ""})
    return '{"sql":"SELECT count(*) FROM plant","explanation":"统计模拟厂站"}'


def test_chat_api_and_followup_history(client, fake_llm):
    fake_llm.responder = api_responder
    first = client.post("/api/ai/chat", json={"question": "一共有多少个厂站？"})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["ok"] and body["citations"][0]["value"] == 7
    assert body["now"] == END.isoformat() and "SELECT" in body["tool_calls"][0]["output"]["sql"]
    second = client.post("/api/ai/chat", json={"question": "再确认一下呢？", "session_id": body["session_id"]})
    assert second.json()["ok"] and second.json()["resolved_question"] == "同一范围的厂站数量"
    stored = client.get(f"/api/ai/chat/sessions/{body['session_id']}").json()
    assert len(stored["turns"]) == 2 and stored["turns"][0]["run_id"] != stored["turns"][1]["run_id"]


def test_sse_events_and_saved_result(client, fake_llm):
    fake_llm.responder = api_responder
    with client.stream("POST", "/api/ai/chat/stream", json={"question": "多少个厂站？"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = []
        event = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                frames.append((event, json.loads(line[6:])))
    assert frames[0][0] == "run_start" and frames[-1][0] == "done"
    assert [n for n, _ in frames].count("tool_end") == 1
    result = frames[-1][1]
    assert result["ok"] and result["citations"][0]["value"] == 7
    assert client.get(f"/api/ai/chat/sessions/{result['session_id']}").json()["turns"][-1] == result


def test_chat_input_unknown_sessions_and_busy_session(client, fake_llm, sync_redis):
    fake_llm.responder = api_responder
    assert client.post("/api/ai/chat", json={"question": "   "}).status_code == 422
    assert client.post("/api/ai/chat", json={"question": "查询数量", "messages": []}).status_code == 422
    assert client.post("/api/ai/chat", json={"question": "查询数量", "session_id": "x"}).status_code == 422
    assert client.post("/api/ai/chat", json={"question": "查询数量", "session_id": "0" * 32}).status_code == 404
    assert client.get("/api/ai/chat/sessions/bad-id").status_code == 404
    session = client.post("/api/ai/chat", json={"question": "有几个厂站？"}).json()["session_id"]
    sync_redis.set(f"ai:session:{session}:lock", "other-request", ex=20)
    for path in ["/api/ai/chat", "/api/ai/chat/stream"]:
        assert client.post(path, json={"question": "再问一次", "session_id": session}).status_code == 409


def recommendation_only():
    answer = draft(kind="recommendation")
    answer["claims"][0]["text"] = "加药通路异常仅为待核查可能，建议核实泵状态与末端出药。"
    return answer


async def test_diagnosis_without_hypothesis_is_repaired_once():
    outputs = [decision([("sql_query", "昨天氨氮"), ("rag_search", "排查")]), decision(), recommendation_only(), draft()]

    def respond(messages):
        if len(outputs) == 1:
            assert "缺少原因假设" in messages[-1]["content"]
        return json.dumps(outputs.pop(0), ensure_ascii=False)

    result = await Agent(FakeLLM(respond), SQL(), Rag()).run(QUESTION, END)
    assert result.ok and [c["kind"] for c in result.claims] == ["hypothesis"]
    assert result.validation_errors and "缺少原因假设" in result.validation_errors[0]
    assert not any("原因假设" in w for w in result.warnings)


async def test_diagnosis_still_without_hypothesis_is_kept_with_warning():
    llm = scripted(decision([("sql_query", "昨天氨氮"), ("rag_search", "排查")]), decision(),
                   recommendation_only(), recommendation_only())
    result = await Agent(llm, SQL(), Rag()).run(QUESTION, END)
    assert result.ok and not result.error and [c["kind"] for c in result.claims] == ["recommendation"]
    assert "诊断回答未给出原因假设，仅含事实与核查建议" in result.warnings


@pytest.mark.parametrize("intent,tools", [("data", ("t1", "t2")), ("diagnosis", ("t1",))])
def test_hypothesis_is_required_only_for_diagnosis_with_data_and_knowledge(intent, tools):
    result = evidence_result()
    result.intent = intent
    result.tool_calls = [r for r in result.tool_calls if r.id in tools]
    answer = draft(kind="fact")
    if "t2" not in tools:
        answer["claims"][0]["knowledge_refs"] = []
    validate_answer(json.dumps(answer), result, require_hypothesis=True)
    with pytest.raises(ValueError, match="缺少原因假设"):
        validate_answer(json.dumps(draft(kind="fact")), evidence_result(), require_hypothesis=True)


def test_prompts_limit_scope_to_listed_metrics_and_moment():
    assert "用户已列明指标、设备和时刻时，sql_query子问题只查这些项和时刻" in PLAN_SYSTEM
    assert "不调用plant_statistics" in PLAN_SYSTEM
    assert "陈述只引用这些指标和时刻的单元格" in ANSWER_SYSTEM
    assert "至少写一条hypothesis" in ANSWER_SYSTEM

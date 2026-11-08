"""并行、取消、完整计时和服务端统计引用的行为回归；不调用外部模型。"""

import asyncio
import copy
import json
from dataclasses import replace

import pytest

from app.ai.agent import Agent, AgentConfig
from app.ai.agent.evidence import observations, validate_answer
from app.ai.agent.models import AgentResult, Decision, ToolCall, ToolRecord
from app.ai.agent.service import Runtime, discard_event
from app.ai.gateway import AttemptRecord, ChatResult, LLMError
from app.ai.text2sql.service import Text2SQLResult
from tests.conftest import END
from tests.test_agent import SQL, decision, draft, evidence_result, scripted


def state_for(questions, emit=discard_event):
    runtime = Runtime(AgentResult("查询统计", END.isoformat()), END, [], emit)
    plan = Decision.model_validate(decision([("sql_query", q) for q in questions], intent="data"))
    return {"runtime": runtime, "decision": plan}


async def test_tools_overlap_and_reverse_completion_keeps_ids():
    started, finished = [], []
    all_started = asyncio.Event()
    releases = {q: asyncio.Event() for q in ("查数甲", "查数乙", "查数丙")}
    ended = {q: asyncio.Event() for q in releases}
    events = []

    class ControlledSQL(SQL):
        async def ask(self, question, now):
            started.append(question)
            if len(started) == 3:
                all_started.set()
            await releases[question].wait()
            return await super().ask(question, now)

    async def emit(name, data):
        events.append((name, data["id"]))
        if name == "tool_end":
            finished.append(data["id"])
            ended[data["question"]].set()

    state = state_for(list(releases), emit)
    worker = asyncio.create_task(Agent(None, ControlledSQL(), None)._tools(state))
    try:
        async with asyncio.timeout(2):
            await all_started.wait()  # 串行实现会在这里超时，不能靠耗时阈值假装并发。
            for q in reversed(releases):
                releases[q].set()
                await ended[q].wait()
            await worker
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
    records = state["runtime"].result.tool_calls
    assert [r.id for r in records] == ["t1", "t2", "t3"]
    assert [r.question for r in records] == list(releases)
    assert finished == ["t3", "t2", "t1"]
    assert events[:3] == [("tool_start", f"t{i}") for i in (1, 2, 3)]
    assert all(r.status == "ok" and r.total_ms >= 0 and r.queue_ms >= 0 for r in records)


async def test_concurrency_limit_is_shared_between_sessions():
    active = peak = 0
    two_started, release = asyncio.Event(), asyncio.Event()

    class LimitedSQL(SQL):
        async def ask(self, question, now):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                two_started.set()
            try:
                await release.wait()
                return await super().ask(question, now)
            finally:
                active -= 1

    agent = Agent(None, LimitedSQL(), None, AgentConfig(tool_concurrency=2))
    states = [state_for([f"会话{s}查数{i}" for i in range(3)]) for s in range(2)]
    workers = [asyncio.create_task(agent._tools(s)) for s in states]
    try:
        async with asyncio.timeout(2):
            await two_started.wait()
            assert sum(r.status == "running" for s in states for r in s["runtime"].result.tool_calls) == 2
            release.set()
            await asyncio.gather(*workers)
    finally:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    assert peak == 2 and active == 0
    records = [r for s in states for r in s["runtime"].result.tool_calls]
    assert len(records) == 6 and all(r.status == "ok" for r in records)


async def test_one_tool_failure_does_not_cancel_independent_tools():
    class OneBadSQL(SQL):
        async def ask(self, question, now):
            if question == "查询失败":
                raise RuntimeError("不能公开的错误正文")
            return await super().ask(question, now)

    state = state_for(["查询失败", "查询成功"])
    await Agent(None, OneBadSQL(), None)._tools(state)
    records = state["runtime"].result.tool_calls
    assert [r.status for r in records] == ["error", "ok"]
    assert "RuntimeError" in records[0].error and "错误正文" not in records[0].error


@pytest.mark.parametrize("stop", ["disconnect", "deadline"])
async def test_all_running_and_queued_tools_are_cancelled_and_slots_released(stop):
    started, cleaned = asyncio.Event(), asyncio.Event()

    class WaitingSQL(SQL):
        async def ask(self, question, now):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

    # 仅一个额度：同时覆盖一个执行中任务和两个排队任务。
    agent = Agent(scripted(decision([("sql_query", f"等待{i}") for i in range(3)])), WaitingSQL(), None,
                  AgentConfig(timeout_s=1.0, tool_concurrency=1))
    worker = asyncio.create_task(agent.run("等待查数", END))
    async with asyncio.timeout(3):
        await started.wait()
        if stop == "disconnect":
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker
        else:
            result = await worker
            assert "超过" in result.error and not result.ok
            assert all(r.status == "cancelled" for r in result.tool_calls)
    assert cleaned.is_set()
    # 取消后能重新取得额度，不发生连接断开后占住信号量的泄漏。
    async with asyncio.timeout(0.5):
        async with agent._tool_slots:
            pass


async def test_followup_and_its_repair_disable_thinking_and_count_full_usage():
    outputs = iter([
        decision([("sql_query", "查询氨氮")], intent="data"),
        decision([("sql_query", "查询氨氮")], intent="data"),  # 重复，引发修复。
        decision(intent="data"),
        {"answerable": True, "claims": [{"kind": "fact", "text": "该次氨氮读数为6.2。",
          "data_refs": [{"call_id": "t1", "row": 0, "column": 0, "value": 6.2}]}]},
    ])
    seen = []

    class TimedLLM:
        async def chat(self, messages, **kwargs):
            seen.append(kwargs["thinking"])
            return ChatResult(json.dumps(next(outputs)), "fake", "fake-model", 2, 3, latency_ms=10,
                              total_latency_ms=110, attempt_history=[
                                  AttemptRecord("fake", 1, "error", 30, 70, "LLMError", 429),
                                  AttemptRecord("fake", 2, "success", 10)])

    result = await Agent(TimedLLM(), SQL(), None).run("查询氨氮", END)
    assert result.ok and seen == [None, False, False, False]
    assert result.llm_ms == 440
    assert [c["round"] for c in result.generation_calls] == [1, 2, 2, 2]
    assert all(c["llm_ms"] == 110 and len(c["attempt_history"]) == 2 for c in result.generation_calls)


async def test_terminal_model_failure_still_records_all_attempt_time():
    class FailedLLM:
        async def chat(self, *args, **kwargs):
            exc = LLMError("不可展示的厂商错误", status=503)
            exc.total_latency_ms = 900
            exc.attempt_history = [AttemptRecord("fake", 1, "error", 900, status_code=503)]
            raise exc

    result = await Agent(FailedLLM(), None, None).run("查询氨氮", END)
    assert result.error and result.llm_ms == 900
    call = result.generation_calls[0]
    assert call["status"] == "error" and call["attempt_history"][0]["status_code"] == 503
    assert "不可展示" not in json.dumps(result.to_dict(), ensure_ascii=False)


def stats_output(start="2026-11-05T10:00:00+08:00", end="2026-11-05T12:00:00+08:00"):
    return {"period": {"start": start, "end": end, "timezone": "Asia/Shanghai"},
            "period_text": f"{start}至{end}（北京时间）", "metric": {"key": "nh3n_out"},
            "facts": [{"id": "P06.nh3n_out.exceed_hours", "statistic": "exceed_hours", "value": 0.5,
                       "metric": "nh3n_out", "text": "黄官镇按5分钟均值统计，累计超标0.5小时；有效24/24个桶",
                       "unit": "小时", "aggregation": "5分钟均值累计异常时长",
                       "coverage": {"P06": {"expected_buckets": 24, "observed_buckets": 24, "coverage": 1.0}}}],
            "coverage_policy": "缺失打断连续计数", "warnings": []}


def statistic_draft(*refs):
    return {"answerable": True, "claims": [{"kind": "statistic", "text": "", "statistic_refs": [
        {"call_id": cid, "fact_id": fid} for cid, fid in refs]}]}


def statistic_result():
    result = evidence_result()
    result.tool_calls.append(ToolRecord("t3", "plant_statistics", "统计氨氮", "ok", stats_output()))
    return result


def test_statistical_sentence_and_metadata_are_written_by_server():
    result = statistic_result()
    ref = ("t3", "P06.nh3n_out.exceed_hours")
    text, claims, citations, reason = validate_answer(json.dumps(statistic_draft(ref, ref)), result)
    fact = result.tool_calls[-1].output["facts"][0]
    assert reason is None and fact["text"] in text and "连续超标" not in text
    assert len(citations) == 1 and citations[0]["value"] == 0.5
    assert citations[0]["text"] == fact["text"] and citations[0]["period"] == stats_output()["period"]
    assert citations[0]["coverage_policy"] == stats_output()["coverage_policy"]
    assert claims[0]["citation_numbers"] == [1]
    obs = observations(result.tool_calls)[-1]
    assert obs["facts"] == stats_output()["facts"]


@pytest.mark.parametrize("mutation", ["fake_id", "old_call", "failed_call", "wrong_tool", "text", "mixed_cell", "period"])
def test_invalid_statistical_evidence_cannot_be_emitted(mutation):
    result = statistic_result()
    body = statistic_draft(("t3", "P06.nh3n_out.exceed_hours"))
    claim = body["claims"][0]
    if mutation == "fake_id":
        claim["statistic_refs"][0]["fact_id"] = "P06.nh3n_out.no_such_fact"
    elif mutation == "old_call":
        claim["statistic_refs"][0]["call_id"] = "last-session-t3"
    elif mutation == "failed_call":
        result.tool_calls[-1].status = "error"
    elif mutation == "wrong_tool":
        claim["statistic_refs"][0]["call_id"] = "t1"
    elif mutation == "text":
        claim["text"] = "连续超标0.5小时"
    elif mutation == "mixed_cell":
        claim["data_refs"] = [{"call_id": "t1", "row": 0, "column": 0, "value": 6.2}]
    else:
        other = copy.deepcopy(result.tool_calls[-1])
        other.id = "t4"
        other.output = stats_output(start="2026-11-04T10:00:00+08:00")
        result.tool_calls.append(other)
        claim["statistic_refs"].append({"call_id": "t4", "fact_id": "P06.nh3n_out.exceed_hours"})
    with pytest.raises(ValueError):
        validate_answer(json.dumps(body), result)


@pytest.mark.parametrize("text", ["连续超标4小时", "累计超标4小时", "瞬时峰值为6.2", "排名第一名", "高于其他厂站"])
def test_cell_citation_cannot_be_relabelled_as_a_statistic(text):
    body = draft(kind="fact")
    body["claims"][0]["text"] = text
    with pytest.raises(ValueError, match="kind=statistic"):
        validate_answer(json.dumps(body), evidence_result())


@pytest.mark.parametrize("text", ["鼓风机60台运行", "鼓风机运行台数由60降至30", "鼓风机60/120运行"])
def test_running_sample_ratio_cannot_be_presented_as_device_count(text):
    result = evidence_result()
    result.tool_calls[0].output["rows"] = [["60/120 运行"]]
    body = draft(kind="fact", value="60/120 运行")
    body["claims"][0]["text"] = text
    with pytest.raises(ValueError, match="采样"):
        validate_answer(json.dumps(body), result)
    body["claims"][0]["text"] = "该小时鼓风机运行采样60/120。"
    assert "运行采样60/120" in validate_answer(json.dumps(body), result)[0]


def test_statistics_duplicate_detection_compares_parameters():
    base = {"plant": "P06", "metric": "nh3n_out", "start": "2026-11-05T00:00:00+08:00",
            "end": END.isoformat(), "statistics": ["instant_max"]}
    call = {"tool": "plant_statistics", "question": "查询指标", "parameters": base}
    one = ToolCall.model_validate_json(json.dumps(call))
    two = ToolCall.model_validate_json(json.dumps({**call, "parameters": {**base, "metric": "tp_out"}}))
    same = ToolCall.model_validate_json(json.dumps({**call, "question": "另一个描述"}))
    assert Agent._call_key(one) != Agent._call_key(two)
    assert Agent._call_key(one) == Agent._call_key(same)

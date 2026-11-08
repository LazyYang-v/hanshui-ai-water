"""LangGraph：规划 → 只读工具 → 补查/综合回答。会话持久化独立于图执行。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.ai.agent.evidence import (
    conversation,
    missing_hypothesis,
    observations,
    validate_answer,
)
from app.ai.agent.models import AgentResult, Decision, ToolRecord
from app.ai.agent.prompts import ANSWER_SYSTEM, PLAN_SYSTEM
from app.ai.gateway import LLMError, LLMGateway
from app.ai.rag.service import RAG
from app.ai.text2sql import Text2SQL
from app.ai.usage import call_usage, elapsed_ms
from app.analytics.statistics import PlantStatistics
from app.domain.metrics import PLANT_METRICS

log = logging.getLogger(__name__)
EventSink = Callable[[str, dict], Awaitable[None]]


@dataclass(frozen=True)
class AgentConfig:
    max_tools: int = 5
    max_plans: int = 2
    timeout_s: float = 180
    tool_timeout_s: float = 60
    max_tokens: int = 4000
    answer_max_tokens: int = 4000
    # 综合回答关闭 Flash 思考：开启时推理与 JSON 共用预算，复杂诊断曾用满 8000 token 仍无输出。
    # None 表示沿用厂商默认（规划阶段即如此）。
    answer_thinking: bool | None = False
    followup_thinking: bool | None = False
    tool_concurrency: int = 3

    def __post_init__(self):
        if not 1 <= self.tool_concurrency <= 5:
            raise ValueError("工具并发数须为1至5")


async def discard_event(event: str, data: dict):
    pass


@dataclass
class Runtime:
    result: AgentResult
    now: datetime
    history: list[dict]
    emit: EventSink
    seen: set[tuple[str, str]] = field(default_factory=set)


class State(TypedDict):
    runtime: Runtime
    decision: Decision | None


# 修复反馈要让模型知道改什么：只给 too_long 这类类型名时，真实诊断曾连续两次超出陈述上限。
ERROR_HINTS = {
    "missing": "缺少字段",
    "extra_forbidden": "不允许的字段",
    "too_long": "最多{max_length}项，当前{actual_length}项，请合并或删减",
    "too_short": "至少{min_length}项",
    "string_too_long": "超过{max_length}字",
    "string_too_short": "少于{min_length}字",
    "literal_error": "只能取{expected}",
    "greater_than_equal": "不能小于{ge}",
    "json_invalid": "不是合法JSON",
}


def error_detail(exc: ValueError) -> str:
    if not isinstance(exc, ValidationError):
        return str(exc)
    parts = []
    for e in exc.errors(include_input=False, include_url=False)[:3]:
        if e["type"] == "value_error":
            hint = e["msg"].removeprefix("Value error, ")
        else:
            try:
                hint = ERROR_HINTS[e["type"]].format(**e.get("ctx", {}))
            except (KeyError, IndexError):
                hint = e["type"]
        parts.append(f"{'.'.join(map(str, e['loc'])) or 'JSON'}: {hint}")
    return "; ".join(parts)


def account(result: AgentResult, usage, stage: str | None = None):
    model = getattr(usage, "model", None)
    if model and model not in result.models:
        result.models.append(model)
    result.prompt_tokens += getattr(usage, "prompt_tokens", 0)
    result.completion_tokens += getattr(usage, "completion_tokens", 0)
    result.llm_ms += elapsed_ms(usage)
    if stage:
        result.generation_calls.append({"stage": stage, **call_usage(usage)})


class Agent:
    def __init__(self, gateway: LLMGateway, text2sql: Text2SQL | None, rag: RAG | None,
                 cfg: AgentConfig = AgentConfig(), *, statistics: PlantStatistics | None = None):
        self.gateway, self.text2sql, self.rag, self.cfg = gateway, text2sql, rag, cfg
        self.statistics = statistics
        # 同一Agent实例共享额度，多个会话也不会把并发无限放大。
        self._tool_slots = asyncio.Semaphore(cfg.tool_concurrency)
        graph = StateGraph(State)
        graph.add_node("plan", self._plan)
        graph.add_node("tools", self._tools)
        graph.add_node("answer", self._answer)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", self._after_plan, {"tools": "tools", "answer": "answer", "end": END})
        graph.add_conditional_edges("tools", self._after_tools, {"plan": "plan", "answer": "answer"})
        graph.add_edge("answer", END)
        self.graph = graph.compile()

    async def run(self, question: str, now: datetime, *, history: list[dict] | None = None,
                  session_id: str | None = None, emit: EventSink = discard_event) -> AgentResult:
        start = time.perf_counter()
        result = AgentResult(question=question, now=now.isoformat(), session_id=session_id)
        runtime = Runtime(result, now, conversation(history or []), emit)
        await emit("run_start", {"run_id": result.run_id, "session_id": session_id, "now": result.now})
        await self._guarded(result, lambda: self.graph.ainvoke({"runtime": runtime, "decision": None},
                                                               {"recursion_limit": 12}))
        result.total_ms = round((time.perf_counter() - start) * 1000, 1)
        result.llm_ms = round(result.llm_ms, 1)
        return result

    async def answer_with_evidence(self, question: str, now: datetime, records: list[ToolRecord], *,
                                   intent: str = "diagnosis") -> AgentResult:
        """跳过规划与工具调用，用调用方准备好的证据走同一套综合回答与引用校验。

        告警处置建议使用：证据是服务端按告警类型执行的固定查询与知识检索，记录格式与工具输出一致。
        """
        start = time.perf_counter()
        result = AgentResult(question=question, now=now.isoformat(), resolved_question=question, intent=intent,
                             tool_calls=list(records))
        runtime = Runtime(result, now, [], discard_event)
        await self._guarded(result, lambda: self._answer({"runtime": runtime, "decision": None}))
        result.total_ms = round((time.perf_counter() - start) * 1000, 1)
        result.llm_ms = round(result.llm_ms, 1)
        return result

    async def _guarded(self, result: AgentResult, step: Callable[[], Awaitable]) -> None:
        try:
            async with asyncio.timeout(self.cfg.timeout_s):
                await step()
        except TimeoutError:
            result.error = f"本轮超过{self.cfg.timeout_s:g}秒，已停止后续调用"
        except LLMError:
            result.error = "模型服务暂时不可用，请稍后重试"
        except Exception as exc:
            log.error("Agent 执行失败（%s）", type(exc).__name__)
            result.error = "Agent 执行失败，请稍后重试"
        # CancelledError不捕获；HTTP断开应传播取消，避免继续调用付费模型。
        if result.error:
            result.ok, result.refused = False, False
            result.answer, result.claims, result.citations = "", [], []
            result.reason = None

    def _payload(self, runtime: Runtime) -> dict:
        res = runtime.result
        return {"question": res.question, "resolved_question": res.resolved_question,
                "intent": res.intent, "now": res.now, "history": runtime.history,
                "observations": observations(res.tool_calls),
                "remaining_tools": self.cfg.max_tools - len(res.tool_calls),
                "available_tools": [name for name, tool in (("sql_query", self.text2sql), ("rag_search", self.rag),
                                                           ("plant_statistics", self.statistics)) if tool is not None],
                "statistics_metrics": {k: {"label": m.label, "unit": m.unit, "threshold": m.limit}
                                       for k, m in PLANT_METRICS.items()} if self.statistics is not None else {}}

    async def _generate(self, result: AgentResult, messages: list[dict], stage: str, **kwargs):
        try:
            output = await self.gateway.chat(messages, json_mode=True, temperature=0.0, **kwargs)
        except (LLMError, asyncio.CancelledError) as exc:
            account(result, exc, stage)
            result.generation_calls[-1].update(round=result.planning_rounds, thinking=kwargs.get("thinking"))
            raise
        account(result, output, stage)
        result.generation_calls[-1].update(round=result.planning_rounds, thinking=kwargs.get("thinking"))
        return output

    @staticmethod
    def _call_key(call) -> tuple[str, str]:
        # 固定统计按规范化参数去重；同一句描述可以查不同指标/时段。
        return call.tool, call.parameters.model_dump_json() if call.parameters is not None else call.question

    async def _plan(self, state: State) -> dict:
        runtime, res = state["runtime"], state["runtime"].result
        res.planning_rounds += 1
        await runtime.emit("planning", {"round": res.planning_rounds})
        messages = [{"role": "system", "content": PLAN_SYSTEM},
                    {"role": "user", "content": json.dumps(self._payload(runtime), ensure_ascii=False)}]
        truncated = False
        for _ in range(2):
            # 首轮沿用厂商默认思考；补查、以及首轮因推理过长被截断后的修复关闭思考，避免再想满预算
            thinking = None if res.planning_rounds == 1 and not truncated else self.cfg.followup_thinking
            output = await self._generate(res, messages, "plan", max_tokens=self.cfg.max_tokens, thinking=thinking)
            try:
                if output.finish_reason == "length":
                    truncated = True
                    raise ValueError("规划生成达到token上限（推理过长），请不要展开推理，直接输出JSON")
                decision = Decision.model_validate_json(output.content)
                if res.intent:  # 补查沿用首轮的意图与补全后的问题，模型中途改写不生效
                    decision = decision.model_copy(update={"intent": res.intent,
                                                           "resolved_question": res.resolved_question})
                if len(res.tool_calls) + len(decision.calls) > self.cfg.max_tools:
                    raise ValueError("超过本轮剩余工具额度，请精简调用或answer")
                keys = [self._call_key(c) for c in decision.calls]
                if len(set(keys)) != len(keys) or any(k in runtime.seen for k in keys):
                    raise ValueError("工具问题重复，请利用已有结果或只补查缺少的证据")
                if decision.intent == "unsupported" and decision.action != "refuse":
                    raise ValueError("不支持的问题应refuse")
                for call in decision.calls:
                    if {"sql_query": self.text2sql, "rag_search": self.rag,
                        "plant_statistics": self.statistics}[call.tool] is None:
                        raise ValueError("该工具当前不可用，请说明缺少能力或使用已有证据")
                res.intent, res.resolved_question = decision.intent, decision.resolved_question
                if decision.action == "refuse":
                    res.refused, res.reason = True, decision.reason
                return {"decision": decision}
            except ValueError as exc:
                detail = error_detail(exc)
                res.validation_errors.append("plan: " + detail)
                messages += [{"role": "assistant", "content": output.content},
                             {"role": "user", "content": "规划校验失败：" + detail
                              + "。请重新输出完整JSON，保留intent、resolved_question、action、calls、reason。"
                              "plant_statistics调用必须带parameters（plant、metric、start、end、statistics）；"
                              "sql_query和rag_search不得带parameters。拒答必须给出非空reason。"}]
        if res.tool_calls:  # 已有证据时，补查规划失败不应丢弃整轮
            res.warnings.append("补查规划连续两次未通过校验，按现有证据作答")
            return {"decision": Decision(intent=res.intent, resolved_question=res.resolved_question,
                                         action="answer", calls=[], reason="")}
        res.error = "工具规划连续两次未通过校验"
        return {"decision": None}

    @staticmethod
    def _after_plan(state: State) -> str:
        res, decision = state["runtime"].result, state["decision"]
        if res.error or res.refused or decision is None:
            return "end"
        return "tools" if decision.action == "tools" else "answer"

    async def _tools(self, state: State) -> dict:
        runtime, res = state["runtime"], state["runtime"].result
        pending = []
        # 先按规划顺序分配id；完成先后不会改动引用编号。
        for call in state["decision"].calls:
            runtime.seen.add(self._call_key(call))
            record = ToolRecord(f"t{len(res.tool_calls) + 1}", call.tool, call.question, status="queued")
            res.tool_calls.append(record)
            pending.append((call, record))
        async with asyncio.TaskGroup() as tasks:
            for call, record in pending:
                tasks.create_task(self._execute_tool(runtime, call, record))
        return {}

    async def _execute_tool(self, runtime: Runtime, call, record: ToolRecord) -> None:
        res, queued_at = runtime.result, time.perf_counter()
        start = None
        try:
            async with self._tool_slots:
                record.queue_ms = round((time.perf_counter() - queued_at) * 1000, 1)
                record.status = "running"
                await runtime.emit("tool_start", {"id": record.id, **call.model_dump(mode="json", exclude_none=True)})
                start = time.perf_counter()
                async with asyncio.timeout(self.cfg.tool_timeout_s):
                    if call.tool == "plant_statistics":
                        record.output = await self.statistics.query(call.parameters, runtime.now)
                        record.status = "ok" if record.output["facts"] else "empty"
                        res.warnings.extend(f"{record.id}：{w}" for w in record.output["warnings"])
                    elif call.tool == "sql_query":
                        output = await self.text2sql.ask(call.question, runtime.now)
                        account(res, output)
                        record.output = output.to_dict()
                        record.status = ("ok" if output.rows else "empty") if output.ok else (
                            "refused" if output.refused else "error")
                        record.error = output.error
                        if output.truncated:
                            res.warnings.append(f"{record.id}查询结果被截断，不能代表完整数据范围")
                        if len(output.rows) > 30:
                            res.warnings.append(f"{record.id}仅将前30行送入模型，完整返回表见tool_calls")
                    else:
                        output = await self.rag.search(call.question, top_k=5, mode="hybrid", rerank=True)
                        account(res, output)
                        record.output = output.to_dict()
                        record.status = "ok" if output.hits else "empty"
                        res.warnings.extend(f"{record.id}：{w}" for w in output.warnings)
        except TimeoutError:
            record.status, record.error = "error", "工具执行超时"
        except asyncio.CancelledError:
            record.status, record.error = "cancelled", "本轮已停止"
            raise
        except Exception as exc:
            record.status, record.error = "error", f"工具执行失败（{type(exc).__name__}）"
        finally:
            if start is not None:
                record.total_ms = round((time.perf_counter() - start) * 1000, 1)
            else:
                record.queue_ms = round((time.perf_counter() - queued_at) * 1000, 1)
        if record.status in {"error", "refused"}:
            res.warnings.append(f"{record.id}未获得证据：{record.error}")
        await runtime.emit("tool_end", record.to_dict())

    def _after_tools(self, state: State) -> str:
        res = state["runtime"].result
        if res.planning_rounds < self.cfg.max_plans and len(res.tool_calls) < self.cfg.max_tools:
            return "plan"
        res.warnings.append("已达到规划或工具调用上限，按现有证据作答")
        return "answer"

    async def _answer(self, state: State) -> dict:
        runtime, res = state["runtime"], state["runtime"].result
        if not any(record.status == "ok" for record in res.tool_calls):
            if any(record.status in {"error", "cancelled"} for record in res.tool_calls):
                res.error = "工具未获得有效证据，暂时无法回答"
            else:
                res.refused, res.reason = True, "本轮没有可引用的数据或知识证据，无法作答"
            return {}
        await runtime.emit("answer_start", {"message": "正在综合证据并校验引用"})
        messages = [{"role": "system", "content": ANSWER_SYSTEM},
                    {"role": "user", "content": json.dumps(self._payload(runtime), ensure_ascii=False)}]
        for attempt in range(2):
            output = await self._generate(res, messages, "answer", max_tokens=self.cfg.answer_max_tokens,
                                          thinking=self.cfg.answer_thinking)
            try:
                if output.finish_reason == "length":
                    raise ValueError("回答生成达到token上限，请只保留最关键的陈述（不超过5条）并减少引用")
                # 原因假设是完整性要求、不是证据要求：只在首轮强制，修复轮仍缺时保留回答并告警，不整轮失败。
                answer, claims, citations, reason = validate_answer(output.content, res, require_hypothesis=attempt == 0)
                if reason is None and missing_hypothesis(claims, res):
                    res.warnings.append("诊断回答未给出原因假设，仅含事实与核查建议")
                res.answer, res.claims, res.citations, res.reason = answer, claims, citations, reason
                res.refused, res.ok = reason is not None, reason is None
                return {}
            except ValueError as exc:
                detail = error_detail(exc)
                res.validation_errors.append("answer: " + detail)
                messages += [{"role": "assistant", "content": output.content},
                             {"role": "user", "content": "回答校验失败：" + detail +
                              "。核对本轮call_id、从0起的行列、单元格原值和原文摘录；统计结论选kind=statistic，"
                              "text留空并引用真实statistic_refs；缺少统计事实则删去该结论或明确拒答。重新输出完整JSON。"}]
        res.error = "回答连续两次未通过证据校验"
        return {}

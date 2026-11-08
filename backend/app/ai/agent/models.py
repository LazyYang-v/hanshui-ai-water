"""Agent 的规划协议与证据协议；模型只能选择已注册工具。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.analytics.statistics import StatisticsQuery


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class ToolCall(StrictModel):
    tool: Literal["sql_query", "rag_search", "plant_statistics"]
    question: str = Field(min_length=2, max_length=500)
    parameters: StatisticsQuery | None = None

    @model_validator(mode="after")
    def tool_parameters(self):
        if (self.tool == "plant_statistics") != (self.parameters is not None):
            raise ValueError("plant_statistics必须提供parameters；其他工具不得带parameters")
        return self


class Decision(StrictModel):
    intent: Literal["data", "knowledge", "diagnosis", "unsupported"]
    resolved_question: str = Field(min_length=2, max_length=500)
    action: Literal["tools", "answer", "refuse"]
    calls: list[ToolCall] = Field(max_length=3)
    # tools/answer 的空理由不承载事实；省略时补空，refuse仍在下方强制非空理由。
    reason: str = Field(default="", max_length=400)

    @model_validator(mode="after")
    def consistent(self):
        if (self.action == "tools") != bool(self.calls):
            raise ValueError("tools 必须带工具调用，answer/refuse 不得带调用")
        if self.action == "refuse" and not self.reason:
            raise ValueError("拒答需要原因")
        return self


class DataRef(StrictModel):
    # 单元格字符串是原始数据，不能随协议字段一起strip。
    model_config = ConfigDict(str_strip_whitespace=False)
    call_id: str
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    value: str | int | float | bool | None


class KnowledgeRef(StrictModel):
    call_id: str
    chunk_id: str
    quote: str = Field(min_length=8, max_length=160)


# 提示词的目标上限为每轮5条陈述、每条6个单元格和2段摘录；校验硬上限各留余量，
# 模型多写一项时不必整轮重写。按硬上限估算最坏约 6×(8×25+3×60+150)≈3200 token，仍在回答预算内。
MAX_CLAIMS, MAX_DATA_REFS, MAX_KNOWLEDGE_REFS = 6, 8, 3


class StatisticRef(StrictModel):
    call_id: str
    fact_id: str


class Claim(StrictModel):
    kind: Literal["fact", "statistic", "hypothesis", "recommendation"]
    text: str = Field(default="", max_length=300)
    # statistic只选事实，整句由服务端生成，防止把累计时长重新写成连续时长。
    statistic_refs: list[StatisticRef] = Field(default_factory=list, max_length=6)
    data_refs: list[DataRef] = Field(default_factory=list, max_length=MAX_DATA_REFS)
    knowledge_refs: list[KnowledgeRef] = Field(default_factory=list, max_length=MAX_KNOWLEDGE_REFS)

    @model_validator(mode="after")
    def statistic_is_server_written(self):
        if self.kind == "statistic":
            if self.text or self.data_refs or self.knowledge_refs or not self.statistic_refs:
                raise ValueError("statistic只填statistic_refs，text须为空，不得混入单元格或知识引用")
        elif not self.text or self.statistic_refs:
            raise ValueError("普通陈述必须有text，统计事实请使用kind=statistic")
        return self


class Answer(StrictModel):
    answerable: bool
    claims: list[Claim] = Field(default_factory=list, max_length=MAX_CLAIMS)
    reason: str = Field(default="", max_length=600)


@dataclass
class ToolRecord:
    id: str
    tool: str
    question: str
    status: str = "running"
    output: dict = field(default_factory=dict)
    error: str | None = None
    total_ms: float = 0.0
    queue_ms: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class AgentResult:
    question: str
    now: str
    run_id: str = field(default_factory=lambda: uuid4().hex)
    session_id: str | None = None
    resolved_question: str = ""
    intent: str | None = None
    ok: bool = False
    refused: bool = False
    answer: str = ""
    reason: str | None = None
    error: str | None = None
    claims: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    tool_calls: list[ToolRecord] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    generation_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_ms: float = 0.0
    total_ms: float = 0.0
    planning_rounds: int = 0
    validation_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

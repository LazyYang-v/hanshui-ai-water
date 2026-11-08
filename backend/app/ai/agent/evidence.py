"""把工具结果投影为有界上下文，并核验回答中的单元格与原文引用。"""

from __future__ import annotations

import json
import re

from app.ai.agent.models import AgentResult, Answer, ToolRecord
from app.ai.rag.service import compact

PREVIEW_ROWS = 30
PREFIX = {"fact": "", "statistic": "", "hypothesis": "原因假设（待核查）：", "recommendation": "核查建议："}
# 这是针对已知统计误读的护栏，不声称解析任意自然语言。统计结论统一走服务端事实。
STATISTIC_LANGUAGE = re.compile(
    r"峰值|最高|最低|最大值|最小值|排名|并列|第[一二三四五六七八九十0-9]+名|"
    r"(?:累计|合计|连续|持续).{0,16}(?:小时|分钟|超标时长|低于下限时长)|"
    r"(?:超标|异常).{0,12}(?:小时|分钟|时长)|(?:高于|低于|多于|少于)(?:其他|其余)"
)
# 模型有时自己写了同类标签；只去掉与 kind 一致的标签，避免重复，不改变陈述类型。
OWN_LABEL = {"hypothesis": re.compile(r"^原因假设(?:（待核查）|\(待核查\))?\s*[：:]\s*"),
             "recommendation": re.compile(r"^核查建议\s*[：:]\s*")}


def observations(records: list[ToolRecord]) -> list[dict]:
    items = []
    for record in records:
        out = record.output
        item = {"call_id": record.id, "tool": record.tool, "question": record.question,
                "status": record.status, "error": record.error}
        if record.tool == "sql_query" and record.status in {"ok", "empty"}:
            item.update(columns=[{"index": i, "name": name} for i, name in enumerate(out.get("columns", []))],
                        rows=[{"index": i, "values": row} for i, row in enumerate(out.get("rows", [])[:PREVIEW_ROWS])],
                        returned_rows=len(out.get("rows", [])),
                        preview_truncated=len(out.get("rows", [])) > PREVIEW_ROWS,
                        query_truncated=out.get("truncated", False))
        elif record.tool == "plant_statistics" and record.status in {"ok", "empty"}:
            item.update(period=out["period"], metric=out["metric"], facts=out["facts"],
                        coverage_policy=out["coverage_policy"], warnings=out["warnings"])
        elif record.tool == "rag_search" and record.status in {"ok", "empty"}:
            item.update(hits=[{k: hit.get(k) for k in ("id", "title", "heading", "source_type", "text")}
                              for hit in out.get("hits", [])], warnings=out.get("warnings", []))
        items.append(item)
    return items


def conversation(turns: list[dict]) -> list[dict]:
    # 不把旧工具结果重新塞入上下文；追问必须重新获取可引用的证据。
    # 用户在回答之后的人工操作（如确认派单）只在发生过时带上，没有时上下文与之前完全一致。
    out = []
    for t in turns[-6:]:
        h = {"question": t["question"], "resolved_question": t.get("resolved_question", ""),
             "now": t["now"], "answer": t.get("answer", "")[:1800],
             "reason": t.get("reason"), "ok": t.get("ok", False)}
        if t.get("followups"):
            h["user_actions_after_answer"] = [f["text"] for f in t["followups"]][-3:]
        out.append(h)
    return out


def statistic_sentence(facts: list[dict]) -> str:
    """多条统计事实合成一句：覆盖说明相同时只在句末写一次（每条引用里仍保留完整文本）。"""
    notes = {f.get("note", "") for f in facts}
    if len(facts) > 1 and len(notes) == 1 and (note := notes.pop()):
        return "；".join(f["text"].removesuffix(note) for f in facts) + note
    return "；".join(f["text"] for f in facts)


def same_cell(actual, expected) -> bool:
    # JSON数值允许6与6.0；布尔不可冒充0/1，字符串不可转成数值。
    if type(actual) in (int, float) and type(expected) in (int, float):
        return actual == expected
    return type(actual) is type(expected) and actual == expected


def missing_hypothesis(claims: list[dict], result: AgentResult) -> bool:
    """诊断题已有数据和知识证据却没有原因假设：通常是把“可能原因”写进了核查建议。"""
    tools = {r.tool for r in result.tool_calls if r.status == "ok"}
    return (result.intent == "diagnosis" and "rag_search" in tools and bool(tools & {"sql_query", "plant_statistics"})
            and not any(c["kind"] == "hypothesis" for c in claims))


def validate_answer(content: str, result: AgentResult, *,
                    require_hypothesis: bool = False) -> tuple[str, list[dict], list[dict], str | None]:
    draft = Answer.model_validate_json(content)
    if not draft.answerable:
        if draft.claims or not draft.reason:
            raise ValueError("拒答必须说明原因且不得携带陈述")
        return "", [], [], draft.reason
    if not draft.claims:
        raise ValueError("可答结果至少需要一条带证据的陈述")
    # 可答时 reason 没有出处，既不展示也不作为证据（模型常填写补充说明，不应因此判失败）。
    records = {r.id: r for r in result.tool_calls if r.status == "ok"}
    citations, claims, paragraphs, index = [], [], [], {}

    def add(citation):
        key = json.dumps(citation, ensure_ascii=False, sort_keys=True)
        if key not in index:
            index[key] = len(citations) + 1
            citations.append({"number": index[key], **citation})
        return index[key]

    for claim in draft.claims:
        if claim.kind == "statistic":
            numbers, picked, periods = [], [], set()
            for ref in claim.statistic_refs:
                record = records.get(ref.call_id)
                if record is None or record.tool != "plant_statistics":
                    raise ValueError("统计引用不是本轮成功的服务端统计调用")
                out = record.output
                fact = next((f for f in out["facts"] if f["id"] == ref.fact_id), None)
                if fact is None:
                    raise ValueError("统计事实id不在本轮结果中，缺失数据不能当成零")
                period = (out["period"]["start"], out["period"]["end"])
                periods.add(period)
                number = add({"kind": "statistic", "call_id": ref.call_id, "fact_id": fact["id"],
                              "value": fact["value"], "statistic": fact["statistic"], "text": fact["text"],
                              "metric": fact["metric"], "unit": fact["unit"],
                              "aggregation": fact["aggregation"], "coverage": fact["coverage"],
                              "period": out["period"], "coverage_policy": out["coverage_policy"],
                              "source": "服务端固定只读统计"})
                if number not in numbers:
                    numbers.append(number)
                    picked.append(fact)
            if len(periods) != 1:
                raise ValueError("一条统计陈述只能合并同一统计时段的事实，请分别列出")
            text = f"统计时段{record.output['period_text']}：" + statistic_sentence(picked) + "。"
            claims.append({"kind": "statistic", "text": text, "citation_numbers": numbers})
            paragraphs.append(text + " " + "".join(f"[{n}]" for n in numbers))
            continue
        if claim.data_refs and STATISTIC_LANGUAGE.search(claim.text):
            raise ValueError("累计/连续时长、峰值、排名和跨厂比较必须使用kind=statistic引用plant_statistics事实；"
                             "普通单元格只能描述该时刻的读数，不能改写为统计结论")
        if not claim.data_refs and not claim.knowledge_refs:
            raise ValueError("每条陈述必须有本轮证据")
        if claim.kind == "hypothesis" and not (claim.data_refs and claim.knowledge_refs):
            raise ValueError("原因假设必须同时引用数据与知识")
        if claim.kind == "recommendation" and not claim.knowledge_refs:
            raise ValueError("核查建议必须引用知识资料")
        numbers = []
        for ref in claim.data_refs:
            record = records.get(ref.call_id)
            if record is None or record.tool != "sql_query":
                raise ValueError("数据引用不是本轮成功的查数调用")
            out = record.output
            if (ref.row >= min(len(out["rows"]), PREVIEW_ROWS) or ref.column >= len(out["columns"])
                    or ref.column >= len(out["rows"][ref.row])):
                raise ValueError("数据引用超出已展示的行列范围")
            value = out["rows"][ref.row][ref.column]
            if not same_cell(value, ref.value):
                raise ValueError("数据引用值与实际单元格不一致")
            if isinstance(value, str) and re.search(r"\d+\s*/\s*\d+\s*运行", value):
                if (re.search(r"台数|\d+\s*台", claim.text) or
                        (re.search(r"\d+\s*/\s*\d+", claim.text) and "采样" not in claim.text)):
                    raise ValueError("运行采样比例必须明确采样口径，不能解释为设备台数")
            numbers.append(add({"kind": "data", "call_id": ref.call_id, "row": ref.row,
                                "column": ref.column, "column_name": out["columns"][ref.column],
                                "value": value, "now": result.now, "source": "模拟运行数据库"}))
        for ref in claim.knowledge_refs:
            record = records.get(ref.call_id)
            if record is None or record.tool != "rag_search":
                raise ValueError("知识引用不是本轮成功的检索调用")
            hit = next((h for h in record.output["hits"] if h["id"] == ref.chunk_id), None)
            if hit is None:
                raise ValueError("知识片段未由该调用检索到")
            if len(compact(ref.quote)) < 8 or compact(ref.quote) not in compact(hit["text"]):
                raise ValueError("引用摘录不在知识原文中")
            metadata = {key: hit.get(key) for key in (
                "document_id", "title", "heading", "source_type", "source", "source_url", "source_page",
                "line_start", "line_end", "url")}
            numbers.append(add({"kind": "knowledge", "call_id": ref.call_id, "chunk_id": ref.chunk_id,
                                "quote": ref.quote, **metadata}))
        numbers = sorted(set(numbers))
        text = claim.text
        if claim.kind in OWN_LABEL:
            text = OWN_LABEL[claim.kind].sub("", text, count=1).strip()
            if not text:
                raise ValueError("陈述内容为空")
        claims.append({"kind": claim.kind, "text": text, "citation_numbers": numbers})
        paragraphs.append(PREFIX[claim.kind] + text + " " + "".join(f"[{n}]" for n in numbers))

    kinds = {c["kind"] for c in citations}
    if result.intent in {"data", "diagnosis"} and not kinds.intersection({"data", "statistic"}):
        raise ValueError("具体厂站查数或诊断缺少本轮数据证据；应说明缺口或拒答")
    if result.intent == "knowledge" and "knowledge" not in kinds:
        raise ValueError("知识回答缺少知识出处")
    if require_hypothesis and missing_hypothesis(claims, result):
        raise ValueError("诊断回答缺少原因假设：请把可能原因写成kind=hypothesis，同时引用数据单元格和知识摘录；"
                         "核查建议只写怎么查")
    return "\n\n".join(paragraphs), claims, citations, None

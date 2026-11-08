"""上层业务应使用网关完整耗时，失败也计入；旧回放明确保留总耗时未知。"""

import json

import pytest

from app.ai.gateway import AttemptRecord, ChatResult, LLMError
from app.ai.rag.retrieval import KnowledgeIndex
from app.ai.rag.service import RAG
from app.ai.reporter import Fact, Reporter
from app.ai.text2sql.schema import SchemaInfo
from app.ai.text2sql.service import Text2SQL, Text2SQLConfig
from app.ai.usage import call_usage, elapsed_ms
from tests.conftest import END
from tests.test_rag import document, responder


class TimedLLM:
    available = True

    def __init__(self, content, fail=False):
        self.content, self.fail = content, fail

    async def chat(self, messages, **kwargs):
        history = [AttemptRecord("fake", 1, "error", 20, 80, status_code=429),
                   AttemptRecord("fake", 2, "success", 10)]
        if self.fail:
            exc = LLMError("不可公开的厂商响应")
            exc.total_latency_ms, exc.attempt_history = 110, history[:1]
            raise exc
        return ChatResult(self.content(messages), "fake", "fake-model", 3, 4, latency_ms=10,
                          total_latency_ms=110, attempt_history=history)


def test_old_cassette_falls_back_without_inventing_complete_timing():
    old = ChatResult("ok", "fake", "fake", latency_ms=7)
    assert elapsed_ms(old) == 7 and call_usage(old)["total_latency_ms"] is None
    hit = ChatResult("ok", "fake", "fake", latency_ms=7, total_latency_ms=0)
    assert elapsed_ms(hit) == 0


@pytest.mark.parametrize("fail", [False, True])
async def test_text2sql_accounts_success_and_failure(fail):
    llm = TimedLLM(lambda m: '{"sql":"","explanation":"没有相关数据"}', fail)
    out = await Text2SQL(llm, None, SchemaInfo({}, []), cfg=Text2SQLConfig(max_repairs=0)).ask("查不存在的信息", END)
    assert out.llm_ms == 110 and len(out.generation_calls) == 1
    call = out.generation_calls[0]
    assert call["total_latency_ms"] == 110
    assert call["status"] == ("error" if fail else "success")
    assert "不可公开" not in json.dumps(out.to_dict(), ensure_ascii=False)


@pytest.mark.parametrize("fail", [False, True])
async def test_rag_accounts_rerank_and_answer_including_failure(fail):
    index = KnowledgeIndex([document("air", "鼓风机跳停后供气下降，溶解氧不足可能影响硝化。")])
    out = await RAG(index, TimedLLM(responder, fail)).ask("鼓风机跳停后怎么办")
    # 重排失败可降级保留原始检索，再尝试回答；两次付费调用的完整耗时均须保留。
    assert out.llm_ms == 220 and len(out.generation_calls) == 2
    assert [c["stage"] for c in out.generation_calls] == ["rerank", "answer"]
    assert out.ok == (not fail)
    assert "不可公开" not in json.dumps(out.to_dict(), ensure_ascii=False)


@pytest.mark.parametrize("fail", [False, True])
async def test_reporter_accounts_success_and_template_fallback(fail):
    facts = {key: Fact(key, key, "零", 0) for key in ("alarms.total", "alarms.unclosed", "orders.open")}
    data = {"kind": "日报", "facts": facts, "events": [], "order_items": [], "devices": [], "plants": []}
    llm = TimedLLM(lambda m: '{"overview":["运行数据见事实表。"],"consumption":["能耗见统计表。"]}', fail)
    _, meta = await Reporter(llm).narrate(data)
    assert meta["llm_ms"] == 110 and len(meta["generation_calls"]) == 1
    assert meta["narrative"] == ("template" if fail else "llm")
    assert "不可公开" not in json.dumps(meta, ensure_ascii=False)

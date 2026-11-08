"""检索预算、已验证缓存、并发、超时与取消；假模型，不发付费请求。"""

import asyncio
import json
import time
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from app.ai.gateway import ChatResult, LLMGateway, OpenAICompatProvider
from app.ai.rag.retrieval import Hit, SearchResult
from app.ai.rag.service import RAG, RerankConfig
from tests.test_rag import document


class Index:
    fingerprint = "corpus-one"
    embedder = SimpleNamespace(model_name="test-embedding")

    def __init__(self, confidence=None, empty=False):
        self.calls = []
        self.confidence = confidence or {}
        self.hits = [] if empty else [
            Hit(document(str(i), "鼓风机的定义与排查资料。" * 3).chunks[0], 10 - i, 10 - i, 1.0)
            for i in range(8)
        ]

    def search(self, question, top_k, mode):
        self.calls.append((question, top_k, mode))
        return SearchResult(deepcopy(self.hits[:top_k]), mode, mode, 2.0, [],
                            confidence=deepcopy(self.confidence))


class Model:
    available = True

    def __init__(self, *, block=False, invalid=False):
        self.calls = []
        self.started, self.release = asyncio.Event(), asyncio.Event()
        self.block, self.invalid = block, invalid
        self.providers = [SimpleNamespace(name="fake", model="fake-flash")]

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        self.started.set()
        if self.block:
            await self.release.wait()
        candidates = json.loads(messages[1]["content"])["candidates"]
        content = "{}" if self.invalid else json.dumps({"ranking": [
            {"id": c["id"], "relevance": 3} for c in reversed(candidates)
        ]})
        return ChatResult(content, "fake", "fake-flash", 17, 11, latency_ms=7,
                          total_latency_ms=13)


def ids(result):
    return [h.chunk.id for h in result.hits]


async def wait_for_waiters(rag, count=2):
    async with asyncio.timeout(2):
        while not rag._locks or next(iter(rag._locks.values()))[1] < count:
            await asyncio.sleep(0)


async def test_budget_and_cache_do_not_rebill_or_share_mutable_results():
    index, model = Index(), Model()
    rag = RAG(index, model)
    first = await rag.search("鼓风机异常如何排查", rerank=True)
    assert index.calls[0][1] == 6 and first.rerank_status == "model"
    assert model.calls[0][1] == {"json_mode": True, "temperature": 0.0,
                                  "max_tokens": 1200, "thinking": False, "use_cache": False}
    expected = ids(first)
    first.hits.clear()
    first.warnings.append("污染")
    first.generation_calls[0]["stage"] = "污染"
    second = await rag.search("鼓风机异常如何排查", rerank=True)
    assert ids(second) == expected and second.warnings == []
    assert second.cache_hit and second.rerank_status == "model" and second.reranked
    assert (second.prompt_tokens, second.completion_tokens, second.llm_ms,
            second.retrieval_ms, second.generation_calls, second.model) == (0, 0, 0, 0, [], None)
    assert len(index.calls) == len(model.calls) == 1


async def test_ttl_lru_and_context_changes_invalidate_cache(monkeypatch):
    from app.ai.rag import service
    clock = [0.0]
    monkeypatch.setattr(service, "time", SimpleNamespace(
        monotonic=lambda: clock[0], perf_counter=time.perf_counter))
    index, model = Index(), Model()
    rag = RAG(index, model, RerankConfig(cache_size=2, cache_ttl_s=10))
    await rag.search("问题甲", rerank=True)
    await rag.search("问题乙", rerank=True)
    assert (await rag.search("问题甲", rerank=True)).cache_hit
    await rag.search("问题丙", rerank=True)
    assert not (await rag.search("问题乙", rerank=True)).cache_hit
    clock[0] = 11
    assert not (await rag.search("问题乙", rerank=True)).cache_hit
    for change in (lambda: setattr(index, "fingerprint", "corpus-two"),
                   lambda: setattr(index, "embedder", SimpleNamespace(model_name="embedding-two")),
                   lambda: setattr(model.providers[0], "model", "fake-flash-two"),
                   lambda: setattr(rag, "cfg", replace(rag.cfg, max_tokens=1400))):
        change()
        assert not (await rag.search("问题乙", rerank=True)).cache_hit
    assert len(rag._cache) <= 2


async def test_concurrent_same_question_shares_one_successful_model_request():
    index, model = Index(), Model(block=True)
    rag = RAG(index, model)
    leader = asyncio.create_task(rag.search("鼓风机异常", rerank=True))
    await model.started.wait()
    follower = asyncio.create_task(rag.search("鼓风机异常", rerank=True))
    await wait_for_waiters(rag)
    model.release.set()
    first, second = await asyncio.gather(leader, follower)
    assert ids(first) == ids(second) and len(model.calls) == 1
    assert second.cache_hit and second.prompt_tokens == 0 and not second.generation_calls
    assert second.retrieval_ms == 2.0  # 等待者确实已经检索过，不能抹去这部分耗时
    assert rag._locks == {}


@pytest.mark.parametrize("cancel_leader", [True, False])
async def test_cancellation_propagates_and_other_caller_recovers(cancel_leader):
    index, model = Index(), Model(block=True)
    rag = RAG(index, model)
    leader = asyncio.create_task(rag.search("鼓风机异常", rerank=True))
    await model.started.wait()
    follower = asyncio.create_task(rag.search("鼓风机异常", rerank=True))
    await wait_for_waiters(rag)
    cancelled, survivor = (leader, follower) if cancel_leader else (follower, leader)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    model.release.set()
    result = await survivor
    assert result.reranked and rag._locks == {}
    assert len(model.calls) == (2 if cancel_leader else 1)
    assert (await rag.search("鼓风机异常", rerank=True)).cache_hit


async def test_timeout_preserves_original_order_and_full_cancelled_attempt_timing():
    calls = []

    async def slow(request):
        calls.append(request)
        await asyncio.Event().wait()

    provider = OpenAICompatProvider("fake", "https://llm.test", None, "fake-flash",
                                    transport=httpx.MockTransport(slow))
    gateway = LLMGateway([provider], max_retries=0)
    index = Index()
    rag = RAG(index, gateway, RerankConfig(timeout_s=0.025))
    try:
        for _ in range(2):
            result = await rag.search("鼓风机异常", rerank=True)
            assert result.rerank_status == "fallback_timeout" and not result.reranked
            assert ids(result) == [h.chunk.id for h in index.hits[:5]]
            assert result.llm_ms >= 15 and result.total_ms >= result.llm_ms
            call = result.generation_calls[0]
            assert call["status"] == "cancelled" and call["total_latency_ms"] >= 15
            assert call["attempt_history"][0]["status"] == "cancelled"
        assert len(calls) == 2 and rag._locks == {} and not rag._cache
        assert gateway.summary()["requests"]["cancelled"] == 2
    finally:
        await gateway.aclose()


async def test_bad_model_response_is_not_cached_in_either_layer():
    calls = []

    def response(request):
        calls.append(request)
        cs = json.loads(request.content)["messages"][1]["content"]
        ranking = [{"id": c["id"], "relevance": 3}
                   for c in json.loads(cs)["candidates"]]
        content = "{}" if len(calls) == 1 else json.dumps({"ranking": ranking})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    provider = OpenAICompatProvider("fake", "https://llm.test", None, "fake-flash",
                                    transport=httpx.MockTransport(response))
    gateway = LLMGateway([provider])
    rag = RAG(Index(), gateway)
    try:
        assert (await rag.search("鼓风机异常", rerank=True)).rerank_status == "fallback_error"
        assert (await rag.search("鼓风机异常", rerank=True)).reranked
        assert len(calls) == 2 and len(gateway.cache) == 0
    finally:
        await gateway.aclose()


async def test_empty_results_cache_and_no_model_fallback_recovers():
    index, model = Index(empty=True), Model()
    rag = RAG(index, model)
    assert not (await rag.search("问题", rerank=True)).hits
    assert (await rag.search("问题", rerank=True)).cache_hit and not model.calls
    rag = RAG(Index(), model)
    model.available = False
    assert (await rag.search("问题", rerank=True)).rerank_status == "fallback_no_model"
    assert not rag._cache
    model.available = True
    assert (await rag.search("问题", rerank=True)).reranked


@pytest.mark.parametrize("question,confidence,skip", [
    ("DO是什么意思", {"same_top": True, "term_coverage": .8, "lexical_margin": 2.1}, True),
    ("DO是什么，今天是否异常", {"same_top": True, "term_coverage": 1, "lexical_margin": 9}, False),
    ("DO是什么，联系人电话是什么", {"same_top": True, "term_coverage": 1, "lexical_margin": 9}, False),
    ("DO是什么意思", {"same_top": False, "term_coverage": 1, "lexical_margin": 9}, False),
    ("DO是什么意思", {"same_top": True, "term_coverage": .3, "lexical_margin": 9}, False),
    ("DO是什么意思", {"same_top": True, "term_coverage": 1, "lexical_margin": 1.1}, False),
    ("氨氮超标如何排查", {"same_top": True, "term_coverage": 1, "lexical_margin": 9}, False),
])
async def test_only_confident_definitions_can_skip(question, confidence, skip):
    model = Model()
    result = await RAG(Index(confidence), model).search(question, rerank=True)
    assert (result.rerank_status == "skipped_confident") == skip
    assert len(model.calls) == (0 if skip else 1)


async def test_repeated_knowledge_answer_disables_thinking_and_does_not_rebill():
    from tests.test_rag import responder
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"model": "fake-flash", "choices": [
            {"message": {"content": responder(body["messages"])}}],
            "usage": {"prompt_tokens": 17, "completion_tokens": 11}})

    provider = OpenAICompatProvider("fake", "https://llm.test", None, "fake-flash",
                                    transport=httpx.MockTransport(respond))
    gateway = LLMGateway([provider])
    rag = RAG(Index(), gateway)
    try:
        first = await rag.ask("鼓风机异常如何排查")
        second = await rag.ask("鼓风机异常如何排查")
        assert first.ok and second.ok and first.citations == second.citations
        assert all(call["thinking"] == {"type": "disabled"} for call in calls)
        assert len(calls) == 2 and first.prompt_tokens == 34
        assert second.prompt_tokens == second.completion_tokens == second.llm_ms == 0
        assert second.generation_calls[0]["from_cache"]
    finally:
        await gateway.aclose()


async def test_disabled_cache_repeats_rerank_and_does_not_leak_locks():
    model = Model()
    rag = RAG(Index(), model, RerankConfig(cache_size=0))
    for _ in range(2):
        assert not (await rag.search("鼓风机异常", rerank=True)).cache_hit
    assert len(model.calls) == 2 and not rag._cache and not rag._locks

"""RAG 边界和 API 测试：假向量/假模型，无模型下载、无付费请求。"""

import json

import numpy as np
import pytest

from app.ai.rag.corpus import (
    Chunk,
    Document,
    load_corpus,
    load_document,
)
from app.ai.rag.retrieval import DenseUnavailable, KnowledgeIndex
from app.ai.rag.service import RAG, validate_answer
from tests.conftest import FakeLLM


def document(doc_id, text):
    c = Chunk(f"{doc_id}:section:1", doc_id, doc_id, "说明", text, "demo", "测试资料", None, None, 1, 1)
    return Document(doc_id, doc_id, "demo", "测试资料", None, text, (c,))


@pytest.fixture
def small_index():
    return KnowledgeIndex([document("air", "鼓风机跳停后供气下降，溶解氧不足可能影响硝化。"),
                           document("chemical", "PAC 加药泵堵塞影响有效药量，可能引起出水总磷上升。")])


def answer(chunk_id, quote, text="资料支持这一排查方向。"):
    return json.dumps({"answerable": True, "claims": [{"text": text, "citations": [
        {"chunk_id": chunk_id, "quote": quote}]}], "reason": ""}, ensure_ascii=False)


def responder(messages):
    payload = json.loads(messages[1]["content"])
    if "candidates" in payload:
        return json.dumps({"ranking": [{"id": c["id"], "relevance": 3} for c in payload["candidates"]]})
    evidence = payload["evidence"][0]
    return answer(evidence["chunk_id"], evidence["text"][:30])


def test_corpus_sources_and_line_numbers():
    docs = load_corpus()
    for doc in docs:
        for chunk in doc.chunks:
            source_lines = "\n".join(doc.content.splitlines()[chunk.line_start - 1:chunk.line_end])
            assert chunk.text in source_lines
            assert 0 < len(chunk.text) <= 500
            assert chunk.source_type in {"demo", "reference"}
    reference = next(d for d in docs if d.id == "reference-case")
    assert {c.source_page for c in reference.chunks} == {1, 2, 3, 4}
    assert all(c.source_url.startswith("https://") for c in reference.chunks)


def test_long_section_splits_with_overlap_and_stable_ids(tmp_path):
    path = tmp_path / "long.md"
    path.write_text('---\nid: long\ntitle: 测试\nsource_type: demo\nsource: 自编\n---\n'
                    '## 标题 {#section}\n' + '这是一个很长的运行记录段落。' * 100)
    first = load_document(path, max_chars=200, overlap=30)
    second = load_document(path, max_chars=200, overlap=30)
    assert len(first.chunks) > 5 and first == second
    assert first.chunks[0].text[-30:] in first.chunks[1].text
    assert len({c.id for c in first.chunks}) == len(first.chunks)


def test_missing_provenance_and_duplicate_anchors_rejected(tmp_path):
    path = tmp_path / "bad.md"
    path.write_text('无来源资料')
    with pytest.raises(ValueError, match="frontmatter"):
        load_document(path)
    path.write_text('---\nid: bad\ntitle: 测试\nsource_type: demo\nsource: 自编\n---\n'
                    '## 一 {#same}\n内容1\n## 二 {#same}\n内容2')
    with pytest.raises(ValueError, match="唯一"):
        load_document(path)


def test_bm25_and_explicit_dense_fallback(small_index):
    hit = small_index.search("鼓风机跳停导致溶解氧下降", mode="bm25").hits[0]
    assert hit.chunk.document_id == "air"
    degraded = small_index.search("加药泵堵塞", mode="hybrid")
    assert degraded.mode == "bm25" and degraded.requested_mode == "hybrid" and degraded.warnings
    with pytest.raises(DenseUnavailable):
        small_index.search("加药泵堵塞", mode="dense")


class FixedEmbedder:
    model_name = "fake-unit-vectors"

    def __init__(self):
        self.calls = 0

    def documents(self, texts):
        self.calls += 1
        return np.array([[1., 0.], [0., 1.]], dtype=np.float32)

    def query(self, text):
        return np.array([0., 2.], dtype=np.float32)


def test_dense_and_rrf_use_vectors_and_cache(small_index, tmp_path):
    emb = FixedEmbedder()
    docs = list(small_index.documents.values())
    index = KnowledgeIndex(docs, emb, tmp_path)
    dense = index.search("表达不同但语义相近的问题", mode="dense", top_k=1)
    assert dense.hits[0].chunk.document_id == "chemical" and dense.hits[0].dense_score == 1.0
    hybrid = index.search("PAC 加药泵堵塞", mode="hybrid", top_k=1)
    assert hybrid.mode == "hybrid" and hybrid.hits[0].chunk.document_id == "chemical"
    assert emb.calls == 1
    KnowledgeIndex(docs, emb, tmp_path)
    assert emb.calls == 1  # 同语料复用缓存
    changed = [document("air", "更新后的鼓风机资料"), docs[1]]
    KnowledgeIndex(changed, emb, tmp_path)
    assert emb.calls == 2  # 原文变化必须重建


async def test_rerank_rejects_unknown_candidate(small_index):
    llm = FakeLLM(lambda m: '{"ranking":[{"id":"invented","relevance":3}]}')
    result = await RAG(small_index, llm).search("鼓风机", rerank=True)
    assert not result.reranked and result.warnings
    assert result.hits[0].chunk.document_id == "air"


async def test_all_irrelevant_refuses_without_generation(small_index):
    def irrelevant(messages):
        cs = json.loads(messages[1]["content"])["candidates"]
        return json.dumps({"ranking": [{"id": c["id"], "relevance": 0} for c in cs]})
    llm = FakeLLM(irrelevant)
    result = await RAG(small_index, llm).ask("这台鼓风机多少钱")
    assert result.refused and not result.ok and not result.error and not result.citations
    assert len(llm.calls) == 1 and result.model == "fake-model"


def test_citation_validation_requires_actual_retrieved_quote(small_index):
    search = small_index.search("鼓风机", mode="bm25")
    chunk = search.hits[0].chunk
    text, sources, reason = validate_answer(answer(chunk.id, chunk.text[:20]), search)
    assert "[1]" in text and sources[0]["quote"] == chunk.text[:20] and reason is None
    with pytest.raises(ValueError, match="片段"):
        validate_answer(answer("unknown", chunk.text[:20]), search)
    with pytest.raises(ValueError, match="原文"):
        validate_answer(answer(chunk.id, "这段摘录其实并不存在于资料里"), search)


async def test_invalid_citations_fail_closed(small_index):
    def bad(messages):
        if "candidates" in json.loads(messages[1]["content"]):
            return responder(messages)
        return answer("not-retrieved", "完全捏造的出处和运行结论")
    llm = FakeLLM(bad)
    result = await RAG(small_index, llm).ask("鼓风机跳停后怎么办")
    assert not result.ok and not result.refused and not result.answer and not result.citations
    assert result.attempts == 2 and "校验" in result.error


async def test_overlong_answer_gets_specific_repair_feedback(small_index):
    generations = 0

    def too_long_then_fixed(messages):
        nonlocal generations
        if "candidates" in json.loads(messages[1]["content"]):
            return responder(messages)
        generations += 1
        draft = json.loads(responder(messages))
        if generations == 1:
            draft["claims"] *= 5
        else:
            assert "claims: too_long" in messages[-1]["content"]
        return json.dumps(draft, ensure_ascii=False)

    result = await RAG(small_index, FakeLLM(too_long_then_fixed)).ask("鼓风机跳停后怎么办")
    assert result.ok and result.attempts == 2
    assert result.validation_errors == ["claims: too_long"]


async def test_generation_with_verified_citations(small_index):
    result = await RAG(small_index, FakeLLM(responder)).ask("鼓风机跳停后怎么办")
    assert result.ok and result.citations and "[1]" in result.answer
    assert result.retrieval.reranked and result.prompt_tokens == 200


def test_knowledge_and_search_endpoints(client):
    catalog = client.get("/api/ai/knowledge").json()
    assert catalog["chunks"] > 20 and not catalog["dense_available"]
    search = client.post("/api/ai/rag/search", json={"question": "碳源管道渗漏", "mode": "hybrid"})
    assert search.status_code == 200 and search.json()["mode"] == "bm25"
    hit = search.json()["hits"][0]
    source = client.get(hit["url"])
    assert source.status_code == 200 and source.json()["text"] == hit["text"]
    doc = client.get('/api/ai/knowledge/' + hit['document_id'])
    assert doc.status_code == 200 and hit["text"] in doc.json()["content"]
    assert client.get("/api/ai/knowledge/chunks/missing").status_code == 404
    assert client.post("/api/ai/rag/search", json={"question": "碳源投加", "mode": "dense"}).status_code == 503
    assert client.post("/api/ai/rag/search", json={"question": "  "}).status_code == 422


def test_rag_endpoint(client, fake_llm):
    fake_llm.responder = responder
    response = client.post("/api/ai/rag", json={"question": "碳源投加管道渗漏怎么排查？"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"], body
    assert body["model"] == "fake-model" and body["citations"]
    assert client.get(body["citations"][0]["url"]).status_code == 200

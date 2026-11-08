"""本地中文向量 + BM25 + Reciprocal Rank Fusion，明确暴露降级状态。"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from app.ai.rag.corpus import Chunk, Document, corpus_fingerprint, load_corpus

log = logging.getLogger(__name__)
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


class Embedder(Protocol):
    model_name: str

    def documents(self, texts: list[str]) -> np.ndarray: ...
    def query(self, text: str) -> np.ndarray: ...


class LocalEmbedder:
    """ONNX CPU 推理；服务启动只读本地模型，下载由 build 命令显式完成。"""

    model_name = EMBEDDING_MODEL

    def __init__(self, cache_dir: Path, *, allow_download: bool = False):
        from fastembed import TextEmbedding
        self.lock = threading.Lock()
        self.model = TextEmbedding(self.model_name, cache_dir=str(cache_dir), threads=2,
                                   providers=["CPUExecutionProvider"], local_files_only=not allow_download)

    def documents(self, texts: list[str]) -> np.ndarray:
        with self.lock:
            return np.asarray(list(self.model.passage_embed(texts)), dtype=np.float32)

    def query(self, text: str) -> np.ndarray:
        with self.lock:
            return np.asarray(next(self.model.query_embed(text)), dtype=np.float32)


def tokens(text: str) -> list[str]:
    result = re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower())
    for word in re.findall(r"[㐀-鿿]+", text):
        result += [word] if len(word) == 1 else [word[i:i + 2] for i in range(len(word) - 1)]
    return result


class BM25:
    def __init__(self, texts: list[str]):
        self.docs = [Counter(tokens(text)) for text in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lengths) / max(1, len(texts)) or 1
        df = Counter(term for doc in self.docs for term in doc)
        self.idf = {term: math.log(1 + (len(texts) - n + 0.5) / (n + 0.5)) for term, n in df.items()}

    def scores(self, question: str) -> np.ndarray:
        terms = set(tokens(question))
        return np.asarray([
            sum(self.idf.get(term, 0) * doc[term] * 2.5 /
                (doc[term] + 1.5 * (0.25 + 0.75 * length / self.avgdl))
                for term in terms if term in doc)
            for doc, length in zip(self.docs, self.lengths)
        ])


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    lexical_score: float
    dense_score: float | None
    relevance: int | None = None

    def to_dict(self) -> dict:
        return {**self.chunk.to_dict(), "score": round(self.score, 6),
                "lexical_score": round(self.lexical_score, 4),
                "dense_score": round(self.dense_score, 4) if self.dense_score is not None else None,
                "relevance": self.relevance}


@dataclass
class SearchResult:
    hits: list[Hit]
    mode: str
    requested_mode: str
    retrieval_ms: float
    warnings: list[str]
    reranked: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_ms: float = 0.0
    model: str | None = None
    generation_calls: list[dict] = field(default_factory=list)
    total_ms: float = 0.0
    rerank_status: str = "not_requested"
    cache_hit: bool = False
    cache_age_s: float = 0.0
    confidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"hits": [h.to_dict() for h in self.hits], "mode": self.mode,
                "requested_mode": self.requested_mode, "retrieval_ms": self.retrieval_ms,
                "warnings": self.warnings, "reranked": self.reranked,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "llm_ms": self.llm_ms, "model": self.model, "generation_calls": self.generation_calls,
                "total_ms": self.total_ms, "rerank_status": self.rerank_status,
                "cache_hit": self.cache_hit, "cache_age_s": self.cache_age_s, "confidence": self.confidence}


class DenseUnavailable(RuntimeError):
    pass


class KnowledgeIndex:
    def __init__(self, documents: list[Document], embedder: Embedder | None = None,
                 cache_dir: Path | None = None, dense_error: str | None = None):
        self.documents = {doc.id: doc for doc in documents}
        self.chunks = [chunk for doc in documents for chunk in doc.chunks]
        self.by_id = {chunk.id: chunk for chunk in self.chunks}
        self.fingerprint = corpus_fingerprint(documents)
        self.bm25 = BM25([c.search_text for c in self.chunks])
        self.embedder, self.dense_error = embedder, dense_error
        self.vectors = None
        if embedder is not None:
            self.vectors = self._vectors(embedder, cache_dir)

    def _vectors(self, embedder: Embedder, directory: Path | None) -> np.ndarray:
        # 将切分结果和模型纳入 key，修改文档或切分策略不会误用旧向量。
        identity = json.dumps({"model": embedder.model_name,
                               "chunks": [(c.id, c.search_text) for c in self.chunks]}, ensure_ascii=False)
        key = hashlib.sha256(identity.encode()).hexdigest()
        path = directory / f"{key}.npy" if directory else None
        matrix = None
        if path and path.exists():
            try:
                candidate = np.load(path, allow_pickle=False)
                if candidate.ndim == 2 and candidate.shape[0] == len(self.chunks) and np.isfinite(candidate).all():
                    matrix = candidate
            except (OSError, ValueError):
                log.warning("RAG 向量缓存无效，重新构建")
        if matrix is None:
            matrix = embedder.documents([c.search_text for c in self.chunks])
            if matrix.ndim != 2 or matrix.shape[0] != len(self.chunks) or not np.isfinite(matrix).all():
                raise ValueError("嵌入模型返回了无效向量")
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
                    tmp = Path(f.name)
                    np.save(f, matrix, allow_pickle=False)
                try:
                    os.replace(tmp, path)
                finally:
                    tmp.unlink(missing_ok=True)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def status(self) -> dict:
        return {"documents": len(self.documents), "chunks": len(self.chunks),
                "fingerprint": self.fingerprint, "dense_available": self.embedder is not None,
                "embedding_model": self.embedder.model_name if self.embedder else None,
                "dense_error": self.dense_error}

    def search(self, question: str, top_k: int = 5, mode: str = "hybrid") -> SearchResult:
        if mode not in {"bm25", "dense", "hybrid"} or not 1 <= top_k <= 20:
            raise ValueError("检索模式或 top_k 非法")
        t0 = time.perf_counter()
        actual, warnings = mode, []
        if mode == "dense" and self.embedder is None:
            raise DenseUnavailable("本地向量模型未就绪，请先运行 RAG build 命令")
        if mode == "hybrid" and self.embedder is None:
            actual = "bm25"
            warnings.append("本地向量模型未就绪，本次使用 BM25")
        lexical = self.bm25.scores(question)
        dense = None
        if actual in {"dense", "hybrid"}:
            query = self.embedder.query(question)
            query /= max(float(np.linalg.norm(query)), 1e-12)
            dense = self.vectors @ query
        # 每路最多取 20 个候选；融合只比较排名，不混加量纲不同的原始分数。
        lexical_order = sorted(range(len(self.chunks)), key=lambda i: (-lexical[i], self.chunks[i].id))
        lexical_order = [i for i in lexical_order if lexical[i] > 0][:20]
        dense_order = sorted(range(len(self.chunks)), key=lambda i: (-dense[i], self.chunks[i].id))[:20] if dense is not None else []
        if actual == "bm25":
            scores = {i: float(lexical[i]) for i in lexical_order}
        elif actual == "dense":
            scores = {i: float(dense[i]) for i in dense_order}
        else:
            scores = {}
            for order in (lexical_order, dense_order):
                for rank, i in enumerate(order, 1):
                    scores[i] = scores.get(i, 0) + 1 / (60 + rank)
        order = sorted(scores, key=lambda i: (-scores[i], self.chunks[i].id))[:top_k]
        hits = [Hit(self.chunks[i], scores[i], float(lexical[i]),
                    float(dense[i]) if dense is not None else None) for i in order]
        query_terms = set(tokens(question))
        top = lexical_order[0] if lexical_order else None
        coverage = (len(query_terms & self.bm25.docs[top].keys()) / len(query_terms)
                    if top is not None and query_terms else 0.0)
        margin = (float(lexical[top]) / max(float(lexical[lexical_order[1]]), 1e-12)
                  if len(lexical_order) > 1 else None)
        confidence = {"same_top": bool(lexical_order and dense_order and lexical_order[0] == dense_order[0]),
                      "lexical_top": self.chunks[top].id if top is not None else None,
                      "term_coverage": round(coverage, 4), "lexical_margin": margin}
        return SearchResult(hits, actual, mode, round((time.perf_counter() - t0) * 1000, 1), warnings,
                            confidence=confidence)


def load_index(corpus_dir: Path, model_cache: Path, vector_cache: Path, *, dense: bool = True,
               allow_download: bool = False) -> KnowledgeIndex:
    docs = load_corpus(corpus_dir)
    if not dense:
        return KnowledgeIndex(docs, dense_error="配置关闭了向量检索")
    try:
        embedder = LocalEmbedder(model_cache, allow_download=allow_download)
        return KnowledgeIndex(docs, embedder, vector_cache)
    except Exception as exc:
        # 资料和 BM25 可独立使用；显式报告降级，不能伪装成向量检索成功。
        log.warning("RAG 向量检索未启用：%s", type(exc).__name__)
        return KnowledgeIndex(docs, dense_error=f"{type(exc).__name__}：请运行 python -m app.ai.rag.build")

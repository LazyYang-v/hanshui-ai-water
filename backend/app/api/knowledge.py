"""知识库资料、检索调试与带引用的知识问答。"""

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.ai.rag.retrieval import DenseUnavailable
from app.ai.rag.service import RAG

router = APIRouter(prefix="/api/ai", tags=["知识库"])


class RAGRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=8)


class SearchRequest(RAGRequest):
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    rerank: bool = False


def get_rag(request: Request) -> RAG:
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise HTTPException(503, "知识库未就绪")
    return rag


@router.get("/knowledge", summary="知识库状态、来源与文档清单")
async def knowledge(request: Request) -> dict:
    rag = get_rag(request)
    return {**rag.index.status(), "items": [doc.summary() for doc in rag.index.documents.values()]}


@router.get("/knowledge/chunks/{chunk_id}", summary="查看引用片段原文与来源位置")
async def chunk(chunk_id: str, request: Request) -> dict:
    item = get_rag(request).index.by_id.get(chunk_id)
    if item is None:
        raise HTTPException(404, "知识片段不存在")
    return item.to_dict()


@router.get("/knowledge/{document_id}", summary="查看知识文档原文与全部片段")
async def document(document_id: str, request: Request) -> dict:
    item = get_rag(request).index.documents.get(document_id)
    if item is None:
        raise HTTPException(404, "知识文档不存在")
    return {**item.summary(), "content": item.content, "sections": [c.to_dict() for c in item.chunks]}


@router.post("/rag/search", summary="BM25 / 中文向量 / 混合检索，可选 Flash 重排")
async def search(body: SearchRequest, request: Request) -> dict:
    try:
        result = await get_rag(request).search(body.question, body.top_k, body.mode, body.rerank)
    except DenseUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return result.to_dict()


@router.post("/rag", summary="知识问答：混合检索 → Flash 重排 → 带引用回答")
async def ask(body: RAGRequest, request: Request) -> dict:
    rag = get_rag(request)
    if rag.gateway is None or not rag.gateway.available:
        raise HTTPException(503, "模型未配置，请在 .env 中配置 DeepSeek Flash")
    return (await rag.ask(body.question, body.top_k)).to_dict()

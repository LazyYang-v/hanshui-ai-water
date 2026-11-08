"""RAG 检索开发集：BM25、BGE、混合检索、混合 + Flash 重排。

uv run python -m app.ai.rag.build
uv run python -m evals.rag.run
uv run python -m evals.rag.run --mode replay
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import re
from dataclasses import asdict
from pathlib import Path

import yaml

from app.ai.gateway import Cassette, LLMGateway, OpenAICompatProvider
from app.ai.rag.retrieval import load_index
from app.ai.rag.service import RAG, RERANK_SYSTEM, RerankConfig
from app.config import settings

HERE = Path(__file__).parent
CONFIGS = {"bm25": ("bm25", False), "dense": ("dense", False),
           "hybrid": ("hybrid", False), "hybrid_rerank": ("hybrid", True)}


def score(expected: list[str], retrieved: list[str]) -> dict:
    """按章节计分，避免同章节拆成多块后重复得分；库外题不混入 Recall。"""
    if not expected:
        return {"recall_at_5": None, "mrr_at_5": None, "hit_at_5": None}
    sections = [chunk.rsplit(":", 1)[0] for chunk in retrieved[:5]]
    matches = set(expected) & set(sections)
    first = next((i for i, section in enumerate(sections, 1) if section in expected), None)
    return {"recall_at_5": len(matches) / len(set(expected)),
            "mrr_at_5": 1 / first if first else 0, "hit_at_5": bool(first)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["record", "replay"], default="record")
    ap.add_argument("--label", required=True, help="新结果标签，不覆盖历史结果")
    ap.add_argument("--profile", choices=["legacy", "optimized"], default="optimized")
    a = ap.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", a.label):
        ap.error("label只能用字母、数字、下划线或连字符")
    output = HERE / "results" / a.label
    output.mkdir(parents=True, exist_ok=False)
    cfg = (RerankConfig(candidates=8, max_tokens=4000, thinking=None, timeout_s=120,
                        skip_simple=False, cache_size=0) if a.profile == "legacy"
           else RerankConfig(cache_size=0))
    dataset = yaml.safe_load((HERE / "dataset.yaml").read_text())
    index = await asyncio.to_thread(load_index, settings.rag_corpus_dir, settings.rag_model_cache,
                                   settings.rag_vector_cache)
    if not index.status()["dense_available"]:
        raise SystemExit("向量模型未就绪，请先运行 python -m app.ai.rag.build；评测不允许静默降级")
    known_sections = {chunk.id.rsplit(":", 1)[0] for chunk in index.chunks}
    assert len({item['id'] for item in dataset}) == len(dataset)
    for item in dataset:
        if not set(item['relevant']) <= known_sections:
            raise ValueError(f"{item['id']} 引用了不存在的章节")
    identity = index.fingerprint + (HERE / "dataset.yaml").read_text() + RERANK_SYSTEM
    if a.profile != "legacy":
        identity += json.dumps(asdict(cfg), sort_keys=True)
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()[:16]
    providers = [] if a.mode == "replay" else [OpenAICompatProvider.from_env("deepseek")]
    if providers and providers[0].model != "deepseek-flash":
        await providers[0].aclose()
        raise SystemExit("本项目 RAG 评测仅使用 deepseek-flash，请核对 DEEPSEEK_MODEL")
    gateway = LLMGateway(providers, mode=a.mode, cassette=Cassette(HERE / "cassettes" / f"{fingerprint}.jsonl"),
                         cache_size=0)
    rag = RAG(index, gateway, cfg)
    rows, summaries = [], {}
    try:
        for name, (mode, rerank) in CONFIGS.items():
            sem = asyncio.Semaphore(3)

            async def one(item):
                async with sem:
                    result = await rag.search(item['q'], 5, mode, rerank)
                # 超时/失败的回退也参与质量和耗时统计，不能只报告成功重排。
                ids = [h.chunk.id for h in result.hits]
                metrics = score(item['relevant'], ids)
                print(name, item['id'], metrics['recall_at_5'], flush=True)
                return {"config": name, "id": item['id'], "question": item['q'], "expected": item['relevant'],
                        "retrieved": ids, **metrics, "retrieval_ms": result.retrieval_ms,
                        "llm_ms": result.llm_ms, "model": result.model, "total_ms": result.total_ms,
                        "rerank_status": result.rerank_status, "warnings": result.warnings,
                        "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens}

            group = await asyncio.gather(*(one(item) for item in dataset))
            positives = [r for r in group if r['expected']]
            lat = sorted(r['total_ms'] for r in group)
            summaries[name] = {
                "questions": len(positives),
                "recall_at_5": statistics.mean(r['recall_at_5'] for r in positives),
                "hit_at_5": statistics.mean(r['hit_at_5'] for r in positives),
                "mrr_at_5": statistics.mean(r['mrr_at_5'] for r in positives),
                "p50_ms": statistics.median(lat), "p95_ms": lat[int(0.95 * (len(lat) - 1))],
                "mean_tokens": statistics.mean(r['prompt_tokens'] + r['completion_tokens'] for r in group),
            }
            print(name, summaries[name], flush=True)
            rows.extend(group)
    finally:
        await gateway.aclose()
    (output / f"{a.mode}.jsonl").write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    metadata = {**index.status(), "mode": a.mode, "fingerprint": fingerprint,
                "reranker": "deepseek-flash", "top_k": 5, "profile": a.profile, "rerank_config": asdict(cfg),
                "dataset_sha256": hashlib.sha256((HERE / "dataset.yaml").read_bytes()).hexdigest()}
    (output / f"{a.mode}.summary.json").write_text(json.dumps({"metadata": metadata, "summaries": summaries},
                                                            ensure_ascii=False, indent=2))
    lines = ["# 水务知识检索评测", "",
             f"语料 {len(index.documents)} 篇、{len(index.chunks)} 片段；30 道可答题 + 4 道库外题；模式 `{a.mode}`。", "",
             "这是人工整理的小规模开发集，不能代表未知生产资料上的表现。"
             "按章节统计 Recall@5；每道题先算召回比例，再对可答题求平均。"
             "MRR@5 为首个相关片段的倒数排名；Hit@5 为至少命中一个相关章节的题目比例。"
             "四道库外题不计入召回率，其检索结果保留在逐题记录中；检索返回片段并不等同于回答有依据。", "",
             "重排使用 DeepSeek Flash，为语义评分重排而非本地 cross-encoder。"
             "向量使用真实 BGE-small-zh-v1.5（ONNX CPU，512维），混合使用 RRF。"
             "本地缓存关闭，厂商可能命中上下文缓存。延迟记录本次检索/重排的真实墙钟，包括内部等待；"
             "回放墙钟仅表示离线运行速度，LLM字段另保留录制耗时，不能用回放墙钟代表在线延迟。", "",
             "| 配置 | Recall@5 | Hit@5 | MRR@5 | P50 | P95 | 平均 token |",
             "|---|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['recall_at_5']:.1%} | {s['hit_at_5']:.1%} | {s['mrr_at_5']:.3f} | "
                     f"{s['p50_ms']:.1f}ms | {s['p95_ms']:.1f}ms | {s['mean_tokens']:.0f} |")
    lines += ["", "## 未完全召回的问题", ""]
    for row in rows:
        if row['expected'] and row['recall_at_5'] < 1:
            lines.append(f"- {row['config']} / {row['id']}：{row['question']}；召回 {row['recall_at_5']:.0%}。")
    lines += ["", "## 复现配置", "", "```json", json.dumps(metadata, ensure_ascii=False, indent=2), "```", ""]
    (output / f"{a.mode}.md").write_text('\n'.join(lines))
    print('报告：', output / f'{a.mode}.md')


if __name__ == '__main__':
    asyncio.run(main())

"""同题比较旧重排与限时重排：真实墙钟、检索质量、冷请求与服务缓存分开记。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.ai.gateway import Cassette, LLMGateway, OpenAICompatProvider
from app.ai.rag.retrieval import load_index
from app.ai.rag.service import RAG, RERANK_SYSTEM, RerankConfig
from app.config import settings
from evals.rag.run import score

HERE = Path(__file__).parent
CONFIGS = {
    "legacy": RerankConfig(candidates=8, max_tokens=4000, thinking=None, timeout_s=120,
                           skip_simple=False, cache_size=0),
    "optimized": RerankConfig(),
}


def summarize(rows):
    positives = [r for r in rows if r["expected"]]
    negatives = [r for r in rows if not r["expected"]]
    latencies = sorted(r["client_ms"] for r in rows)
    return {"requests": len(rows), "positive_questions": len(positives),
            "recall_at_5": statistics.mean(r["recall_at_5"] for r in positives) if positives else None,
            "hit_at_5": statistics.mean(r["hit_at_5"] for r in positives) if positives else None,
            "mrr_at_5": statistics.mean(r["mrr_at_5"] for r in positives) if positives else None,
            "p50_ms": statistics.median(latencies),
            "p95_ms": latencies[int(.95 * (len(latencies) - 1))],
            "mean_tokens": statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in rows),
            "fallbacks": sum(r["rerank_status"].startswith("fallback") for r in rows),
            "skipped_confident": sum(r["rerank_status"] == "skipped_confident" for r in rows),
            "out_of_scope_empty": sum(not r["retrieved"] for r in negatives),
            "out_of_scope_questions": len(negatives)}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", choices=["record", "replay"], default="record")
    parser.add_argument("--cassette", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("label只能包含字母、数字、连字符和下划线")
    output = HERE / "results" / args.label
    output.mkdir(parents=True, exist_ok=False)
    dataset_path = HERE / "dataset.yaml"
    dataset = yaml.safe_load(dataset_path.read_text())
    if args.limit is not None:
        if args.limit < 1:
            parser.error("limit必须大于0")
        dataset = dataset[:args.limit]
    index = await asyncio.to_thread(load_index, settings.rag_corpus_dir, settings.rag_model_cache,
                                   settings.rag_vector_cache)
    if not index.status()["dense_available"]:
        raise SystemExit("真实BGE未就绪，评测不能静默降级")
    known = {c.id.rsplit(":", 1)[0] for c in index.chunks}
    assert len({r["id"] for r in dataset}) == len(dataset)
    assert all(set(r["relevant"]) <= known for r in dataset)
    providers = [] if args.mode == "replay" else [OpenAICompatProvider.from_env("deepseek")]
    if providers and providers[0].model != "deepseek-flash":
        await providers[0].aclose()
        raise SystemExit("本评测只允许deepseek-flash")
    cassette = args.cassette or output / "calls.jsonl"
    if args.mode == "replay" and not cassette.is_file():
        raise SystemExit("回放必须指定已有cassette")
    gateway = LLMGateway(providers, mode=args.mode, cassette=Cassette(cassette), cache_size=0)
    engines = {name: RAG(index, gateway, config) for name, config in CONFIGS.items()}
    metadata = {"checked_at": datetime.now(timezone.utc).isoformat(), **index.status(),
                "mode": args.mode, "model": "deepseek-flash", "concurrency": args.concurrency,
                "configs": {name: asdict(cfg) for name, cfg in CONFIGS.items()},
                "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                "rerank_prompt_sha256": hashlib.sha256(RERANK_SYSTEM.encode()).hexdigest(),
                "note": "同题成对请求，交替先后顺序；并发数固定。冷请求含检索、等待和全部重排尝试。缓存命中单列。"
                        "小规模开发集；超时仍计入质量与耗时，不能只统计成功重排。"}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    rows = []
    sem = asyncio.Semaphore(args.concurrency)

    async def one(position, item):
        async with sem:
            names = list(CONFIGS) if position % 2 == 0 else list(reversed(CONFIGS))
            for name in names:
                start = time.perf_counter()
                result = await engines[name].search(item["q"], 5, "hybrid", True)
                client_ms = round((time.perf_counter() - start) * 1000, 1)
                retrieved = [h.chunk.id for h in result.hits]
                row = {"config": name, "id": item["id"], "question": item["q"],
                       "expected": item["relevant"], "retrieved": retrieved,
                       **score(item["relevant"], retrieved), "client_ms": client_ms,
                       **{k: v for k, v in result.to_dict().items() if k != "hits"}}
                if name == "optimized" and not result.rerank_status.startswith("fallback"):
                    warm = await engines[name].search(item["q"], 5, "hybrid", True)
                    assert warm.cache_hit and not warm.generation_calls
                    assert warm.prompt_tokens == warm.completion_tokens == warm.llm_ms == 0
                    assert [h.chunk.id for h in warm.hits] == retrieved
                    row["warm_ms"] = warm.total_ms
                rows.append(row)
                with (output / "rows.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps({"config": name, "id": item["id"], "recall": row["recall_at_5"],
                                  "status": result.rerank_status, "ms": client_ms}, ensure_ascii=False), flush=True)

    try:
        await asyncio.gather(*(one(i, item) for i, item in enumerate(dataset)))
    finally:
        (output / "gateway.json").write_text(json.dumps(gateway.summary(), ensure_ascii=False, indent=2) + "\n")
        await gateway.aclose()
    summaries = {name: summarize([r for r in rows if r["config"] == name]) for name in CONFIGS}
    warm = [r["warm_ms"] for r in rows if "warm_ms" in r]
    report = {"metadata": metadata, "summaries": summaries,
              "warm": {"requests": len(warm), "p50_ms": statistics.median(warm) if warm else None,
                       "max_ms": max(warm) if warm else None}}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# RAG同题质量与耗时对照", "", metadata["note"], "",
             "库外题只检查过滤边界，不混入Recall；有片段不代表可以据此回答。", "",
             "|配置|Recall@5|Hit@5|MRR@5|冷请求P50|冷请求P95|平均token|降级|库外空结果|",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        lines.append(f"|{name}|{s['recall_at_5']:.1%}|{s['hit_at_5']:.1%}|{s['mrr_at_5']:.3f}|"
                     f"{s['p50_ms']:.1f}ms|{s['p95_ms']:.1f}ms|{s['mean_tokens']:.0f}|{s['fallbacks']}|"
                     f"{s['out_of_scope_empty']}/{s['out_of_scope_questions']}|")
    lines += ["", "逐题原始结果、完整尝试计时和调用录制保存在同目录。暖缓存不重复累计模型用量。", ""]
    (output / "report.md").write_text("\n".join(lines))
    print(json.dumps(report["summaries"], ensure_ascii=False), flush=True)
    print(f"报告：{output / 'report.md'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

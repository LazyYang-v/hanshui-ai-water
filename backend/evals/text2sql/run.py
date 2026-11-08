"""Text2SQL 评测：结果匹配率 + 消融实验 + 延迟/token。

    uv run python -m evals.text2sql.build_db                # 首次建立独立评测库
    uv run python -m evals.text2sql.run --validate-only     # 只执行金标，不调用模型
    uv run python -m evals.text2sql.run                    # Flash，五组配置，录制响应
    uv run python -m evals.text2sql.run --mode replay      # 使用相同配置离线复现

正式结果写入 results/{model}-{mode}.*，回放不会覆盖真实调用报告。
每个配置使用单独的 cassette，避免相同提示词的多次采样互相覆盖。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import statistics
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import yaml
from psycopg_pool import AsyncConnectionPool

from app import config  # noqa: F401 加载 .env
from app.ai.gateway import PRESETS, Cassette, LLMGateway, OpenAICompatProvider
from app.ai.text2sql import ExampleIndex, Text2SQL, Text2SQLConfig
from app.ai.text2sql.guard import guard
from app.ai.text2sql.schema import load_schema
from app.ai.text2sql.service import SYSTEM, execute_readonly
from evals.text2sql.build_db import DAYS, EVAL_NOW, SEED, STEP_S, readonly_url
from evals.text2sql.score import match

HERE = Path(__file__).parent
CONFIGS = {
    "full": Text2SQLConfig(),
    "no_fewshot": Text2SQLConfig(use_fewshot=False),
    "no_prune": Text2SQLConfig(prune_schema=False),
    "no_repair": Text2SQLConfig(max_repairs=0),
    "bare": Text2SQLConfig(use_fewshot=False, prune_schema=False, max_repairs=0),
}
CONFIG_DESC = {
    "full": "完整（few-shot 检索 + schema 裁剪 + 自修复）",
    "no_fewshot": "去掉 few-shot",
    "no_prune": "去掉 schema 裁剪（全量 schema）",
    "no_repair": "去掉自修复",
    "bare": "只有规则和全量 schema",
}


def load_dataset(ids: set[str] | None = None) -> list[dict]:
    items = yaml.safe_load((HERE / "dataset.yaml").read_text())
    known = {it["id"] for it in items}
    if len(known) != len(items):
        raise ValueError("评测题 id 重复")
    if ids and ids - known:
        raise ValueError(f"未知题号：{sorted(ids - known)}")
    examples = {e["q"] for e in ExampleIndex.load().examples}
    if any(it["q"] in examples for it in items):
        raise ValueError("评测题与 few-shot 示例重复")
    return [it for it in items if not ids or it["id"] in ids]


async def gold_results(pool, schema, items) -> dict[str, list]:
    out = {}
    for it in items:
        if it["sql"] is None:
            continue
        _, rows, _ = await execute_readonly(pool, guard(it["sql"], schema.names, EVAL_NOW))
        if not rows or all(v is None for v in rows[0]):
            raise ValueError(f"金标 {it['id']} 结果为空，请检查评测库数据")
        out[it["id"]] = rows
    return out


def judge(it: dict, res, gold: dict[str, list]) -> dict:
    if it["sql"] is None:
        return {"correct": res.refused, "refused": res.refused}
    correct = res.ok and match(gold[it["id"]], res.rows, it.get("ordered", False),
                               time_only=it.get("time_only", False))
    return {"correct": correct, "refused": res.refused}


async def run_config(name, cfg, gateway, pool, schema, items, gold, concurrency) -> list[dict]:
    t2s = Text2SQL(gateway, pool, schema, ExampleIndex.load(), cfg)
    sem = asyncio.Semaphore(concurrency)

    async def one(it):
        async with sem:
            res = await t2s.ask(it["q"], EVAL_NOW)
        verdict = judge(it, res, gold)
        mark = "✓" if verdict["correct"] else "✗"
        print(f"  {mark} {it['id']} {it['q'][:28]}" +
              ("" if verdict["correct"] else f"  ← {res.error or '结果不一致'}"[:100]), flush=True)
        return {"config": name, "id": it["id"], "cat": it["cat"], "q": it["q"], **verdict,
                "ok": res.ok, "repaired": len(res.attempts) > 1, "attempts": len(res.attempts),
                "attempt_details": [asdict(a) for a in res.attempts],
                "sql": res.sql, "explanation": res.explanation, "error": res.error,
                "columns": res.columns, "rows": res.rows, "gold_rows": gold.get(it["id"], []),
                "provider": res.provider, "model": res.model,
                "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                "llm_ms": res.llm_ms, "total_ms": res.total_ms}

    print(f"\n== {name}：{CONFIG_DESC[name]}", flush=True)
    return await asyncio.gather(*(one(it) for it in items))


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    answerable = [r for r in rows if r["cat"] != "拒答"]
    rep = [r for r in rows if r["repaired"]]
    # 使用记录的 LLM 延迟，回放时保留它；total_ms 则是本次实际执行的耗时。
    lat = [r["llm_ms"] for r in rows]
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["cat"]].append(r["correct"])
    return {
        "n": n, "correct": sum(r["correct"] for r in rows),
        "ex": sum(r["correct"] for r in rows) / n,
        "ex_answerable": sum(r["correct"] for r in answerable) / max(1, len(answerable)),
        "false_refusal": sum(r["refused"] for r in answerable),
        "repair_rate": len(rep) / n,
        "repair_success": sum(r["correct"] for r in rep) / max(1, len(rep)),
        "lat_p50": statistics.median(lat),
        "lat_p95": sorted(lat)[int(0.95 * (len(lat) - 1))],
        "tokens": statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in rows),
        "prompt_tokens": statistics.mean(r["prompt_tokens"] for r in rows),
        "by_cat": {c: sum(v) / len(v) for c, v in by_cat.items()},
    }


def write_report(model: str, summaries: dict, results: list[dict], path: Path, metadata: dict) -> None:
    cats = list(next(iter(summaries.values()))["by_cat"])
    lines = [f"# Text2SQL 评测报告：{model}", "",
             f"固定评测时钟：{EVAL_NOW:%Y-%m-%d %H:%M %Z}；"
             f"{next(iter(summaries.values()))['n']} 题；运行模式：`{metadata['mode']}`。",
             f"生成时间：{metadata['generated_at']}。实验指纹：`{metadata['fingerprint']}`。", "",
             "## 评测口径", "",
             "这是本项目模拟数据上的开发集，包含 60 道可答题和 4 道拒答题；"
             "已用于发现 schema 裁剪等问题，并非独立保留的测试集。",
             "结果匹配要求行数相同、金标列一一对应，允许列顺序不同和额外说明列。"
             "数值容差为相对 1% 或绝对 0.01；日期统一为北京时间。"
             "仅 E02 允许只答时刻，标记 ordered 的题保留行序。"
             "格式错误、调用失败不算正确拒答。此口径不能与官方 Spider/BIRD EX 直接比较。", "",
             "每配置独立调用且关闭本地缓存；厂商上下文缓存可能命中。"
             "LLM 延迟为各轮调用延迟之和，回放沿用录制值；"
             "逐题 total_ms 是当次本地端到端耗时。token 是请求用量，非人民币成本。"
             "单次运行的准确率与延迟会有波动。", "",
             "## 消融对比", "",
             "| 配置 | 正确题数 | 结果匹配率 | 可答题匹配率 | 误拒 | 触发修复 | 修复后正确 | LLM P50 | LLM P95 | 平均 token |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        lines.append(f"| {CONFIG_DESC[name]} | {s['correct']}/{s['n']} | **{s['ex']:.1%}** | "
                     f"{s['ex_answerable']:.1%} | {s['false_refusal']} | {s['repair_rate']:.0%} | "
                     f"{s['repair_success']:.0%} | {s['lat_p50'] / 1000:.2f}s | {s['lat_p95'] / 1000:.2f}s | "
                     f"{s['tokens']:.0f} |")
    lines += ["", "## 分类匹配率", "", "| 配置 | " + " | ".join(cats) + " |",
              "|---" * (len(cats) + 1) + "|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | " + " | ".join(f"{s['by_cat'].get(c, 0):.0%}" for c in cats) + " |")
    lines += ["", "## 完整配置的错题", ""]
    failures = [r for r in results if r["config"] == "full" and not r["correct"]]
    if not failures:
        lines += ["本次完整配置没有错题。", ""]
    for r in failures:
        lines += [f"### {r['id']} {r['q']}", "", f"- 错误：{r['error'] or '结果与金标不一致'}",
                  f"- 预测前几行：`{json.dumps(r['rows'][:3], ensure_ascii=False)}`",
                  f"- 金标前几行：`{json.dumps(r['gold_rows'][:3], ensure_ascii=False)}`", "",
                  "```sql", r["sql"] or "-- 无", "```", ""]
    lines += ["## 复现配置", "", "```json", json.dumps(metadata, ensure_ascii=False, indent=2), "```", ""]
    path.write_text("\n".join(lines))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--provider", default="deepseek", choices=list(PRESETS))
    ap.add_argument("--model", default=None, help="覆盖默认模型；当前默认 deepseek-flash")
    ap.add_argument("--mode", default="record", choices=["live", "record", "replay"])
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--ids", default="", help="只跑部分题，逗号分隔")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--label", default="", help="结果标签：写入 {model}-{mode}[-subset]-{label}.*，已存在则拒绝覆盖")
    a = ap.parse_args()
    if a.label and not re.fullmatch(r"[A-Za-z0-9_.-]+", a.label):
        ap.error("label 只能包含字母、数字、点、下划线和连字符")
    names = a.configs.split(",")
    if any(n not in CONFIGS for n in names) or len(names) != len(set(names)):
        ap.error("configs 必须是已知且不重复的配置名称")
    if not 1 <= a.concurrency <= 16 or (a.max_tokens is not None and a.max_tokens < 1):
        ap.error("concurrency 范围为 1~16，max-tokens 必须为正整数")
    if a.model:
        os.environ[f"{a.provider.upper()}_MODEL"] = a.model
    model = a.model or os.getenv(f"{a.provider.upper()}_MODEL", PRESETS[a.provider][2])
    model_path = re.sub(r"[^a-zA-Z0-9_.-]", "_", model)
    out = HERE / "results"
    stem = f"{model_path}-{a.mode}{'-subset' if a.ids else ''}{'-' + a.label if a.label else ''}"
    if a.label and not a.validate_only and any((out / f"{stem}{ext}").exists()
                                               for ext in (".jsonl", ".summary.json", ".md")):
        ap.error(f"结果 {stem}.* 已存在；请换一个 label，不覆盖已有评测记录")
    items = load_dataset({x for x in a.ids.split(",") if x})
    configs = {n: CONFIGS[n] if a.max_tokens is None else replace(CONFIGS[n], max_tokens=a.max_tokens) for n in names}
    pool = AsyncConnectionPool(readonly_url(), min_size=1, max_size=a.concurrency + 1, open=False,
                               kwargs={"autocommit": True})
    try:
        await pool.open(wait=True, timeout=10)
        async with pool.connection() as conn:
            schema = await load_schema(conn)
        gold = await gold_results(pool, schema, items)
        print(f"{len(items)} 题，{len(gold)} 条金标 SQL 全部通过校验并执行成功", flush=True)
        if a.validate_only:
            return

        def sha(text):
            return hashlib.sha256(text.encode()).hexdigest()

        signature = {"dataset_sha256": sha((HERE / 'dataset.yaml').read_text()),
                     "prompt_sha256": sha(SYSTEM),
                     "examples_sha256": sha(json.dumps(ExampleIndex.load().examples, ensure_ascii=False)),
                     "schema_sha256": sha(schema.render(list(schema.tables))),
                     "max_tokens": a.max_tokens or Text2SQLConfig().max_tokens,
                     "now": EVAL_NOW.isoformat(), "seed": SEED, "days": DAYS, "step_s": STEP_S}
        fingerprint = sha(json.dumps(signature, sort_keys=True))[:16]
        metadata = {**signature, "fingerprint": fingerprint, "model_requested": model,
                    "provider": a.provider, "mode": a.mode, "concurrency": a.concurrency,
                    "generated_at": datetime.now().astimezone().isoformat(),
                    "configs": {n: asdict(c) for n, c in configs.items()}}
        print(f"模型 {model}，模式 {a.mode}，指纹 {fingerprint}", flush=True)
        results, summaries = [], {}
        for name, cfg in configs.items():
            cassette = Cassette(HERE / "cassettes" / model_path / fingerprint / f"{name}.jsonl")
            providers = [] if a.mode == "replay" else [OpenAICompatProvider.from_env(a.provider)]
            gateway = LLMGateway(providers, cassette=cassette, mode=a.mode, max_retries=3, cache_size=0)
            try:
                rows = await run_config(name, cfg, gateway, pool, schema, items, gold, a.concurrency)
                if a.mode == "replay" and any("回放模式" in (r["error"] or "") for r in rows):
                    raise RuntimeError("录制不完整或配置已变化；本次回放结果不写入正式报告")
                results += rows
                summaries[name] = s = summarize(rows)
                print(f"  → {s['correct']}/{s['n']} ({s['ex']:.1%})  "
                      f"LLM P50 {s['lat_p50'] / 1000:.2f}s  平均 token {s['tokens']:.0f}", flush=True)
            finally:
                await gateway.aclose()
    finally:
        await pool.close()

    metadata["models_returned"] = sorted({r["model"] for r in results if r["model"]})
    metadata["label"] = a.label or None
    out.mkdir(exist_ok=True)
    (out / f"{stem}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    (out / f"{stem}.summary.json").write_text(json.dumps(
        {"metadata": metadata, "summaries": summaries}, ensure_ascii=False, indent=2))
    write_report(model, summaries, results, out / f"{stem}.md", metadata)
    print(f"\n报告：{out / f'{stem}.md'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

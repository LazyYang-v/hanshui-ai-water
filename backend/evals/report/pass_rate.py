"""日报叙述的一次通过率评测：对固定日期重复生成叙述（不写入 report 表），统计一次通过、修复后通过与模板兜底的比例。

只用 Flash；关闭网关缓存，每次都是真实调用（付费）。同时记录第一次尝试是否依赖解析容错才通过，
从同一批样本中同时得到“有容错 / 无容错”两个一次通过率。
用法（backend 目录，先载入 .env）：
  uv run python -m evals.report.pass_rate --dates 2026-11-06 2026-11-07 2026-11-08 --repeats 3 --label after-tolerance
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import date, datetime
from pathlib import Path

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.ai.gateway import LLMGateway, OpenAICompatProvider
from app.ai.reporter import Reporter, collect_facts, period_for
from app.config import settings
from app.deps import LOCAL_TZ

HERE = Path(__file__).parent


async def one(reporter: Reporter, data: dict, day: str, k: int) -> dict:
    t0 = time.perf_counter()
    _, meta = await reporter.narrate(data)
    attempts = meta["attempts"]
    first = attempts[0] if attempts else {"errors": ["无尝试"]}
    rec = {"date": day, "repeat": k, "narrative": meta["narrative"], "attempts": len(attempts),
           "first_pass": not first["errors"], "first_pass_needed_fixes": not first["errors"] and bool(first.get("fixes")),
           "prompt_tokens": meta["prompt_tokens"], "completion_tokens": meta["completion_tokens"],
           "ms": round((time.perf_counter() - t0) * 1000), "model": meta["model"], "attempt_detail": attempts}
    print(f"{day} #{k}: {rec['narrative']} 尝试{rec['attempts']}次 "
          f"{'一次通过' + ('（靠容错）' if rec['first_pass_needed_fixes'] else '') if rec['first_pass'] else '首次失败：' + '；'.join(first['errors'][:2])}"
          f"  {rec['prompt_tokens'] + rec['completion_tokens']} tokens {rec['ms']} ms", flush=True)
    return rec


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    tokens = [r["prompt_tokens"] + r["completion_tokens"] for r in rows]
    return {"n": n,
            "first_pass": sum(r["first_pass"] for r in rows),
            "first_pass_without_fixes": sum(r["first_pass"] and not r["first_pass_needed_fixes"] for r in rows),
            "llm_after_repair": sum(r["narrative"] == "llm" and not r["first_pass"] for r in rows),
            "template": sum(r["narrative"] == "template" for r in rows),
            "tokens_total": sum(tokens), "tokens_mean": round(statistics.mean(tokens)) if tokens else 0,
            "ms_mean": round(statistics.mean(r["ms"] for r in rows)) if rows else 0}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    provider = OpenAICompatProvider.from_env("deepseek")
    if "flash" not in provider.model:
        raise SystemExit(f"只允许 Flash，当前模型 {provider.model}")
    gateway = LLMGateway([provider], cache_size=0)
    reporter = Reporter(gateway)
    pool = AsyncConnectionPool(settings.database_url, min_size=1, max_size=2, open=False)
    await pool.open(wait=True)
    try:
        async with pool.connection() as conn:
            conn.row_factory = dict_row
            now = (await (await conn.execute("SELECT max(ts) AS t FROM plant_reading")).fetchone())["t"].astimezone(LOCAL_TZ)
            datas = {}
            for d in a.dates:
                start, end = period_for("日报", date.fromisoformat(d), now)
                datas[d] = await collect_facts(conn, "日报", start, end, now)
        sem = asyncio.Semaphore(a.concurrency)

        async def guarded(d, k):
            async with sem:
                return await one(reporter, datas[d], d, k)
        rows = await asyncio.gather(*(guarded(d, k) for d in a.dates for k in range(a.repeats)))
    finally:
        await pool.close()
        await gateway.aclose()
    summary = summarize(rows)
    out = {"label": a.label, "generated_at": datetime.now().astimezone().isoformat(), "data_now": now.isoformat(),
           "model": provider.model, "dates": a.dates, "repeats": a.repeats, "summary": summary,
           "by_date": {d: summarize([r for r in rows if r["date"] == d]) for d in a.dates}, "rows": rows}
    path = HERE / "results" / f"pass-rate-{a.label}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    s = summary
    print(f"\n一次通过 {s['first_pass']}/{s['n']}（其中不靠容错 {s['first_pass_without_fixes']}），修复后通过 "
          f"{s['llm_after_repair']}，模板兜底 {s['template']}；平均 {s['tokens_mean']} tokens、{s['ms_mean']} ms；共 {s['tokens_total']} tokens\n→ {path}")


if __name__ == "__main__":
    asyncio.run(main())

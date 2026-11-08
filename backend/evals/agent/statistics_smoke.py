"""真实 Flash 统计语义验收：答案须选择固定统计事实，引用值、文本、时段逐项匹配。

    uv run python -m evals.agent.statistics_smoke --label optimized

纯函数测试使用人工给定金标；本脚本检验自然语言规划到只读统计及回答引用的整条链路。
"""

import argparse
import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx

from evals.agent.smoke import check_citations
from evals.http import alogin

PERIOD = {"start": "2026-11-06T00:00:00+08:00", "end": "2026-11-07T00:00:00+08:00"}
CASES = [
    ("duration", "黄官镇2026年11月6日出水氨氮按5分钟均值统计，累计超标多长时间、最长连续超标多长时间？"
     "另外有几个小时桶内曾出现超标原始采样？请分别引用固定统计，不把小时桶数量当作连续时长。",
     {"plant": "P06", "metric": "nh3n_out", **PERIOD,
      "statistics": ["exceed_hours", "longest_exceed_hours", "hours_with_exceed"]}),
    ("peaks", "黄官镇2026年11月6日出水氨氮的原始采样瞬时最大值与5分钟均值最大值分别是多少？请区分口径。",
     {"plant": "P06", "metric": "nh3n_out", **PERIOD,
      "statistics": ["instant_max", "bucket_mean_max"]}),
    ("ranking", "2026年11月6日，按5分钟均值等权计算各厂站出水氨氮的时段平均值。"
     "黄官镇在全区降序密集排名第几？全区这个平均值的范围和最高的厂站分别是什么？",
     {"plant": "P06", "metric": "nh3n_out", **PERIOD, "rank_by": "mean",
      "statistics": ["rank", "range", "top"]}),
]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label):
        parser.error("label只允许字母、数字、下划线和连字符")
    path = Path(__file__).parent / "results" / f"statistics-{args.label}.json"
    if path.exists():
        parser.error(f"不覆盖已有记录：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"checked_at": datetime.now(UTC).isoformat(), "cases": []}
    async with httpx.AsyncClient(base_url=args.url, timeout=220, trust_env=False) as client:
        await alogin(client)
        gateway = (await client.get("/api/ai/gateway")).raise_for_status().json()
        assert set(gateway["providers"]) == {"deepseek"}
        assert gateway["providers"]["deepseek"]["model"] == "deepseek-flash"
        for name, question, query in CASES:
            reference = (await client.post("/api/ai/statistics", json=query)).raise_for_status().json()
            assert reference["facts"] and not reference["warnings"], reference["warnings"]
            expected = {f["id"]: f for f in reference["facts"]}
            result = (await client.post("/api/ai/chat", json={"question": question})).raise_for_status().json()
            verified = await check_citations(client, result)
            found = {c["fact_id"]: c for c in result["citations"] if c["kind"] == "statistic"}
            matched = all(fid in found and all(found[fid][k] == f[k] for k in (
                "value", "statistic", "text", "metric", "unit", "aggregation", "coverage"))
                and found[fid]["period"]["start"] == reference["period"]["start"]
                and found[fid]["period"]["end"] == reference["period"]["end"]
                for fid, f in expected.items())
            passed = bool(result["ok"] and result["models"] == ["deepseek-flash"] and verified and matched)
            report["cases"].append({"case": name, "question": question, "passed": passed,
                                    "citations_verified": verified, "statistics_matched": matched,
                                    "reference": reference, "result": result})
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"case": name, "passed": passed, "ms": result["total_ms"],
                              "validation_errors": result["validation_errors"], "answer": result["answer"],
                              "error": result["error"]}, ensure_ascii=False), flush=True)
    if not all(c["passed"] for c in report["cases"]):
        raise SystemExit(f"统计验收未全部通过，原始记录：{path}")
    print(f"3/3统计语义场景通过，记录：{path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

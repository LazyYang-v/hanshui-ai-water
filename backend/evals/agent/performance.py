"""固定历史问题的 Flash API 计时；每次单独存档，不覆盖原有 smoke。

    uv run python -m evals.agent.performance --label before --repeats 2

会产生 Flash API 费用；只读对话，不确认工单或修改运行数据。
"""

import argparse
import asyncio
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from evals.agent.smoke import check_citations
from evals.http import alogin

QUESTION = "黄官镇2026年11月6日是否有氨氮超标？请结合当时的DO、进水量和鼓风机状态分析可能原因，并给出核查建议。"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=2, choices=range(1, 6))
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label):
        parser.error("label 只能用字母、数字、下划线或连字符")
    path = Path(__file__).parent / "results" / f"performance-{args.label}.json"
    if path.exists():
        parser.error(f"记录已存在，请另用 label：{path}")
    path.parent.mkdir(exist_ok=True)
    report = {"checked_at": datetime.now(UTC).isoformat(), "question": QUESTION,
              "note": "固定历史日期；同一运行服务连续请求，缓存和模型采样会影响耗时；小样本不报告P95。",
              "runs": []}
    async with httpx.AsyncClient(base_url=args.url, timeout=220, trust_env=False) as client:
        await alogin(client)
        response = await client.get("/api/ai/gateway")
        response.raise_for_status()
        report["gateway_before"] = response.json()
        providers = report["gateway_before"].get("providers", {})
        assert set(providers) == {"deepseek"} and providers["deepseek"]["model"] == "deepseek-flash"
        plant = (await client.get("/api/plants/P06")).raise_for_status().json()
        blowers = [d["code"] for d in plant["devices"] if d["kind"] == "鼓风机"]
        report["blower_catalog"] = blowers
        for i in range(args.repeats):
            start = time.perf_counter()
            response = await client.post("/api/ai/chat", json={"question": QUESTION})
            response.raise_for_status()
            result = response.json()
            assert result["models"] == ["deepseek-flash"], result["models"]
            client_ms = round((time.perf_counter() - start) * 1000, 1)
            verified = await check_citations(client, result)
            diagnosis_evidence = ({"data", "knowledge", "statistic"} <= {c["kind"] for c in result["citations"]}
                                  and {"hypothesis", "recommendation"} <= {c["kind"] for c in result["claims"]})
            count_checks = []
            for call in result["tool_calls"]:
                if call["tool"] != "sql_query" or call["status"] != "ok":
                    continue
                out = call["output"]
                for column, name in enumerate(out["columns"]):
                    if "鼓风机" in name and any(word in name for word in ("台数", "设备数")):
                        count_checks.extend(type(row[column]) in (int, float) and 0 <= row[column] <= len(blowers)
                                            for row in out["rows"] if row[column] is not None)
            for claim in result["claims"]:
                if "鼓风机" in claim["text"]:
                    count_checks.extend(float(v) <= len(blowers)
                                        for v in re.findall(r"(\d+(?:\.\d+)?)\s*台", claim["text"]))
            counts_plausible = all(count_checks)
            entry = {"client_ms": client_ms, "citations_verified": verified,
                     "diagnosis_evidence_complete": diagnosis_evidence,
                     "device_counts_plausible": counts_plausible, "result": result}
            report["runs"].append(entry)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"run": i + 1, "ok": result["ok"], "error": result["error"],
                              "total_ms": result["total_ms"], "prompt_tokens": result["prompt_tokens"],
                              "completion_tokens": result["completion_tokens"],
                              "validation_errors": result["validation_errors"]}, ensure_ascii=False), flush=True)
        response = await client.get("/api/ai/gateway")
        response.raise_for_status()
        report["gateway_after"] = response.json()
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not all(run["result"]["ok"] and run["citations_verified"] and run["diagnosis_evidence_complete"] and run["device_counts_plausible"]
               for run in report["runs"]):
        raise SystemExit(f"存在失败，原始结果已保存：{path}")
    print(f"已保存 {len(report['runs'])} 次结果：{path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

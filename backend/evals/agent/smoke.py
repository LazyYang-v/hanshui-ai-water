"""验收运行中的 Agent API（真实调用 DeepSeek Flash）。

    uv run python -m evals.agent.smoke

核验跨工具诊断、连续追问、SSE和能力拒答；不是通用原因诊断准确率评测。
"""

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from evals.http import alogin

DIAGNOSIS = "黄官镇昨天是否有氨氮超标？请结合当时的DO、进水量和鼓风机状态分析可能原因，并给出核查建议。"
FOLLOWUP = "那它今天最新的氨氮和DO是多少？"
REFUSALS = ["黄官镇污水处理站站长的手机号码是多少？", "请立即把黄官镇鼓风机频率设为50Hz并创建工单。"]


async def check_citations(client, result):
    calls = {c["id"]: c for c in result["tool_calls"]}
    for cite in result["citations"]:
        call = calls.get(cite["call_id"])
        if call is None or call["status"] != "ok":
            return False
        if cite["kind"] == "data":
            out = call["output"]
            if (call["tool"] != "sql_query" or not out["sql"] or
                    out["rows"][cite["row"]][cite["column"]] != cite["value"]):
                return False
        elif cite["kind"] == "statistic":
            if call["tool"] != "plant_statistics":
                return False
            out = call["output"]
            fact = next((f for f in out["facts"] if f["id"] == cite["fact_id"]), None)
            if (fact is None or any(cite[k] != fact[k] for k in ("value", "statistic", "text", "metric", "unit", "aggregation", "coverage"))
                    or cite["period"] != out["period"] or cite["coverage_policy"] != out["coverage_policy"]):
                return False
        elif cite["kind"] == "knowledge":
            if call["tool"] != "rag_search" or cite["chunk_id"] not in {h["id"] for h in call["output"]["hits"]}:
                return False
            original = await client.get(cite["url"])
            original.raise_for_status()
            if "".join(cite["quote"].split()) not in "".join(original.json()["text"].split()):
                return False
        else:
            return False
    return True


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "api-smoke-optimized.json")
    args = parser.parse_args()
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "cases": []}
    path = args.output
    if path.exists():
        parser.error(f"不覆盖已有验收记录：{path}；请指定新的--output")
    path.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(base_url=args.url, timeout=220, trust_env=False) as client:
        await alogin(client)
        gateway = await client.get("/api/ai/gateway")
        gateway.raise_for_status()
        providers = gateway.json().get("providers", {})
        assert set(providers) == {"deepseek"} and providers["deepseek"]["model"] == "deepseek-flash", "必须只启用Flash"
        spec = await client.get("/openapi.json")
        spec.raise_for_status()
        assert "/api/ai/chat/stream" in spec.json()["paths"], "请先构建并启动带Agent的新API"
        catalog = await client.get("/api/ai/knowledge")
        catalog.raise_for_status()
        assert catalog.json()["dense_available"], "真实验收要求本地中文向量可用"

        async def save_case(question, result, expected_refusal, extra=True, **details):
            citations_ok = await check_citations(client, result)
            passed = (result["models"] == ["deepseek-flash"] and not result["error"]
                      and result["refused"] == expected_refusal and result["ok"] == (not expected_refusal)
                      and citations_ok and (expected_refusal or bool(result["citations"])) and extra)
            report["cases"].append({"question": question, "passed": passed, "expected_refusal": expected_refusal,
                                    "citations_verified": citations_ok, "result": result, **details})
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps({"question": question, "passed": passed, "resolved_question": result["resolved_question"],
                              "answer": result["answer"], "reason": result["reason"], "error": result["error"],
                              "tools": [c["tool"] for c in result["tool_calls"]], "models": result["models"],
                              "validation_errors": result["validation_errors"], "ms": result["total_ms"]},
                             ensure_ascii=False), flush=True)

        # 先以真实流式接口完成一次跨工具诊断，保留各事件到达耗时。
        start = time.perf_counter()
        events, result, event = [], None, None
        async with client.stream("POST", "/api/ai/chat/stream", json={"question": DIAGNOSIS}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    body = json.loads(line[6:])
                    events.append({"event": event, "elapsed_ms": round((time.perf_counter() - start) * 1000, 1)})
                    if event == "done":
                        result = body
                    elif event == "error":
                        raise RuntimeError(body["error"])
        assert result is not None, "SSE未收到完成事件"
        tool_types = {c["tool"] for c in result["tool_calls"]}
        citation_types = {c["kind"] for c in result["citations"]}
        await save_case(DIAGNOSIS, result, False,
                        tool_types == {"sql_query", "rag_search", "plant_statistics"}
                        and citation_types == {"data", "knowledge", "statistic"}
                        and events[0]["event"] == "run_start" and events[-1]["event"] == "done",
                        events=events)
        session_id = result["session_id"]

        response = await client.post("/api/ai/chat", json={"question": FOLLOWUP, "session_id": session_id})
        response.raise_for_status()
        followup = response.json()
        await save_case(FOLLOWUP, followup, False,
                        ("黄官" in followup["resolved_question"] or "P06" in followup["resolved_question"])
                        and any(c["tool"] == "sql_query" for c in followup["tool_calls"]))
        stored = await client.get(f"/api/ai/chat/sessions/{session_id}")
        stored.raise_for_status()
        report["session_verified"] = ([t["run_id"] for t in stored.json()["turns"]]
                                      == [result["run_id"], followup["run_id"]])
        for question in REFUSALS:
            response = await client.post("/api/ai/chat", json={"question": question})
            response.raise_for_status()
            body = response.json()
            await save_case(question, body, True, not body["tool_calls"])

    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["session_verified"] or not all(c["passed"] for c in report["cases"]):
        raise SystemExit(f"验收未全部通过，记录：{path}")
    print(f"4/4 Agent API场景通过；SSE、连续会话及引用均核验。记录：{path}")


if __name__ == "__main__":
    asyncio.run(main())

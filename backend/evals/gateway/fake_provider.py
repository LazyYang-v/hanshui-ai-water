"""压测用的假厂商：OpenAI 兼容 /v1/chat/completions，行为可配置，不需要网络和费用。

    uv run uvicorn evals.gateway.fake_provider:app --port 18090 --log-level warning

POST /_config 设置行为（未给的字段保持不变），POST /_reset 清零计数，GET /_stats 查看计数：
  delay_ms     非流式：首字前延迟中位数（对数正态，sigma 控制离散度）
  sigma        对数正态离散度，0 为固定延迟
  tokens       输出 token 数；token_ms 为每个 token 的间隔（流式逐个发送，非流式一次性等完）
  capacity     同时处理上限，超过返回 429（模拟厂商限流）；0 为不限
  retry_after  429 时的 Retry-After 秒数；None 不带
  error_rate   随机返回 503 的概率
非流式响应的 content 是 {"server_ms": 实际处理毫秒}，压测端据此扣除厂商耗时，得到网关与 HTTP 开销。
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

CFG = {"delay_ms": 500.0, "sigma": 0.0, "tokens": 50, "token_ms": 0.0, "capacity": 0, "retry_after": None,
       "error_rate": 0.0}
STATS = {"requests": 0, "ok": 0, "429": 0, "503": 0, "inflight": 0, "max_inflight": 0}


def _delay() -> float:
    base = CFG["delay_ms"] / 1000
    return base * math.exp(random.gauss(0, CFG["sigma"])) if CFG["sigma"] else base


def _usage() -> dict:
    return {"prompt_tokens": 100, "completion_tokens": CFG["tokens"], "prompt_cache_hit_tokens": 0}


async def chat(request: Request):
    body = await request.json()
    STATS["requests"] += 1
    if CFG["capacity"] and STATS["inflight"] >= CFG["capacity"]:
        STATS["429"] += 1
        headers = {"Retry-After": str(CFG["retry_after"])} if CFG["retry_after"] is not None else {}
        return JSONResponse({"error": {"message": "rate limited"}}, status_code=429, headers=headers)
    if random.random() < CFG["error_rate"]:
        STATS["503"] += 1
        return JSONResponse({"error": {"message": "overloaded"}}, status_code=503)
    STATS["inflight"] += 1
    STATS["max_inflight"] = max(STATS["max_inflight"], STATS["inflight"])
    t0 = time.perf_counter()
    if not body.get("stream"):
        try:
            await asyncio.sleep(_delay() + CFG["tokens"] * CFG["token_ms"] / 1000)
        finally:
            STATS["inflight"] -= 1
        STATS["ok"] += 1
        server_ms = (time.perf_counter() - t0) * 1000
        return JSONResponse({"model": body.get("model"), "usage": _usage(), "choices": [
            {"message": {"role": "assistant", "content": json.dumps({"server_ms": server_ms})},
             "finish_reason": "stop"}]})

    async def gen():
        try:
            await asyncio.sleep(_delay())
            ttft = (time.perf_counter() - t0) * 1000
            for i in range(CFG["tokens"]):
                if i:
                    await asyncio.sleep(CFG["token_ms"] / 1000)
                text = json.dumps({"server_ttft_ms": ttft}) if i == 0 else "x"
                yield f"data: {json.dumps({'model': body.get('model'), 'choices': [{'delta': {'content': text}}]})}\n\n"
            yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'usage': _usage()})}\n\n"
            yield "data: [DONE]\n\n"
            STATS["ok"] += 1
        finally:
            STATS["inflight"] -= 1

    return StreamingResponse(gen(), media_type="text/event-stream")


async def config(request: Request):
    CFG.update(await request.json())
    return JSONResponse(CFG)


async def reset(request: Request):
    for k in STATS:
        STATS[k] = 0
    return JSONResponse(STATS)


async def stats(request: Request):
    return JSONResponse(STATS)


app = Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"]),
                        Route("/_config", config, methods=["POST"]), Route("/_reset", reset, methods=["POST"]),
                        Route("/_stats", stats)])

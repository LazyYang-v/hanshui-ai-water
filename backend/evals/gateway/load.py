"""LLM 网关并发压测。

两部分：
  fake   本地假厂商（evals/gateway/fake_provider.py，独立进程），测网关自身开销、限流、降级、熔断与流式首字，
         不花钱、可复现；
  live   真实 DeepSeek Flash 小规模并发（须显式 --live 且确认费用），测首字延迟与总延迟。

    uv run python -m evals.gateway.load fake --label baseline
    uv run python -m evals.gateway.load live --label flash --n 100 --concurrency 50

结果写 evals/gateway/results/{part}-{label}.json。
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx

from app.ai.gateway import ChatResult, LLMError, LLMGateway, OpenAICompatProvider

RESULTS = Path(__file__).resolve().parent / "results"
ROOT = Path(__file__).resolve().parents[2]
PORTS = (18090, 18091)


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))], 1)


def dist(xs: list[float]) -> dict:
    return {"n": len(xs), "p50": pct(xs, 50), "p95": pct(xs, 95), "p99": pct(xs, 99), "max": pct(xs, 100)}


def machine() -> dict:
    return {"platform": platform.platform(), "cpu_count": os.cpu_count(), "load_avg_1m": round(os.getloadavg()[0], 1),
            "proxy_env": sorted(k for k in os.environ if k.lower() in ("http_proxy", "https_proxy", "all_proxy")),
            "no_proxy": os.environ.get("NO_PROXY") or os.environ.get("no_proxy")}


# ---------------------------------------------------------------- 假厂商进程

class FakeServers:
    def __init__(self):
        self.procs: list[subprocess.Popen] = []

    def __enter__(self):
        for port in PORTS:
            self.procs.append(subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "evals.gateway.fake_provider:app", "--port", str(port),
                 "--log-level", "warning", "--backlog", "2048"], cwd=ROOT))
        for port in PORTS:
            for _ in range(100):
                try:
                    httpx.get(f"http://127.0.0.1:{port}/_stats", timeout=1)
                    break
                except httpx.TransportError:
                    time.sleep(0.1)
        return self

    def __exit__(self, *exc):
        for p in self.procs:
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:      # hang 场景留下的长请求会让 uvicorn 优雅退出一直等
                p.kill()
                p.wait(5)


def configure(port: int, **cfg) -> None:
    defaults = {"delay_ms": 500.0, "sigma": 0.0, "tokens": 50, "token_ms": 0.0, "capacity": 0, "retry_after": None,
                "error_rate": 0.0}
    httpx.post(f"http://127.0.0.1:{port}/_config", json={**defaults, **cfg}).raise_for_status()
    httpx.post(f"http://127.0.0.1:{port}/_reset").raise_for_status()


def server_stats(port: int) -> dict:
    return httpx.get(f"http://127.0.0.1:{port}/_stats").json()


def fake_provider(name: str, port: int, timeout: float = 30) -> OpenAICompatProvider:
    return OpenAICompatProvider(name, f"http://127.0.0.1:{port}/v1", "k", "fake-model", timeout=timeout)


def attempt_key(a) -> str:
    return f"{a.provider}:{a.status_code or a.error_type or 'ok'}"


# ---------------------------------------------------------------- 负载

def messages() -> list[dict]:
    return [{"role": "user", "content": f"压测 {uuid.uuid4().hex}"}]     # 每条不同，避免命中网关缓存


async def run_chat(gw: LLMGateway, n: int, concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    lat, overhead, attempts, errors, providers = [], [], [], Counter(), Counter()
    retry_waits, queue, attempt_outcomes, failed_lat = [], [], Counter(), []

    async def one():
        async with sem:
            t0 = time.perf_counter()
            try:
                r: ChatResult = await gw.chat(messages(), use_cache=False)
            except LLMError as e:
                errors[str(e).split("：")[0][:40] if "所有厂商" in str(e) else type(e).__name__] += 1
                attempt_outcomes.update(attempt_key(a) for a in e.attempt_history)
                failed_lat.append((time.perf_counter() - t0) * 1000)
                return
            ms = (time.perf_counter() - t0) * 1000
            lat.append(ms)
            attempt_outcomes.update(attempt_key(a) for a in r.attempt_history)
            queue.append(sum(getattr(a, "queue_ms", 0.0) for a in r.attempt_history))
            providers[r.provider] += 1
            attempts.append(len(r.attempt_history))
            retry_waits.append(sum(a.wait_ms for a in r.attempt_history))
            try:
                overhead.append(ms - json.loads(r.content)["server_ms"])
            except (ValueError, KeyError):
                pass

    t0 = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(n)))
    wall = time.perf_counter() - t0
    return {"n": n, "concurrency": concurrency, "ok": len(lat), "errors": dict(errors),
            "success_rate": round(len(lat) / n, 4), "throughput_rps": round(n / wall, 1),
            "latency_ms": dist(lat), "overhead_ms": dist(overhead) if overhead else None,
            "attempts": dict(Counter(attempts)), "retry_wait_ms": dist([w for w in retry_waits if w]),
            "queue_ms": dist(queue) if any(queue) else None, "failed_latency_ms": dist(failed_lat) if failed_lat else None,
            "attempt_outcomes": dict(sorted(attempt_outcomes.items())), "providers": dict(providers)}


async def run_raw(port: int, n: int, concurrency: int) -> dict:
    """对照：不经网关，直接用同样配置的 httpx 客户端请求假厂商。"""
    client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}/v1", timeout=30)
    sem = asyncio.Semaphore(concurrency)
    overhead, status = [], Counter()

    async def one():
        async with sem:
            t0 = time.perf_counter()
            r = await client.post("/chat/completions", json={"model": "fake-model", "messages": messages(),
                                                             "temperature": 0})
            status[r.status_code] += 1
            if r.status_code != 200:
                return
            ms = (time.perf_counter() - t0) * 1000
            overhead.append(ms - json.loads(r.json()["choices"][0]["message"]["content"])["server_ms"])

    t0 = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(n)))
    wall = time.perf_counter() - t0
    await client.aclose()
    return {"n": n, "concurrency": concurrency, "status": dict(status), "throughput_rps": round(n / wall, 1),
            "overhead_ms": dist(overhead)}


async def run_stream(gw: LLMGateway, n: int, concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    ttft, ttft_overhead, total, errors = [], [], [], Counter()

    async def one():
        async with sem:
            t0 = time.perf_counter()
            first = None
            try:
                async for part in gw.stream(messages()):
                    if isinstance(part, str) and first is None:
                        first = (time.perf_counter() - t0) * 1000
                        ttft.append(first)
                        try:
                            ttft_overhead.append(first - json.loads(part)["server_ttft_ms"])
                        except (ValueError, KeyError):
                            pass
            except LLMError as e:
                errors[type(e).__name__] += 1
                return
            total.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(n)))
    wall = time.perf_counter() - t0
    return {"n": n, "concurrency": concurrency, "ok": len(total), "errors": dict(errors),
            "throughput_rps": round(n / wall, 1), "ttft_ms": dist(ttft),
            "ttft_overhead_ms": dist(ttft_overhead) if ttft_overhead else None, "total_ms": dist(total)}


# ---------------------------------------------------------------- 场景

def gateway_for(*ports, timeout: float = 30, **kw) -> LLMGateway:
    names = {PORTS[0]: "primary", PORTS[1]: "secondary"}
    return LLMGateway([fake_provider(names[p], p, timeout) for p in ports], **kw)


def provider_brief(gw: LLMGateway) -> dict:
    keys = ("calls", "errors", "rate_limited", "retries", "circuit", "times_opened", "queue_p95_ms")
    return {k: {kk: v.get(kk) for kk in keys} for k, v in gw.summary()["providers"].items()}


async def scenario_overhead(out: dict) -> None:
    """固定 500 ms 厂商延迟，网关与裸 httpx 在不同并发下的额外开销。"""
    configure(PORTS[0], delay_ms=500)
    res = {}
    for c in (1, 10, 50, 100):
        n = 40 if c == 1 else 500
        gw = gateway_for(PORTS[0])
        res[f"gateway_c{c}"] = await run_chat(gw, n, c)
        await gw.aclose()
        res[f"raw_c{c}"] = await run_raw(PORTS[0], n, c)
        print("overhead", c, res[f"gateway_c{c}"]["overhead_ms"], res[f"raw_c{c}"]["overhead_ms"], flush=True)
    out["overhead"] = res


async def scenario_ratelimit(out: dict) -> None:
    """厂商最多同时处理 20 个，超过返回 429；客户端并发 50。对比有无 Retry-After、有无网关并发上限。"""
    res = {}
    variants = {"retry_after_1": ({"retry_after": 1}, {}), "no_retry_after": ({"retry_after": None}, {}),
                "bulkhead_20": ({"retry_after": 1}, {"max_concurrency": 20}),
                "bulkhead_16": ({"retry_after": 1}, {"max_concurrency": 16})}
    for name, (cfg, kw) in variants.items():
        if kw.keys() - inspect.signature(LLMGateway).parameters.keys():
            continue                      # 对照旧网关时跳过它不支持的参数
        configure(PORTS[0], delay_ms=500, capacity=20, **cfg)
        gw = gateway_for(PORTS[0], **kw)
        r = await run_chat(gw, 300, 50)
        r["server"] = server_stats(PORTS[0])
        r["gateway"] = provider_brief(gw)
        await gw.aclose()
        res[name] = r
        print("ratelimit", name, r["success_rate"], r["errors"], r["latency_ms"], r["server"], flush=True)
    out["ratelimit"] = res


async def scenario_fallback(out: dict) -> None:
    """主厂商 30% 返回 503，备用厂商正常；并发 50。"""
    configure(PORTS[0], delay_ms=500, error_rate=0.3)
    configure(PORTS[1], delay_ms=800)
    gw = gateway_for(PORTS[0], PORTS[1], backoff_s=0.2)
    r = await run_chat(gw, 300, 50)
    r["gateway"] = provider_brief(gw)
    await gw.aclose()
    out["fallback"] = r
    print("fallback", r["success_rate"], r["errors"], r["providers"], r["gateway"], flush=True)


async def scenario_outage(out: dict) -> None:
    """主厂商故障、备用正常；看熔断前后的延迟。

    不用「连一个没人监听的端口」：macOS 对关闭端口的 RST 限速，单次连接要 2～4 秒，测的是操作系统而不是网关。
    改为两种可控故障：fast_503 主厂商立即返回 503；hang 主厂商不响应，靠 2 秒读超时发现。
    """
    res = {}
    for name, (cfg, timeout) in {"fast_503": ({"error_rate": 1.0}, 30), "hang": ({"delay_ms": 60_000}, 2)}.items():
        configure(PORTS[0], **cfg)
        configure(PORTS[1], delay_ms=500)
        gw = gateway_for(PORTS[0], PORTS[1], backoff_s=0.2, timeout=timeout)
        r = await run_chat(gw, 300, 50)
        r["gateway"] = provider_brief(gw)
        await gw.aclose()
        res[name] = r
        print("outage", name, r["success_rate"], r["errors"], r["latency_ms"], r["gateway"], flush=True)
    out["outage"] = res


async def scenario_stream(out: dict) -> None:
    """流式：首字延迟中位数 400 ms（对数正态 σ=0.3），之后 100 个 token、每 20 ms 一个。"""
    configure(PORTS[0], delay_ms=400, sigma=0.3, tokens=100, token_ms=20)
    res = {}
    for c in (1, 50):
        gw = gateway_for(PORTS[0])
        res[f"c{c}"] = await run_stream(gw, 30 if c == 1 else 300, c)
        await gw.aclose()
        print("stream", c, res[f"c{c}"]["ttft_ms"], res[f"c{c}"]["ttft_overhead_ms"], flush=True)
    out["stream"] = res


SCENARIOS = {"overhead": scenario_overhead, "ratelimit": scenario_ratelimit, "fallback": scenario_fallback,
             "stream": scenario_stream, "outage": scenario_outage}   # outage 最后：hang 会在假厂商里留下长请求


async def run_fake(args) -> dict:
    # 本机若设了 HTTP_PROXY，httpx 访问 127.0.0.1 也会走代理：并发高时代理返回 502，开销也算进代理。
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    out = {"generated_at": datetime.now().isoformat(timespec="seconds"), "machine_before": machine()}
    with FakeServers():
        for name in args.scenarios:
            await SCENARIOS[name](out)
    out["machine_after"] = machine()
    return out


# ---------------------------------------------------------------- 真实 Flash

LIVE_PROMPT = "用一句话（不超过40字）说明污水处理厂曝气池溶解氧偏低可能的一个原因。编号{i}。"


async def run_live(args) -> dict:
    import app.config  # noqa: F401  读取本地 .env（密钥只在本地）
    gw = LLMGateway.from_env()
    if not gw.providers or any(p.model != "deepseek-flash" for p in gw.providers):
        raise SystemExit("只允许 deepseek-flash（项目约定不调用 Pro），请检查 LLM_PROVIDERS / DEEPSEEK_MODEL")
    out = {"generated_at": datetime.now().isoformat(timespec="seconds"), "machine_before": machine(),
           "model": [p.model for p in gw.providers], "prompt": LIVE_PROMPT, "n": args.n,
           "concurrency": args.concurrency, "thinking": args.thinking, "max_tokens": 120}
    thinking = {"default": None, "off": False}[args.thinking]
    sem = asyncio.Semaphore(args.concurrency)

    def msgs(i):
        return [{"role": "user", "content": LIVE_PROMPT.format(i=i)}]

    # 流式：首字延迟
    ttft, total, tokens, errors, finish = [], [], Counter(), Counter(), Counter()

    async def one_stream(i):
        async with sem:
            t0 = time.perf_counter()
            first = None
            try:
                async for part in gw.stream(msgs(i), temperature=0.3, max_tokens=120, thinking=thinking):
                    if isinstance(part, str) and first is None:
                        first = (time.perf_counter() - t0) * 1000
                    elif isinstance(part, ChatResult):
                        finish[f"{part.finish_reason}{'' if part.content.strip() else '(正文为空)'}"] += 1
                        tokens["prompt"] += part.prompt_tokens
                        tokens["completion"] += part.completion_tokens
                        tokens["retries"] += max(0, len(part.attempt_history) - 1)
            except LLMError as e:
                errors[f"{type(e).__name__}:{e.status}"] += 1
                return
            if first is not None:
                ttft.append(first)
            total.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    await asyncio.gather(*(one_stream(i) for i in range(args.n)))
    out["stream"] = {"wall_s": round(time.perf_counter() - t0, 1), "ok": len(total), "errors": dict(errors),
                     "finish": dict(finish), "ttft_ms": dist(ttft), "total_ms": dist(total), "tokens": dict(tokens)}
    print("live stream", json.dumps(out["stream"], ensure_ascii=False), flush=True)

    # 非流式（业务实际用法）：总延迟
    lat, tokens2, errors2, finish2, samples = [], Counter(), Counter(), Counter(), []

    async def one_chat(i):
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await gw.chat(msgs(i + 10_000), temperature=0.3, max_tokens=120, use_cache=False,
                                  thinking=thinking)
            except LLMError as e:
                errors2[f"{type(e).__name__}:{e.status}"] += 1
                return
            lat.append((time.perf_counter() - t0) * 1000)
            finish2[f"{r.finish_reason}{'' if r.content.strip() else '(正文为空)'}"] += 1
            if len(samples) < 3:
                samples.append(r.content)
            tokens2["prompt"] += r.prompt_tokens
            tokens2["completion"] += r.completion_tokens
            tokens2["retries"] += max(0, len(r.attempt_history) - 1)

    t0 = time.perf_counter()
    await asyncio.gather(*(one_chat(i) for i in range(args.n)))
    out["chat"] = {"wall_s": round(time.perf_counter() - t0, 1), "ok": len(lat), "errors": dict(errors2),
                   "finish": dict(finish2), "latency_ms": dist(lat), "tokens": dict(tokens2), "samples": samples}
    print("live chat", json.dumps(out["chat"], ensure_ascii=False), flush=True)
    out["gateway"] = gw.summary()
    out["machine_after"] = machine()
    await gw.aclose()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("part", choices=["fake", "live"])
    p.add_argument("--label", default="v1")
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--thinking", choices=["default", "off"], default="off",
                   help="live：Flash 默认开思考，120 token 上限常在推理阶段用完、没有正文")
    a = p.parse_args()
    out = asyncio.run(run_fake(a) if a.part == "fake" else run_live(a))
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{a.part}-{a.label}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(path)


if __name__ == "__main__":
    main()

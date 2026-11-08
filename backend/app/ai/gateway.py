"""LLM 网关：统一 OpenAI 兼容协议，屏蔽厂商差异。

能力
- 多厂商按顺序降级（如 DeepSeek → 通义千问 → 本地 Ollama）
- 可重试错误（超时、连接失败、429、5xx）指数退避重试，尊重 Retry-After
- 并发隔离（bulkhead）：每个厂商同时在途请求数上限，超出的在网关内排队，排队时间单独记录
- 熔断：最近 N 次尝试的失败率超过阈值后在冷却期内直接跳过；429 是限流背压而不是故障，不计入熔断。
  分母包含在途请求（即在途全部成功时失败率仍达标才熔断）：高并发下失败（如立即返回的 503）比成功先回来，
  只看已完成的会把 30% 的失败率误判成 100%
- temperature=0 的调用走 LRU 缓存（Text2SQL 评测、重复提问）
- 流式输出（SSE），记录首字延迟
- 统计：调用数、错误数、P50/P95 延迟、token 用量、缓存命中
- 录制 / 回放（cassette）：离线演示和测试时不依赖网络
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
from asyncio import Semaphore     # 直接导入：测试会替换模块里的 asyncio（假时钟）
from collections import OrderedDict, deque
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import httpx

log = logging.getLogger(__name__)


@dataclass
class AttemptRecord:
    """一次实际厂商请求；attempt 跨厂商从 1 编号，wait_ms 是随后实际退避时间，queue_ms 是之前在并发隔离里的排队时间。"""

    provider: str
    attempt: int
    status: str                      # success / error / cancelled
    latency_ms: float
    wait_ms: float = 0.0
    error_type: str | None = None    # 仅异常类名，不保存异常正文
    status_code: int | None = None
    queue_ms: float = 0.0


class LLMError(Exception):
    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None,
                 retry_after: float | None = None):
        super().__init__(message)
        self.retryable, self.status, self.retry_after = retryable, status, retry_after
        self.total_latency_ms: float | None = None
        self.attempt_history: list[AttemptRecord] = []


@dataclass
class ChatResult:
    content: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0          # 厂商侧上下文缓存命中（DeepSeek 返回）
    latency_ms: float = 0.0           # 保持兼容：最终成功的单次厂商请求耗时
    ttft_ms: float | None = None       # 流式首字延迟，相对最终厂商请求
    attempts: int = 1                 # 保持兼容：最终厂商的尝试次数
    from_cache: bool = False           # 网关本地缓存命中
    finish_reason: str | None = None   # stop / length 等，区分输出截断与普通格式错误
    total_latency_ms: float | None = None  # 全部尝试、退避及厂商切换；旧录制未知
    attempt_history: list[AttemptRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, item: dict) -> ChatResult:
        values = {k: v for k, v in item.items() if k in cls.__dataclass_fields__}
        values["attempt_history"] = [AttemptRecord(**a) for a in values.get("attempt_history", [])]
        return cls(**values)


# ---------------------------------------------------------------- 厂商

class Provider(Protocol):
    name: str
    model: str

    async def complete(self, body: dict) -> dict: ...
    def stream(self, body: dict) -> AsyncIterator[dict]: ...


PRESETS = {
    # name: (base_url, key 环境变量, 默认模型)
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-flash"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "QWEN_API_KEY", "qwen-plus"),
    "ollama": ("http://localhost:11434/v1", None, "qwen2.5:7b"),
}


class OpenAICompatProvider:
    def __init__(self, name: str, base_url: str, api_key: str | None, model: str,
                 timeout: float = 60.0, transport: httpx.AsyncBaseTransport | None = None):
        self.name, self.model = name, model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(base_url=base_url, headers=headers, transport=transport,
                                        timeout=httpx.Timeout(timeout, connect=10.0))

    @classmethod
    def from_env(cls, name: str) -> OpenAICompatProvider:
        if name not in PRESETS:
            raise ValueError(f"未知厂商 {name}，可选：{', '.join(PRESETS)}")
        base_url, key_env, model = PRESETS[name]
        prefix = name.upper()
        api_key = os.getenv(key_env) if key_env else None
        if key_env and not api_key:
            raise ValueError(f"缺少环境变量 {key_env}")
        return cls(name, os.getenv(f"{prefix}_BASE_URL", base_url), api_key,
                   os.getenv(f"{prefix}_MODEL", model), float(os.getenv("LLM_TIMEOUT_S", "60")))

    @staticmethod
    def _retry_after(value: str | None) -> float | None:
        # 只支持秒数（可带小数）；HTTP 日期格式少见，按未给处理，走指数退避。
        try:
            seconds = float(value) if value else None
        except ValueError:
            return None
        return seconds if seconds is not None and 0 <= seconds < float("inf") else None

    @classmethod
    def _error(cls, resp: httpx.Response) -> LLMError:
        try:
            detail = resp.json().get("error", {}).get("message") or resp.text
        except ValueError:
            detail = resp.text
        return LLMError(f"HTTP {resp.status_code}: {detail[:300]}",
                        retryable=resp.status_code == 429 or resp.status_code >= 500,
                        status=resp.status_code,
                        retry_after=cls._retry_after(resp.headers.get("retry-after")))

    async def complete(self, body: dict) -> dict:
        try:
            resp = await self.client.post("/chat/completions", json={**body, "model": self.model})
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise LLMError(f"{type(e).__name__}: {e}", retryable=True) from e
        if resp.status_code != 200:
            raise self._error(resp)
        return resp.json()

    async def stream(self, body: dict) -> AsyncIterator[dict]:
        payload = {**body, "model": self.model, "stream": True, "stream_options": {"include_usage": True}}
        try:
            async with self.client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise self._error(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    yield json.loads(data)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise LLMError(f"{type(e).__name__}: {e}", retryable=True) from e

    async def aclose(self) -> None:
        await self.client.aclose()


# ---------------------------------------------------------------- 录制回放

def request_key(messages: list[dict], params: dict) -> str:
    raw = json.dumps({"messages": messages, **params}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class Cassette:
    """JSONL 文件：每行 {key, content, provider, model, usage}。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.items: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    item = json.loads(line)
                    self.items[item["key"]] = item

    def get(self, key: str) -> dict | None:
        return self.items.get(key)

    def put(self, key: str, result: ChatResult) -> None:
        # 重新录制同一请求时追加最新响应，读回时最后一条生效。
        item = {"key": key, **result.to_dict()}
        self.items[key] = item
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 统计与熔断

def _percentile(xs, p):
    if not xs:
        return None
    values = sorted(xs)
    return round(values[min(len(values) - 1, int(p / 100 * len(values)))], 1)


@dataclass
class ProviderStats:
    calls: int = 0
    errors: int = 0
    retries: int = 0
    cancelled: int = 0
    rate_limited: int = 0             # 429 次数（也计入 errors）
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    open_until: float = 0.0
    times_opened: int = 0
    inflight: int = 0
    outcomes: deque = field(default_factory=lambda: deque(maxlen=50))   # 熔断窗口：True 成功 / False 失败
    queue_waits: deque = field(default_factory=lambda: deque(maxlen=500))
    latencies: deque = field(default_factory=lambda: deque(maxlen=500))
    attempt_latencies: deque = field(default_factory=lambda: deque(maxlen=500))
    ttfts: deque = field(default_factory=lambda: deque(maxlen=500))

    def summary(self) -> dict:
        return {
            "calls": self.calls, "errors": self.errors, "retries": self.retries, "cancelled": self.cancelled,
            "rate_limited": self.rate_limited, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "latency_p50_ms": _percentile(self.latencies, 50), "latency_p95_ms": _percentile(self.latencies, 95),
            "attempt_latency_p50_ms": _percentile(self.attempt_latencies, 50),
            "attempt_latency_p95_ms": _percentile(self.attempt_latencies, 95),
            "ttft_p50_ms": _percentile(self.ttfts, 50), "ttft_p95_ms": _percentile(self.ttfts, 95),
            "queue_p50_ms": _percentile(self.queue_waits, 50), "queue_p95_ms": _percentile(self.queue_waits, 95),
            "window_failure_rate": round(self.outcomes.count(False) / len(self.outcomes), 3) if self.outcomes else None,
            "circuit": "open" if self.open_until > time.monotonic() else "closed", "times_opened": self.times_opened,
            "inflight": self.inflight,
        }


@dataclass
class RequestStats:
    """实时逻辑调用统计；缓存和回放不重复计入，分位数保留最近 500 次。"""

    calls: int = 0
    successes: int = 0
    errors: int = 0
    cancelled: int = 0
    total_latencies: deque = field(default_factory=lambda: deque(maxlen=500))

    def summary(self) -> dict:
        return {
            "calls": self.calls, "successes": self.successes, "errors": self.errors, "cancelled": self.cancelled,
            "total_latency_p50_ms": _percentile(self.total_latencies, 50),
            "total_latency_p95_ms": _percentile(self.total_latencies, 95),
        }


# ---------------------------------------------------------------- 网关

class LLMGateway:
    def __init__(self, providers: list[Provider], *, max_retries: int = 2, backoff_s: float = 0.5,
                 max_retry_after_s: float = 10.0, max_concurrency: int | None = None,
                 breaker_threshold: int = 20, breaker_window: int = 50, breaker_failure_rate: float = 0.5,
                 breaker_cooldown_s: float = 30.0, cache_size: int = 512,
                 cassette: Cassette | None = None, mode: str = "live"):
        """breaker_threshold 是熔断判定所需的最少尝试数：窗口（最近 breaker_window 次）里至少这么多次、
        且失败率 ≥ breaker_failure_rate 才熔断。max_concurrency 为每个厂商的在途上限，None 不限。
        Retry-After 超过 max_retry_after_s 时不在本厂商等待，直接降级到下一个。"""
        if mode not in ("live", "record", "replay"):
            raise ValueError("mode 须为 live / record / replay")
        if mode != "live" and cassette is None:
            raise ValueError(f"{mode} 模式需要 cassette")
        if mode != "replay" and not providers:
            raise ValueError("没有可用的厂商")
        self.providers = providers
        self.max_retries, self.backoff_s, self.max_retry_after_s = max_retries, backoff_s, max_retry_after_s
        self.breaker_threshold, self.breaker_cooldown_s = breaker_threshold, breaker_cooldown_s
        self.breaker_failure_rate = breaker_failure_rate
        self.max_concurrency = max_concurrency
        self._slots = {p.name: Semaphore(max_concurrency) for p in providers} if max_concurrency else {}
        self.cache: OrderedDict[str, ChatResult] = OrderedDict()
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cassette, self.mode = cassette, mode
        self.stats: dict[str, ProviderStats] = {
            p.name: ProviderStats(outcomes=deque(maxlen=breaker_window)) for p in providers}
        self.request_stats = RequestStats()

    @classmethod
    def from_env(cls) -> LLMGateway:
        names = [n.strip() for n in os.getenv("LLM_PROVIDERS", "deepseek").split(",") if n.strip()]
        mode = os.getenv("LLM_MODE", "live")
        providers = []
        for n in names:
            try:
                providers.append(OpenAICompatProvider.from_env(n))
            except ValueError as e:
                log.warning("跳过厂商 %s：%s", n, e)
        cassette = Cassette(os.getenv("LLM_CASSETTE", "data/llm_cassette.jsonl")) if mode != "live" else None
        return cls(providers, cassette=cassette, mode=mode,
                   max_concurrency=int(os.getenv("LLM_MAX_CONCURRENCY", "0")) or None)

    @property
    def available(self) -> bool:
        return bool(self.providers) or self.mode == "replay"

    # ------------------------------------------------------------ 非流式

    async def chat(self, messages: list[dict], *, temperature: float = 0.0, json_mode: bool = False,
                   max_tokens: int | None = None, thinking: bool | None = None,
                   use_cache: bool = True) -> ChatResult:
        params: dict[str, Any] = {"temperature": temperature}
        # DeepSeek扩展；None不发送，保留已有接口与其他厂商的默认行为。
        if thinking is not None:
            params["thinking"] = {"type": "enabled" if thinking else "disabled"}
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        if max_tokens:
            params["max_tokens"] = max_tokens
        key = request_key(messages, params)

        if use_cache and temperature == 0 and key in self.cache:
            self.cache.move_to_end(key)
            self.cache_hits += 1
            return replace(self.cache[key], from_cache=True, latency_ms=0.0, attempts=0,
                           prompt_tokens=0, completion_tokens=0, cache_hit_tokens=0,
                           total_latency_ms=0.0, attempt_history=[])

        if self.mode == "replay":
            item = self.cassette.get(key)
            if item is None:
                raise LLMError("回放模式：录制中没有这条请求（提示词或参数变了？需重新录制）")
            # 新录制完整还原；旧录制缺少总耗时时保留 None，不能当作真实总耗时。
            return ChatResult.from_dict(item)

        result = await self._call_with_fallback(messages, params)
        if use_cache and temperature == 0 and self.cache_size:
            self.cache[key] = result
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        if self.mode == "record":
            self.cassette.put(key, result)
        return result

    async def _call_with_fallback(self, messages: list[dict], params: dict) -> ChatResult:
        started, history = time.perf_counter(), []
        errors = []
        try:
            for p in self.providers:
                st = self.stats[p.name]
                if st.open_until > time.monotonic():
                    errors.append(f"{p.name}: 熔断中")
                    continue
                try:
                    result = await self._call_with_retry(p, {"messages": messages, **params}, history)
                except LLMError as e:
                    errors.append(f"{p.name}: {e}")
                    log.warning("LLM fallback provider=%s error_type=%s status_code=%s",
                                p.name, history[-1].error_type, e.status)
                    continue
                self._record_request(started, history, result=result)
                return result
            raise LLMError("所有厂商均不可用：" + "；".join(errors))
        except (Exception, asyncio.CancelledError) as e:
            self._record_request(started, history, error=e)
            raise

    async def _call_with_retry(self, p: Provider, body: dict, history: list[AttemptRecord]) -> ChatResult:
        st = self.stats[p.name]
        attempt = 0
        while True:
            attempt += 1
            queue_ms = await self._acquire(p)
            st.calls += 1
            st.inflight += 1
            t0 = time.perf_counter()
            try:
                try:
                    data = await p.complete(body)
                finally:
                    self._release(p)
                usage = data.get("usage") or {}
                result = ChatResult(
                    content=data["choices"][0]["message"].get("content") or "",
                    provider=p.name, model=data.get("model", p.model),
                    prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=usage.get("completion_tokens", 0),
                    cache_hit_tokens=usage.get("prompt_cache_hit_tokens", 0), attempts=attempt,
                    finish_reason=data["choices"][0].get("finish_reason"),
                )
            except (Exception, asyncio.CancelledError) as e:
                record = self._record_attempt(p, history, t0, error=e, queue_ms=queue_ms)
                if not isinstance(e, LLMError) or not self._should_retry(st, e, attempt):
                    raise
                await self._wait_before_retry(p, record, e, attempt)
                if st.open_until > time.monotonic():
                    raise                       # 退避期间被其他请求的失败熔断了：不再打这个厂商
                continue
            result.latency_ms = self._record_attempt(p, history, t0, queue_ms=queue_ms).latency_ms
            self._record_success(st, result)
            return result

    async def _acquire(self, p: Provider) -> float:
        """占用该厂商的一个在途名额，返回排队毫秒数；未配置并发上限时不排队。"""
        slot = self._slots.get(p.name)
        if slot is None:
            return 0.0
        started = time.perf_counter()
        await slot.acquire()
        return round((time.perf_counter() - started) * 1000, 1)

    def _release(self, p: Provider) -> None:
        slot = self._slots.get(p.name)
        if slot is not None:
            slot.release()

    def _should_retry(self, st: ProviderStats, e: LLMError, attempt: int) -> bool:
        if not e.retryable or attempt > self.max_retries or st.open_until > time.monotonic():
            return False
        # 厂商要求等太久：与其占着请求干等，不如降级到下一个厂商。
        return e.retry_after is None or e.retry_after <= self.max_retry_after_s

    def _record_attempt(self, p: Provider, history: list[AttemptRecord], started: float,
                        *, error: BaseException | None = None, queue_ms: float = 0.0) -> AttemptRecord:
        st = self.stats[p.name]
        st.inflight -= 1
        cancelled = isinstance(error, (asyncio.CancelledError, GeneratorExit))
        record = AttemptRecord(
            provider=p.name, attempt=len(history) + 1,
            status="cancelled" if cancelled else "error" if error is not None else "success",
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            error_type=type(error.__cause__ or error).__name__ if error is not None else None,
            status_code=error.status if isinstance(error, LLMError) else None, queue_ms=queue_ms,
        )
        history.append(record)
        st.attempt_latencies.append(record.latency_ms)
        if self._slots:
            st.queue_waits.append(queue_ms)
        if cancelled:
            st.cancelled += 1
        elif error is not None:
            st.errors += 1
            if record.status_code == 429:
                st.rate_limited += 1        # 背压：靠退避和并发上限处理，不让它把厂商熔断掉
            else:
                self._record_outcome(p.name, st, ok=False)
            log.warning("LLM attempt provider=%s attempt=%d status=%s error_type=%s status_code=%s latency_ms=%.1f",
                        p.name, record.attempt, record.status, record.error_type, record.status_code, record.latency_ms)
        return record

    async def _wait_before_retry(self, p: Provider, record: AttemptRecord, error: LLMError, attempt: int) -> None:
        self.stats[p.name].retries += 1
        delay = error.retry_after if error.retry_after is not None else self.backoff_s * 2 ** (attempt - 1)
        delay *= 1 + 0.2 * random.random()
        log.warning("LLM retry provider=%s attempt=%d error_type=%s status_code=%s delay_ms=%.1f",
                    p.name, record.attempt, record.error_type, record.status_code, delay * 1000)
        started = time.perf_counter()
        try:
            await asyncio.sleep(delay)
        finally:
            record.wait_ms = round((time.perf_counter() - started) * 1000, 1)

    def _record_request(self, started: float, history: list[AttemptRecord], *,
                        result: ChatResult | None = None, error: BaseException | None = None) -> None:
        total = round((time.perf_counter() - started) * 1000, 1)
        st = self.request_stats
        st.calls += 1
        st.total_latencies.append(total)
        if isinstance(error, (asyncio.CancelledError, GeneratorExit)):
            st.cancelled += 1
        elif error is not None:
            st.errors += 1
        else:
            st.successes += 1
        target = result if result is not None else error
        if isinstance(target, (ChatResult, LLMError, asyncio.CancelledError, GeneratorExit)):
            target.total_latency_ms = total
            target.attempt_history = list(history)

    def _record_outcome(self, name: str, st: ProviderStats, *, ok: bool) -> None:
        now = time.monotonic()
        if st.open_until > now:
            return                          # 熔断期间仍在途的请求结果不计入下一轮窗口
        st.outcomes.append(ok)
        failures = st.outcomes.count(False)
        if (not ok and len(st.outcomes) >= self.breaker_threshold
                and failures / (len(st.outcomes) + st.inflight) >= self.breaker_failure_rate):
            st.open_until = now + self.breaker_cooldown_s
            st.times_opened += 1
            log.warning("熔断 provider=%s failures=%d/%d inflight=%d cooldown_s=%.0f",
                        name, failures, len(st.outcomes), st.inflight, self.breaker_cooldown_s)
            st.outcomes.clear()

    def _record_success(self, st: ProviderStats, r: ChatResult) -> None:
        self._record_outcome(r.provider, st, ok=True)
        st.latencies.append(r.latency_ms)
        if r.ttft_ms is not None:
            st.ttfts.append(r.ttft_ms)
        st.prompt_tokens += r.prompt_tokens
        st.completion_tokens += r.completion_tokens
        st.cache_hit_tokens += r.cache_hit_tokens

    # ------------------------------------------------------------ 流式

    async def stream(self, messages: list[dict], *, temperature: float = 0.3, max_tokens: int | None = None,
                     thinking: bool | None = None) -> AsyncIterator[str | ChatResult]:
        """逐段产出文本，最后产出一个 ChatResult（含用量与延迟）。

        只在尚未输出任何内容时才会重试或降级，避免前端收到重复文本。
        """
        body: dict[str, Any] = {"messages": messages, "temperature": temperature}
        if thinking is not None:      # 与 chat 一致：DeepSeek 扩展，None 不发送
            body["thinking"] = {"type": "enabled" if thinking else "disabled"}
        if max_tokens:
            body["max_tokens"] = max_tokens
        if self.mode == "replay":
            r = await self.chat(messages, temperature=temperature, max_tokens=max_tokens, thinking=thinking)
            yield r.content
            yield r
            return

        started, history = time.perf_counter(), []
        errors, completed = [], False
        try:
            for p in self.providers:
                st = self.stats[p.name]
                if st.open_until > time.monotonic():
                    errors.append(f"{p.name}: 熔断中")
                    continue
                for attempt in range(1, self.max_retries + 2):
                    queue_ms = await self._acquire(p)
                    st.calls += 1
                    st.inflight += 1
                    t0 = time.perf_counter()
                    ttft, parts, usage, model, emitted = None, [], {}, p.model, False
                    finish_reason = None
                    try:
                        try:
                            async for chunk in p.stream(body):
                                model = chunk.get("model", model)
                                if chunk.get("usage"):
                                    usage = chunk["usage"]
                                for ch in chunk.get("choices") or []:
                                    finish_reason = ch.get("finish_reason") or finish_reason
                                    text = (ch.get("delta") or {}).get("content")
                                    if text:
                                        if ttft is None:
                                            ttft = (time.perf_counter() - t0) * 1000
                                        parts.append(text)
                                        emitted = True
                                        yield text
                        finally:
                            self._release(p)
                        result = ChatResult(
                            "".join(parts), p.name, model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                            usage.get("prompt_cache_hit_tokens", 0),
                            ttft_ms=round(ttft, 1) if ttft is not None else None,
                            attempts=attempt, finish_reason=finish_reason)
                    except (Exception, asyncio.CancelledError, GeneratorExit) as e:
                        record = self._record_attempt(p, history, t0, error=e, queue_ms=queue_ms)
                        if not isinstance(e, LLMError) or emitted:
                            raise
                        errors.append(f"{p.name}: {e}")
                        if not self._should_retry(st, e, attempt):
                            break
                        await self._wait_before_retry(p, record, e, attempt)
                        if st.open_until > time.monotonic():
                            break
                        continue
                    result.latency_ms = self._record_attempt(p, history, t0, queue_ms=queue_ms).latency_ms
                    self._record_success(st, result)
                    self._record_request(started, history, result=result)
                    completed = True
                    if self.mode == "record":
                        self.cassette.put(request_key(messages, {k: v for k, v in body.items() if k != "messages"}),
                                          result)
                    yield result
                    return
            raise LLMError("所有厂商均不可用：" + "；".join(errors))
        except (Exception, asyncio.CancelledError, GeneratorExit) as e:
            if not completed:
                self._record_request(started, history, error=e)
            raise

    # ------------------------------------------------------------ 其他

    def summary(self) -> dict:
        return {"mode": self.mode, "cache_entries": len(self.cache), "cache_hits": self.cache_hits,
                "requests": self.request_stats.summary(),
                "providers": {p.name: {"model": p.model, **self.stats[p.name].summary()} for p in self.providers}}

    async def aclose(self) -> None:
        for p in self.providers:
            if hasattr(p, "aclose"):
                await p.aclose()

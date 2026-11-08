"""LLM 网关单元测试：用 httpx.MockTransport 模拟厂商，不需要网络和 API Key。"""

import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest

import app.ai.gateway as gateway_module
from app.ai.gateway import (
    AttemptRecord,
    Cassette,
    LLMError,
    LLMGateway,
    OpenAICompatProvider,
    request_key,
)

MSG = [{"role": "user", "content": "你好"}]


def ok(content="好的", model="m", finish_reason="stop"):
    return httpx.Response(200, json={
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "prompt_cache_hit_tokens": 4},
    })


class Script:
    """按顺序返回预设响应，并记录请求。"""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r


def provider(name, script, model="m"):
    return OpenAICompatProvider(name, "https://llm.test/v1", "k", model, transport=httpx.MockTransport(script))


def gateway(*providers, **kw):
    kw.setdefault("backoff_s", 0.0)
    return LLMGateway(list(providers), **kw)


async def test_success_parses_content_and_usage():
    s = Script(ok("你好呀"))
    r = await gateway(provider("a", s)).chat(MSG, json_mode=True)
    assert r.content == "你好呀" and r.provider == "a"
    assert (r.prompt_tokens, r.completion_tokens, r.cache_hit_tokens) == (10, 3, 4)
    assert s.requests[0]["response_format"] == {"type": "json_object"}
    assert s.requests[0]["model"] == "m"


async def test_retries_5xx_then_succeeds():
    s = Script(httpx.Response(502, text="bad gateway"), ok())
    gw = gateway(provider("a", s))
    r = await gw.chat(MSG)
    assert r.attempts == 2
    assert gw.summary()["providers"]["a"]["retries"] == 1


async def test_retries_timeout_and_429_with_retry_after():
    s = Script(httpx.ReadTimeout("slow"), httpx.Response(429, headers={"retry-after": "0"}, json={}), ok())
    r = await gateway(provider("a", s)).chat(MSG)
    assert r.attempts == 3
    assert [a.error_type for a in r.attempt_history] == ["ReadTimeout", "LLMError", None]
    assert [a.status_code for a in r.attempt_history] == [None, 429, None]


async def test_client_error_is_not_retried_but_falls_back():
    a = Script(httpx.Response(401, json={"error": {"message": "invalid key"}}))
    b = Script(ok("来自备用"))
    r = await gateway(provider("a", a), provider("b", b)).chat(MSG)
    assert len(a.requests) == 1
    assert r.provider == "b" and r.content == "来自备用"


async def test_all_providers_fail():
    a = Script(httpx.Response(500, text="x"))
    with pytest.raises(LLMError, match="所有厂商均不可用"):
        await gateway(provider("a", a), max_retries=1).chat(MSG)
    assert len(a.requests) == 2  # 1 次 + 1 次重试


async def test_circuit_breaker_skips_failing_provider():
    a = Script(httpx.Response(500, text="x"))
    b = Script(ok())
    gw = gateway(provider("a", a), provider("b", b), max_retries=5, breaker_threshold=3, breaker_cooldown_s=60)
    await gw.chat(MSG, temperature=0.5)
    assert len(a.requests) == 3  # 窗口里已有 3 次尝试且全部失败 → 熔断，不再重试
    await gw.chat(MSG, temperature=0.5)
    assert len(a.requests) == 3  # 熔断期间直接跳过
    assert gw.summary()["providers"]["a"]["circuit"] == "open"


async def test_rate_limit_429_does_not_open_circuit():
    """429 是限流背压：照常按 Retry-After 重试，但不计入熔断窗口。"""
    s = Script(httpx.Response(429, headers={"retry-after": "0.01"}, json={}),
               httpx.Response(429, headers={"retry-after": "0.01"}, json={}), ok())
    gw = gateway(provider("a", s), breaker_threshold=1)
    r = await gw.chat(MSG)
    assert r.attempts == 3 and [a.status_code for a in r.attempt_history] == [429, 429, None]
    summary = gw.summary()["providers"]["a"]
    assert summary["circuit"] == "closed" and summary["rate_limited"] == 2 and summary["errors"] == 2
    assert summary["window_failure_rate"] == 0.0


def test_retry_after_accepts_fractional_seconds_only():
    parse = OpenAICompatProvider._retry_after
    assert parse("1") == 1.0 and parse("0.5") == 0.5
    assert parse(None) is None and parse("Wed, 21 Oct 2026 07:28:00 GMT") is None
    assert parse("-1") is None and parse("inf") is None


async def test_long_retry_after_falls_back_instead_of_waiting():
    a = Script(httpx.Response(429, headers={"retry-after": "120"}, json={}))
    b = Script(ok("备用"))
    gw = gateway(provider("a", a), provider("b", b), max_retry_after_s=10)
    r = await gw.chat(MSG)
    assert r.provider == "b" and len(a.requests) == 1
    assert [a.wait_ms for a in r.attempt_history] == [0.0, 0.0]


async def test_breaker_uses_failure_rate_not_streak():
    """30% 失败率不熔断（旧实现连续 3 次失败就会切走）；失败率过半才熔断。"""
    pattern = [True, False, True, True, False, True, True, False, True, True] * 3   # 30% 失败
    gw = gateway(provider("a", Script(ok())), breaker_threshold=10, breaker_window=20)
    st = gw.stats["a"]
    for ok_ in pattern:
        gw._record_outcome("a", st, ok=ok_)
    assert st.times_opened == 0
    for _ in range(5):        # 窗口里失败数 6 → 9（滑出的旧记录里也有失败），仍不到一半
        gw._record_outcome("a", st, ok=False)
    assert st.times_opened == 0
    gw._record_outcome("a", st, ok=False)      # 10/20 = 50%
    assert st.times_opened == 1 and gw.summary()["providers"]["a"]["circuit"] == "open"
    assert len(st.outcomes) == 0
    gw._record_outcome("a", st, ok=False)     # 熔断期间在途请求的失败不计入下一轮
    assert len(st.outcomes) == 0


async def test_breaker_counts_inflight_as_not_yet_failed():
    """高并发下快速失败先回来：在途请求按「可能成功」计入分母，不因时间偏差误熔断。"""
    gw = gateway(provider("a", Script(ok())), breaker_threshold=5)
    st = gw.stats["a"]
    st.inflight = 10
    for _ in range(9):                         # 9 / (9 + 10) < 50%
        gw._record_outcome("a", st, ok=False)
    assert st.times_opened == 0
    gw._record_outcome("a", st, ok=False)      # 10 / 20 = 50%：即使在途全部成功也过半
    assert st.times_opened == 1


async def test_breaker_needs_minimum_calls():
    gw = gateway(provider("a", Script(ok())), breaker_threshold=5)
    st = gw.stats["a"]
    for _ in range(4):
        gw._record_outcome("a", st, ok=False)
    assert st.times_opened == 0
    gw._record_outcome("a", st, ok=False)
    assert st.times_opened == 1


@pytest.mark.parametrize("kind", ["chat", "stream"])
async def test_bulkhead_limits_inflight_and_records_queue_time(kind):
    inflight, peak, release = 0, 0, asyncio.Event()

    class Slow:
        name, model = "a", "m"

        async def enter(self):
            nonlocal inflight, peak
            inflight += 1
            peak = max(peak, inflight)
            await release.wait()
            inflight -= 1

        async def complete(self, body):
            await self.enter()
            return ok().json()

        async def stream(self, body):
            await self.enter()
            yield text_chunk("好")

    gw = gateway(Slow(), max_concurrency=2)
    tasks = [asyncio.create_task(invoke(gw, kind)) for _ in range(5)]
    await asyncio.sleep(0.02)
    assert peak == inflight == 2               # 其余 3 个在网关里排队，没有打到厂商
    release.set()
    results = await asyncio.gather(*tasks)
    assert peak == 2 and all(r.content == "好的" if kind == "chat" else r.content == "好" for r in results)
    queued = [r.attempt_history[0].queue_ms for r in results]
    assert sum(q > 0 for q in queued) == 3
    assert gw.summary()["providers"]["a"]["queue_p95_ms"] > 0


async def test_bulkhead_slot_released_on_error_and_cancel():
    s = Script(httpx.Response(400, json={}))
    gw = gateway(provider("a", s), max_concurrency=1, max_retries=0)
    for _ in range(3):
        with pytest.raises(LLMError):
            await gw.chat(MSG, temperature=0.5)
    gen = gw.stream(MSG)
    with pytest.raises(LLMError):
        await gen.__anext__()
    assert gw._slots["a"]._value == 1


async def test_cache_only_for_deterministic_calls():
    s = Script(ok())
    gw = gateway(provider("a", s))
    await gw.chat(MSG)
    r = await gw.chat(MSG)
    assert r.from_cache and len(s.requests) == 1
    await gw.chat(MSG, temperature=0.7)
    await gw.chat(MSG, temperature=0.7)
    assert len(s.requests) == 3


async def test_per_call_cache_bypass_does_not_read_or_write_cached_result():
    script = Script(ok("缓存结果"), ok("本次重新请求"))
    gw = gateway(provider("a", script))
    await gw.chat(MSG)
    fresh = await gw.chat(MSG, use_cache=False)
    assert fresh.content == "本次重新请求" and not fresh.from_cache
    assert (await gw.chat(MSG)).content == "缓存结果"
    assert len(script.requests) == 2 and "use_cache" not in script.requests[-1]


async def test_thinking_switch_is_explicit_and_part_of_cache_key():
    s = Script(ok("默认"), ok("关闭思考"))
    gw = gateway(provider("a", s))
    assert (await gw.chat(MSG)).content == "默认"
    assert (await gw.chat(MSG, thinking=False)).content == "关闭思考"  # 不能命中默认调用的缓存
    assert "thinking" not in s.requests[0]
    assert s.requests[1]["thinking"] == {"type": "disabled"}


def sse(*chunks):
    body = "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


async def test_stream_thinking_switch_matches_chat_and_replays(tmp_path):
    chunk = {"model": "m", "choices": [{"delta": {"content": "好"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
    s = Script(sse(chunk))
    path = tmp_path / "c.jsonl"
    rec = gateway(provider("a", s), cassette=Cassette(path), mode="record")
    items = [x async for x in rec.stream(MSG, thinking=False)]
    assert items[0] == "好" and s.requests[0]["thinking"] == {"type": "disabled"}
    default = Script(sse(chunk))
    [x async for x in gateway(provider("b", default)).stream(MSG)]
    assert "thinking" not in default.requests[0]
    rep = LLMGateway([], cassette=Cassette(path), mode="replay")
    assert [x async for x in rep.stream(MSG, thinking=False)][-1].content == "好"   # 录制键与 chat 一致


async def test_stream_yields_text_then_result():
    s = Script(sse(
        {"model": "m", "choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "氨氮"}}]},
        {"choices": [{"delta": {"content": "升高"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7, "completion_tokens": 2}},
    ))
    out = [x async for x in gateway(provider("a", s)).stream(MSG)]
    assert out[:2] == ["氨氮", "升高"]
    result = out[-1]
    assert result.content == "氨氮升高" and result.completion_tokens == 2 and result.ttft_ms is not None
    assert result.finish_reason == "stop"
    assert s.requests[0]["stream"] is True


async def test_stream_falls_back_before_any_output():
    a = Script(httpx.Response(503, text="busy"))
    b = Script(sse({"choices": [{"delta": {"content": "备用"}}]}))
    out = [x async for x in gateway(provider("a", a), provider("b", b), max_retries=0).stream(MSG)]
    assert out[0] == "备用" and out[-1].provider == "b"


async def test_record_then_replay(tmp_path):
    path = tmp_path / "c.jsonl"
    s = Script(ok("录下来的回答", finish_reason="length"))
    rec = gateway(provider("a", s), cassette=Cassette(path), mode="record")
    original = await rec.chat(MSG, json_mode=True)

    rep = LLMGateway([], cassette=Cassette(path), mode="replay")
    r = await rep.chat(MSG, json_mode=True)
    assert r.content == "录下来的回答"
    assert r.latency_ms == original.latency_ms
    assert r.finish_reason == original.finish_reason == "length"
    with pytest.raises(LLMError, match="回放模式"):
        await rep.chat([{"role": "user", "content": "没录过"}])


async def test_rerecord_replays_latest_response(tmp_path):
    path = tmp_path / "c.jsonl"
    rec = gateway(provider("a", Script(ok("第一次"), ok("第二次"))),
                  cassette=Cassette(path), mode="record", cache_size=0)
    await rec.chat(MSG)
    latest = await rec.chat(MSG)
    rep = LLMGateway([], cassette=Cassette(path), mode="replay")
    replayed = await rep.chat(MSG)
    assert replayed.content == "第二次" and replayed.latency_ms == latest.latency_ms


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.sleep_extra_s = 0.0
        self.sleep_fraction = 1.0
        self.sleep_error = None

    def perf_counter(self):
        return self.now

    def monotonic(self):
        return self.now

    def advance(self, milliseconds):
        self.now += milliseconds / 1000

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds * self.sleep_fraction + self.sleep_extra_s
        if self.sleep_error is not None:
            raise self.sleep_error


@pytest.fixture
def clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(gateway_module, "time", clock)
    monkeypatch.setattr(gateway_module, "random", SimpleNamespace(random=lambda: 0.0))
    monkeypatch.setattr(gateway_module, "asyncio", SimpleNamespace(
        sleep=clock.sleep, CancelledError=asyncio.CancelledError))
    return clock


class TimedProvider:
    """每个脚本事件推进假时钟，完全离线，不构造真实厂商客户端。"""

    model = "fake-model"

    def __init__(self, clock, name="a", *, responses=(), streams=()):
        self.clock, self.name = clock, name
        self.responses, self.streams = list(responses), list(streams)
        self.calls = 0

    def event(self, elapsed, value):
        self.clock.advance(elapsed)
        if isinstance(value, BaseException):
            raise value
        return value

    async def complete(self, body):
        self.calls += 1
        return self.event(*self.responses.pop(0))

    async def stream(self, body):
        self.calls += 1
        for event in self.streams.pop(0):
            yield self.event(*event)


def text_chunk(text):
    return {"choices": [{"delta": {"content": text}}]}


def timed_provider(clock, kind, name="a", *events):
    if kind == "chat":
        return TimedProvider(clock, name, responses=events)
    streams = []
    for elapsed, value in events:
        if isinstance(value, BaseException):
            # 角色元数据已经发出仍可重试，因为没有文本传递给调用者。
            streams.append([(0, {"choices": [{"delta": {"role": "assistant"}}]}), (elapsed, value)])
        else:
            streams.append([(elapsed, {
                "model": value["model"], "choices": [{
                    "delta": {"content": value["choices"][0]["message"]["content"]},
                    "finish_reason": value["choices"][0]["finish_reason"],
                }], "usage": value["usage"],
            })])
    return TimedProvider(clock, name, streams=streams)


async def invoke(gw, kind, **params):
    params.setdefault("temperature", 0.0)
    if kind == "chat":
        return await gw.chat(MSG, **params)
    return [x async for x in gw.stream(MSG, **params)][-1]


@pytest.mark.parametrize("kind", ["chat", "stream"])
async def test_retry_trace_counts_failures_and_actual_wait(clock, caplog, kind):
    sensitive = "private-prompt-and-test-credential"
    failure = LLMError(sensitive, retryable=True, status=429, retry_after=0.2)
    p = timed_provider(clock, kind, "a", (400, failure), (100, ok().json()))
    gw = gateway(p, backoff_s=10)
    clock.sleep_extra_s = 0.05  # 实际等待比计划值多 50ms，记录不能仅使用计划值。
    with caplog.at_level(logging.WARNING, logger="app.ai.gateway"):
        r = await invoke(gw, kind)

    assert r.latency_ms == 100.0 and r.total_latency_ms == 750.0 and r.attempts == 2
    assert r.attempt_history == [
        AttemptRecord("a", 1, "error", 400.0, 250.0, "LLMError", 429),
        AttemptRecord("a", 2, "success", 100.0),
    ]
    assert clock.sleeps == [0.2]
    assert "LLM retry" in caplog.text and "provider=a" in caplog.text
    assert sensitive not in caplog.text and sensitive not in json.dumps(r.to_dict())
    stats = gw.summary()
    assert stats["requests"] == {
        "calls": 1, "successes": 1, "errors": 0, "cancelled": 0,
        "total_latency_p50_ms": 750.0, "total_latency_p95_ms": 750.0,
    }
    assert stats["providers"]["a"]["latency_p95_ms"] == 100.0
    assert stats["providers"]["a"]["attempt_latency_p95_ms"] == 400.0


@pytest.mark.parametrize("kind", ["chat", "stream"])
async def test_final_failure_has_trace_and_total_latency(clock, kind):
    p = timed_provider(clock, kind, "a",
                       (100, LLMError("busy", retryable=True, status=503)),
                       (200, LLMError("busy", retryable=True, status=503)))
    gw = gateway(p, max_retries=1, backoff_s=0.25)
    with pytest.raises(LLMError, match="所有厂商均不可用") as caught:
        await invoke(gw, kind)
    assert caught.value.total_latency_ms == 550.0
    assert caught.value.attempt_history == [
        AttemptRecord("a", 1, "error", 100.0, 250.0, "LLMError", 503),
        AttemptRecord("a", 2, "error", 200.0, 0.0, "LLMError", 503),
    ]
    assert clock.sleeps == [0.25] and p.calls == 2
    stats = gw.summary()
    assert stats["requests"] == {
        "calls": 1, "successes": 0, "errors": 1, "cancelled": 0,
        "total_latency_p50_ms": 550.0, "total_latency_p95_ms": 550.0,
    }
    assert stats["providers"]["a"]["errors"] == 2
    assert stats["providers"]["a"]["latency_p95_ms"] is None
    assert stats["providers"]["a"]["attempt_latency_p95_ms"] == 200.0


@pytest.mark.parametrize("kind", ["chat", "stream"])
async def test_fallback_trace_spans_providers_and_keeps_legacy_attempts(clock, caplog, kind):
    a = timed_provider(clock, kind, "a", (100, LLMError("private-error", status=401)))
    b = timed_provider(clock, kind, "b",
                       (200, LLMError("busy", retryable=True, status=502, retry_after=0.05)),
                       (300, ok("备用").json()))
    gw = gateway(a, b)
    with caplog.at_level(logging.WARNING, logger="app.ai.gateway"):
        r = await invoke(gw, kind)
    assert r.provider == "b" and r.attempts == 2
    assert r.latency_ms == 300.0 and r.total_latency_ms == 650.0
    assert [(a.provider, a.attempt, a.status) for a in r.attempt_history] == [
        ("a", 1, "error"), ("b", 2, "error"), ("b", 3, "success"),
    ]
    assert [a.wait_ms for a in r.attempt_history] == [0.0, 50.0, 0.0]
    assert a.calls == 1 and b.calls == 2 and "private-error" not in caplog.text
    assert gw.summary()["requests"]["calls"] == 1


@pytest.mark.parametrize("kind", ["chat", "stream"])
async def test_open_circuit_does_not_add_attempt_or_sleep(clock, kind):
    p = timed_provider(clock, kind, "a", (100, LLMError("busy", retryable=True, status=503)))
    gw = gateway(p, max_retries=5, breaker_threshold=1, breaker_cooldown_s=60, backoff_s=1)
    with pytest.raises(LLMError) as first:
        await invoke(gw, kind)
    assert first.value.total_latency_ms == 100.0 and len(first.value.attempt_history) == 1
    with pytest.raises(LLMError) as second:
        await invoke(gw, kind)
    assert second.value.total_latency_ms == 0.0 and second.value.attempt_history == []
    assert p.calls == 1 and clock.sleeps == []
    assert gw.summary()["providers"]["a"]["retries"] == 0
    assert gw.summary()["requests"]["errors"] == 2


@pytest.mark.parametrize("kind", ["chat", "stream"])
async def test_breaker_opened_during_backoff_stops_retrying(clock, kind):
    a = timed_provider(clock, kind, "a", (100, LLMError("busy", retryable=True, status=503)), (100, ok().json()))
    b = timed_provider(clock, kind, "b", (50, ok("备用").json()))
    gw = gateway(a, b, max_retries=2, backoff_s=1)

    async def sleep(seconds):
        await clock.sleep(seconds)
        gw.stats["a"].open_until = clock.monotonic() + 30    # 并发的其他请求在退避期间把 a 熔断了

    gateway_module.asyncio.sleep = sleep
    r = await invoke(gw, kind)
    assert r.provider == "b" and a.calls == 1 and b.calls == 1
    assert [(x.provider, x.status) for x in r.attempt_history] == [("a", "error"), ("b", "success")]


async def test_cache_hit_clears_trace_without_mutating_original(clock):
    p = timed_provider(clock, "chat", "a",
                       (100, LLMError("busy", retryable=True)), (200, ok().json()))
    gw = gateway(p, backoff_s=0.5)
    original = await gw.chat(MSG)
    cached = await gw.chat(MSG)
    assert cached.from_cache and cached.latency_ms == cached.total_latency_ms == 0.0
    assert cached.attempts == 0 and cached.attempt_history == []
    assert cached.prompt_tokens == cached.completion_tokens == cached.cache_hit_tokens == 0
    assert (original.prompt_tokens, original.completion_tokens, original.cache_hit_tokens) == (10, 3, 4)
    assert original.total_latency_ms == 800.0 and len(original.attempt_history) == 2
    assert original.attempt_history[-1].status == "success" and not original.from_cache
    assert p.calls == 2
    assert gw.summary()["cache_hits"] == 1 and gw.summary()["requests"]["calls"] == 1


@pytest.mark.parametrize("kind", ["chat", "stream"])
async def test_new_cassette_preserves_complete_trace_and_usage(clock, tmp_path, kind):
    path = tmp_path / "trace.jsonl"
    p = timed_provider(clock, kind, "a",
                       (100, LLMError("private-error", retryable=True)),
                       (200, ok("录制结果", finish_reason="length").json()))
    rec = gateway(p, cassette=Cassette(path), mode="record", backoff_s=0.5)
    original = await invoke(rec, kind)
    rep = LLMGateway([], cassette=Cassette(path), mode="replay")
    replayed = await invoke(rep, kind)
    assert replayed.to_dict() == original.to_dict()
    assert all(isinstance(a, AttemptRecord) for a in replayed.attempt_history)
    assert replayed.total_latency_ms == 800.0 and replayed.cache_hit_tokens == 4
    assert replayed.ttft_ms == (200.0 if kind == "stream" else None)
    assert "private-error" not in path.read_text()
    assert rep.summary()["requests"]["calls"] == 0


@pytest.mark.parametrize("kind", ["chat", "stream"])
async def test_old_cassette_keeps_unknown_total_latency(clock, tmp_path, kind):
    path = tmp_path / "legacy.jsonl"
    item = {"key": request_key(MSG, {"temperature": 0.0}), "content": "旧结果",
            "provider": "old", "model": "old-model", "latency_ms": 42.5,
            "prompt_tokens": 8, "completion_tokens": 2, "finish_reason": "stop"}
    path.write_text(json.dumps(item) + "\n")
    rep = LLMGateway([], cassette=Cassette(path), mode="replay")
    r = await invoke(rep, kind)
    assert r.content == "旧结果" and r.latency_ms == 42.5 and r.attempts == 1
    assert r.total_latency_ms is None and r.attempt_history == []
    assert r.finish_reason == "stop" and rep.summary()["requests"]["calls"] == 0


async def test_stream_failure_after_output_never_retries_or_falls_back(clock):
    failure = LLMError("断流", retryable=True, status=503)
    a = TimedProvider(clock, streams=[[(30, text_chunk("已输出")), (70, failure)]])
    b = timed_provider(clock, "stream", "b", (50, ok("备用").json()))
    gw = gateway(a, b, max_retries=3, backoff_s=1)
    output = []
    with pytest.raises(LLMError) as caught:
        async for item in gw.stream(MSG):
            output.append(item)
    assert caught.value is failure and output == ["已输出"]
    assert failure.total_latency_ms == 100.0
    assert failure.attempt_history == [AttemptRecord("a", 1, "error", 100.0, 0.0, "LLMError", 503)]
    assert a.calls == 1 and b.calls == 0 and clock.sleeps == []
    assert gw.summary()["requests"]["errors"] == 1


@pytest.mark.parametrize("kind", ["chat", "stream"])
async def test_task_cancellation_propagates_and_never_falls_back(clock, kind):
    started = asyncio.Event()
    blocked = asyncio.Event()

    class BlockingProvider:
        name, model = "a", "fake-model"

        async def wait(self):
            clock.advance(80)
            started.set()
            await blocked.wait()

        async def complete(self, body):
            await self.wait()
            return ok().json()

        async def stream(self, body):
            yield text_chunk("已输出")
            await self.wait()

    b = timed_provider(clock, kind, "b", (50, ok().json()))
    gw = gateway(BlockingProvider(), b)
    task = asyncio.create_task(invoke(gw, kind))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel("test cancellation")
    with pytest.raises(asyncio.CancelledError, match="test cancellation") as caught:
        await task
    assert caught.value.total_latency_ms == 80.0
    assert caught.value.attempt_history == [AttemptRecord("a", 1, "cancelled", 80.0,
                                                         error_type="CancelledError")]
    assert b.calls == 0 and clock.sleeps == []
    stats = gw.summary()
    assert stats["requests"] == {
        "calls": 1, "successes": 0, "errors": 0, "cancelled": 1,
        "total_latency_p50_ms": 80.0, "total_latency_p95_ms": 80.0,
    }
    assert stats["providers"]["a"]["cancelled"] == 1
    assert stats["providers"]["a"]["errors"] == 0


@pytest.mark.parametrize("kind", ["chat", "stream"])
async def test_cancellation_during_backoff_keeps_partial_wait(clock, kind):
    cancelled = asyncio.CancelledError("stop waiting")
    clock.sleep_error, clock.sleep_fraction = cancelled, 0.4
    p = timed_provider(clock, kind, "a", (100, LLMError("busy", retryable=True)))
    gw = gateway(p, backoff_s=1)
    with pytest.raises(asyncio.CancelledError) as caught:
        await invoke(gw, kind)
    assert caught.value is cancelled and cancelled.total_latency_ms == 500.0
    assert cancelled.attempt_history == [AttemptRecord("a", 1, "error", 100.0, 400.0, "LLMError")]
    assert p.calls == 1 and gw.summary()["requests"]["cancelled"] == 1
    assert gw.summary()["providers"]["a"]["cancelled"] == 0  # 取消时没有实际厂商请求在途


async def test_closing_stream_records_cancellation_without_retry(clock):
    p = TimedProvider(clock, streams=[[(30, text_chunk("片段")), (70, text_chunk("未拉取"))]])
    gw = gateway(p)
    stream = gw.stream(MSG)
    assert await anext(stream) == "片段"
    clock.advance(20)
    await stream.aclose()
    stats = gw.summary()
    assert stats["requests"]["cancelled"] == 1 and stats["requests"]["calls"] == 1
    assert stats["requests"]["total_latency_p95_ms"] == 50.0
    assert stats["providers"]["a"]["cancelled"] == 1
    assert p.calls == 1 and clock.sleeps == []


@pytest.mark.parametrize("kind", ["chat", "stream"])
async def test_percentiles_include_failed_and_cancelled_requests(clock, kind):
    p = timed_provider(clock, kind, "a",
                       (100, LLMError("failed")), (20, ok().json()),
                       (50, asyncio.CancelledError("cancelled")))
    gw = gateway(p, max_retries=0, cache_size=0)
    with pytest.raises(LLMError):
        await invoke(gw, kind)
    await invoke(gw, kind)
    with pytest.raises(asyncio.CancelledError):
        await invoke(gw, kind)
    stats = gw.summary()
    assert stats["requests"] == {
        "calls": 3, "successes": 1, "errors": 1, "cancelled": 1,
        "total_latency_p50_ms": 50.0, "total_latency_p95_ms": 100.0,
    }
    assert stats["providers"]["a"]["latency_p50_ms"] == 20.0
    assert stats["providers"]["a"]["attempt_latency_p50_ms"] == 50.0
    assert stats["providers"]["a"]["attempt_latency_p95_ms"] == 100.0


async def test_closing_after_stream_result_does_not_count_cancellation(clock):
    p = timed_provider(clock, "stream", "a", (100, ok("完成").json()))
    gw = gateway(p)
    stream = gw.stream(MSG)
    assert await anext(stream) == "完成"
    result = await anext(stream)
    assert result.total_latency_ms == 100.0
    await stream.aclose()
    assert gw.summary()["requests"]["successes"] == 1
    assert gw.summary()["requests"]["cancelled"] == 0
    assert gw.summary()["providers"]["a"]["calls"] == 1

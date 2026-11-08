"""Redis 会话单元测试；可通过 --noconftest 独立运行，不访问外部服务。"""

import asyncio
import json
from dataclasses import is_dataclass
from uuid import UUID, uuid4

import fakeredis
import pytest
from redis.asyncio.client import Pipeline

from app.ai.agent.sessions import (
    SessionBusy,
    SessionExpired,
    SessionLease,
    SessionNotFound,
    SessionStore,
)


@pytest.fixture(params=[False, True], ids=["bytes", "decoded"])
async def redis_client(request):
    async with fakeredis.FakeAsyncRedis(decode_responses=request.param) as client:
        yield client


@pytest.fixture
def store(redis_client):
    return SessionStore(redis_client)


def _before_next_exec(monkeypatch, action):
    """在读完 WATCH 的键之后、EXEC 之前插入一次真实 Redis 写入。"""
    original = Pipeline.execute
    pending = True

    async def execute(pipe, *args, **kwargs):
        nonlocal pending
        if pending:
            pending = False
            await action()
        return await original(pipe, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "execute", execute)


async def test_create_read_append_and_renew_ttl(store, redis_client):
    lease = await store.acquire()
    assert isinstance(lease, SessionLease) and is_dataclass(lease)
    assert UUID(hex=lease.session_id).version == 4
    assert UUID(hex=lease.session_id).hex == lease.session_id
    assert lease.token and lease.turns == []
    key = f"ai:session:{lease.session_id}"
    lock_key = f"{key}:lock"
    assert await store.get(lease.session_id) == {"session_id": lease.session_id, "turns": []}
    assert 86398 <= await redis_client.ttl(key) <= 86400
    assert 238 <= await redis_client.ttl(lock_key) <= 240
    assert await redis_client.get(lock_key) in (lease.token, lease.token.encode())

    await redis_client.expire(key, 10)
    result = {
        "answer": "水质正常",
        "evidence": [{"rows": [{"plant": "汉中", "value": 1.5, "ok": True, "note": None}]}],
        "trace": {"steps": [{"tool": "query", "complete": True}]},
    }
    await store.finish(lease, result)
    assert await store.get(lease.session_id) == {"session_id": lease.session_id, "turns": [result]}
    assert 86398 <= await redis_client.ttl(key) <= 86400
    # finish 保留租约，直到调用方 finally 中显式释放。
    with pytest.raises(SessionBusy):
        await store.acquire(lease.session_id)
    await store.release(lease)
    await store.release(lease)
    assert not await redis_client.exists(lock_key)
    assert (await store.get(lease.session_id))["turns"] == [result]

    next_lease = await store.acquire(lease.session_id)
    assert next_lease.token != lease.token
    assert next_lease.turns == [result]
    await store.release(next_lease)


async def test_keep_six_complete_results(store):
    lease = await store.acquire()
    session_id = lease.session_id
    results = []
    for turn in range(8):
        result = {
            "answer": f"第 {turn} 轮",
            "evidence": {"rows": [{"n": row, "details": [None, True, "原始数据"]} for row in range(80)]},
            "trace": [{"round": turn}],
        }
        results.append(result)
        await store.finish(lease, result)
        await store.release(lease)
        assert (await store.get(session_id))["turns"] == results[-6:]
        lease = await store.acquire(session_id)
        assert lease.turns == results[-6:]
    await store.release(lease)
    assert len((await store.get(session_id))["turns"]) == 6


async def test_custom_ttl_lock_and_max_turns(redis_client):
    store = SessionStore(redis_client, ttl_s=90, lock_s=15, max_turns=2)
    lease = await store.acquire()
    key = f"ai:session:{lease.session_id}"
    assert 88 <= await redis_client.ttl(key) <= 90
    assert 13 <= await redis_client.ttl(f"{key}:lock") <= 15
    for turn in range(3):
        await store.finish(lease, {"turn": turn})
    assert (await store.get(lease.session_id))["turns"] == [{"turn": 1}, {"turn": 2}]
    await store.release(lease)


@pytest.mark.parametrize("session_id", ["0" * 32, "F" * 32])
async def test_unknown_valid_id_never_creates_session(store, redis_client, session_id):
    with pytest.raises(SessionNotFound):
        await store.get(session_id)
    with pytest.raises(SessionNotFound):
        await store.acquire(session_id)
    assert await redis_client.dbsize() == 0


@pytest.mark.parametrize("release_first", [False, True])
async def test_expired_session_never_recreated(store, redis_client, release_first):
    lease = await store.acquire()
    key = f"ai:session:{lease.session_id}"
    lock_key = f"{key}:lock"
    if release_first:
        await store.release(lease)
    await redis_client.pexpire(key, 0)
    previous_lock = await redis_client.get(lock_key)

    with pytest.raises(SessionNotFound):
        await store.get(lease.session_id)
    with pytest.raises(SessionNotFound):
        await store.acquire(lease.session_id)
    with pytest.raises(SessionExpired):
        await store.finish(lease, {"answer": "迟到的结果"})
    assert not await redis_client.exists(key)
    assert await redis_client.get(lock_key) == previous_lock
    await store.release(lease)
    assert await redis_client.dbsize() == 0


async def test_existing_lock_is_immediately_busy(store):
    lease = await store.acquire()
    with pytest.raises(SessionBusy):
        await asyncio.wait_for(store.acquire(lease.session_id), timeout=1)
    await store.release(lease)


async def test_concurrent_acquire_has_one_winner(store):
    lease = await store.acquire()
    await store.release(lease)
    outcomes = await asyncio.gather(
        *(store.acquire(lease.session_id) for _ in range(8)), return_exceptions=True
    )
    winners = [outcome for outcome in outcomes if isinstance(outcome, SessionLease)]
    assert len(winners) == 1
    assert sum(isinstance(outcome, SessionBusy) for outcome in outcomes) == 7
    await store.release(winners[0])


async def test_different_sessions_are_concurrent_and_isolated(store):
    left, right = await asyncio.gather(store.acquire(), store.acquire())
    assert left.session_id != right.session_id
    assert left.token != right.token
    await asyncio.gather(
        store.finish(left, {"answer": "左会话"}),
        store.finish(right, {"answer": "右会话"}),
    )
    assert (await store.get(left.session_id))["turns"] == [{"answer": "左会话"}]
    assert (await store.get(right.session_id))["turns"] == [{"answer": "右会话"}]
    await store.release(left)
    next_left = await store.acquire(left.session_id)
    with pytest.raises(SessionBusy):
        await store.acquire(right.session_id)
    await asyncio.gather(store.release(next_left), store.release(right))


async def test_old_lease_cannot_release_or_overwrite_new_owner(store, redis_client):
    old = await store.acquire()
    key = f"ai:session:{old.session_id}"
    lock_key = f"{key}:lock"
    await redis_client.pexpire(lock_key, 0)
    current = await store.acquire(old.session_id)
    assert current.token != old.token
    await store.finish(current, {"answer": "新租约结果"})

    await store.release(old)
    assert await redis_client.get(lock_key) in (current.token, current.token.encode())
    with pytest.raises(SessionExpired):
        await store.finish(old, {"answer": "旧租约结果"})
    assert (await store.get(old.session_id))["turns"] == [{"answer": "新租约结果"}]
    assert await redis_client.get(lock_key) in (current.token, current.token.encode())
    await store.finish(current, {"answer": "继续写入"})
    await store.release(current)
    assert (await store.get(old.session_id))["turns"] == [
        {"answer": "新租约结果"}, {"answer": "继续写入"}
    ]


async def test_expired_lock_cannot_write_or_renew_session(store, redis_client):
    lease = await store.acquire()
    key = f"ai:session:{lease.session_id}"
    original = await redis_client.get(key)
    await redis_client.expire(key, 10)
    await redis_client.pexpire(f"{key}:lock", 0)
    with pytest.raises(SessionExpired):
        await store.finish(lease, {"answer": "过期结果"})
    assert await redis_client.get(key) == original
    assert 0 < await redis_client.ttl(key) <= 10
    await store.release(lease)


async def test_cancelled_task_can_release_in_finally(store, redis_client):
    acquired = asyncio.Event()
    blocked = asyncio.Event()
    leases = []

    async def work():
        lease = await store.acquire()
        leases.append(lease)
        try:
            acquired.set()
            await blocked.wait()
        finally:
            await store.release(lease)

    task = asyncio.create_task(work())
    try:
        await asyncio.wait_for(acquired.wait(), timeout=1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    lease = leases[0]
    assert not await redis_client.exists(f"ai:session:{lease.session_id}:lock")
    next_lease = await store.acquire(lease.session_id)
    assert next_lease.turns == []
    await store.release(next_lease)


async def test_finish_retries_without_losing_concurrent_results(store, redis_client, monkeypatch):
    lease = await store.acquire()
    key = f"ai:session:{lease.session_id}"
    concurrent = {"answer": "并发写入的完整结果", "rows": [{"value": 1}]}
    result = {"answer": "本次结果"}
    lease.turns.append({"answer": "仅在内存中修改的历史"})

    async def concurrent_write():
        session = await store.get(lease.session_id)
        session["turns"].append(concurrent)
        await redis_client.set(key, json.dumps(session), keepttl=True)

    _before_next_exec(monkeypatch, concurrent_write)
    await store.finish(lease, result)
    assert (await store.get(lease.session_id))["turns"] == [concurrent, result]
    await store.release(lease)


@pytest.mark.parametrize("change", ["replace_lock", "expire_lock", "expire_session"])
async def test_finish_rechecks_session_and_token_on_exec_conflict(
    store, redis_client, monkeypatch, change
):
    lease = await store.acquire()
    key = f"ai:session:{lease.session_id}"
    lock_key = f"{key}:lock"
    original = await redis_client.get(key)
    replacement = uuid4().hex

    async def race():
        if change == "replace_lock":
            await redis_client.set(lock_key, replacement, ex=240)
        else:
            await redis_client.pexpire(key if change == "expire_session" else lock_key, 0)

    _before_next_exec(monkeypatch, race)
    with pytest.raises(SessionExpired):
        await store.finish(lease, {"answer": "不能提交"})
    assert await redis_client.get(key) == (None if change == "expire_session" else original)
    if change == "replace_lock":
        assert await redis_client.get(lock_key) in (replacement, replacement.encode())


async def test_release_rechecks_token_on_exec_conflict(store, redis_client, monkeypatch):
    lease = await store.acquire()
    key = f"ai:session:{lease.session_id}"
    lock_key = f"{key}:lock"
    original = await redis_client.get(key)
    replacement = uuid4().hex

    async def replace_lock():
        await redis_client.set(lock_key, replacement, ex=240)

    _before_next_exec(monkeypatch, replace_lock)
    await store.release(lease)
    await store.release(lease)
    assert await redis_client.get(lock_key) in (replacement, replacement.encode())
    assert await redis_client.get(key) == original


@pytest.mark.parametrize("change", ["expire_session", "lock_session"])
async def test_acquire_rechecks_session_and_lock_on_exec_conflict(
    store, redis_client, monkeypatch, change
):
    lease = await store.acquire()
    await store.release(lease)
    key = f"ai:session:{lease.session_id}"
    lock_key = f"{key}:lock"
    replacement = uuid4().hex

    async def race():
        if change == "expire_session":
            await redis_client.pexpire(key, 0)
        else:
            await redis_client.set(lock_key, replacement, ex=240)

    _before_next_exec(monkeypatch, race)
    with pytest.raises(SessionNotFound if change == "expire_session" else SessionBusy):
        await store.acquire(lease.session_id)
    if change == "expire_session":
        assert not await redis_client.exists(key, lock_key)
    else:
        assert await redis_client.get(lock_key) in (replacement, replacement.encode())
        assert (await store.get(lease.session_id))["turns"] == []


@pytest.mark.parametrize("session_id", [
    "", "a" * 31, "a" * 33, "g" * 32, "a" * 32 + "\n", " " + "a" * 32,
    "a" * 32 + ":lock", "00000000-0000-4000-8000-000000000000", "ａ" * 32, 123,
])
async def test_invalid_ids_are_rejected_without_writing(store, redis_client, session_id):
    with pytest.raises(ValueError, match="32 位十六进制"):
        await store.acquire(session_id)
    with pytest.raises(ValueError, match="32 位十六进制"):
        await store.get(session_id)
    assert await redis_client.dbsize() == 0

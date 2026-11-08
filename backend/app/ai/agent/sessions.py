"""只读 Agent 的 Redis 会话与排他租约；调用方须在 finally 中 release。"""

import json
import re
import secrets
from dataclasses import dataclass
from uuid import uuid4

from redis.asyncio import Redis
from redis.exceptions import WatchError


class SessionNotFound(Exception):
    """显式指定的会话不存在或已经过期。"""


class SessionBusy(Exception):
    """同一会话已有正在使用的租约。"""


class SessionExpired(Exception):
    """租约失去锁，或租约对应的会话已经过期。"""


@dataclass
class SessionLease:
    session_id: str
    token: str
    turns: list[dict]


def _session_key(session_id: str) -> str:
    if not isinstance(session_id, str) or re.fullmatch(r"[0-9a-fA-F]{32}", session_id) is None:
        raise ValueError("session_id 必须是 32 位十六进制字符串")
    return f"ai:session:{session_id}"


def _owns_lock(owner: str | bytes | None, token: str) -> bool:
    # 同时兼容 decode_responses=True 和 redis.asyncio 默认的 bytes 响应。
    return owner == token or owner == token.encode("ascii")


class SessionStore:
    def __init__(self, redis: Redis, ttl_s: int = 86400, lock_s: int = 240, max_turns: int = 6):
        if ttl_s <= 0 or lock_s <= 0 or max_turns <= 0:
            raise ValueError("ttl_s、lock_s 和 max_turns 必须为正数")
        self._redis = redis
        self._ttl_s = ttl_s
        self._lock_s = lock_s
        self._max_turns = max_turns

    async def acquire(self, session_id: str | None = None) -> SessionLease:
        """仅无 ID 时新建会话；锁已被占用时立即报忙，不等待锁释放。"""
        creating = session_id is None
        if not creating:
            _session_key(session_id)
        token = secrets.token_hex(32)
        while True:
            candidate = uuid4().hex if creating else session_id
            key = _session_key(candidate)
            lock_key = f"{key}:lock"
            async with self._redis.pipeline() as pipe:
                try:
                    await pipe.watch(key, lock_key)
                    payload, owner = await pipe.mget(key, lock_key)
                    if creating:
                        # 极小概率的 ID 碰撞也不能覆盖既有会话或锁。
                        if payload is not None or owner is not None:
                            continue
                        session = {"session_id": candidate, "turns": []}
                    else:
                        if payload is None:
                            raise SessionNotFound("会话不存在或已过期")
                        if owner is not None:
                            raise SessionBusy("会话正在处理中")
                        session = json.loads(payload)

                    pipe.multi()
                    if creating:
                        pipe.set(key, json.dumps(session), ex=self._ttl_s)
                    pipe.set(lock_key, token, ex=self._lock_s)
                    await pipe.execute()
                    return SessionLease(candidate, token, session["turns"])
                except WatchError:
                    # 只重试乐观事务冲突；下一次读到锁就立即报忙。
                    continue

    async def get(self, session_id: str) -> dict:
        payload = await self._redis.get(_session_key(session_id))
        if payload is None:
            raise SessionNotFound("会话不存在或已过期")
        return json.loads(payload)

    async def finish(self, lease: SessionLease, result: dict) -> None:
        """追加完整 JSON 结果并续会话 TTL；锁由调用方 finally 中释放。"""
        key = _session_key(lease.session_id)
        lock_key = f"{key}:lock"
        while True:
            async with self._redis.pipeline() as pipe:
                try:
                    await pipe.watch(key, lock_key)
                    payload, owner = await pipe.mget(key, lock_key)
                    if payload is None or not _owns_lock(owner, lease.token):
                        raise SessionExpired("会话或租约已过期")
                    # 读取 Redis 中的最新结果，不以 lease 的历史快照覆盖并发写入。
                    session = json.loads(payload)
                    session["turns"] = (session["turns"] + [result])[-self._max_turns:]
                    encoded = json.dumps(session, ensure_ascii=False, allow_nan=False)
                    pipe.multi()
                    pipe.set(key, encoded, ex=self._ttl_s)
                    await pipe.execute()
                    return
                except WatchError:
                    continue

    async def add_followup(self, session_id: str, run_id: str, event: dict, *, alarm_id: int | None = None) -> bool:
        """把用户在某轮回答之后的人工操作（如确认派单）记到该轮上，追问时 Agent 能看到。

        不需要会话锁：只改已完成的轮次，和 finish 一样用 WATCH 乐观重试，不会覆盖并发写入。
        找不到该轮、或该轮没有给过这条告警的派单入口时不记录，返回 False。
        """
        key = _session_key(session_id)
        while True:
            async with self._redis.pipeline() as pipe:
                try:
                    await pipe.watch(key)
                    payload = await pipe.get(key)
                    if payload is None:
                        return False
                    session = json.loads(payload)
                    turn = next((t for t in session["turns"] if t.get("run_id") == run_id), None)
                    if turn is None or (alarm_id is not None and alarm_id not in
                                        {a.get("alarm_id") for a in turn.get("actions", [])}):
                        return False
                    turn.setdefault("followups", []).append(event)
                    ttl = await pipe.ttl(key)
                    pipe.multi()
                    pipe.set(key, json.dumps(session, ensure_ascii=False, allow_nan=False),
                             ex=ttl if ttl and ttl > 0 else self._ttl_s)
                    await pipe.execute()
                    return True
                except WatchError:
                    continue

    async def release(self, lease: SessionLease) -> None:
        """原子比较令牌后释放本租约的锁；过期租约和重复释放均无副作用。"""
        lock_key = f"{_session_key(lease.session_id)}:lock"
        while True:
            async with self._redis.pipeline() as pipe:
                try:
                    await pipe.watch(lock_key)
                    owner = await pipe.get(lock_key)
                    if not _owns_lock(owner, lease.token):
                        return
                    pipe.multi()
                    pipe.delete(lock_key)
                    await pipe.execute()
                    return
                except WatchError:
                    continue

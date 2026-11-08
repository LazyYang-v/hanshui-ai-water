"""连接池、Redis 和「演示时钟」。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated
from zoneinfo import ZoneInfo

from fastapi import Depends, Request
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis

LOCAL_TZ = ZoneInfo("Asia/Shanghai")
KEY_CLOCK = "sim:clock"
KEY_SPEED = "sim:speed"


async def get_conn(request: Request) -> AsyncIterator[AsyncConnection]:
    pool: AsyncConnectionPool = request.app.state.pool
    async with pool.connection() as conn:
        conn.row_factory = dict_row
        yield conn


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


Conn = Annotated[AsyncConnection, Depends(get_conn)]
Rds = Annotated[Redis, Depends(get_redis)]


async def sim_now(redis: Redis, conn: AsyncConnection) -> datetime:
    """系统的「现在」：优先取演示时钟（加速演示时领先墙上时间），否则取最新数据时间。"""
    raw = await redis.get(KEY_CLOCK)
    if raw:
        return datetime.fromisoformat(raw).astimezone(LOCAL_TZ)
    row = await (await conn.execute("SELECT max(ts) AS ts FROM plant_reading")).fetchone()
    if row and row["ts"]:
        return row["ts"].astimezone(LOCAL_TZ)
    return datetime.now(LOCAL_TZ)


async def get_now(conn: Conn, redis: Rds) -> datetime:
    return await sim_now(redis, conn)


Now = Annotated[datetime, Depends(get_now)]


async def get_now_released(request: Request) -> datetime:
    """同一演示时钟，但取完立即归还连接：随后要长时间调用模型的接口用它，避免占住连接池。"""
    async with request.app.state.pool.connection() as conn:
        conn.row_factory = dict_row
        return await sim_now(request.app.state.redis, conn)


ReleasedNow = Annotated[datetime, Depends(get_now_released)]

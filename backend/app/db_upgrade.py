"""启动时补齐表结构。

新库由 Postgres 容器按序执行 db/init/*.sql；已经初始化过的库不会再执行，
因此可重复执行的增量脚本（只用 IF NOT EXISTS 一类语句）在 API 启动时再跑一遍。
多个 API 实例同时启动时用事务级 advisory lock 串行执行。
"""

from __future__ import annotations

import logging
from pathlib import Path

from psycopg_pool import AsyncConnectionPool

log = logging.getLogger(__name__)

INIT_DIR = Path(__file__).resolve().parents[1] / "db" / "init"
UPGRADE_SCRIPTS = ("005_ops.sql", "006_vision.sql")
LOCK_KEY = 0x5348_5549   # 任意固定值，仅用于本项目的升级锁


async def upgrade(pool: AsyncConnectionPool) -> list[str]:
    applied = []
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_KEY,))
        for name in UPGRADE_SCRIPTS:
            await conn.execute((INIT_DIR / name).read_text(encoding="utf-8"))
            applied.append(name)
    log.info("表结构已检查：%s", ", ".join(applied))
    return applied

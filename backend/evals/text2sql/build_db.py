"""建评测库 shuiwu_eval：固定时钟 + 7 天历史（含暴雨、管网渗漏、加药堵塞、风机故障等事件）。

    uv run python -m evals.text2sql.build_db

评测库与线上库隔离，「现在」固定为 EVAL_NOW，所以金标 SQL 的结果是确定的，可复现。
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg

from simulator import runner

ADMIN_URL = os.getenv("EVAL_ADMIN_DATABASE_URL", "postgresql://shuiwu:shuiwu@localhost:15432/shuiwu")
EVAL_DB = "shuiwu_eval"
EVAL_NOW = datetime(2026, 11, 5, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai"))   # 周四中午
DAYS, STEP_S, SEED = 7, 60.0, 7
INIT_DIR = Path(__file__).resolve().parents[2] / "db" / "init"


def eval_url(user: str = "shuiwu", password: str = "shuiwu") -> str:
    base = ADMIN_URL.rsplit("/", 1)[0].split("@", 1)[1]
    return f"postgresql://{user}:{password}@{base}/{EVAL_DB}"


def readonly_url() -> str:
    return eval_url("text2sql_ro", os.getenv("TEXT2SQL_RO_PASSWORD", "readonly"))


def build() -> None:
    with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {EVAL_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {EVAL_DB}")
    with psycopg.connect(eval_url(), autocommit=True) as conn:
        for f in sorted(INIT_DIR.glob("*.sql")):
            conn.execute(f.read_text())
        sim, _ = runner.load_simulator(conn, seed=SEED)
        runner.backfill(conn, sim, EVAL_NOW, DAYS, STEP_S, SEED)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build()

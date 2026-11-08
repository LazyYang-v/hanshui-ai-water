"""API 集成测试夹具。

在 compose 的 PostgreSQL 里建独立的测试库 shuiwu_test，用模拟器生成确定性的 2 天数据；
Redis 用 fakeredis，避免与正在运行的模拟器共用 Pub/Sub 频道。数据库不可用时跳过 API 测试。
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import fakeredis
import psycopg
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from simulator import runner
from simulator.scenarios import Scenario

ADMIN_URL = os.getenv("TEST_ADMIN_DATABASE_URL", "postgresql://shuiwu:shuiwu@localhost:15432/shuiwu")
TEST_DB = "shuiwu_test"
INIT_DIR = Path(__file__).resolve().parents[1] / "db" / "init"
TZ = ZoneInfo("Asia/Shanghai")

# 测试数据的「现在」。此刻：P02 加药泵堵塞已 5 小时（总磷超标），P06 鼓风机已跳停 1.5 小时
END = datetime(2026, 11, 5, 12, 0, tzinfo=TZ)


def _test_url() -> str:
    return ADMIN_URL.rsplit("/", 1)[0] + "/" + TEST_DB


@pytest.fixture(scope="session")
def db_url() -> str:
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as e:
        pytest.skip(f"数据库不可用（先 docker compose up -d db）：{e}")
    with admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {TEST_DB}")

    url = _test_url()
    with psycopg.connect(url, autocommit=True) as conn:
        for f in sorted(INIT_DIR.glob("*.sql")):
            conn.execute(f.read_text())
        sim, _ = runner.load_simulator(conn, seed=42)
        sim.book.add(Scenario("dosing_block", END - timedelta(hours=5), 600, "P02"))
        sim.book.add(Scenario("blower_fault", END - timedelta(hours=2), 600, "P06"))
        writer = runner.Writer(conn)
        start, step = END - timedelta(days=2), 60
        for i in range(int((END - start).total_seconds() // step) + 1):
            writer.add(sim.step(start + timedelta(seconds=i * step), step))
        writer.flush()
        runner.refresh_aggregates(conn)
    return url


@pytest.fixture
def redis_server():
    return fakeredis.FakeServer()


@pytest.fixture
def sync_redis(redis_server):
    return fakeredis.FakeRedis(server=redis_server, decode_responses=True)


@pytest.fixture(scope="session")
def ro_url(db_url) -> str:
    """测试库的只读账号连接串（角色在集群级别，由 init 脚本创建并授权）。"""
    pwd = os.getenv("TEXT2SQL_RO_PASSWORD", "readonly")
    return db_url.replace("shuiwu:shuiwu@", f"text2sql_ro:{pwd}@", 1)


class FakeLLM:
    """假模型：按脚本返回内容，记录收到的消息。responder(messages) -> str。"""

    def __init__(self, responder):
        self.responder = responder
        self.calls: list[list[dict]] = []
        self.available = True

    async def chat(self, messages, **kw):
        from app.ai.gateway import ChatResult
        self.calls.append(messages)
        return ChatResult(self.responder(messages), "fake", "fake-model", 100, 20, latency_ms=1.0)

    def summary(self):
        return {"mode": "fake", "calls": len(self.calls)}

    async def aclose(self):
        pass


@pytest.fixture
def fake_llm():
    """默认回答：查厂站数量。测试里可以替换 fake_llm.responder。"""
    return FakeLLM(lambda m: '{"sql": "SELECT count(*) AS n FROM plant", "explanation": "数厂站"}')


ADMIN_PASSWORD = "test-admin-pw"


@pytest.fixture
def client(db_url, ro_url, redis_server, sync_redis, fake_llm):
    sync_redis.set("sim:clock", END.isoformat())
    # 后台巡检关闭：测试按需调用 /api/alarms/scan，避免后台任务与断言并发改告警表
    app = create_app(Settings(database_url=db_url, readonly_database_url=ro_url, rag_dense_enabled=False,
                              alarm_monitor_enabled=False, admin_password=ADMIN_PASSWORD),
                     redis=fakeredis.FakeAsyncRedis(server=redis_server, decode_responses=True),
                     gateway=fake_llm)
    with TestClient(app) as c:
        # 绝大多数测试以管理员身份调用写接口；未登录的行为见 test_auth.py
        assert c.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD}).status_code == 200
        yield c

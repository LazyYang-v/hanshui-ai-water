"""汉水智控 API。

    uv run uvicorn app.main:app --reload
文档：http://localhost:8000/docs
"""

import asyncio
import contextlib
import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis

from app.ai.advisor import AlarmAdvisor
from app.ai.agent import Agent
from app.ai.agent.sessions import SessionStore
from app.ai.gateway import LLMGateway
from app.ai.rag.retrieval import KnowledgeIndex, load_index
from app.ai.rag.service import RAG
from app.ai.reporter import Reporter
from app.ai.text2sql import Text2SQL
from app.ai.text2sql.schema import load_schema
from app.analytics.alarms import AlarmMonitor
from app.analytics.statistics import PlantStatistics
from app.api import (aeration, ai, alarms, chat, demo, devices, knowledge, meta, overview, plants, realtime, reports,
                     vision, work_orders)
from app.api import auth as auth_api
from app.auth import AdminOnlyWrites
from app.config import Settings, settings as default_settings
from app.db_upgrade import upgrade
from app.vision.service import VisionService

log = logging.getLogger(__name__)


def _setup_logging() -> None:
    """uvicorn 只配置它自己的 logger，app.* 的 INFO（升级、补扫耗时）默认不输出；重复调用不重复添加。"""
    logger = logging.getLogger("app")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _gateway_from_env() -> LLMGateway | None:
    try:
        return LLMGateway.from_env()
    except ValueError as e:
        log.warning("LLM 网关未启用：%s", e)
        return None


def create_app(settings: Settings = default_settings, redis: Redis | None = None,
               gateway: LLMGateway | None = None, rag_index: KnowledgeIndex | None = None) -> FastAPI:
    """依赖可注入，测试使用 fakeredis、假模型和轻量索引，不发起外部模型请求。"""
    _setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            app.state.pool = AsyncConnectionPool(settings.database_url, min_size=1, max_size=10, open=False,
                                                 kwargs={"autocommit": True})
            stack.push_async_callback(app.state.pool.close)
            await app.state.pool.open(wait=True, timeout=10)
            try:
                await upgrade(app.state.pool)
            except Exception as e:  # 升级失败时只影响告警/工单/报告，其余接口照常
                log.error("表结构升级失败：%s", e)
            app.state.redis = redis or Redis.from_url(settings.redis_url, decode_responses=True)
            stack.push_async_callback(app.state.redis.aclose)

            app.state.gateway = gateway or _gateway_from_env()
            if app.state.gateway is not None and gateway is None:
                stack.push_async_callback(app.state.gateway.aclose)
            app.state.ro_pool = AsyncConnectionPool(settings.readonly_database_url, min_size=1, max_size=5,
                                                    open=False, kwargs={"autocommit": True})
            stack.push_async_callback(app.state.ro_pool.close)
            app.state.text2sql = None
            app.state.statistics = None
            try:
                await app.state.ro_pool.open(wait=True, timeout=10)
                app.state.statistics = PlantStatistics(app.state.ro_pool)
                async with app.state.ro_pool.connection() as conn:
                    schema = await load_schema(conn)
                if app.state.gateway is not None:
                    app.state.text2sql = Text2SQL(app.state.gateway, app.state.ro_pool, schema)
            except Exception as e:  # 只读库不可用时其余接口仍可工作
                log.warning("Text2SQL 未启用：%s", e)

            app.state.rag = None
            try:
                index = rag_index or await asyncio.to_thread(
                    load_index, settings.rag_corpus_dir, settings.rag_model_cache, settings.rag_vector_cache,
                    dense=settings.rag_dense_enabled)
                app.state.rag = RAG(index, app.state.gateway)
            except Exception as e:
                log.warning("知识库未启用：%s", e)
            app.state.agent_sessions = SessionStore(app.state.redis)
            app.state.agent = (Agent(app.state.gateway, app.state.text2sql, app.state.rag,
                                     statistics=app.state.statistics)
                               if app.state.gateway is not None else None)
            app.state.advisor = AlarmAdvisor(app.state.agent, app.state.rag) if app.state.agent else None
            app.state.reporter = Reporter(app.state.gateway)
            app.state.vision = None
            if settings.vision_enabled:
                try:
                    app.state.vision = await asyncio.to_thread(
                        VisionService.load, settings.vision_model_path, concurrency=settings.vision_concurrency,
                        threads=settings.vision_threads, samples_dir=settings.vision_samples_dir)
                except Exception as e:  # 模型缺失只影响视觉巡检接口
                    log.warning("视觉巡检未启用：%s", e)

            # 巡检器始终创建（手动扫描接口要用），后台循环按配置启动；先于连接池关闭而停止
            app.state.alarm_monitor = AlarmMonitor(app.state.pool, app.state.redis,
                                                   interval_s=settings.alarm_scan_interval_s)
            if settings.alarm_monitor_enabled:
                stack.push_async_callback(_cancel, asyncio.create_task(app.state.alarm_monitor.run(),
                                                                       name="alarm-monitor"))
            yield

    app = FastAPI(
        title="汉水智控 API",
        description="AI+水务智慧运营平台（Demo）。所有运行数据由模拟器生成。时间均以演示时钟为准。",
        version="0.4.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    if not settings.admin_password:
        log.warning("未配置 ADMIN_PASSWORD：管理员无法登录，所有写入与模型调用接口将返回 401")
    # 先加的在内层：CORS 在最外层，401 响应也带 CORS 头
    app.add_middleware(AdminOnlyWrites)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["*"], allow_headers=["*"], allow_credentials=True)
    for module in (auth_api, meta, overview, plants, devices, demo, realtime, ai, knowledge, chat, alarms, work_orders,
                   reports, vision, aeration):
        app.include_router(module.router)
    return app


app = create_app()

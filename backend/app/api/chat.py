"""Agent JSON/SSE 接口；只接受用户文本与服务端签发的会话 id。"""

import asyncio
import json
import logging

import anyio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, Field
from redis.exceptions import RedisError

from app.ai.agent.service import Agent
from app.ai.agent.sessions import (
    SessionBusy,
    SessionExpired,
    SessionLease,
    SessionNotFound,
    SessionStore,
)
from app.deps import ReleasedNow
from app.services.chat_actions import suggest_actions

router = APIRouter(prefix="/api/ai", tags=["Agent 对话"])
log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=2, max_length=500)
    session_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


ChatNow = ReleasedNow   # 在长时间模型/SSE调用前归还数据库连接


def services(request: Request) -> tuple[Agent, SessionStore]:
    agent = request.app.state.agent
    if agent is None or not agent.gateway.available:
        raise HTTPException(503, "Agent 未就绪，请配置 DeepSeek Flash")
    return agent, request.app.state.agent_sessions


async def acquire(store: SessionStore, session_id: str | None) -> SessionLease:
    try:
        return await store.acquire(session_id)
    except SessionNotFound as exc:
        raise HTTPException(404, "会话不存在或已过期，请不带session_id新建会话") from exc
    except SessionBusy as exc:
        raise HTTPException(409, "同一会话正在回答，请等待完成") from exc
    except RedisError as exc:
        raise HTTPException(503, "会话存储暂时不可用") from exc


async def actions_for(pool, body: dict, now) -> list[dict]:
    """派单入口由服务端按本轮查询范围查告警表得到，不经模型；失败只是不显示入口。"""
    if pool is None:
        return []
    try:
        async with pool.connection() as conn:
            conn.row_factory = dict_row
            return await suggest_actions(conn, body, now)
    except Exception as exc:  # noqa: BLE001 —— 入口是附加信息，不能让已完成的回答失败
        log.warning("派单入口查询失败：%s", exc)
        return []


async def execute(agent, store, lease, question, now, *, pool, **kwargs) -> dict:
    result = await agent.run(question, now, history=lease.turns, session_id=lease.session_id, **kwargs)
    body = result.to_dict()
    body["actions"] = await actions_for(pool, body, now)
    await store.finish(lease, body)
    return body


async def release(store, lease):
    # Starlette断开连接时取消流，释放必须位于受保护清理区。
    with anyio.CancelScope(shield=True):
        try:
            await store.release(lease)
        except RedisError:
            pass  # Redis断开时租约到期后自动释放


class SessionStream(StreamingResponse):
    """响应头尚未送出时断开，也须释放在路由里取得的会话锁。"""

    def __init__(self, events, store, lease):
        super().__init__(events, media_type="text/event-stream",
                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        self.store, self.lease = store, lease

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                finally:
                    await release(self.store, self.lease)


@router.post("/chat", summary="Agent 连续对话：只读查数、检索、证据校验")
async def chat(body: ChatRequest, request: Request, now: ChatNow) -> dict:
    agent, store = services(request)
    lease = await acquire(store, body.session_id)
    try:
        return await execute(agent, store, lease, body.question, now, pool=request.app.state.pool)
    except SessionExpired as exc:
        raise HTTPException(409, "会话租约已失效，本轮未保存") from exc
    except RedisError as exc:
        raise HTTPException(503, "会话保存失败，本轮未确认完成") from exc
    finally:
        await release(store, lease)


@router.post("/chat/stream", summary="SSE：规划进度、工具调用和校验后的完整回答")
async def stream(body: ChatRequest, request: Request, now: ChatNow):
    agent, store = services(request)
    lease = await acquire(store, body.session_id)  # 忙/过期在发送200前返回HTTP错误

    async def events():
        queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue(maxsize=32)

        async def emit(event, data):
            await queue.put((event, data))

        async def produce():
            try:
                result = await execute(agent, store, lease, body.question, now, emit=emit,
                                       pool=request.app.state.pool)
                await emit("done", result)
            except SessionExpired:
                await emit("error", {"session_id": lease.session_id, "error": "会话租约失效，本轮未保存"})
            except RedisError:
                await emit("error", {"session_id": lease.session_id, "error": "会话保存失败，本轮未确认完成"})
            except Exception:
                await emit("error", {"session_id": lease.session_id, "error": "对话执行失败"})

        task = asyncio.create_task(produce())
        try:
            while True:
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                if event in {"done", "error"}:
                    break
        finally:
            task.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(task, return_exceptions=True)

    return SessionStream(events(), store, lease)


@router.get("/chat/sessions/{session_id}", summary="查看最近6轮已完成对话、SQL与引用")
async def session(session_id: str, request: Request) -> dict:
    try:
        return await request.app.state.agent_sessions.get(session_id)
    except (SessionNotFound, ValueError) as exc:
        raise HTTPException(404, "会话不存在或已过期") from exc
    except RedisError as exc:
        raise HTTPException(503, "会话存储暂时不可用") from exc

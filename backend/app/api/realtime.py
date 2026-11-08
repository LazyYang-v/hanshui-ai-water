"""WebSocket 实时推送：连接后先发最新一帧，之后转发模拟器的每一帧和告警变化。

模拟器帧没有 type 字段（含 ts、plants、devices）；告警消息为 {"type": "alarm", "event": new/escalated/recovered/backfill, ...}。
"""

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.analytics.alarms import CH_ALARM

router = APIRouter(tags=["实时"])
log = logging.getLogger(__name__)

CH_TICK, KEY_LATEST = "realtime:tick", "realtime:latest"


@router.websocket("/ws/realtime")
async def realtime(ws: WebSocket):
    await ws.accept()
    redis = ws.app.state.redis
    async with redis.pubsub(ignore_subscribe_messages=True) as ps:
        await ps.subscribe(CH_TICK, CH_ALARM)
        if latest := await redis.get(KEY_LATEST):
            await ws.send_text(latest)

        async def forward():
            async for msg in ps.listen():
                if msg["type"] == "message":
                    await ws.send_text(msg["data"])

        async def watch_client():
            # 客户端不需要发消息；读到断开即结束
            while True:
                await ws.receive_text()

        tasks = [asyncio.create_task(forward()), asyncio.create_task(watch_client())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if (e := t.exception()) and not isinstance(e, WebSocketDisconnect):
                    log.warning("实时推送中断：%s", e)
        finally:
            for t in tasks:
                t.cancel()

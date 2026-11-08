"""演示控制台：通过 Redis 向模拟器发命令，并等待带相同 id 的回复。"""

import asyncio
import json
import uuid

from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.deps import Rds
from app.schemas import CommandResult, ScenarioIn, SpeedIn

router = APIRouter(prefix="/api/demo", tags=["演示控制"])

CH_CONTROL, CH_EVENTS, KEY_SCENARIOS = "sim:control", "sim:events", "sim:scenarios"


async def send(redis, cmd: dict) -> str:
    cmd = {**cmd, "id": uuid.uuid4().hex}
    async with redis.pubsub(ignore_subscribe_messages=True) as ps:
        await ps.subscribe(CH_EVENTS)
        if await redis.publish(CH_CONTROL, json.dumps(cmd, ensure_ascii=False)) == 0:
            raise HTTPException(503, "模拟器未运行")
        try:
            async with asyncio.timeout(settings.sim_reply_timeout_s):
                async for msg in ps.listen():
                    if msg["type"] != "message":
                        continue
                    reply = json.loads(msg["data"])
                    if reply.get("id") == cmd["id"]:
                        if not reply["ok"]:
                            raise HTTPException(400, reply["message"])
                        return reply["message"]
        except TimeoutError:
            raise HTTPException(504, "模拟器未及时响应") from None
    raise HTTPException(504, "模拟器未及时响应")


@router.get("/scenarios", summary="正在进行的场景")
async def active_scenarios(redis: Rds) -> list[dict]:
    return json.loads(await redis.get(KEY_SCENARIOS) or "[]")


@router.post("/scenarios", response_model=CommandResult, summary="注入场景")
async def inject(body: ScenarioIn, redis: Rds):
    plant = body.plant.upper() if body.plant else None
    msg = await send(redis, {"action": "inject", "kind": body.kind, "plant": plant,
                             "duration_min": body.duration_min, "params": body.params})
    return {"message": msg}


@router.delete("/scenarios", response_model=CommandResult, summary="结束场景（不带参数则结束全部）")
async def clear(redis: Rds, plant: str | None = Query(None), kind: str | None = Query(None)):
    return {"message": await send(redis, {"action": "clear", "plant": plant.upper() if plant else None, "kind": kind})}


@router.put("/speed", response_model=CommandResult, summary="设置演示时钟倍速")
async def speed(body: SpeedIn, redis: Rds):
    return {"message": await send(redis, {"action": "speed", "value": body.value})}

"""工单：确认草稿后写库，状态流转 待处理 → 处理中 → 已关闭。

草稿由 POST /api/alarms/{id}/work-order-draft 生成并存在 Redis（30 分钟），确认时一次性取出，
重复提交同一草稿会得到 404；同一告警同一时间只能有一张未关闭的工单（数据库部分唯一索引兜底）。
加锁顺序统一为先告警行、后工单行，避免与派单、巡检并发时死锁。
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from psycopg import sql
from psycopg.errors import UniqueViolation
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field
from redis.exceptions import RedisError

from app.api.alarms import DRAFT_KEY, ChatRef
from app.deps import Conn, Now, Rds
from app.services.ops import WO_OPEN, WORK_ORDER_SELECT, get_work_order, work_order_out

router = APIRouter(prefix="/api/work-orders", tags=["告警与工单"])

STATUSES = ("待处理", "处理中", "已关闭")
Step = Annotated[str, Field(min_length=1, max_length=300)]


class WorkOrderCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    draft_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    assignee: str = Field(min_length=1, max_length=32, description="负责人")
    title: str | None = Field(None, min_length=2, max_length=100, description="不填则用草稿标题")
    steps: list[Step] | None = Field(None, min_length=1, max_length=12, description="不填则用草稿步骤")
    chat: ChatRef | None = Field(None, description="从 AI 对话派单时的会话与轮次")


class CloseWorkOrder(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    resolution: str = Field(min_length=2, max_length=500, description="处理结果")


def _dumps(o) -> str:
    return json.dumps(o, ensure_ascii=False, default=str)


@router.post("", status_code=201, summary="确认工单草稿并写入")
async def create_work_order(body: WorkOrderCreate, request: Request, conn: Conn, redis: Rds, now: Now) -> dict:
    raw = await redis.getdel(DRAFT_KEY.format(body.draft_id))
    if raw is None:
        raise HTTPException(404, "草稿不存在或已过期（有效期 30 分钟），请重新生成")
    draft = json.loads(raw)
    steps = draft["steps"]
    if body.steps is not None:   # 未改动的步骤保留来源与引用，新写的标为 user
        by_text = {s["text"]: s for s in draft["steps"]}
        steps = [by_text.get(t) or {"text": t, "source": "user", "citations": []} for t in body.steps]
    title = body.title or draft["title"]
    source = {**draft["source"], "draft_id": body.draft_id, "draft_created_at": draft["created_at"],
              "edited": {"title": title != draft["title"], "steps": [s["text"] for s in steps] !=
                         [s["text"] for s in draft["steps"]]}}
    if body.chat:
        source["via"] = {"channel": "chat", **body.chat.model_dump()}
    try:
        async with conn.transaction():
            alarm = await (await conn.execute("SELECT id, status FROM alarm WHERE id = %s FOR UPDATE",
                                              (draft["alarm_id"],))).fetchone()
            if alarm is None:
                raise HTTPException(404, f"告警 {draft['alarm_id']} 不存在")
            if alarm["status"] == "已关闭":
                raise HTTPException(409, "告警已关闭，不能再派单")
            row = await (await conn.execute("""
                INSERT INTO work_order (alarm_id, plant_id, title, assignee, status, steps, created_at, updated_at, source)
                VALUES (%s, %s, %s, %s, '待处理', %s, %s, %s, %s) RETURNING id""",
                (alarm["id"], draft["plant_id"], title, body.assignee, Jsonb(steps, _dumps), now, now,
                 Jsonb(source, _dumps)))).fetchone()
            await conn.execute("UPDATE alarm SET status = '处理中', updated_at = %s WHERE id = %s AND status = '未处理'",
                               (now, alarm["id"]))
    except UniqueViolation:
        raise HTTPException(409, "该告警已有未关闭的工单") from None
    out = work_order_out(await get_work_order(conn, row["id"]))
    if body.chat:
        out["chat_recorded"] = await _record_in_chat(request, body.chat, out, draft["alarm_id"], now)
    return out


async def _record_in_chat(request: Request, chat: ChatRef, wo: dict, alarm_id: int, now) -> bool:
    """工单已提交，记回会话失败（会话过期、Redis 抖动）不影响派单结果。"""
    text = f"用户已人工确认派单：工单 #{wo['id']}（告警 #{alarm_id}，负责人 {wo['assignee']}）"
    try:
        return await request.app.state.agent_sessions.add_followup(
            chat.session_id, chat.run_id, {"type": "work_order", "work_order_id": wo["id"], "alarm_id": alarm_id,
                                           "at": now.isoformat(), "text": text}, alarm_id=alarm_id)
    except (RedisError, ValueError):
        return False


@router.get("", summary="工单列表（附各状态数量，供看板使用）")
async def list_work_orders(
    conn: Conn,
    status: Literal["待处理", "处理中", "已关闭"] | None = None,
    plant: str | None = Query(None, description="厂站编码"),
    assignee: str | None = None,
    alarm_id: int | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    conds, params = [sql.SQL("TRUE")], {}
    if plant:
        conds.append(sql.SQL("p.code = %(plant)s"))
        params["plant"] = plant.upper()
    if assignee:
        conds.append(sql.SQL("w.assignee = %(assignee)s"))
        params["assignee"] = assignee
    if alarm_id is not None:
        conds.append(sql.SQL("w.alarm_id = %(alarm_id)s"))
        params["alarm_id"] = alarm_id
    base = sql.SQL(" AND ").join(conds)
    groups = await (await conn.execute(sql.SQL("""
        SELECT w.status, count(*) AS n FROM work_order w LEFT JOIN plant p ON p.id = w.plant_id
        WHERE {base} GROUP BY 1""").format(base=base), params)).fetchall()
    counts = dict.fromkeys(STATUSES, 0) | {g["status"]: g["n"] for g in groups}
    where = sql.SQL("{} AND w.status = %(status)s").format(base) if status else base
    rows = await (await conn.execute(sql.SQL(WORK_ORDER_SELECT + """
        WHERE {where} ORDER BY w.id DESC LIMIT %(limit)s OFFSET %(offset)s""").format(where=where),
        {**params, "status": status, "limit": limit, "offset": offset})).fetchall()
    return {"total": counts[status] if status else sum(counts.values()), "limit": limit, "offset": offset,
            "counts": counts, "items": [work_order_out(r) for r in rows]}


@router.get("/{wo_id}", summary="工单详情")
async def work_order_detail(wo_id: int, conn: Conn) -> dict:
    row = await get_work_order(conn, wo_id)
    if row is None:
        raise HTTPException(404, f"工单 {wo_id} 不存在")
    return work_order_out(row)


async def _conflict(conn, wo_id: int, action: str) -> HTTPException:
    row = await (await conn.execute("SELECT status FROM work_order WHERE id = %s", (wo_id,))).fetchone()
    if row is None:
        return HTTPException(404, f"工单 {wo_id} 不存在")
    return HTTPException(409, f"工单当前为{row['status']}，不能{action}")


@router.post("/{wo_id}/start", summary="开始处理")
async def start_work_order(wo_id: int, conn: Conn, now: Now) -> dict:
    row = await (await conn.execute("""
        UPDATE work_order SET status = '处理中', started_at = %s, updated_at = %s
        WHERE id = %s AND status = '待处理' RETURNING id""", (now, now, wo_id))).fetchone()
    if row is None:
        raise await _conflict(conn, wo_id, "开始处理")
    return work_order_out(await get_work_order(conn, wo_id))


@router.post("/{wo_id}/close", summary="关闭工单")
async def close_work_order(wo_id: int, body: CloseWorkOrder, conn: Conn, now: Now) -> dict:
    """关联告警的数据已恢复时一并关闭告警；未恢复时告警保持处理中，关闭工单不等于异常消除。"""
    alarm_closed, notice = False, None
    async with conn.transaction():
        wo = await (await conn.execute("SELECT alarm_id FROM work_order WHERE id = %s", (wo_id,))).fetchone()
        if wo is None:
            raise HTTPException(404, f"工单 {wo_id} 不存在")
        alarm = None
        if wo["alarm_id"] is not None:
            alarm = await (await conn.execute("SELECT id, status, ended_at FROM alarm WHERE id = %s FOR UPDATE",
                                              (wo["alarm_id"],))).fetchone()
        row = await (await conn.execute("""
            UPDATE work_order SET status = '已关闭', closed_at = %s, updated_at = %s, resolution = %s
            WHERE id = %s AND status = ANY(%s) RETURNING id""",
            (now, now, body.resolution, wo_id, list(WO_OPEN)))).fetchone()
        if row is None:
            raise await _conflict(conn, wo_id, "关闭")
        if alarm is not None and alarm["status"] != "已关闭":
            if alarm["ended_at"] is not None:
                await conn.execute("""
                    UPDATE alarm SET status = '已关闭', closed_at = %s, close_note = %s, updated_at = %s
                    WHERE id = %s""", (now, f"随工单 #{wo_id} 关闭：{body.resolution}"[:300], now, alarm["id"]))
                alarm_closed = True
            else:
                notice = f"告警 #{alarm['id']} 的数据尚未恢复，告警保持{alarm['status']}；恢复后可在告警页关闭"
    return {**work_order_out(await get_work_order(conn, wo_id)), "alarm_closed": alarm_closed, "notice": notice}

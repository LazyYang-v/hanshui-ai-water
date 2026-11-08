"""告警：列表与详情、巡检状态与手动扫描、AI 处置建议、工单草稿、关闭。

告警由巡检（app/analytics/alarms.py）按规则写入，本模块不新建告警。
ended_at 表示数据恢复；status 是人工处理状态（未处理 / 处理中 / 已关闭），两者相互独立。
工单草稿只存在 Redis 里，用户在 /api/work-orders 确认后才写库。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field
from redis.exceptions import RedisError

from app.ai.agent.sessions import SessionNotFound
from app.analytics.rules import BUCKET, RULES
from app.api.common import aware
from app.deps import Conn, Now, Rds, sim_now
from app.vision.alarms import KIND as VISION_KIND, VISION_RULE_IDS
from app.services.ops import (ALARM_SELECT, WO_OPEN, WORK_ORDER_SELECT, alarm_out, get_alarm, metric_meta,
                              work_order_out)

router = APIRouter(prefix="/api/alarms", tags=["告警与工单"])

LEVELS = ("告警", "预警", "提示")
STATUSES = ("未处理", "处理中", "已关闭")
KINDS = tuple(dict.fromkeys([*(r.kind for r in RULES.values()), VISION_KIND]))
RELATED_SPAN = timedelta(hours=6)
DRAFT_KEY = "wo:draft:{}"
DRAFT_TTL_S = 1800
ADVICE_LOCK = "alarm:advice:lock:{}"
ADVICE_LOCK_S = 240          # 长于 Agent 单轮超时（180 秒），生成期间锁不会先过期

# 没有可用的 AI 核查建议时的默认步骤（明确标注为模板，不是 AI 建议）
DEFAULT_STEPS = {
    "水质超标": ["复核{metric}在线仪表读数，必要时取样化验确认", "检查与{metric}相关的工艺单元（曝气、加药、回流）运行状态"],
    "水质预警": ["复核{metric}在线仪表读数与变化趋势", "检查与{metric}相关的工艺单元运行状态"],
    "工艺异常": ["检查鼓风机与曝气管路运行状态", "复核溶解氧仪表读数"],
    "进水冲击": ["核实进水流量与降雨情况", "检查提升泵与调节池运行状态"],
    "进水负荷": ["复核进水流量与对应污染物浓度，按同采样流量×浓度÷1000核算kg/h负荷",
                 "对照进出水、DO与曝气风量变化，区分浓度冲击与雨天稀释",
                 "必要时人工取样并核查上游来源，由有权限人员按规程处置"],
    "药耗异常": ["核对加药计量与药剂库存消耗是否一致", "检查加药泵与管路有无渗漏、堵塞"],
    "设备异常": ["现场检查{device}运行状态、振动与温度", "确认同类备用设备状态"],
    VISION_KIND: ["调取对应摄像头实时画面，复核{metric}的位置与范围（模型可能误检）",
                  "现场确认后处置：漂浮物安排打捞并检查格栅、撇渣设施；渗漏先隔离、标记渗漏点并报修，处置前后拍照留档"],
}
CLOSING_STEP = "处置后跟踪{metric}恢复情况，并在工单中记录处理结果"


class CloseAlarm(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    note: str = Field(min_length=2, max_length=300, description="关闭说明，如处理结果、误报原因")


def _where(plant, status, level, kind, active, start, end) -> tuple[sql.Composable, dict]:
    conds, params = [sql.SQL("TRUE")], {}
    if plant:
        conds.append(sql.SQL("p.code = %(plant)s"))
        params["plant"] = plant.upper()
    if status == "未关闭":
        conds.append(sql.SQL("a.status <> '已关闭'"))
    elif status:
        conds.append(sql.SQL("a.status = %(status)s"))
        params["status"] = status
    if level:
        conds.append(sql.SQL("a.level = %(level)s"))
        params["level"] = level
    if kind:
        conds.append(sql.SQL("a.kind = %(kind)s"))
        params["kind"] = kind
    if active is not None:
        conds.append(sql.SQL("a.ended_at IS NULL" if active else "a.ended_at IS NOT NULL"))
    if end:   # 事件与 [start, end) 相交
        conds.append(sql.SQL("a.ts < %(end)s"))
        params["end"] = end
    if start:
        conds.append(sql.SQL("(a.ended_at IS NULL OR a.ended_at >= %(start)s)"))
        params["start"] = start
    return sql.SQL(" AND ").join(conds), params


@router.get("", summary="告警列表")
async def list_alarms(
    conn: Conn, now: Now,
    plant: str | None = Query(None, description="厂站编码，如 P06"),
    status: Literal["未处理", "处理中", "已关闭", "未关闭"] | None = Query(None, description="处理状态"),
    level: Literal["告警", "预警", "提示"] | None = None,
    kind: str | None = Query(None, description=" / ".join(KINDS)),
    active: bool | None = Query(None, description="true：数据仍异常；false：数据已恢复"),
    start: datetime | None = Query(None, description="与 [start, end) 相交的告警；不带时区按北京时间"),
    end: datetime | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    if kind and kind not in KINDS:
        raise HTTPException(422, f"kind 须为 {' / '.join(KINDS)}")
    start, end = (aware(start) if start else None), (aware(end) if end else None)
    if start and end and start >= end:
        raise HTTPException(422, "start 必须早于 end")
    where, params = _where(plant, status, level, kind, active, start, end)
    rows = await (await conn.execute(sql.SQL(ALARM_SELECT + """
        WHERE {where} ORDER BY a.ts DESC, a.id DESC LIMIT %(limit)s OFFSET %(offset)s""").format(where=where),
        {**params, "limit": limit, "offset": offset})).fetchall()
    groups = await (await conn.execute(sql.SQL("""
        SELECT a.level, a.status, a.ended_at IS NULL AS active, count(*) AS n
        FROM alarm a JOIN plant p ON p.id = a.plant_id WHERE {where} GROUP BY 1, 2, 3""").format(where=where),
        params)).fetchall()
    summary = {"total": 0, "active": 0, "by_level": dict.fromkeys(LEVELS, 0), "by_status": dict.fromkeys(STATUSES, 0)}
    for g in groups:
        summary["total"] += g["n"]
        summary["active"] += g["n"] if g["active"] else 0
        summary["by_level"][g["level"]] += g["n"]
        summary["by_status"][g["status"]] += g["n"]
    return {"total": summary["total"], "limit": limit, "offset": offset, "summary": summary,
            "items": [alarm_out(r, now) for r in rows]}


@router.get("/monitor", summary="巡检状态与规则")
async def monitor_status(request: Request) -> dict:
    return {**request.app.state.alarm_monitor.status(), "bucket_min": BUCKET.total_seconds() / 60,
            "rules": [{"id": r.id, "kind": r.kind, "level": r.level, "on_buckets": r.on, "off_buckets": r.off}
                      for r in RULES.values()]}


@router.post("/scan", summary="立即扫描最近若干小时")
async def scan_now(request: Request, hours: float = Query(3, gt=0, le=168)) -> dict:
    """超过 3 小时视为补扫：变化只汇总推送一条 backfill，避免前端把旧事件当新告警弹出。"""
    stats = await request.app.state.alarm_monitor.scan_once(timedelta(hours=hours), quiet=hours > 3)
    if stats is None:
        raise HTTPException(409, "巡检正在进行，请稍后重试")
    return stats.__dict__


@router.get("/{alarm_id}", summary="告警详情（含前后关联告警、工单与最近一次处置建议）")
async def alarm_detail(alarm_id: int, conn: Conn, now: Now) -> dict:
    row = await get_alarm(conn, alarm_id)
    if row is None:
        raise HTTPException(404, f"告警 {alarm_id} 不存在")
    related = await (await conn.execute(ALARM_SELECT + """
        WHERE a.plant_id = %(pid)s AND a.id <> %(id)s AND a.ts >= %(lo)s AND a.ts <= %(hi)s
        ORDER BY a.ts LIMIT 20""", {"pid": row["plant_id"], "id": alarm_id, "lo": row["ts"] - RELATED_SPAN,
                                    "hi": row["ts"] + RELATED_SPAN})).fetchall()
    orders = await (await conn.execute(WORK_ORDER_SELECT + " WHERE w.alarm_id = %s ORDER BY w.id",
                                       (alarm_id,))).fetchall()
    return {**alarm_out(row, now, advice=True), "related": [alarm_out(r, now) for r in related],
            "work_orders": [work_order_out(o) for o in orders]}


@router.post("/{alarm_id}/advice", summary="生成 AI 处置建议（固定证据查询 + 知识检索 + 引用校验）")
async def alarm_advice(alarm_id: int, request: Request,
                       refresh: bool = Query(False, description="true：忽略已保存的建议重新生成")) -> dict:
    """证据由服务端按告警类型查询，结论必须逐条引用证据表单元格或知识原文。

    只保存通过校验的建议；拒答照常返回但不保存，模型失败返回 502，都不覆盖已有建议。
    查证据与写结果各用一次连接，模型生成期间不占用连接池。
    """
    advisor = request.app.state.advisor
    if advisor is None or not advisor.agent.gateway.available:
        raise HTTPException(503, "处置建议未就绪，请配置 DeepSeek Flash")
    pool, redis = request.app.state.pool, request.app.state.redis
    async with pool.connection() as conn:
        conn.row_factory = dict_row
        alarm = await get_alarm(conn, alarm_id)
        if alarm is None:
            raise HTTPException(404, f"告警 {alarm_id} 不存在")
        if alarm["rule"] in VISION_RULE_IDS:
            raise HTTPException(422, "视觉巡检告警暂不生成 AI 处置建议（知识库没有对应规程），可直接生成工单草稿")
        if alarm["advice"] and not refresh:
            return {"alarm_id": alarm_id, "cached": True, "saved": True, "advice": alarm["advice"]}
        now = await sim_now(redis, conn)

    lock, token = ADVICE_LOCK.format(alarm_id), uuid.uuid4().hex
    if not await redis.set(lock, token, nx=True, ex=ADVICE_LOCK_S):
        raise HTTPException(409, "该告警的处置建议正在生成，请稍后查看")
    try:
        async with pool.connection() as conn:
            conn.row_factory = dict_row
            records = await advisor.evidence(conn, alarm, now)
        advice = await advisor.advise(alarm, now, records)
        if advice["error"]:
            raise HTTPException(502, f"处置建议生成失败：{advice['error']}")
        if advice["ok"]:
            async with pool.connection() as conn:
                await conn.execute("UPDATE alarm SET advice = %s, ai_suggestion = %s WHERE id = %s",
                                   (Jsonb(advice, lambda o: json.dumps(o, ensure_ascii=False, default=str)),
                                    advice["answer"], alarm_id))
        return {"alarm_id": alarm_id, "cached": False, "saved": advice["ok"], "advice": advice}
    finally:
        if await redis.get(lock) == token:
            await redis.delete(lock)


class ChatRef(BaseModel):
    """从 AI 对话里派单时带上：草稿优先用该轮回答的核查建议，工单来源记下会话与轮次，派单结果也记回会话。"""
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")


class DraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chat: ChatRef | None = Field(None, description="从 AI 对话生成草稿时的会话与轮次")


def _brief(c: dict) -> dict:
    if c["kind"] == "knowledge":
        return {"kind": "knowledge", "title": c.get("title"), "heading": c.get("heading"), "quote": c.get("quote")}
    if c["kind"] == "statistic":
        return {k: c.get(k) for k in ("kind", "call_id", "fact_id", "value", "unit")}
    return {k: c.get(k) for k in ("kind", "call_id", "row", "column_name", "value")}


def _spaced(name: str) -> str:
    """嵌进中文句子的名称以西文开头或结尾时补空格：“出水 SS在线” → “出水 SS 在线”。"""
    return (" " if name[:1].isascii() and name[:1].isalnum() else "") + name + \
           (" " if name[-1:].isascii() and name[-1:].isalnum() else "")


def _recommendations(result: dict) -> list[dict]:
    """回答里的核查建议（服务端已校验过引用知识资料），连同引用一起带进草稿。"""
    if not result.get("ok"):
        return []
    cites = {c["number"]: c for c in result.get("citations", [])}
    return [{"text": c["text"], "source": "ai",
             "citations": [_brief(cites[n]) for n in c.get("citation_numbers", []) if n in cites]}
            for c in result.get("claims", []) if c["kind"] == "recommendation"]


def build_draft(alarm: dict, now: datetime, chat_turn: dict | None = None) -> dict:
    """步骤优先取本次对话回答中的核查建议，其次取已保存处置建议中的核查建议（都连同引用）；
    都没有时用默认模板并注明。chat_turn 须是给过该告警派单入口的那一轮。"""
    label, _ = metric_meta(alarm["metric"])
    where = alarm["plant_name"] + (f" {alarm['device_code']}" if alarm.get("device_code") else "")
    advice = alarm.get("advice") or {}
    steps = _recommendations(chat_turn) if chat_turn else []
    if steps:
        source = {"kind": "chat_answer", "session_id": chat_turn.get("session_id"), "run_id": chat_turn.get("run_id"),
                  "question": chat_turn.get("question"), "answered_at": chat_turn.get("now"),
                  "models": chat_turn.get("models", []),
                  "note": "步骤来自本次对话的 AI 核查建议，针对的是提问而不专指本告警，请核对后派单"}
    elif steps := _recommendations(advice):
        source = {"kind": "ai_advice", "advice_generated_at": advice.get("generated_at"),
                  "models": advice.get("models", [])}
    else:
        values = {"metric": _spaced(label), "device": _spaced(alarm.get("device_code") or "设备")}
        steps = [{"text": t.format(**values), "source": "template", "citations": []}
                 for t in DEFAULT_STEPS.get(alarm["kind"], [])]
        source = {"kind": "template", "note": "没有可用的 AI 核查建议，以下为默认步骤，不是 AI 建议"}
    steps.append({"text": CLOSING_STEP.format(metric=_spaced(label)), "source": "template", "citations": []})
    return {"draft_id": uuid.uuid4().hex, "alarm_id": alarm["id"], "plant_id": alarm["plant_id"],
            "title": f"{where} {alarm['kind']}（{label}）", "steps": steps, "source": source,
            "created_at": now.isoformat(), "expires_in_s": DRAFT_TTL_S}


async def chat_turn_for(request: Request, chat: ChatRef, alarm_id: int) -> dict | None:
    """该轮存在且给过这条告警的派单入口才采用；会话过期、Redis 故障时退回处置建议或模板，不影响派单。"""
    try:
        session = await request.app.state.agent_sessions.get(chat.session_id)
    except (SessionNotFound, RedisError, ValueError):
        return None
    turn = next((t for t in session.get("turns", []) if t.get("run_id") == chat.run_id), None)
    if turn is None or alarm_id not in {a.get("alarm_id") for a in turn.get("actions", [])}:
        return None
    return {**turn, "session_id": chat.session_id}


@router.post("/{alarm_id}/work-order-draft", summary="生成工单草稿（不写库，需用户确认）")
async def work_order_draft(alarm_id: int, conn: Conn, redis: Rds, now: Now, request: Request,
                           body: DraftRequest | None = None) -> dict:
    alarm = await get_alarm(conn, alarm_id)
    if alarm is None:
        raise HTTPException(404, f"告警 {alarm_id} 不存在")
    if alarm["status"] == "已关闭":
        raise HTTPException(409, "告警已关闭，不能再派单")
    open_wo = await (await conn.execute(
        "SELECT id FROM work_order WHERE alarm_id = %s AND status = ANY(%s) LIMIT 1",
        (alarm_id, list(WO_OPEN)))).fetchone()
    if open_wo:
        raise HTTPException(409, f"该告警已有未关闭的工单 #{open_wo['id']}")
    turn = await chat_turn_for(request, body.chat, alarm_id) if body and body.chat else None
    draft = build_draft(alarm, now, turn)
    await redis.set(DRAFT_KEY.format(draft["draft_id"]), json.dumps(draft, ensure_ascii=False), ex=DRAFT_TTL_S)
    return draft


@router.post("/{alarm_id}/close", summary="关闭告警（处理状态）")
async def close_alarm(alarm_id: int, body: CloseAlarm, conn: Conn, now: Now) -> dict:
    """有未关闭的工单时须先关闭工单。数据未恢复也允许关闭（如确认误报、计划内检修），须写明说明。"""
    async with conn.transaction():
        alarm = await (await conn.execute("SELECT id, status FROM alarm WHERE id = %s FOR UPDATE",
                                          (alarm_id,))).fetchone()
        if alarm is None:
            raise HTTPException(404, f"告警 {alarm_id} 不存在")
        if alarm["status"] == "已关闭":
            raise HTTPException(409, "告警已关闭")
        open_wo = await (await conn.execute(
            "SELECT id FROM work_order WHERE alarm_id = %s AND status = ANY(%s) LIMIT 1",
            (alarm_id, list(WO_OPEN)))).fetchone()
        if open_wo:
            raise HTTPException(409, f"请先关闭工单 #{open_wo['id']}")
        await conn.execute("""UPDATE alarm SET status = '已关闭', closed_at = %s, close_note = %s, updated_at = %s
                              WHERE id = %s""", (now, body.note, now, alarm_id))
    return alarm_out(await get_alarm(conn, alarm_id), now)

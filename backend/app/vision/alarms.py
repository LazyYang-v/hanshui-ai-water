"""视觉巡检结果 → 巡检记录 + 告警事件。

一个事件 = 同一厂站、同一摄像头、同一类别的一段连续异常（与信号告警的 episode 同义）：
- 本次检测有目标且分数 ≥ 告警阈值：已有未恢复事件且距上次命中不超过 DEDUP_WINDOW → 合并（更新次数、峰值、
  级别、最近图片）；否则新建。超过窗口的旧事件先以上次命中时间结束，再新建。
- 本次检测该类最高分 < 恢复阈值：已有未恢复事件 → 以本次时间恢复（ended_at）。
- 介于两个阈值之间：不新建也不恢复（滞回，避免同一画面在阈值附近反复开合）。
已被人工关闭（status=已关闭）的事件不再合并新的检测：用户判定处理完或误报后，再出现就是新事件。

告警写入同一张 alarm 表，rule 为 vision_*，device_id 为空（摄像头不是工艺设备台账），
摄像头编号在 detail.camera。信号巡检（app/analytics/alarms.py）只处理自己的规则编码，互不干扰。
阈值和级别是 Demo 运行管理口径。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from psycopg import AsyncConnection
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.vision.detector import Detection, Result

KIND = "视觉巡检"
DEDUP_WINDOW = timedelta(minutes=30)
KEEP_IMAGES = 5
KEEP_BOXES = 20


@dataclass(frozen=True)
class VisionRule:
    id: str
    cls: str
    label: str
    alarm_conf: float      # 达到才告警
    clear_conf: float      # 低于才恢复
    escalate_count: int    # 单张图目标数达到即升为「预警」，否则「提示」；0 表示一律「预警」


VISION_RULES = {r.cls: r for r in [
    VisionRule("vision_debris", "floating_debris", "漂浮物", 0.45, 0.25, 3),
    VisionRule("vision_leak", "leak", "渗漏", 0.50, 0.30, 0),
]}
VISION_RULE_IDS = [r.id for r in VISION_RULES.values()]
LABELS = {r.cls: r.label for r in VISION_RULES.values()}


def level_of(rule: VisionRule, count: int) -> str:
    return "预警" if rule.escalate_count == 0 or count >= rule.escalate_count else "提示"


def decide(rule: VisionRule, detections: list[Detection]) -> tuple[str, list[Detection]]:
    """返回 (hit / clear / hold, 达到告警阈值的检测)。"""
    scores = [d.score for d in detections if d.cls == rule.cls]
    hits = sorted((d for d in detections if d.cls == rule.cls and d.score >= rule.alarm_conf),
                  key=lambda d: -d.score)
    if hits:
        return "hit", hits
    if not scores or max(scores) < rule.clear_conf:
        return "clear", []
    return "hold", []


def _message(camera: str, rule: VisionRule, hits: list[Detection]) -> str:
    return f"{camera} 视觉巡检发现{rule.label} {len(hits)} 处（最高置信度 {hits[0].score:.2f}）"


async def _insert_alarm(cur, row: dict) -> dict:
    """演示时钟可能静止，同一时刻同厂站同类别的另一个摄像头会撞上事件唯一索引：错开 1 毫秒重试。"""
    for k in range(5):
        try:
            async with cur.connection.transaction():
                return await (await cur.execute("""
                    INSERT INTO alarm (ts, plant_id, device_id, level, kind, metric, value, threshold, message, status,
                                       rule, peak, peak_at, detail, updated_at)
                    VALUES (%(ts)s, %(plant_id)s, NULL, %(level)s, %(kind)s, %(metric)s, %(value)s, %(threshold)s,
                            %(message)s, '未处理', %(rule)s, %(value)s, %(ts)s, %(detail)s, %(now)s)
                    RETURNING *""", {**row, "ts": row["ts"] + timedelta(milliseconds=k)})).fetchone()
        except UniqueViolation:
            continue
    raise RuntimeError("视觉告警写入冲突，请重试")


async def record(conn: AsyncConnection, *, plant_id: int | None, camera: str | None, source: str,
                 result: Result, model: str, now: datetime, write_alarm: bool) -> dict:
    """写巡检记录；write_alarm 且给了厂站和摄像头时合并告警。返回 {inspection_id, events}。"""
    events: list[dict] = []
    async with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        alarm_ids: list[int] = []
        if write_alarm and plant_id is not None and camera:
            # 同一摄像头的并发请求串行处理，避免同一事件被重复新建
            await cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"vision:{plant_id}:{camera}",))
            for rule in VISION_RULES.values():
                action, hits = decide(rule, result.detections)
                if action == "hold":
                    continue
                open_ = await (await cur.execute("""
                    SELECT * FROM alarm WHERE plant_id = %s AND rule = %s AND detail->>'camera' = %s
                      AND ended_at IS NULL ORDER BY ts DESC LIMIT 1 FOR UPDATE""",
                    (plant_id, rule.id, camera))).fetchone()
                if action == "clear":
                    if open_ is not None:
                        row = await (await cur.execute("""
                            UPDATE alarm SET ended_at = %s, updated_at = %s, detail = detail || %s
                            WHERE id = %s RETURNING *""",
                            (now, now, Jsonb({"cleared_by": source}), open_["id"]))).fetchone()
                        events.append({"event": "recovered", "alarm_id": row["id"], "rule": rule.id})
                        alarm_ids.append(row["id"])
                    continue
                level = level_of(rule, len(hits))
                boxes = [d.out() for d in hits[:KEEP_BOXES]]
                image = {"ts": now.isoformat(), "sha256": result.sha256, "source": source, "count": len(hits),
                         "max_score": round(hits[0].score, 4)}
                if open_ is not None:
                    last_seen = datetime.fromisoformat(open_["detail"].get("last_seen", open_["ts"].isoformat()))
                    if open_["status"] == "已关闭" or now - last_seen > DEDUP_WINDOW:
                        await cur.execute("UPDATE alarm SET ended_at = %s, updated_at = %s WHERE id = %s",
                                          (max(last_seen, open_["ts"]), now, open_["id"]))
                        open_ = None
                if open_ is None:
                    row = await _insert_alarm(cur, {
                        "ts": now, "plant_id": plant_id, "level": level, "kind": KIND, "metric": rule.cls,
                        "value": round(hits[0].score, 4), "threshold": rule.alarm_conf,
                        "message": _message(camera, rule, hits), "rule": rule.id, "now": now,
                        "detail": Jsonb({"camera": camera, "unit": "置信度", "count": len(hits), "boxes": boxes,
                                         "images": [image], "hits": 1, "model": model,
                                         "last_seen": now.isoformat(), "detected_at": now.isoformat()})})
                    events.append({"event": "new", "alarm_id": row["id"], "rule": rule.id, "level": row["level"]})
                else:
                    d = open_["detail"]
                    new_level = "预警" if "预警" in (open_["level"], level) else open_["level"]
                    peak = max(open_["peak"] or 0, hits[0].score)
                    detail = {**d, "count": len(hits), "boxes": boxes, "hits": d.get("hits", 1) + 1, "model": model,
                              "images": (d.get("images", []) + [image])[-KEEP_IMAGES:], "last_seen": now.isoformat()}
                    row = await (await cur.execute("""
                        UPDATE alarm SET level = %s, value = %s, message = %s, peak = %s, peak_at = %s, detail = %s,
                               updated_at = %s WHERE id = %s RETURNING *""",
                        (new_level, round(hits[0].score, 4), _message(camera, rule, hits), round(peak, 4),
                         now if peak == hits[0].score else open_["peak_at"], Jsonb(detail), now, open_["id"])
                    )).fetchone()
                    events.append({"event": "escalated" if new_level != open_["level"] else "updated",
                                   "alarm_id": row["id"], "rule": rule.id, "level": row["level"]})
                alarm_ids.append(row["id"])
        ins = await (await cur.execute("""
            INSERT INTO vision_inspection (ts, plant_id, camera, source, image_sha256, width, height, model,
                                           detections, timings_ms, alarm_ids)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (now, plant_id, camera, source, result.sha256, result.width, result.height, model,
             Jsonb([d.out() for d in result.detections]), Jsonb(result.timings_ms), alarm_ids))).fetchone()
    return {"inspection_id": ins["id"], "events": events}

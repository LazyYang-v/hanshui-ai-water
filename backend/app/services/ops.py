"""告警与工单的读取和输出格式，供告警、工单、报告接口共用。"""

from __future__ import annotations

from datetime import datetime

from psycopg import AsyncConnection

from app.domain.metrics import DERIVED_METRICS, DEVICE_METRICS, PLANT_METRICS
from app.vision.alarms import LABELS as VISION_LABELS

ALARM_SELECT = """
    SELECT a.*, p.code AS plant_code, p.name AS plant_name, d.code AS device_code, d.kind AS device_kind
    FROM alarm a JOIN plant p ON p.id = a.plant_id LEFT JOIN device d ON d.id = a.device_id"""

WORK_ORDER_SELECT = """
    SELECT w.*, p.code AS plant_code, p.name AS plant_name, a.level AS alarm_level, a.kind AS alarm_kind,
           a.message AS alarm_message, a.status AS alarm_status
    FROM work_order w LEFT JOIN alarm a ON a.id = w.alarm_id LEFT JOIN plant p ON p.id = w.plant_id"""

WO_OPEN = ("待处理", "处理中")


def metric_meta(metric: str | None) -> tuple[str, str]:
    if metric in PLANT_METRICS:
        m = PLANT_METRICS[metric]
    elif metric in DERIVED_METRICS:
        m = DERIVED_METRICS[metric]
    elif metric in DEVICE_METRICS:
        m = DEVICE_METRICS[metric]
    elif metric == "running":
        return "运行状态", ""
    elif metric in VISION_LABELS:
        return VISION_LABELS[metric], "置信度"
    else:
        return metric or "", ""
    return m.label, m.unit


def _iso(t: datetime | None) -> str | None:
    return t.isoformat() if t else None


def alarm_out(r: dict, now: datetime, *, advice: bool = False) -> dict:
    label, unit = metric_meta(r["metric"])
    end = r.get("ended_at")
    out = {
        "id": r["id"], "ts": _iso(r["ts"]), "ended_at": _iso(end), "active": end is None,
        "duration_min": round(((end or now) - r["ts"]).total_seconds() / 60, 1),
        "plant": {"id": r["plant_id"], "code": r["plant_code"], "name": r["plant_name"]},
        "device": ({"id": r["device_id"], "code": r["device_code"], "kind": r["device_kind"]}
                   if r.get("device_id") else None),
        "level": r["level"], "kind": r["kind"], "rule": r.get("rule"), "metric": r["metric"],
        "metric_label": label, "unit": (r.get("detail") or {}).get("unit", unit),
        "value": r["value"], "threshold": r["threshold"], "peak": r.get("peak"), "peak_at": _iso(r.get("peak_at")),
        "message": r["message"], "status": r["status"], "detail": r.get("detail") or {},
        "updated_at": _iso(r.get("updated_at")), "closed_at": _iso(r.get("closed_at")),
        "close_note": r.get("close_note"),
        "advice_at": (r.get("advice") or {}).get("generated_at"),
    }
    if advice:
        out["advice"] = r.get("advice")
    return out


def work_order_out(r: dict) -> dict:
    return {
        "id": r["id"], "alarm_id": r["alarm_id"], "title": r["title"], "assignee": r["assignee"],
        "status": r["status"], "steps": r["steps"], "resolution": r["resolution"],
        "plant": {"id": r["plant_id"], "code": r.get("plant_code"), "name": r.get("plant_name")},
        "alarm": {"level": r.get("alarm_level"), "kind": r.get("alarm_kind"), "message": r.get("alarm_message"),
                  "status": r.get("alarm_status")} if r["alarm_id"] else None,
        "created_at": _iso(r["created_at"]), "started_at": _iso(r.get("started_at")),
        "closed_at": _iso(r["closed_at"]), "updated_at": _iso(r.get("updated_at")), "source": r.get("source") or {},
    }


async def get_alarm(conn: AsyncConnection, alarm_id: int) -> dict | None:
    return await (await conn.execute(ALARM_SELECT + " WHERE a.id = %s", (alarm_id,))).fetchone()


async def get_work_order(conn: AsyncConnection, wo_id: int) -> dict | None:
    return await (await conn.execute(WORK_ORDER_SELECT + " WHERE w.id = %s", (wo_id,))).fetchone()

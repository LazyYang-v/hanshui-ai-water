"""对话里的派单入口：按 Agent 本轮实际查询的范围，找出可能相关的未关闭告警。

模型不参与这一步，也不能写库：服务端只读本轮结果里已经确定的信息——
统计工具的厂站与时间段、规划后问题里写明的厂站/时间/指标、用户显式写出的告警编号——
再查告警表。前端据此显示“生成工单草稿”，草稿与确认派单仍走原有接口和人工确认。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from psycopg import AsyncConnection

from app.deps import LOCAL_TZ
from app.services.ops import ALARM_SELECT, WO_OPEN, metric_meta

MAX_ACTIONS = 3
MAX_PLANTS = 3                      # 涉及更多厂站的全区排名类问题不给派单入口
DEFAULT_WINDOW = timedelta(hours=24)
MARGIN = timedelta(hours=1)         # 告警开始时间略早于查询时段时仍算相关
# 告警编号只认 “#25”“告警25”“告警 #25”“告警编号 25” 这类明确写法；“告警 2 条”“1#池”不算
ALARM_ID = re.compile(r"(?:#|告警(?=\d)|告警编号\s*)(\d{1,7})(?!\d)")
# 规划后的问题里时间写法不一：“2026-11-07 18:00”“2026-11-08T10:19:30+08:00”“2026年11月8日10:19:30至16:19:30”。
# 没写日期的时刻沿用前一个日期（跨午夜则加一天）
DATETIME = re.compile(
    r"(?:(\d{4})[-年/](\d{1,2})[-月/](\d{1,2})日?\s*(T)?\s*)?(?<![\d:])(\d{1,2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?"
    r"(?(4)(Z|[+-]\d{2}:?\d{2})?)(?![\d:])")      # 只有 ISO 写法（带 T）才认时区：“18:00-20:00”的“-20:00”是结束时刻
DEVICE_WORDS = ("鼓风机", "设备", "振动", "轴承", "温度", "停机", "运行")


@dataclass
class Scope:
    plants: set[str] = field(default_factory=set)
    start: datetime | None = None
    end: datetime | None = None
    words: str = ""                 # 规划后问题与工具问题，去空白，用于指标相关性排序
    alarm_ids: list[int] = field(default_factory=list)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _label_key(label: str) -> str:
    """“好氧区 DO（1#）” → “好氧区DO”：问题里通常不写池号。"""
    return re.sub(r"[（(].*?[）)]", "", _norm(label))


def scope_from_result(result: dict, plants: list[dict], now: datetime) -> Scope:
    """只读取本轮结果里服务端已校验或模型规划时明确写出的范围；纯函数，便于单测。"""
    scope = Scope(alarm_ids=[int(x) for x in ALARM_ID.findall(result.get("question", ""))])
    resolved = result.get("resolved_question", "") or ""
    texts = [resolved] + [t.get("question", "") for t in result.get("tool_calls", [])]
    scope.words = _norm(" ".join(texts))

    starts: list[datetime] = []
    ends: list[datetime] = []
    for call in result.get("tool_calls", []):
        out = call.get("output") or {}
        if call.get("tool") == "plant_statistics" and call.get("status") == "ok":
            codes = [p["code"] for p in out.get("plants", []) if p.get("code")]
            if 0 < len(codes) <= MAX_PLANTS:
                scope.plants.update(codes)
            period = out.get("period") or {}
            if period.get("start") and period.get("end"):
                starts.append(datetime.fromisoformat(period["start"]))
                ends.append(datetime.fromisoformat(period["end"]))

    # 规划后问题会写明厂站（如“黄官镇（P06）”）
    for p in plants:
        short = re.sub(r"(中心)?污水处理(厂|站)$", "", p["name"])
        if p["code"] in resolved or p["name"] in resolved or (short and short in resolved):
            scope.plants.add(p["code"])
    if len(scope.plants) > MAX_PLANTS:
        scope.plants = set()

    if not starts:
        found = [t for text in texts for t in datetimes(text)]
        if len(found) >= 2:
            starts.append(min(found))
            ends.append(max(found))
    if starts:
        scope.start, scope.end = min(starts), max(ends)
    else:
        scope.start, scope.end = now - DEFAULT_WINDOW, now
    return scope


def datetimes(text: str) -> list[datetime]:
    """按出现顺序取出文本里的时刻；没有日期可沿用的时刻忽略。"""
    out: list[datetime] = []
    day: tuple[int, int, int] | None = None
    for y, mo, d, _t, h, mi, sec, tz in DATETIME.findall(text):
        if y:
            day = (int(y), int(mo), int(d))
        if day is None or int(h) > 23 or int(mi) > 59:
            continue
        try:
            t = datetime(*day, int(h), int(mi), int(sec or 0))
        except ValueError:
            continue
        if tz:
            t = datetime.fromisoformat(t.isoformat() + ("+00:00" if tz == "Z" else tz))
            t = t.astimezone(LOCAL_TZ)
        else:
            t = t.replace(tzinfo=LOCAL_TZ)
        if not y and out and t < out[-1]:
            t += timedelta(days=1)
        out.append(t)
    return out


def relevance(alarm: dict, words: str) -> int:
    """问题里提到了该告警的指标（或设备类关键词）排前面。"""
    label, _ = metric_meta(alarm.get("metric"))
    if label and _label_key(label) and _label_key(label) in words:
        return 2
    if alarm.get("device_id") and any(w in words for w in DEVICE_WORDS):
        return 1
    return 0


def _action(a: dict, matched_by: str, open_wo: int | None) -> dict:
    return {
        "type": "work_order", "alarm_id": a["id"], "plant_code": a["plant_code"], "plant_name": a["plant_name"],
        "device_code": a.get("device_code"), "level": a["level"], "kind": a["kind"], "metric": a.get("metric"),
        "metric_label": metric_meta(a.get("metric"))[0], "message": a["message"],
        "ts": a["ts"].isoformat(), "ended_at": a["ended_at"].isoformat() if a.get("ended_at") else None,
        "status": a["status"], "open_work_order_id": open_wo, "matched_by": matched_by,
    }


async def suggest_actions(conn: AsyncConnection, result: dict, now: datetime) -> list[dict]:
    """显式编号总是给入口（用户说“给 #25 派单”时 Agent 会拒答，入口仍应出现）。
    按查询范围匹配不要求回答成功：查数超时、证据不足而拒答时，问到的厂站时段里的告警仍是库里的事实，
    此时入口反而最有用；只有能力边界外的问题（intent=unsupported）不按范围匹配。"""
    plants = await (await conn.execute("SELECT code, name FROM plant")).fetchall()
    scope = scope_from_result(result, plants, now)
    if result.get("intent") == "unsupported":
        scope.plants = set()
    picked: list[tuple[dict, str]] = []
    if scope.alarm_ids:
        rows = await (await conn.execute(ALARM_SELECT + " WHERE a.id = ANY(%s)", (scope.alarm_ids[:MAX_ACTIONS],))).fetchall()
        order = {i: k for k, i in enumerate(scope.alarm_ids)}
        picked += [(r, "explicit") for r in sorted(rows, key=lambda r: order[r["id"]])]
    if scope.plants and len(picked) < MAX_ACTIONS:
        rows = await (await conn.execute(
            ALARM_SELECT + """ WHERE p.code = ANY(%s) AND a.status <> '已关闭'
              AND a.ts < %s AND (a.ended_at IS NULL OR a.ended_at > %s)
            ORDER BY a.ts DESC LIMIT 50""",
            (sorted(scope.plants), scope.end + MARGIN, scope.start - MARGIN))).fetchall()
        level = {"告警": 2, "预警": 1, "提示": 0}
        rows.sort(key=lambda r: (relevance(r, scope.words), level.get(r["level"], 0), r["ts"]), reverse=True)
        seen = {r["id"] for r, _ in picked}
        picked += [(r, "scope") for r in rows if r["id"] not in seen]
    picked = picked[:MAX_ACTIONS]
    if not picked:
        return []
    open_rows = await (await conn.execute(
        "SELECT alarm_id, id FROM work_order WHERE alarm_id = ANY(%s) AND status = ANY(%s)",
        ([r["id"] for r, _ in picked], list(WO_OPEN)))).fetchall()
    open_wo = {r["alarm_id"]: r["id"] for r in open_rows}
    return [_action(r, how, open_wo.get(r["id"])) for r, how in picked]

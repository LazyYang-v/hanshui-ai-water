import json
from collections import Counter
from datetime import datetime, time, timedelta

from fastapi import APIRouter

from app.deps import Conn, Now, Rds
from app.schemas import Overview
from app.services import queries

router = APIRouter(prefix="/api", tags=["总览"])

KEY_SCENARIOS = "sim:scenarios"


@router.get("/overview", response_model=Overview, summary="集控总览：天气、当日 KPI、厂站状态、24 小时趋势")
async def overview(conn: Conn, redis: Rds, now: Now):
    day_start = datetime.combine(now.date(), time(0), now.tzinfo)
    plants = await queries.plants_with_latest(conn, now)
    stats = await queries.period_stats(conn, day_start, now)
    hourly = await queries.hourly_totals(conn, now - timedelta(hours=24), now)
    weather = await (await conn.execute(
        "SELECT ts, rain_mm_h, temp_c FROM weather WHERE ts <= %s ORDER BY ts DESC LIMIT 1", (now,))).fetchone()
    scenarios = json.loads(await redis.get(KEY_SCENARIOS) or "[]")
    counts = Counter(p["status"] for p in plants)
    return {
        "now": now,
        "weather": weather or {"ts": None, "rain_mm_h": None, "temp_c": None},
        "today": queries.summarize(stats),
        "status_counts": {s: counts.get(s, 0) for s in ("normal", "warn", "alarm", "offline")},
        "plants": plants,
        "hourly": hourly,
        "scenarios": scenarios,
    }

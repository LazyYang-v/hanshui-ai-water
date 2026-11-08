from datetime import datetime, time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.common import TimeRange, parse_metrics
from app.deps import Conn, Now
from app.domain.metrics import PLANT_METRICS
from app.schemas import PlantDetail, PlantSummary, Series
from app.services import queries

router = APIRouter(prefix="/api/plants", tags=["厂站"])


async def _plant(conn, now: datetime, code: str) -> dict:
    rows = await queries.plants_with_latest(conn, now, code.upper())
    if not rows:
        raise HTTPException(404, f"厂站 {code} 不存在")
    return rows[0]


@router.get("", response_model=list[PlantSummary], summary="厂站列表（含最新读数与状态）")
async def list_plants(conn: Conn, now: Now):
    return await queries.plants_with_latest(conn, now)


@router.get("/{code}", response_model=PlantDetail, summary="厂站详情（含当日统计与设备）")
async def get_plant(code: str, conn: Conn, now: Now):
    plant = await _plant(conn, now, code)
    day_start = datetime.combine(now.date(), time(0), now.tzinfo)
    stats = await queries.period_stats(conn, day_start, now, plant["id"])
    devices = await queries.devices_with_latest(conn, plant["id"], now)
    return {**plant, "today": stats[0] if stats else None, "devices": devices}


@router.get("/{code}/readings", response_model=Series, summary="厂站指标时间序列")
async def plant_readings(
    code: str,
    conn: Conn,
    now: Now,
    rng: Annotated[TimeRange, Depends()],
    metrics: str | None = Query(None, description="逗号分隔的指标，见 /api/metrics；默认出水五项",
                                examples=["nh3n_out,do_aerobic_1,inflow_m3h"]),
):
    plant = await _plant(conn, now, code)
    ms = parse_metrics(metrics, PLANT_METRICS, ["cod_out", "nh3n_out", "tn_out", "tp_out", "ss_out"])
    try:
        interval, name = queries.pick_bucket(rng.start, rng.end, rng.bucket)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    data = await queries.series(conn, "plant_reading", "plant_id", plant["id"], ms, rng.start, rng.end, interval)
    return {"entity": plant["code"], "bucket": name, "start": rng.start, "end": rng.end,
            "metrics": [m.info() for m in ms], **data}

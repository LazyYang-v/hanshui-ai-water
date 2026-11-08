from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.common import TimeRange, parse_metrics
from app.deps import Conn
from app.domain.metrics import DEVICE_METRICS
from app.schemas import Series
from app.services import queries

router = APIRouter(prefix="/api/devices", tags=["设备"])


@router.get("/{code}/readings", response_model=Series, summary="设备监测时间序列")
async def device_readings(
    code: str,
    conn: Conn,
    rng: Annotated[TimeRange, Depends()],
    metrics: str | None = Query(None, description="current_a / vibration_mms / temp_c，默认全部"),
):
    row = await (await conn.execute("SELECT id, code FROM device WHERE code = %s", (code.upper(),))).fetchone()
    if not row:
        raise HTTPException(404, f"设备 {code} 不存在")
    ms = parse_metrics(metrics, DEVICE_METRICS, list(DEVICE_METRICS))
    try:
        interval, name = queries.pick_bucket(rng.start, rng.end, rng.bucket)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    data = await queries.series(conn, "device_reading", "device_id", row["id"], ms, rng.start, rng.end, interval)
    return {"entity": row["code"], "bucket": name, "start": rng.start, "end": rng.end,
            "metrics": [m.info() for m in ms], **data}

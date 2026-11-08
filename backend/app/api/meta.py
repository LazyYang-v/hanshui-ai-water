from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.deps import KEY_SPEED, LOCAL_TZ, Conn, Now, Rds
from app.domain.metrics import DERIVED_METRICS, DEVICE_METRICS, PLANT_METRICS
from app.schemas import Clock, MetricInfo

router = APIRouter(prefix="/api", tags=["基础"])


@router.get("/metrics", summary="指标目录（名称、单位、限值）")
async def metrics() -> dict[str, list[MetricInfo]]:
    return {"plant": [m.info() for m in PLANT_METRICS.values()],
            "device": [m.info() for m in DEVICE_METRICS.values()],
            "derived": [m.info() for m in DERIVED_METRICS.values()]}


@router.get("/clock", response_model=Clock, summary="演示时钟")
async def clock(now: Now, redis: Rds):
    wall = datetime.now(LOCAL_TZ)
    return {"now": now, "wall": wall, "ahead_s": round((now - wall).total_seconds(), 1),
            "speed": float(await redis.get(KEY_SPEED) or 1)}


@router.get("/health", summary="健康检查：数据库、Redis、模拟器")
async def health(conn: Conn, redis: Rds):
    checks = {}
    try:
        await conn.execute("SELECT 1")
        checks["db"] = "ok"
    except Exception as e:  # noqa: BLE001
        checks["db"] = f"error: {e}"
    try:
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as e:  # noqa: BLE001
        checks["redis"] = f"error: {e}"
    # 模拟器在跑 ⇔ 有人订阅控制频道
    try:
        subs = dict(await redis.pubsub_numsub("sim:control"))
        checks["simulator"] = "ok" if subs.get("sim:control", 0) > 0 else "not running"
    except Exception:  # noqa: BLE001
        checks["simulator"] = "unknown"
    healthy = checks["db"] == "ok" and checks["redis"] == "ok"
    return JSONResponse({"status": "ok" if healthy else "degraded", **checks}, status_code=200 if healthy else 503)

"""视觉巡检：上传图片或用内置样例推理，结果写巡检记录，命中阈值时合并为告警。

上传用原始请求体（Content-Type: image/jpeg 等），不走 multipart，不额外引入表单解析依赖：
    curl -X POST 'localhost:8000/api/vision/inspect?plant=P01&camera=CAM-AER-01' \\
         -H 'Content-Type: image/jpeg' --data-binary @pool.jpg
模型说明见 GET /api/vision/model；训练数据、指标与许可见 evals/vision/results/。
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from psycopg.rows import dict_row

from app.deps import sim_now
from app.vision.alarms import record
from app.vision.detector import ImageError

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/vision", tags=["视觉巡检"])

MAX_UPLOAD = 10 * 1024 * 1024
CAMERA = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp", "image/bmp", "application/octet-stream")


def _service(request: Request):
    svc = getattr(request.app.state, "vision", None)
    if svc is None:
        raise HTTPException(503, "视觉巡检模型未加载（缺少 models/vision/inspect.onnx 或加载失败）")
    return svc


async def _read_body(request: Request) -> bytes:
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype not in IMAGE_TYPES:
        raise HTTPException(415, f"请求体须为图片（Content-Type: {', '.join(IMAGE_TYPES[:4])}）")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD:
        raise HTTPException(413, f"图片超过 {MAX_UPLOAD // 1024 // 1024} MB")
    buf = bytearray()
    async for chunk in request.stream():            # 不信任 Content-Length，边读边限
        buf += chunk
        if len(buf) > MAX_UPLOAD:
            raise HTTPException(413, f"图片超过 {MAX_UPLOAD // 1024 // 1024} MB")
    if not buf:
        raise HTTPException(422, "请求体为空")
    return bytes(buf)


async def _inspect(request: Request, data: bytes, *, source: str, plant: str | None, camera: str | None,
                   alarm: bool, conf: float | None) -> dict:
    svc = _service(request)
    if camera is not None and not CAMERA.match(camera):
        raise HTTPException(422, "camera 只能包含字母、数字、- 和 _，长度 1-32")
    pool = request.app.state.pool
    plant_row = None
    if plant:
        async with pool.connection() as conn:
            conn.row_factory = dict_row
            plant_row = await (await conn.execute("SELECT id, code, name FROM plant WHERE code = %s",
                                                  (plant,))).fetchone()
        if plant_row is None:
            raise HTTPException(404, f"厂站 {plant} 不存在")
    try:
        result = await svc.detect(data, conf)
    except ImageError as e:
        raise HTTPException(422, str(e)) from e
    out = {"model_id": svc.model_id, "source": source, "camera": camera,
           "plant": dict(plant_row) if plant_row else None, **result.out(),
           "inspection_id": None, "events": [], "alarm_note": None}
    write_alarm = alarm and plant_row is not None and camera is not None
    if alarm and not write_alarm:
        out["alarm_note"] = "未同时指定厂站和摄像头，只记录推理结果，不写告警"
    try:
        async with pool.connection() as conn:
            conn.row_factory = dict_row
            now = await sim_now(request.app.state.redis, conn)
            saved = await record(conn, plant_id=plant_row["id"] if plant_row else None, camera=camera,
                                 source=source, result=result, model=svc.model_id, now=now, write_alarm=write_alarm)
        out.update(saved, ts=now.isoformat())
    except Exception as e:  # 记录失败不影响返回检测结果，但要明说
        log.exception("视觉巡检记录写入失败")
        out["alarm_note"] = f"检测完成，但巡检记录/告警写入失败：{type(e).__name__}"
    return out


@router.get("/model", summary="模型卡：类别、阈值、训练数据、评测指标、许可与运行配置")
async def model_card(request: Request) -> dict:
    return _service(request).info()


@router.get("/samples", summary="内置样例图片（合成图 + 公开数据集图）")
async def samples(request: Request) -> dict:
    return {"items": [s.out() for s in _service(request).samples.values()]}


@router.get("/samples/{name}", summary="样例图片原图", response_class=FileResponse)
async def sample_image(name: str, request: Request):
    path = _service(request).sample_path(name)
    if path is None:
        raise HTTPException(404, f"样例 {name} 不存在")
    return FileResponse(path)


@router.post("/samples/{name}/inspect", summary="对内置样例离线推理（可写告警）")
async def inspect_sample(
    name: str, request: Request,
    plant: str | None = Query(None, description="厂站编码，如 P01；与 camera 同时给出才写告警"),
    camera: str | None = Query(None, description="摄像头编号；默认用样例自带的编号"),
    alarm: bool = Query(True, description="是否按阈值写入/合并告警"),
    conf: float | None = Query(None, ge=0.05, le=0.95, description="展示用置信度下限，默认取模型卡"),
) -> dict:
    svc = _service(request)
    path = svc.sample_path(name)
    if path is None:
        raise HTTPException(404, f"样例 {name} 不存在")
    return await _inspect(request, path.read_bytes(), source=f"sample:{name}", plant=plant,
                          camera=camera or svc.samples[name].camera, alarm=alarm, conf=conf)


@router.post("/inspect", summary="上传图片推理（原始请求体，≤10 MB）")
async def inspect_upload(
    request: Request,
    plant: str | None = Query(None, description="厂站编码，如 P01；与 camera 同时给出才写告警"),
    camera: str | None = Query(None, description="摄像头编号"),
    alarm: bool = Query(True, description="是否按阈值写入/合并告警"),
    conf: float | None = Query(None, ge=0.05, le=0.95, description="展示用置信度下限，默认取模型卡"),
) -> dict:
    data = await _read_body(request)
    return await _inspect(request, data, source="upload", plant=plant, camera=camera, alarm=alarm, conf=conf)


@router.get("/inspections", summary="最近的巡检记录")
async def inspections(request: Request, plant: str | None = Query(None), camera: str | None = Query(None),
                      limit: int = Query(20, ge=1, le=200)) -> dict:
    async with request.app.state.pool.connection() as conn:
        conn.row_factory = dict_row
        rows = await (await conn.execute("""
            SELECT v.id, v.ts, p.code AS plant, v.camera, v.source, v.image_sha256, v.width, v.height, v.model,
                   v.detections, v.timings_ms, v.alarm_ids
            FROM vision_inspection v LEFT JOIN plant p ON p.id = v.plant_id
            WHERE (%(plant)s::text IS NULL OR p.code = %(plant)s) AND (%(camera)s::text IS NULL OR v.camera = %(camera)s)
            ORDER BY v.ts DESC, v.id DESC LIMIT %(limit)s""", {"plant": plant, "camera": camera, "limit": limit})
        ).fetchall()
    return {"items": [{**r, "ts": r["ts"].isoformat()} for r in rows]}

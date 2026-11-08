"""运行报告：生成日报/周报，列表、详情与 Markdown 下载。"""

from __future__ import annotations

import asyncio
from datetime import date as Date  # 请求字段名就叫 date，类型用别名避免被字段遮住
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from app.ai.reporter import NoData, period_for
from app.deps import LOCAL_TZ, Conn, ReleasedNow
from app.services.pdf import render_pdf

router = APIRouter(prefix="/api/reports", tags=["运行报告"])

KIND_SLUG = {"日报": "daily", "周报": "weekly"}


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["日报", "周报"]
    date: Date | None = Field(None, description="北京时间日期，默认演示时钟当天；周报为截至该日的 7 天")


@router.post("", status_code=201, summary="生成运行报告")
async def create_report(body: ReportRequest, request: Request, now: ReleasedNow) -> dict:
    """统计时段截至演示时钟；SQL 汇总事实后由 Flash 写叙述，校验失败时用模板叙述。"""
    start, end = period_for(body.kind, body.date or now.date(), now)
    if start >= now:
        raise HTTPException(422, "所选日期晚于演示时钟，还没有数据")
    try:
        return await request.app.state.reporter.generate(request.app.state.pool, body.kind, start, end, now)
    except NoData as e:
        raise HTTPException(422, str(e)) from None


@router.get("", summary="报告列表")
async def list_reports(conn: Conn, kind: Literal["日报", "周报"] | None = None,
                       limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)) -> dict:
    rows = await (await conn.execute("""
        SELECT id, kind, period_start, period_end, created_at, meta->>'narrative' AS narrative,
               meta->>'model' AS model, (meta->>'total_ms')::float8 AS total_ms, count(*) OVER () AS total
        FROM report WHERE %(kind)s::text IS NULL OR kind = %(kind)s
        ORDER BY id DESC LIMIT %(limit)s OFFSET %(offset)s""",
        {"kind": kind, "limit": limit, "offset": offset})).fetchall()
    return {"total": rows[0]["total"] if rows else 0, "limit": limit, "offset": offset,
            "items": [{k: v for k, v in r.items() if k != "total"} for r in rows]}


async def _get(conn, report_id: int) -> dict:
    row = await (await conn.execute("SELECT * FROM report WHERE id = %s", (report_id,))).fetchone()
    if row is None:
        raise HTTPException(404, f"报告 {report_id} 不存在")
    return row


@router.get("/{report_id}", summary="报告详情（含 Markdown、统计事实与生成信息）")
async def report_detail(report_id: int, conn: Conn) -> dict:
    return await _get(conn, report_id)


@router.get("/{report_id}/pdf", summary="下载 PDF（服务端由同一份 Markdown 排版）")
async def report_pdf(report_id: int, conn: Conn) -> Response:
    row = await _get(conn, report_id)
    day = row["period_start"].astimezone(LOCAL_TZ)
    name = f"shuiwu-{KIND_SLUG.get(row['kind'], 'report')}-{day:%Y%m%d}-{report_id}.pdf"
    title = row["content_md"].splitlines()[0].lstrip("# ").strip() if row["content_md"] else "汉水智控运行报告"
    # 排版是纯 CPU 计算，放到线程里，不阻塞事件循环
    data = await asyncio.to_thread(render_pdf, row["content_md"], title=title)
    return Response(data, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/{report_id}/markdown", summary="下载 Markdown")
async def report_markdown(report_id: int, conn: Conn) -> Response:
    row = await _get(conn, report_id)
    day = row["period_start"].astimezone(LOCAL_TZ)
    name = f"shuiwu-{KIND_SLUG.get(row['kind'], 'report')}-{day:%Y%m%d}-{report_id}.md"
    return Response(row["content_md"], media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})

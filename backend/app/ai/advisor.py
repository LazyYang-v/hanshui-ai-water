"""告警处置建议：固定查询取证据 → 知识检索 → 复用 Agent 的综合回答与引用校验。

与 Agent 对话不同，证据由服务端按告警类型确定，不经过 Text2SQL：
同一条告警每次看到的都是同一组表，引用可以复核；结论仍须逐条带单元格或知识原文出处，
原因只能写成待核查的假设，建议只能是资料支持的核查项，不写未经验证的操作参数。
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

from psycopg import AsyncConnection, sql

from app.ai.agent import Agent
from app.ai.agent.models import ToolRecord
from app.ai.rag.service import RAG
from app.analytics.alarms import bucket_floor
from app.analytics.statistics import StatisticsQuery
from app.deps import LOCAL_TZ
from app.domain.metrics import DERIVED_METRICS, LOAD_SOURCES, PLANT_METRICS
from app.services.ops import metric_meta

# 厂站级告警：主指标之外一起看的工艺指标
RELATED = {
    "nh3n_out": ["nh3n_out", "tn_out", "do_aerobic_1", "air_flow_m3h", "inflow_m3h", "mlss"],
    "tn_out": ["tn_out", "nh3n_out", "carbon_dose_kgh", "do_aerobic_1", "inflow_m3h"],
    "tp_out": ["tp_out", "pac_dose_kgh", "tp_in", "inflow_m3h"],
    "cod_out": ["cod_out", "cod_in", "do_aerobic_1", "inflow_m3h"],
    "ss_out": ["ss_out", "inflow_m3h", "mlss"],
    "do_aerobic_1": ["do_aerobic_1", "air_flow_m3h", "blower_kw", "nh3n_out"],
    "inflow_m3h": ["inflow_m3h", "ss_out", "nh3n_out", "mlss"],
    "carbon_dose_kgh": ["carbon_dose_kgh", "tn_out", "tn_in", "inflow_m3h"],
    "pac_dose_kgh": ["pac_dose_kgh", "tp_out", "tp_in", "inflow_m3h"],
}
LOWER_IS_BAD = {"do_aerobic_1"}
# 厂站级告警关联的设备类型
DEVICE_KINDS = {
    "nh3n_out": ["鼓风机"], "tn_out": ["鼓风机", "回流泵", "加药泵"], "tp_out": ["加药泵"], "cod_out": ["鼓风机"],
    "ss_out": ["提升泵", "污泥泵"], "do_aerobic_1": ["鼓风机"], "inflow_m3h": ["提升泵"],
    "carbon_dose_kgh": ["加药泵"], "pac_dose_kgh": ["加药泵"],
}
METRIC_QUERY = {
    "nh3n_out": "出水氨氮超标 低溶解氧 鼓风机故障 冲击负荷鉴别",
    "tn_out": "总氮升高 碳源 反硝化 回流",
    "tp_out": "总磷上升 PAC 加药泵堵塞 有效投加核查",
    "cod_out": "出水COD升高 冲击负荷 排查",
    "ss_out": "出水SS升高 暴雨 污泥流失 夹带",
    "do_aerobic_1": "溶解氧偏低 鼓风机故障 供气能力 水质响应",
    "inflow_m3h": "暴雨入流 进水量激增 稀释 初期冲刷 大流量处置",
    "carbon_dose_kgh": "碳源管道渗漏 药耗异常 有效送达 计量",
    "pac_dose_kgh": "PAC 泵堵塞 投加恢复验证 总磷",
}
for load_metric, source in LOAD_SOURCES.items():
    RELATED[load_metric] = ["inflow_m3h", source, source.replace("_in", "_out"), "do_aerobic_1", "air_flow_m3h"]
    DEVICE_KINDS[load_metric] = ["提升泵", "鼓风机"]
    METRIC_QUERY[load_metric] = (f"{DERIVED_METRICS[load_metric].label} 流量乘浓度 kg/h 冲击负荷 雨天稀释 仪表复核 人工取样")

DEVICE_QUERY = {"鼓风机": "鼓风机 振动 温度 跳停 证据边界 检查隔离与恢复验证",
                "加药泵": "加药泵 堵塞 有效投加 恢复验证", "提升泵": "提升泵 停机 进水 大流量"}
MAX_ROWS = 24
KNOWLEDGE_TOP_K = 4


def evidence_window(alarm: dict, now: datetime) -> tuple[datetime, datetime, str]:
    """告警前 1 小时到恢复后 30 分钟（未恢复则到现在）；按跨度选桶宽，行数不超过 24。"""
    end = min(alarm["ended_at"] + timedelta(minutes=30), now) if alarm.get("ended_at") else now
    start = alarm["ts"] - timedelta(hours=1)
    span = end - start
    if span <= timedelta(hours=6):
        bucket = "15 minutes"
    elif span <= timedelta(hours=12):
        bucket = "30 minutes"
    else:
        bucket, start = "1 hour", max(start, end - timedelta(hours=MAX_ROWS - 1))
    return start.astimezone(LOCAL_TZ), end.astimezone(LOCAL_TZ), bucket


def _round(expr: sql.Composable, digits: int = 3) -> sql.Composable:
    return sql.SQL("round(({})::numeric, {})::float8").format(expr, sql.Literal(digits))


def _local_text(expr: str) -> sql.Composable:
    return sql.SQL("to_char({} AT TIME ZONE 'Asia/Shanghai', 'MM-DD HH24:MI')").format(sql.SQL(expr))


async def _table(conn: AsyncConnection, record_id: str, question: str, query: sql.Composable, params: dict) -> ToolRecord:
    t0 = time.perf_counter()
    cur = await conn.execute(query, params)
    rows = await cur.fetchall()
    columns = [c.name for c in cur.description if c.name != "t"]   # 分桶时间戳只用于排序，展示用北京时间文本列
    values = [[r[c] for c in columns] for r in rows]
    record = ToolRecord(record_id, "sql_query", question, status="ok" if values else "empty",
                        output={"sql": query.as_string(conn), "params": {k: str(v) for k, v in params.items()},
                                "columns": columns, "rows": values[:MAX_ROWS], "truncated": len(values) > MAX_ROWS})
    record.total_ms = round((time.perf_counter() - t0) * 1000, 1)
    return record


def _metric_columns(metrics: list[str], primary: str | None) -> tuple[sql.Composable, list[str]]:
    parts, names = [], []
    for key in metrics:
        m = PLANT_METRICS[key]
        unit = f"({m.unit})" if m.unit else ""
        name = f"{m.label}均值{unit}"
        parts.append(sql.SQL("{} AS {}").format(_round(sql.SQL("avg({})").format(sql.Identifier(key))),
                                               sql.Identifier(name)))
        names.append(name)
        if key == primary:
            worst = "min" if key in LOWER_IS_BAD else "max"
            name = f"{m.label}{'最低' if worst == 'min' else '最高'}{unit}"
            parts.append(sql.SQL("{} AS {}").format(_round(sql.SQL(worst + "({})").format(sql.Identifier(key))),
                                                   sql.Identifier(name)))
            names.append(name)
    return sql.SQL(", ").join(parts), names


async def plant_series(conn, record_id, alarm, metrics, primary, start, end, bucket) -> ToolRecord:
    cols, _ = _metric_columns(metrics, primary)
    query = sql.SQL("""
        WITH b AS (
            SELECT time_bucket(%(bucket)s::interval, ts) AS t, {cols}
            FROM plant_reading WHERE plant_id = %(pid)s AND ts >= %(start)s AND ts < %(end)s GROUP BY 1
        ), w AS (
            SELECT time_bucket(%(bucket)s::interval, ts) AS t, max(rain_mm_h) AS rain
            FROM weather WHERE ts >= %(start)s AND ts < %(end)s GROUP BY 1
        )
        SELECT {t} AS "时间", b.*, {rain} AS "降雨强度最大(mm/h)"
        FROM b LEFT JOIN w USING (t) ORDER BY b.t""").format(
        cols=cols, t=_local_text("b.t"), rain=_round(sql.SQL("coalesce(w.rain, 0)"), 2))
    return await _table(conn, record_id, f"{alarm['plant_name']} {start:%m-%d %H:%M} 至 {end:%m-%d %H:%M} "
                        f"按 {bucket} 统计的相关工艺指标", query,
                        {"bucket": bucket, "pid": alarm["plant_id"], "start": start, "end": end})


async def load_series(conn, record_id, alarm, metrics, start, end, bucket) -> ToolRecord:
    """先按检测口径算5分钟负荷，再压缩展示；派生指标不当成原始列。"""
    metric = alarm["metric"]
    source = LOAD_SOURCES[metric]
    label = DERIVED_METRICS[metric].label
    raw = sql.SQL(", ").join(sql.SQL("avg({c}) AS {c}").format(c=sql.Identifier(k)) for k in metrics)
    cols, _ = _metric_columns(metrics, None)
    query = sql.SQL("""
        WITH five AS (
            SELECT time_bucket('5 minutes', ts) AS ts, {raw},
                   avg(inflow_m3h::float8 * {source} / 1000) FILTER
                   (WHERE inflow_m3h >= 0 AND {source} >= 0) AS load
            FROM plant_reading WHERE plant_id = %(pid)s AND ts >= %(start)s AND ts < %(end)s GROUP BY 1
        ), b AS (
            SELECT time_bucket(%(bucket)s::interval, ts) AS t, {cols},
                   {mean} AS {mean_name}, {peak} AS {peak_name} FROM five GROUP BY 1
        ) SELECT {t} AS "时间", b.* FROM b ORDER BY b.t
    """).format(raw=raw, source=sql.Identifier(source), cols=cols,
                mean=_round(sql.SQL("avg(load)")), peak=_round(sql.SQL("max(load)")),
                mean_name=sql.Identifier(label + "5分钟均值的时段均值(kg/h)"),
                peak_name=sql.Identifier(label + "5分钟均值最高值(kg/h)"), t=_local_text("b.t"))
    aligned_start = bucket_floor(start)
    if aligned_start < start:
        aligned_start += timedelta(minutes=5)
    return await _table(conn, record_id, f"{alarm['plant_name']}进水负荷及浓度/流量背景；按5分钟桶等权压缩展示", query,
                        {"pid": alarm["plant_id"], "start": aligned_start, "end": bucket_floor(end), "bucket": bucket})


async def device_series(conn, record_id, alarm, start, end, bucket) -> ToolRecord:
    query = sql.SQL("""
        WITH b AS (
            SELECT time_bucket(%(bucket)s::interval, ts) AS t,
                   {run} AS "运行占比(%%)", {vib} AS "最大振动(mm/s)", {temp} AS "最高轴承温度(℃)",
                   {cur} AS "运行平均电流(A)"
            FROM device_reading WHERE device_id = %(did)s AND ts >= %(start)s AND ts < %(end)s GROUP BY 1
        )
        SELECT {t} AS "时间", b.* FROM b ORDER BY b.t""").format(
        t=_local_text("b.t"),
        run=_round(sql.SQL("100 * avg(running::int)"), 1), vib=_round(sql.SQL("max(vibration_mms)"), 2),
        temp=_round(sql.SQL("max(temp_c)"), 1), cur=_round(sql.SQL("avg(current_a) FILTER (WHERE running)"), 1))
    return await _table(conn, record_id, f"{alarm['device_code']} {start:%m-%d %H:%M} 至 {end:%m-%d %H:%M} "
                        f"按 {bucket} 统计的运行、振动、温度与电流", query,
                        {"bucket": bucket, "did": alarm["device_id"], "start": start, "end": end})


async def related_alarms(conn, record_id, alarm, now) -> ToolRecord:
    lo, hi = alarm["ts"] - timedelta(hours=6), min(alarm["ts"] + timedelta(hours=6), now)
    query = sql.SQL("""
        SELECT a.id AS "告警ID", {ts} AS "开始", coalesce({end}, '持续中') AS "恢复", a.level AS "级别",
               a.kind AS "类型", coalesce(d.code, '') AS "设备", a.message AS "说明"
        FROM alarm a LEFT JOIN device d ON d.id = a.device_id
        WHERE a.plant_id = %(pid)s AND a.id <> %(id)s AND a.ts >= %(lo)s AND a.ts <= %(hi)s
        ORDER BY a.ts LIMIT 25""").format(ts=_local_text("a.ts"), end=_local_text("a.ended_at"))
    return await _table(conn, record_id, f"{alarm['plant_name']} 本告警前后 6 小时内的其他告警", query,
                        {"pid": alarm["plant_id"], "id": alarm["id"], "lo": lo, "hi": hi})


async def device_summary(conn, record_id, alarm, kinds, start, end) -> ToolRecord:
    query = sql.SQL("""
        SELECT d.code AS "设备", d.kind AS "类型", d.status AS "当前台账状态",
               {run} AS "窗口内运行占比(%%)", {vib} AS "最大振动(mm/s)", {temp} AS "最高轴承温度(℃)",
               coalesce({last}, '窗口内未运行') AS "最后运行时间"
        FROM device d LEFT JOIN device_reading r ON r.device_id = d.id AND r.ts >= %(start)s AND r.ts < %(end)s
        WHERE d.plant_id = %(pid)s AND d.kind = ANY(%(kinds)s)
        GROUP BY d.id, d.code, d.kind, d.status ORDER BY d.code""").format(
        run=_round(sql.SQL("100 * avg(r.running::int)"), 1), vib=_round(sql.SQL("max(r.vibration_mms)"), 2),
        temp=_round(sql.SQL("max(r.temp_c)"), 1), last=_local_text("max(r.ts) FILTER (WHERE r.running)"))
    return await _table(conn, record_id, f"{alarm['plant_name']} {'、'.join(kinds)} {start:%m-%d %H:%M} 至 "
                        f"{end:%m-%d %H:%M} 的运行汇总（含备用设备）", query,
                        {"pid": alarm["plant_id"], "kinds": kinds, "start": start, "end": end})


def knowledge_query(alarm: dict) -> str:
    if alarm.get("device_id"):
        return DEVICE_QUERY.get(alarm["device_kind"], f"{alarm['device_kind']} 故障 排查")
    return METRIC_QUERY.get(alarm["metric"], alarm["message"])


def question_for(alarm: dict, now: datetime) -> str:
    label, unit = metric_meta(alarm["metric"])
    unit = (alarm.get("detail") or {}).get("unit", unit)
    state = (f"已于 {alarm['ended_at'].astimezone(LOCAL_TZ):%m-%d %H:%M} 恢复" if alarm.get("ended_at")
             else f"截至 {now:%m-%d %H:%M} 仍在持续")
    peak = "" if alarm.get("peak") is None else f"，持续期间最不利值 {alarm['peak']:g} {unit}".rstrip()
    return (f"告警 #{alarm['id']}（{alarm['level']}，{alarm['kind']}）：{alarm['plant_name']}（{alarm['plant_code']}）"
            f"{alarm['message']}。开始于 {alarm['ts'].astimezone(LOCAL_TZ):%m-%d %H:%M}，{state}{peak}。"
            "请依据本轮证据说明数据显示的异常经过，给出待核查的原因假设和核查建议；"
            "不要写未经验证的操作参数，也不要声称已派单或已处置。")


class AlarmAdvisor:
    def __init__(self, agent: Agent, rag: RAG | None):
        self.agent, self.rag = agent, rag

    async def evidence(self, conn: AsyncConnection, alarm: dict, now: datetime) -> list[ToolRecord]:
        start, end, bucket = evidence_window(alarm, now)
        if alarm.get("device_id"):
            records = [await device_series(conn, "t1", alarm, start, end, bucket)]
            kinds = [alarm["device_kind"]]
        else:
            metrics = RELATED.get(alarm["metric"], [alarm["metric"]])
            if alarm["metric"] in DERIVED_METRICS:
                records = [await load_series(conn, "t1", alarm, metrics, start, end, bucket)]
            else:
                records = [await plant_series(conn, "t1", alarm, metrics, alarm["metric"], start, end, bucket)]
            kinds = DEVICE_KINDS.get(alarm["metric"], [])
        records.append(await related_alarms(conn, "t2", alarm, now))
        if kinds:
            records.append(await device_summary(conn, "t3", alarm, kinds, start, end))
        if self.rag is not None:
            rid = f"t{len(records) + 1}"
            t0 = time.perf_counter()
            query = knowledge_query(alarm)
            try:
                found = await self.rag.search(query, top_k=KNOWLEDGE_TOP_K, mode="hybrid", rerank=False)
                record = ToolRecord(rid, "rag_search", query, status="ok" if found.hits else "empty",
                                    output=found.to_dict())
            except Exception as exc:  # 检索失败时仍可只基于数据给出事实陈述
                record = ToolRecord(rid, "rag_search", query, status="error", error=f"检索失败（{type(exc).__name__}）")
            record.total_ms = round((time.perf_counter() - t0) * 1000, 1)
            records.append(record)
        # 追加而不改动既有t1～t4；工艺告警的时长/峰值由同一只读统计服务生成。
        statistics = getattr(self.agent, "statistics", None)
        if not alarm.get("device_id") and alarm["metric"] in PLANT_METRICS and statistics is not None:
            rid, t0 = f"t{len(records) + 1}", time.perf_counter()
            stats = ["instant_max", "instant_min", "bucket_mean_max", "bucket_mean_min"]
            if PLANT_METRICS[alarm["metric"]].limit is not None:
                stats += ["exceed_hours", "longest_exceed_hours"]
            record = ToolRecord(rid, "plant_statistics", "告警证据窗口内的服务端统计事实")
            try:
                query = StatisticsQuery(plant=alarm["plant_code"], metric=alarm["metric"], start=start, end=end,
                                        statistics=stats)
                record.output = await statistics.query(query, now)
                record.status = "ok" if record.output["facts"] else "empty"
            except Exception as exc:
                record.status, record.error = "error", f"统计取证失败（{type(exc).__name__}）"
            record.total_ms = round((time.perf_counter() - t0) * 1000, 1)
            records.append(record)
        return records

    async def advise(self, alarm: dict, now: datetime, records: list[ToolRecord]) -> dict:
        """records 来自 evidence()。分两步调用，模型生成期间不占用数据库连接。"""
        question = question_for(alarm, now)
        result = await self.agent.answer_with_evidence(question, now, records)
        return {
            "generated_at": now.isoformat(), "question": question,
            "ok": result.ok, "refused": result.refused, "reason": result.reason, "error": result.error,
            "answer": result.answer, "claims": result.claims, "citations": result.citations,
            "evidence": [r.to_dict() for r in records],
            "models": result.models, "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens, "llm_ms": result.llm_ms, "total_ms": result.total_ms,
            "validation_errors": result.validation_errors, "warnings": result.warnings,
        }

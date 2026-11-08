"""数据查询。列名只来自 domain.metrics 白名单，经 psycopg.sql.Identifier 转义后拼入 SQL。"""

from __future__ import annotations

from datetime import datetime, timedelta

from psycopg import AsyncConnection, sql

from app.domain.metrics import DEVICE_METRICS, EFFLUENT_KEYS, PLANT_METRICS, Metric, worst

STALE_AFTER = timedelta(minutes=10)

# (秒, PostgreSQL interval, 对外名称)
BUCKETS = [
    (5, "5 seconds", "5s"),
    (60, "1 minute", "1m"),
    (300, "5 minutes", "5m"),
    (900, "15 minutes", "15m"),
    (3600, "1 hour", "1h"),
    (21600, "6 hours", "6h"),
    (86400, "1 day", "1d"),
]
BUCKET_NAMES = [b[2] for b in BUCKETS]
MAX_POINTS = 720


def pick_bucket(start: datetime, end: datetime, name: str = "auto") -> tuple[str, str]:
    """返回 (PostgreSQL interval, 对外名称)。auto 时取点数不超过 MAX_POINTS 的最细粒度，且不细于 1 分钟。"""
    if name != "auto":
        for _, interval, n in BUCKETS:
            if n == name:
                return interval, n
        raise ValueError(f"bucket 须为 auto 或 {'/'.join(BUCKET_NAMES)}")
    span = (end - start).total_seconds()
    for seconds, interval, n in BUCKETS[1:]:
        if span / seconds <= MAX_POINTS:
            return interval, n
    return BUCKETS[-1][1], BUCKETS[-1][2]


def _compliant_expr() -> sql.Composable:
    """出水全部指标达标的布尔表达式（限值来自代码常量，不来自用户输入）。"""
    parts = [sql.SQL("{} <= {}").format(sql.Identifier(k), sql.Literal(PLANT_METRICS[k].limit)) for k in EFFLUENT_KEYS]
    return sql.SQL(" AND ").join(parts)


# ---------------------------------------------------------------- 厂站状态

def assess(latest: dict | None, faulty_devices: int, latest_ts: datetime | None, now: datetime) -> tuple[str, list[dict]]:
    """根据最新读数判断厂站状态：normal / warn / alarm / offline，并列出问题。"""
    if latest is None or latest_ts is None or now - latest_ts > STALE_AFTER:
        return "offline", [{"metric": None, "label": "数据中断", "level": "alarm", "value": None, "limit": None}]
    issues = []
    for key, m in PLANT_METRICS.items():
        level = m.level(latest.get(key))
        if level != "normal":
            issues.append({"metric": key, "label": m.label, "level": level,
                           "value": round(latest[key], 3), "limit": m.limit})
    if faulty_devices:
        issues.append({"metric": None, "label": f"{faulty_devices} 台设备故障", "level": "alarm",
                       "value": None, "limit": None})
    issues.sort(key=lambda i: i["level"] != "alarm")
    return worst(i["level"] for i in issues), issues


async def plants_with_latest(conn: AsyncConnection, now: datetime, code: str | None = None) -> list[dict]:
    cols = sql.SQL(", ").join(sql.SQL("r.{}").format(sql.Identifier(k)) for k in PLANT_METRICS)
    q = sql.SQL("""
        SELECT p.id, p.code, p.name, p.kind, p.process, p.capacity_m3d, p.lng, p.lat,
               r.ts AS latest_ts, {cols},
               (SELECT count(*) FROM device d WHERE d.plant_id = p.id AND d.status = '故障') AS faulty_devices
        FROM plant p
        LEFT JOIN LATERAL (
            SELECT * FROM plant_reading WHERE plant_id = p.id AND ts <= %(now)s ORDER BY ts DESC LIMIT 1
        ) r ON true
        WHERE %(code)s::text IS NULL OR p.code = %(code)s
        ORDER BY p.id
    """).format(cols=cols)
    rows = await (await conn.execute(q, {"now": now, "code": code})).fetchall()
    out = []
    for row in rows:
        latest = None if row["latest_ts"] is None else {k: row[k] for k in PLANT_METRICS}
        status, issues = assess(latest, row["faulty_devices"], row["latest_ts"], now)
        out.append({
            "id": row["id"], "code": row["code"], "name": row["name"], "kind": row["kind"],
            "process": row["process"], "capacity_m3d": row["capacity_m3d"], "lng": row["lng"], "lat": row["lat"],
            "status": status, "issues": issues, "latest_ts": row["latest_ts"],
            "latest": None if latest is None else {k: round(v, 3) for k, v in latest.items() if v is not None},
        })
    return out


# ---------------------------------------------------------------- 当日统计

async def period_stats(conn: AsyncConnection, start: datetime, end: datetime, plant_id: int | None = None) -> list[dict]:
    """按厂站统计时段内的水量、电耗、药耗与达标率。

    原始数据采样间隔不均（回填 1 分钟、实时 5 秒），先按 5 分钟分桶取均值，再对桶做时间加权，
    避免高频段权重过大。
    """
    q = sql.SQL("""
        WITH b AS (
            SELECT plant_id, time_bucket('5 minutes', ts) AS t,
                   avg(inflow_m3h) AS q, avg(blower_kw) AS kw,
                   avg(carbon_dose_kgh) AS carbon, avg(pac_dose_kgh) AS pac,
                   bool_and({ok}) AS ok
            FROM plant_reading
            WHERE ts >= %(start)s AND ts < %(end)s
              AND (%(pid)s::smallint IS NULL OR plant_id = %(pid)s)
            GROUP BY plant_id, t
        )
        SELECT plant_id, count(*) AS buckets,
               avg(q) AS q, avg(kw) AS kw, avg(carbon) AS carbon, avg(pac) AS pac,
               avg(ok::int) AS compliance
        FROM b GROUP BY plant_id ORDER BY plant_id
    """).format(ok=_compliant_expr())
    rows = await (await conn.execute(q, {"start": start, "end": end, "pid": plant_id})).fetchall()
    out = []
    for r in rows:
        hours = r["buckets"] * 5 / 60  # 有数据的时长
        water = r["q"] * hours
        out.append({
            "plant_id": r["plant_id"],
            "hours": round(hours, 2),
            "water_m3": round(water),
            "aeration_kwh": round(r["kw"] * hours, 1),
            "aeration_kwh_per_m3": round(r["kw"] / r["q"], 4) if r["q"] else None,
            "carbon_kg": round(r["carbon"] * hours, 1),
            "pac_kg": round(r["pac"] * hours, 1),
            "compliance_rate": round(float(r["compliance"]), 4),
        })
    return out


def summarize(stats: list[dict]) -> dict:
    water = sum(s["water_m3"] for s in stats)
    kwh = sum(s["aeration_kwh"] for s in stats)
    hours = sum(s["hours"] for s in stats)
    return {
        "water_m3": water,
        "aeration_kwh": round(kwh, 1),
        "aeration_kwh_per_m3": round(kwh / water, 4) if water else None,
        "carbon_kg": round(sum(s["carbon_kg"] for s in stats), 1),
        "pac_kg": round(sum(s["pac_kg"] for s in stats), 1),
        # 按运行时长加权的达标率
        "compliance_rate": round(sum(s["compliance_rate"] * s["hours"] for s in stats) / hours, 4) if hours else None,
    }


# ---------------------------------------------------------------- 时间序列

async def series(conn: AsyncConnection, table: str, id_col: str, entity_id: int, metrics: list[Metric],
                 start: datetime, end: datetime, interval: str) -> dict:
    if table not in ("plant_reading", "device_reading") or id_col not in ("plant_id", "device_id"):
        raise ValueError("非法的表名")
    aggs = sql.SQL(", ").join(
        sql.SQL("round(avg({c})::numeric, 4)::float8 AS {c}").format(c=sql.Identifier(m.key)) for m in metrics)
    q = sql.SQL("""
        SELECT time_bucket(%(interval)s::interval, ts, 'Asia/Shanghai') AS t, {aggs}
        FROM {table}
        WHERE {id_col} = %(id)s AND ts >= %(start)s AND ts < %(end)s
        GROUP BY t ORDER BY t
    """).format(aggs=aggs, table=sql.Identifier(table), id_col=sql.Identifier(id_col))
    rows = await (await conn.execute(q, {"interval": interval, "id": entity_id, "start": start, "end": end})).fetchall()
    return {
        "ts": [r["t"] for r in rows],
        "series": {m.key: [r[m.key] for r in rows] for m in metrics},
    }


async def hourly_totals(conn: AsyncConnection, start: datetime, end: datetime) -> list[dict]:
    """全区逐小时：总进水量、总曝气功率、降雨。用于总览趋势图。"""
    q = """
        WITH p AS (
            SELECT time_bucket('1 hour', ts, 'Asia/Shanghai') AS t, plant_id,
                   avg(inflow_m3h) AS q, avg(blower_kw) AS kw
            FROM plant_reading WHERE ts >= %(start)s AND ts < %(end)s
            GROUP BY t, plant_id
        ), w AS (
            SELECT time_bucket('1 hour', ts, 'Asia/Shanghai') AS t, avg(rain_mm_h) AS rain
            FROM weather WHERE ts >= %(start)s AND ts < %(end)s GROUP BY t
        )
        SELECT p.t, round(sum(p.q)) AS inflow_m3h, round(sum(p.kw)::numeric, 1)::float8 AS blower_kw,
               round(coalesce(max(w.rain), 0)::numeric, 2)::float8 AS rain_mm_h
        FROM p LEFT JOIN w USING (t) GROUP BY p.t ORDER BY p.t
    """
    rows = await (await conn.execute(q, {"start": start, "end": end})).fetchall()
    return [{"ts": r["t"], "inflow_m3h": float(r["inflow_m3h"]), "blower_kw": r["blower_kw"], "rain_mm_h": r["rain_mm_h"]}
            for r in rows]


async def devices_with_latest(conn: AsyncConnection, plant_id: int, now: datetime) -> list[dict]:
    q = """
        SELECT d.id, d.code, d.kind, d.model, d.rated_kw, d.installed_at, d.status,
               r.ts AS latest_ts, r.running, r.current_a, r.vibration_mms, r.temp_c
        FROM device d
        LEFT JOIN LATERAL (
            SELECT * FROM device_reading WHERE device_id = d.id AND ts <= %(now)s ORDER BY ts DESC LIMIT 1
        ) r ON true
        WHERE d.plant_id = %(pid)s ORDER BY d.id
    """
    rows = await (await conn.execute(q, {"pid": plant_id, "now": now})).fetchall()
    out = []
    for r in rows:
        levels = [DEVICE_METRICS[k].level(r[k]) for k in DEVICE_METRICS if r["running"]]
        if r["status"] == "故障":
            levels.append("alarm")
        out.append({**{k: r[k] for k in ("code", "kind", "model", "rated_kw", "installed_at", "status",
                                         "latest_ts", "running", "current_a", "vibration_mms", "temp_c")},
                    "level": worst(levels)})
    return out

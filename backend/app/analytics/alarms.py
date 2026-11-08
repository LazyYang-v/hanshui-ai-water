"""告警扫描：库内 5 分钟信号 → rules.detect → 与已有告警事件合并写库 → 发布变化。

同一事件反复扫描只会更新，不会重复插入：按（厂站、设备、规则、指标）和时间重叠匹配已有记录。
扫描窗口可能截断长事件的开头，因此匹配后保留原开始时间与触发参数，只更新恢复时间、峰值和级别。
有未恢复的告警时，窗口起点前移到其中最早的开始时间（replay_start），整段重放这些事件。
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
import uuid
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis

from app.analytics.rules import (
    DEFAULT_LOAD_CONFIG,
    DOSE_FEATURES,
    LOAD_METRICS,
    LOCAL_TZ,
    LOOKBACK,
    RULES,
    Baseline,
    DeviceSignals,
    Episode,
    LoadConfig,
    PlantSignals,
    Reference,
    detect,
    fit_dose_model,
    worse_level,
)
from app.deps import sim_now
from app.domain.metrics import EFFLUENT_KEYS

RULE_IDS = list(RULES)

log = logging.getLogger(__name__)

SIGNAL_COLS = [*EFFLUENT_KEYS, "do_aerobic_1", "inflow_m3h", "tn_in", "cod_in", "nh3n_in", "tp_in",
               "carbon_dose_kgh", "pac_dose_kgh"]
HOURLY_COLS = ["inflow_m3h", "tn_in", "cod_in", "nh3n_in", "tp_in", "carbon_dose_kgh", "pac_dose_kgh"]
BASELINE_SPAN = timedelta(days=7)
CH_ALARM = "alarm:events"
KEY_LOCK = "alarm:scan:lock"


def bucket_floor(t: datetime) -> datetime:
    return t - timedelta(minutes=t.minute % 5, seconds=t.second, microseconds=t.microsecond)


# ---------------------------------------------------------------- 取信号

def baselines(hourly: list[dict]) -> tuple[dict[int, float], dict]:
    """小时数据 → (北京时间钟点 → 进水中位数, 药剂 → 负荷模型)。纯函数，评测直接复用。"""
    by_hour: dict[int, list[float]] = {}
    for r in hourly:
        if r.get("inflow_m3h"):
            by_hour.setdefault(r["hour"].astimezone(LOCAL_TZ).hour, []).append(r["inflow_m3h"])
    inflow = {h: statistics.median(v) for h, v in by_hour.items()}
    models = {d: m for d in DOSE_FEATURES if (m := fit_dose_model(d, hourly)) is not None}
    return inflow, models


@dataclass
class SignalHistory:
    """SQL 聚合结果的只读快照。线上与回放共用切窗及基线生成，缓存只在这份快照内有效。"""
    plants: list[dict]
    devices: list[dict]
    rows: list[dict]
    hourly: list[dict]
    drows: list[dict]
    _cache: dict[tuple[int, datetime], Baseline] = field(default_factory=dict, init=False)

    def __post_init__(self):
        def group(rows, key, timestamp):
            out = {}
            for row in rows:
                out.setdefault(row[key], []).append(row)
            return {k: (sorted(v, key=lambda r: r[timestamp]), sorted(r[timestamp] for r in v)) for k, v in out.items()}
        self._plants = group(self.rows, "plant_id", "t")
        self._hourly = group(self.hourly, "plant_id", "hour")
        self._devices = group(self.drows, "device_id", "t")

    @staticmethod
    def _slice(grouped, key, lo, hi):
        rows, times = grouped.get(key, ([], []))
        return rows[bisect_left(times, lo):bisect_left(times, hi)]

    def baseline(self, pid: int, anchor: datetime) -> Baseline:
        key = pid, anchor
        if key not in self._cache:
            # 严格排除判定桶所在小时；过去异常可进入历史，由中位数/MAD和稳健拟合压低影响，不看场景标签。
            rows = self._slice(self._hourly, pid, anchor - BASELINE_SPAN, anchor)
            inflow, models = baselines(rows)
            groups = {}
            for r in rows:
                h = r["hour"].astimezone(LOCAL_TZ).hour
                for metric in [*LOAD_METRICS, *LOAD_METRICS.values()]:
                    value = r.get(metric)
                    if value is not None and value >= 0:
                        groups.setdefault((h, metric), []).append(value)
            refs = {}
            for k, values in groups.items():
                med = statistics.median(values)
                refs[k] = Reference(med, statistics.median(abs(v - med) for v in values), len(values))
            self._cache[key] = Baseline(anchor, inflow, models, refs)
        return self._cache[key]

    def window(self, start: datetime, end: datetime) -> tuple[list[PlantSignals], list[DeviceSignals]]:
        lo, hi = start - LOOKBACK, bucket_floor(end)
        signals = []
        for p in self.plants:
            rows = self._slice(self._plants, p["id"], lo, hi)
            anchors = {r["t"].replace(minute=0, second=0, microsecond=0) for r in rows}
            signals.append(PlantSignals(
                p["id"], p["code"], p["name"], [r["t"] for r in rows],
                {c: [r.get(c) for r in rows] for c in [*SIGNAL_COLS, *LOAD_METRICS.values()]},
                eval_from=start, baselines={a: self.baseline(p["id"], a) for a in anchors}))
        devices = []
        for d in self.devices:
            rows = self._slice(self._devices, d["id"], lo, hi)
            devices.append(DeviceSignals(
                d["id"], d["plant_id"], d["code"], d["kind"], d["status"] == "备用", [r["t"] for r in rows],
                [r["vib"] for r in rows], [r["temp"] for r in rows], [r["running"] for r in rows], eval_from=start))
        return signals, devices


def _aggregates(columns: list[str]):
    aggs = [sql.SQL("avg({c})::float8 AS {c}").format(c=sql.Identifier(c)) for c in columns]
    aggs += [sql.SQL("avg(inflow_m3h::float8 * {c} / 1000) FILTER "
                     "(WHERE inflow_m3h >= 0 AND {c} >= 0) AS {m}").format(
                         c=sql.Identifier(c), m=sql.Identifier(m)) for c, m in LOAD_METRICS.items()]
    return sql.SQL(", ").join(aggs)


async def load_history(conn: AsyncConnection, start: datetime, end: datetime) -> SignalHistory:
    """只读已完成桶；小时基线直接取原始数据，迟到数据无需等待连续聚合刷新。"""
    lo, cutoff = bucket_floor(start) - LOOKBACK, bucket_floor(end)
    hour_end = cutoff.replace(minute=0, second=0, microsecond=0)
    params = {"lo": lo, "cutoff": cutoff, "b": lo.replace(minute=0) - BASELINE_SPAN, "hour_end": hour_end}
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute("SELECT id, code, name FROM plant ORDER BY id")
        plants = await cur.fetchall()
        await cur.execute(sql.SQL("""
            SELECT plant_id, time_bucket('5 minutes', ts) AS t, {aggs}
            FROM plant_reading WHERE ts >= %(lo)s AND ts < %(cutoff)s
            GROUP BY 1, 2 ORDER BY 1, 2""").format(aggs=_aggregates(SIGNAL_COLS)), params)
        rows = await cur.fetchall()
        await cur.execute(sql.SQL("""
            SELECT plant_id, time_bucket('1 hour', ts) AS hour, {aggs}
            FROM plant_reading WHERE ts >= %(b)s AND ts < %(hour_end)s
            GROUP BY 1, 2 ORDER BY 1, 2""").format(aggs=_aggregates(HOURLY_COLS)), params)
        hourly = await cur.fetchall()
        await cur.execute("SELECT id, plant_id, code, kind, status FROM device ORDER BY id")
        devices = await cur.fetchall()
        await cur.execute("""
            SELECT device_id, time_bucket('5 minutes', ts) AS t,
                   max(vibration_mms) FILTER (WHERE running)::float8 AS vib,
                   max(temp_c) FILTER (WHERE running)::float8 AS temp, bool_or(running) AS running
            FROM device_reading WHERE ts >= %(lo)s AND ts < %(cutoff)s
            GROUP BY 1, 2 ORDER BY 1, 2""", params)
        drows = await cur.fetchall()
    return SignalHistory(plants, devices, rows, hourly, drows)


async def load_signals(conn: AsyncConnection, start: datetime, end: datetime) -> tuple[list[PlantSignals], list[DeviceSignals]]:
    return (await load_history(conn, start, end)).window(start, end)


def replay_start(start: datetime, end: datetime, open_starts: Iterable[datetime]) -> datetime:
    """扫描起点：前移到最早一条未恢复告警的开始时间，长期未恢复事件也保留完整状态。

    只看最近 3 小时时，窗口里可能已经没有这次事件的开头：设备跳停后振动长时间无数据，暴雨期间
    药耗暂停评估，或者条件时有时无。检不出事件时只能按数据覆盖判恢复，会提前判恢复（还会推送），
    或者把一次事件拆成两条。整段重放时，状态机从事件开头走起，与一次性全量检测的结果一致。
    """
    return min([start, *(t for t in open_starts if t < start)])


def coverage(plants: list[PlantSignals], devices: list[DeviceSignals]) -> dict[tuple[int, int], datetime]:
    """每个厂站/设备在本次窗口内最后一个有数据的桶。"""
    out = {(s.plant_id, 0): s.ts[-1] for s in plants if s.ts}
    out.update({(d.plant_id, d.device_id): d.ts[-1] for d in devices if d.ts})
    return out


# ---------------------------------------------------------------- 合并（纯函数）

@dataclass
class Existing:
    id: int
    plant_id: int
    device_id: int            # 0 表示厂站级
    rule: str
    metric: str
    ts: datetime
    ended_at: datetime | None
    level: str
    peak: float | None
    peak_at: datetime | None
    detail: dict

    @property
    def key(self) -> tuple:
        return self.plant_id, self.device_id, self.rule, self.metric


@dataclass
class Changes:
    inserts: list[Episode] = field(default_factory=list)
    updates: list[tuple[Existing, dict]] = field(default_factory=list)   # (原记录, 要更新的字段)
    recovered: list[Existing] = field(default_factory=list)              # 本次确认恢复的原记录


def _overlaps(a: Existing, e: Episode) -> bool:
    a_end = a.ended_at or datetime.max.replace(tzinfo=a.ts.tzinfo)
    e_end = e.end or datetime.max.replace(tzinfo=e.start.tzinfo)
    return a.ts <= e_end and e.start <= a_end


def _worse_peak(metric: str, a: float | None, b: float | None) -> bool:
    """b 是否比 a 更不利。"""
    if b is None:
        return False
    if a is None:
        return True
    return b < a if metric == "do_aerobic_1" else b > a


def plan(existing: list[Existing], episodes: list[Episode], covered: dict[tuple[int, int], datetime]) -> Changes:
    """每条已有告警最多对应一个事件。

    未恢复的已有告警与之后所有同类事件都“重叠”：停机后补扫时，同一窗口里可能既有它的恢复、
    又有同类新事件，若允许重复匹配，新事件会被并进旧告警而丢失。因此按开始时间处理事件，
    已匹配的告警不再参与；有多个候选时取开始最晚的一条（持续事件的自然延续）。
    """
    changes = Changes()
    matched: set[int] = set()
    by_key: dict[tuple, list[Existing]] = {}
    for a in existing:
        by_key.setdefault(a.key, []).append(a)
    for e in sorted(episodes, key=lambda e: e.start):
        a = max((x for x in by_key.get(e.key, []) if x.id not in matched and _overlaps(x, e)),
                key=lambda x: x.ts, default=None)
        if a is None:
            changes.inserts.append(e)
            continue
        matched.add(a.id)
        upd: dict = {}
        # 恢复时间只会推后或重新打开，不因截断窗口提前
        reopen = e.end is None and a.ended_at is not None
        later = e.end is not None and (a.ended_at is None or e.end > a.ended_at)
        if reopen or later:
            upd["ended_at"] = e.end
        if _worse_peak(e.metric, a.peak, e.peak):
            upd["peak"], upd["peak_at"] = e.peak, e.peak_at
        level = worse_level(a.level, e.level)
        if level != a.level:
            upd["level"] = level
        last_seen = a.detail.get("last_seen")
        if last_seen is None or e.last_seen.isoformat() > last_seen:
            upd["detail"] = {**a.detail, "last_seen": e.last_seen.isoformat()}
        if upd:
            changes.updates.append((a, upd))
            if a.ended_at is None and upd.get("ended_at") is not None:
                changes.recovered.append(a)
    # 未匹配开放事件保持开放。厂站/设备有后续数据不代表本规则有有效正常值，
    # 恢复只能来自 detect 状态机实际数够 off 个正常桶，缺失和暂停评估均不算恢复。
    return changes


# ---------------------------------------------------------------- 写库

def _alarm_dict(row: dict) -> dict:
    return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in row.items() if k != "advice"}


async def persist(conn: AsyncConnection, episodes: list[Episode], window_start: datetime, now: datetime,
                  covered: dict[tuple[int, int], datetime]) -> tuple[list[dict], Changes]:
    async with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        rows = await (await cur.execute("""
            SELECT id, plant_id, coalesce(device_id, 0) AS device_id, rule, metric, ts, ended_at, level,
                   peak, peak_at, detail
            FROM alarm
            WHERE rule = ANY(%(rules)s) AND ts <= %(now)s AND (ended_at IS NULL OR ended_at >= %(ws)s)
            FOR UPDATE""", {"now": now, "ws": window_start - LOOKBACK, "rules": RULE_IDS})).fetchall()
        changes = plan([Existing(**r) for r in rows], episodes, covered)
        events = []
        for e in changes.inserts:
            row = await (await cur.execute("""
                INSERT INTO alarm (ts, plant_id, device_id, level, kind, metric, value, threshold, message, status,
                                   rule, ended_at, peak, peak_at, detail, updated_at)
                VALUES (%(ts)s, %(plant_id)s, %(device_id)s, %(level)s, %(kind)s, %(metric)s, %(value)s,
                        %(threshold)s, %(message)s, '未处理', %(rule)s, %(ended_at)s, %(peak)s, %(peak_at)s,
                        %(detail)s, %(now)s)
                ON CONFLICT (plant_id, (coalesce(device_id, 0)), rule, metric, ts) DO NOTHING
                RETURNING *""", {
                "ts": e.start, "plant_id": e.plant_id, "device_id": e.device_id, "level": e.level, "kind": e.kind,
                "metric": e.metric, "value": e.value, "threshold": e.threshold, "message": e.message,
                "rule": e.rule, "ended_at": e.end, "peak": e.peak, "peak_at": e.peak_at,
                "detail": Jsonb({**e.detail, "last_seen": e.last_seen.isoformat(), "detected_at": now.isoformat()}),
                "now": now})).fetchone()
            if row:
                events.append({"event": "new", "alarm": _alarm_dict(row)})
        for a, upd in changes.updates:
            sets = sql.SQL(", ").join(sql.SQL("{} = {}").format(sql.Identifier(k), sql.Placeholder(k)) for k in upd)
            params = {k: Jsonb(v) if k == "detail" else v for k, v in upd.items()}
            row = await (await cur.execute(
                sql.SQL("UPDATE alarm SET {sets}, updated_at = %(now)s WHERE id = %(id)s RETURNING *").format(sets=sets),
                {**params, "now": now, "id": a.id})).fetchone()
            if "level" in upd:
                events.append({"event": "escalated", "alarm": _alarm_dict(row)})
            elif a in changes.recovered:
                events.append({"event": "recovered", "alarm": _alarm_dict(row)})
    return events, changes


@dataclass
class ScanStats:
    start: str
    end: str
    episodes: int
    inserted: int
    updated: int
    recovered: int
    ms: float


async def scan(conn: AsyncConnection, start: datetime, end: datetime, *,
               history: SignalHistory | None = None,
               load_config: LoadConfig = DEFAULT_LOAD_CONFIG) -> tuple[ScanStats, list[dict]]:
    """扫描 [start, end)；有未恢复的告警时起点前移到其开始时间，统计里的 start 是实际起点。"""
    t0 = time.perf_counter()
    async with conn.cursor(row_factory=dict_row) as cur:
        # 只看本模块的规则：视觉巡检告警（app/vision）由图片驱动，不参与信号补扫
        await cur.execute("SELECT ts FROM alarm WHERE rule = ANY(%s) AND ended_at IS NULL AND ts < %s",
                          (RULE_IDS, start))
        opened = await cur.fetchall()
    start = replay_start(start, end, [r["ts"] for r in opened])
    plants, devices = history.window(start, end) if history is not None else await load_signals(conn, start, end)
    # 回放只替换只读信号来源；实际检测、合并、写库与线上共用。快照缓存不跨数据库可见性变化。
    episodes = await asyncio.to_thread(detect, plants, devices, load_config=load_config)
    events, changes = await persist(conn, episodes, start, end, coverage(plants, devices))
    stats = ScanStats(start.isoformat(), end.isoformat(), len(episodes), len(changes.inserts),
                      len(changes.updates), len(changes.recovered), round((time.perf_counter() - t0) * 1000, 1))
    return stats, events


# ---------------------------------------------------------------- 后台巡检

class AlarmMonitor:
    """API 进程内的周期扫描。首轮补扫最近 7 天，之后每 interval_s 秒扫描最近 3 小时。"""

    def __init__(self, pool: AsyncConnectionPool, redis: Redis, *, interval_s: float = 10.0,
                 window: timedelta = timedelta(hours=3), history: timedelta = BASELINE_SPAN):
        self.pool, self.redis = pool, redis
        self.interval_s, self.window, self.history = interval_s, window, history
        self.last: ScanStats | None = None
        self.last_error: str | None = None
        self.scans = 0
        self.running = False

    async def scan_once(self, span: timedelta | None = None, *, quiet: bool = False) -> ScanStats | None:
        """扫描 [现在 - span, 现在)。quiet：补扫的历史事件不逐条推送，免得前端把旧告警当成新告警弹出。"""
        token = uuid.uuid4().hex
        if not await self.redis.set(KEY_LOCK, token, nx=True, px=120_000):
            return None  # 其他实例正在扫描
        try:
            async with self.pool.connection() as conn:
                now = await sim_now(self.redis, conn)
                stats, events = await scan(conn, now - (span or self.window), now)
            if quiet:
                events = [{"event": "backfill", "inserted": stats.inserted, "updated": stats.updated}] if events else []
            for ev in events:
                await self.redis.publish(CH_ALARM, json.dumps({"type": "alarm", **ev}, ensure_ascii=False))
            self.last, self.last_error = stats, None
            self.scans += 1
            return stats
        finally:
            if await self.redis.get(KEY_LOCK) == token:
                await self.redis.delete(KEY_LOCK)

    async def run(self) -> None:
        first = True
        self.running = True
        try:
            while True:
                try:
                    stats = await self.scan_once(self.history if first else None, quiet=first)
                    if first and stats:
                        log.info("告警补扫完成：%d 个事件，新增 %d，用时 %.0f ms", stats.episodes, stats.inserted, stats.ms)
                    first = first and stats is None
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # 扫描失败不影响 API，下轮重试
                    self.last_error = f"{type(exc).__name__}: {exc}"[:300]
                    log.warning("告警扫描失败：%s", self.last_error)
                await asyncio.sleep(self.interval_s)
        finally:
            self.running = False

    def status(self) -> dict:
        return {"running": self.running, "scans": self.scans, "interval_s": self.interval_s,
                "window_h": self.window.total_seconds() / 3600,
                "last": None if self.last is None else self.last.__dict__, "last_error": self.last_error}

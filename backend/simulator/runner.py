"""把仿真器接到 PostgreSQL（TimescaleDB）和 Redis。

- backfill：生成过去 N 天的历史数据（含预置历史事件），供大屏和 Text2SQL 使用
- live：按「演示时钟」实时推进，每步写库并发布到 Redis
  - 演示时钟可以加速（speed=60 即 1 秒走 1 分钟），加速后时钟会领先墙上时间，
    下游（API、Text2SQL）统一以 Redis 中的 sim:clock 作为「当前时间」
- 控制：订阅 Redis 频道 sim:control，接收 inject / clear / speed 命令
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import psycopg
import redis

from .engine import DeviceConfig, Simulator, Snapshot, plant_config
from .scenarios import DEFAULT_DURATION_MIN, Scenario

log = logging.getLogger("simulator")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://shuiwu:shuiwu@localhost:15432/shuiwu")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:16379/0")
LOCAL = ZoneInfo("Asia/Shanghai")

CH_CONTROL = "sim:control"
CH_TICK = "realtime:tick"
CH_EVENTS = "sim:events"
KEY_CLOCK = "sim:clock"
KEY_SPEED = "sim:speed"
KEY_SCENARIOS = "sim:scenarios"
KEY_LATEST = "realtime:latest"

BASE_TICK_S = 5.0          # 1 倍速时每 5 秒一步
DEVICE_EVERY_S = 30.0      # 设备数据每 30 秒（仿真时间）写一次

PLANT_COLS = ["ts", "plant_id", "inflow_m3h", "cod_in", "nh3n_in", "tn_in", "tp_in", "cod_out", "nh3n_out",
              "tn_out", "tp_out", "ss_out", "do_aerobic_1", "do_aerobic_2", "air_flow_m3h", "blower_kw",
              "carbon_dose_kgh", "pac_dose_kgh", "mlss"]
DEVICE_COLS = ["ts", "device_id", "running", "current_a", "vibration_mms", "temp_c"]


# ---------------------------------------------------------------- 加载与写库

def load_simulator(conn: psycopg.Connection, seed: int) -> tuple[Simulator, dict[str, int]]:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        plants = cur.execute("SELECT id, code, kind, process, capacity_m3d FROM plant ORDER BY id").fetchall()
        devices = cur.execute("SELECT id, plant_id, code, kind, rated_kw, status FROM device ORDER BY id").fetchall()
    devs = [DeviceConfig(d["id"], d["plant_id"], d["code"], d["kind"], d["rated_kw"], d["status"] == "备用")
            for d in devices]
    sim = Simulator([plant_config(p) for p in plants], devs, seed=seed)
    return sim, {p["code"]: p["id"] for p in plants}


class Writer:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn
        self.plant_rows: list[tuple] = []
        self.device_rows: list[tuple] = []
        self.weather_rows: list[tuple] = []
        self.status: dict[int, str] = {}
        self._last_device_ts: datetime | None = None

    def add(self, snap: Snapshot) -> None:
        for pid, r in snap.plants.items():
            self.plant_rows.append((snap.ts, pid, *(round(r[c], 4) for c in PLANT_COLS[2:])))
        self.weather_rows.append((snap.ts, round(snap.weather["rain_mm_h"], 2), round(snap.weather["temp_c"], 1)))
        if self._last_device_ts is None or (snap.ts - self._last_device_ts).total_seconds() >= DEVICE_EVERY_S:
            self._last_device_ts = snap.ts
            self.device_rows.extend(tuple(d[c] for c in DEVICE_COLS) for d in snap.devices)
        self.status.update(snap.device_status)

    def flush(self) -> None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            for table, cols, rows in (("plant_reading", PLANT_COLS, self.plant_rows),
                                      ("device_reading", DEVICE_COLS, self.device_rows),
                                      ("weather", ["ts", "rain_mm_h", "temp_c"], self.weather_rows)):
                if rows:
                    with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN") as cp:
                        for row in rows:
                            cp.write_row(row)
            for device_id, status in self.status.items():
                cur.execute("UPDATE device SET status = %s WHERE id = %s", (status, device_id))
        self.plant_rows.clear()
        self.device_rows.clear()
        self.weather_rows.clear()
        self.status.clear()


def refresh_aggregates(conn: psycopg.Connection) -> None:
    """连续聚合刷新不能在事务内执行，conn 须为 autocommit 连接。"""
    conn.execute("CALL refresh_continuous_aggregate('plant_reading_1h', NULL, NULL)")


# ---------------------------------------------------------------- 历史回填

def history_scenarios(end: datetime, days: int, seed: int) -> list[Scenario]:
    """预置的历史事件，让「上周发生了什么」类问题有答案。时间相对回填终点。"""
    def ago(d: float, hour: int) -> datetime:
        day = (end - timedelta(days=d)).astimezone(LOCAL)
        return day.replace(hour=hour, minute=0, second=0, microsecond=0)

    rng = np.random.default_rng([seed, 999])
    items = [
        Scenario("storm", ago(5, 14), 150, params={"peak_mm_h": 14.0}),
        Scenario("pipe_leak", ago(3.5, 6), 1080, "P05"),
        Scenario("dosing_block", ago(2, 9), 240, "P02"),
        Scenario("blower_fault", ago(1, 15), 120, "P06"),
    ]
    # 零星小雨：平均每 3 天一场
    for d in range(days):
        if rng.random() < 1 / 3:
            items.append(Scenario("storm", ago(d + 0.5, int(rng.integers(0, 24))), float(rng.integers(60, 240)),
                                  params={"peak_mm_h": round(float(rng.uniform(1.5, 5.0)), 1)}))
    start = end - timedelta(days=days)
    return [sc for sc in items if start <= sc.start < end]


def backfill(conn: psycopg.Connection, sim: Simulator, end: datetime, days: int, step_s: float, seed: int) -> None:
    start = end - timedelta(days=days)
    for sc in history_scenarios(end, days, seed):
        sim.book.add(sc)
        log.info("历史事件 %s %s %s", sc.start.astimezone(LOCAL).strftime("%m-%d %H:%M"), sc.kind, sc.plant_code or "")
    writer = Writer(conn)
    n = int((end - start).total_seconds() // step_s)
    for i in range(n):
        writer.add(sim.step(start + timedelta(seconds=i * step_s), step_s))
        if (i + 1) % 1440 == 0:
            writer.flush()
            log.info("回填进度 %d/%d", i + 1, n)
    writer.flush()
    sim.book.prune(end)
    refresh_aggregates(conn)
    log.info("回填完成：%s → %s，共 %d 步", start, end, n)


# ---------------------------------------------------------------- 实时运行

def _publish(r: redis.Redis, sim: Simulator, snap: Snapshot, code_of: dict[int, str]) -> None:
    payload = {
        "ts": snap.ts.astimezone(LOCAL).isoformat(),
        "weather": snap.weather,
        "plants": {code_of[pid]: {k: round(v, 3) for k, v in row.items()} for pid, row in snap.plants.items()},
        "devices": [{k: v for k, v in d.items() if k != "ts"} for d in snap.devices],
        "scenarios": [sc.to_dict() for sc in sim.book.active(snap.ts)],
    }
    msg = json.dumps(payload, ensure_ascii=False)
    pipe = r.pipeline()
    pipe.set(KEY_CLOCK, payload["ts"])
    pipe.set(KEY_LATEST, msg)
    pipe.set(KEY_SCENARIOS, json.dumps(payload["scenarios"], ensure_ascii=False))
    pipe.publish(CH_TICK, msg)
    pipe.execute()


def _reply(r: redis.Redis, cmd: dict, ok: bool, message: str) -> None:
    """回复带上请求 id，调用方据此匹配自己的命令。"""
    r.publish(CH_EVENTS, json.dumps({"id": cmd.get("id"), "ok": ok, "message": message}, ensure_ascii=False))


def handle_command(sim: Simulator, cmd: dict, clock: datetime) -> str:
    action = cmd.get("action")
    if action == "inject":
        kind = cmd["kind"]
        sc = sim.book.add(Scenario(
            kind=kind,
            start=clock,
            duration_min=float(cmd.get("duration_min") or DEFAULT_DURATION_MIN[kind]),
            plant_code=cmd.get("plant"),
            params=cmd.get("params") or {},
        ))
        return f"已注入 {sc.kind} {sc.plant_code or '全区'} 至 {sc.end.astimezone(LOCAL):%H:%M}"
    if action == "clear":
        n = sim.book.clear(plant_code=cmd.get("plant"), kind=cmd.get("kind"), at=clock)
        return f"已结束 {n} 个场景"
    raise ValueError(f"未知命令 {cmd!r}")


def live(conn: psycopg.Connection, r: redis.Redis, sim: Simulator, code_by_id: dict[int, str],
         start: datetime, speed: float) -> None:
    pubsub = r.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(CH_CONTROL)
    r.set(KEY_SPEED, speed)
    writer = Writer(conn)
    clock = start
    next_wall = time.monotonic()
    log.info("实时运行：演示时钟 %s，%g 倍速", clock.astimezone(LOCAL).strftime("%Y-%m-%d %H:%M:%S"), speed)
    while True:
        tick_wall = max(1.0, BASE_TICK_S / speed)
        dt = tick_wall * speed
        # 落后墙上时间（如刚回填完）时用大步长追上，单步最多 60 秒
        lag = (datetime.now(timezone.utc) - clock).total_seconds()
        if lag > dt:
            dt = min(lag, 60.0)
        snap = sim.step(clock, dt)
        writer.add(snap)
        writer.flush()
        _publish(r, sim, snap, code_by_id)
        clock += timedelta(seconds=dt)
        sim.book.prune(clock - timedelta(hours=1))

        next_wall += tick_wall
        while (remaining := next_wall - time.monotonic()) > 0:
            msg = pubsub.get_message(timeout=remaining)
            if not msg:
                continue
            cmd: dict = {}
            try:
                cmd = json.loads(msg["data"])
                if cmd.get("action") == "speed":
                    speed = max(0.1, min(600.0, float(cmd["value"])))
                    r.set(KEY_SPEED, speed)
                    result = f"演示时钟 {speed:g} 倍速"
                else:
                    result = handle_command(sim, cmd, clock)
                    # 先写回场景列表再回复，调用方收到回复后立即查询就能看到变化，不必等下一拍
                    r.set(KEY_SCENARIOS, json.dumps([sc.to_dict() for sc in sim.book.active(clock)], ensure_ascii=False))
                log.info(result)
                _reply(r, cmd, True, result)
            except Exception as e:  # 控制命令出错不能让仿真停下
                log.warning("命令处理失败：%s", e)
                _reply(r, cmd, False, str(e))


def reset(conn: psycopg.Connection, r: redis.Redis) -> None:
    """清空运行数据（保留厂站和设备台账），之后重新回填。"""
    with conn.transaction():
        conn.execute("TRUNCATE plant_reading, device_reading, weather, alarm, work_order, report RESTART IDENTITY CASCADE")
        conn.execute("UPDATE device SET status = '运行' WHERE status IN ('故障', '检修')")
    # TRUNCATE 不会使连续聚合失效，需要手动清掉物化数据
    conn.execute("CALL refresh_continuous_aggregate('plant_reading_1h', NULL, NULL)")
    r.delete(KEY_CLOCK, KEY_LATEST, KEY_SCENARIOS, KEY_SPEED)
    log.info("已清空运行数据")


def run(backfill_days: int, backfill_step: float, speed: float, seed: int, do_reset: bool = False) -> None:
    conn = psycopg.connect(DATABASE_URL, autocommit=True)  # 写入由 Writer 显式开事务
    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    if do_reset:
        reset(conn, r)
    sim, id_by_code = load_simulator(conn, seed)
    code_by_id = {v: k for k, v in id_by_code.items()}

    now = datetime.now(timezone.utc).replace(microsecond=0)
    last = conn.execute("SELECT max(ts) FROM plant_reading").fetchone()[0]
    if last is None and backfill_days > 0:
        backfill(conn, sim, now, backfill_days, backfill_step, seed)
        start = now
    else:
        # 续跑：时钟从上次写到的位置继续（演示加速过则可能领先墙上时间）；先预热 3 小时让状态收敛
        start = max(now, last + timedelta(seconds=BASE_TICK_S)) if last else now
        for i in range(180):
            sim.step(start - timedelta(minutes=180 - i), 60)
    live(conn, r, sim, code_by_id, start, speed)

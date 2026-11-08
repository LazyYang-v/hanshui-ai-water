"""因果告警回放：同一 SignalHistory.window → scan → detect → persist/plan。

uv run python -m evals.alarms.run --label causal-60s --seeds 3 --step 60 --tune
uv run python -m evals.alarms.run --label causal-5s --seeds 1 --step 5

每个判断桶只使用该整点之前7天的完整小时。全程实际写独立评测库的 alarm，
检出延迟使用首次写入的扫描时刻，包含停机补扫的等待。快照预载仅节省重复SQL聚合，
按前缀切窗；每种子另与生产原始SQL取数路径逐字段对照。迟到/缺失用实际增量写库验证。
既有 step60s/step5s 结果保留，新结果必须使用未占用的 label。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import psycopg
from psycopg.rows import dict_row

from app.analytics.alarms import BASELINE_SPAN, SignalHistory, load_history, load_signals, scan
from app.analytics.rules import BUCKET, DEFAULT_LOAD_CONFIG, LOCAL_TZ, LoadConfig, detect, load_episodes
from simulator import runner
from simulator.scenarios import Scenario

HERE = Path(__file__).parent
ADMIN_URL = os.getenv("EVAL_ADMIN_DATABASE_URL", "postgresql://shuiwu:shuiwu@localhost:15432/shuiwu")
EVAL_DB = "shuiwu_alarm_eval"
INIT_DIR = Path(__file__).resolve().parents[2] / "db" / "init"
HISTORY_DAYS, EVAL_DAYS = 7, 6
T0 = datetime(2026, 10, 1, tzinfo=LOCAL_TZ)
EVAL_START = T0 + timedelta(days=HISTORY_DAYS)
EVAL_END = EVAL_START + timedelta(days=EVAL_DAYS)
TAIL = timedelta(hours=6)
SCAN_SPAN = timedelta(hours=3)
OUTAGE = (EVAL_START + timedelta(days=1, hours=10), EVAL_START + timedelta(days=1, hours=16))


@dataclass
class Case:
    day: int
    hour: int
    kind: str
    plant: str | None
    minutes: int
    params: dict = field(default_factory=dict)
    negative: bool = False

    @property
    def start(self):
        return EVAL_START + timedelta(days=self.day, hours=self.hour)

    @property
    def end(self):
        return self.start + timedelta(minutes=self.minutes)

    @property
    def label(self):
        return ("小雨（阴性）" if self.negative else self.kind) + f" {self.plant or '全区'}"


SCHEDULE = [
    Case(0, 2, "shock_load", "P04", 180),
    Case(0, 9, "dosing_block", "P02", 360),
    Case(0, 15, "blower_fault", "P06", 240),
    Case(1, 6, "pipe_leak", "P05", 600),
    Case(2, 3, "blower_fault", "P01", 120),
    Case(2, 10, "dosing_block", "P03", 240),
    Case(2, 19, "shock_load", "P07", 180),
    Case(3, 14, "storm", None, 150, {"peak_mm_h": 14.0}),
    Case(4, 10, "storm", None, 120, {"peak_mm_h": 3.0}, negative=True),
]
EXPECTED = {
    "shock_load": {("effluent_exceed", None), ("effluent_trend", None), ("inflow_load", None)},
    "dosing_block": {("dosing_anomaly", "pac_dose_kgh"), ("effluent_trend", "tp_out"), ("effluent_exceed", "tp_out")},
    "blower_fault": {("device_vibration", None), ("device_temp", None), ("device_stop", None)},
    "pipe_leak": {("dosing_anomaly", "carbon_dose_kgh")},
    "storm": {("inflow_surge", None)},
}


def db_url():
    return ADMIN_URL.rsplit("/", 1)[0] + "/" + EVAL_DB


def build(seed: int, step_eval: float):
    with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {EVAL_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {EVAL_DB}")
    with psycopg.connect(db_url(), autocommit=True) as conn:
        for f in sorted(INIT_DIR.glob("*.sql")):
            conn.execute(f.read_text())
        sim, codes = runner.load_simulator(conn, seed=seed)
        rng = np.random.default_rng([seed, 999])
        for d in range(HISTORY_DAYS):
            if rng.random() < 1 / 3:
                sim.book.add(Scenario("storm", T0 + timedelta(days=d, hours=int(rng.integers(0, 20))),
                                      float(rng.integers(60, 240)), params={"peak_mm_h": round(float(rng.uniform(1.5, 5)), 1)}))
        for c in SCHEDULE:
            sim.book.add(Scenario(c.kind, c.start, c.minutes, c.plant, dict(c.params)))
        writer, t, n = runner.Writer(conn), T0, 0
        while t < EVAL_END:
            dt = 60.0 if t < EVAL_START else step_eval
            writer.add(sim.step(t, dt))
            t += timedelta(seconds=dt)
            n += 1
            if n % 5000 == 0:
                writer.flush()
        writer.flush()
    return codes


def expected(rule, metric, kind):
    return any(r == rule and (m is None or m == metric) for r, m in EXPECTED[kind])


def classify(a, codes):
    inside = [c for c in SCHEDULE if (c.plant is None or codes[c.plant] == a["plant_id"])
              and c.start <= a["ts"] <= c.end + TAIL]
    if any(not c.negative and expected(a["rule"], a["metric"], c.kind) for c in inside):
        return "hit"
    if any(not c.negative for c in inside):
        return "secondary"
    return "negative" if inside else "false"


def detection(a):
    return datetime.fromisoformat(a["detail"]["detected_at"])


def case_results(rows, codes):
    out = []
    for c in SCHEDULE:
        mine = [a for a in rows if (c.plant is None or codes[c.plant] == a["plant_id"])
                and c.start <= a["ts"] <= c.end + TAIL]
        hits = [a for a in mine if not c.negative and expected(a["rule"], a["metric"], c.kind)]
        first = min(hits, key=detection, default=None)
        effluent = [a for a in mine if a["rule"] in ("effluent_exceed", "effluent_trend")]
        load = [a for a in mine if a["rule"] == "inflow_load"]
        first_load = min(load, key=detection, default=None)
        effluent_at = min((detection(a) for a in effluent), default=None)
        out.append({"case": c.label, "kind": c.kind, "negative": c.negative, "start": c.start.isoformat(),
                    "detected": first is not None, "first_rule": first["rule"] + ":" + first["metric"] if first else None,
                    "latency_min": (detection(first) - c.start).total_seconds() / 60 if first else None,
                    "load_detected": bool(load), "effluent_detected": bool(effluent),
                    "load_latency_min": (detection(first_load) - c.start).total_seconds() / 60 if first_load else None,
                    "load_lead_vs_effluent_min": (effluent_at - detection(first_load)).total_seconds() / 60
                    if c.kind == "shock_load" and c.plant is not None and effluent_at and first_load else None,
                    "other_rules": sorted({a["rule"] + ":" + a["metric"] for a in mine
                                           if c.negative or not expected(a["rule"], a["metric"], c.kind)})})
    return out


def consistency(episodes, rows):
    full = {(*e.key, e.start): e for e in episodes}
    inc = {(a["plant_id"], a["device_id"] or 0, a["rule"], a["metric"], a["ts"]): a for a in rows}
    diffs = []
    for key in full.keys() & inc.keys():
        e, a = full[key], inc[key]
        bad = {}
        for field, want in (("ended_at", e.end), ("level", e.level), ("peak", e.peak), ("peak_at", e.peak_at)):
            got = a[field]
            same = abs(got - want) < 1e-3 if field == "peak" and got is not None and want is not None else got == want
            if not same:
                bad[field] = {"full": str(want), "incremental": str(got)}
        if bad:
            diffs.append({"key": str(key), **bad})
    return {"full": len(full), "incremental": len(inc), "missing": [str(k) for k in full.keys() - inc.keys()],
            "extra": [str(k) for k in inc.keys() - full.keys()], "field_diffs": diffs,
            "peak_tolerance": 0.001}  # alarm.peak 为 REAL；纯函数保留4位小数


def tune_load(history, codes):
    """种子1/2用于参数比较，种子3保留为验证；所有候选使用同一因果历史。"""
    plants, _ = history.window(EVAL_START, EVAL_END)
    out = []
    for ratio in (1.4, 1.5, 1.6):
        for on in (2, 3, 4):
            cfg = LoadConfig(ratio=ratio, on=on)
            eps = [e for p in plants for e in load_episodes(p, cfg)]
            rows = [{"plant_id": e.plant_id, "rule": e.rule, "metric": e.metric, "ts": e.start,
                     "detail": {"detected_at": e.detail["condition_confirmed_at"]}} for e in eps]
            shocks = [c for c in SCHEDULE if c.kind == "shock_load"]
            delays = []
            for c in shocks:
                hits = [a for a in rows if a["plant_id"] == codes[c.plant] and c.start <= a["ts"] <= c.end]
                if hits:
                    delays.append((min(detection(a) for a in hits) - c.start).total_seconds() / 60)
            out.append({**asdict(cfg), "detected": len(delays), "cases": len(shocks),
                        "latencies": delays, "false": sum(classify(a, codes) in ("false", "negative") for a in rows),
                        "rain_alerts": sum(any(c.kind == "storm" and c.start <= a["ts"] <= c.end + TAIL
                                                for c in SCHEDULE) for a in rows)})
    return out


async def resilience(conn):
    """有真实写入时序的 SQL 路径验证，不使用预载快照，不接触演示库。"""
    t = datetime(2028, 1, 8, tzinfo=LOCAL_TZ)
    checks = {}
    async with conn.transaction(force_rollback=True):
        for offset in (0, 1):
            await conn.execute("INSERT INTO plant_reading(ts,plant_id,do_aerobic_1) VALUES (%s,1,0.5)", (t + offset * BUCKET,))
        _, events = await scan(conn, t, t + 2 * BUCKET)
        a = next(e["alarm"] for e in events if e["alarm"]["rule"] == "do_low")
        await conn.execute("INSERT INTO plant_reading(ts,plant_id,inflow_m3h) VALUES (%s,1,100)", (t + 4 * BUCKET,))
        stats, _ = await scan(conn, t, t + 5 * BUCKET)
        checks["missing_is_not_recovery"] = stats.recovered == 0
        # 扫描暂停6小时后补扫，始终同一事件，只在3个有效正常桶后恢复。
        for offset in (72, 73, 74):
            await conn.execute("INSERT INTO plant_reading(ts,plant_id,do_aerobic_1) VALUES (%s,1,2)", (t + offset * BUCKET,))
        _, events = await scan(conn, t, t + 75 * BUCKET)
        rec = [e["alarm"] for e in events if e["event"] == "recovered" and e["alarm"]["id"] == a["id"]]
        checks["outage_catchup"] = len(rec) == 1 and rec[0]["ended_at"] == (t + 72 * BUCKET).isoformat()
        late = t + timedelta(days=1)
        await conn.execute("INSERT INTO plant_reading(ts,plant_id,do_aerobic_1) VALUES (%s,1,0.5)", (late,))
        stats, _ = await scan(conn, late, late + 2 * BUCKET)
        assert stats.inserted == 0
        await conn.execute("INSERT INTO plant_reading(ts,plant_id,do_aerobic_1) VALUES (%s,1,0.5)", (late + BUCKET,))
        _, events = await scan(conn, late, late + 6 * BUCKET)
        row = next(e["alarm"] for e in events if e["event"] == "new" and e["alarm"]["ts"] == late.isoformat())
        checks["late_report_actual_detection"] = row["detail"]["detected_at"] == (late + 6 * BUCKET).isoformat()
        # 已存储了未来采样，也不应在它的桶完成以前触发。
        future = late + timedelta(hours=1)
        for offset in (0, 1):
            await conn.execute("INSERT INTO plant_reading(ts,plant_id,do_aerobic_1) VALUES (%s,2,0.5)", (future + offset * BUCKET,))
        stats, _ = await scan(conn, late, late + 7 * BUCKET)
        checks["future_samples_hidden"] = stats.inserted == 0
        old = t + timedelta(days=20)
        for offset in (0, 1):
            await conn.execute("INSERT INTO device_reading(ts,device_id,running,vibration_mms) VALUES (%s,21,true,5)", (old + offset * BUCKET,))
        await scan(conn, old, old + 2 * BUCKET)
        now = old + timedelta(days=9)
        await conn.execute("INSERT INTO device_reading(ts,device_id,running) VALUES (%s,21,false)", (now,))
        _, events = await scan(conn, now - SCAN_SPAN, now + BUCKET)
        checks["over_7_days_stays_open"] = not any(e["event"] == "recovered" for e in events)
        for offset in (1, 2, 3):
            await conn.execute("INSERT INTO device_reading(ts,device_id,running,vibration_mms) VALUES (%s,21,true,3)", (now + offset * BUCKET,))
        _, events = await scan(conn, now - SCAN_SPAN, now + 4 * BUCKET)
        rec = [e["alarm"] for e in events if e["event"] == "recovered" and e["alarm"]["rule"] == "device_vibration"]
        checks["over_7_days_recovers_from_normal_values"] = len(rec) == 1 and rec[0]["ended_at"] == (now + BUCKET).isoformat()
    assert all(checks.values()), checks
    return checks


def equivalent(a, b):
    """PostgreSQL 聚合顺序可能产生浮点尾差，其他类型仍精确比较。"""
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-9
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    return a == b


async def replay(codes, *, tune=False):
    t0 = time.perf_counter()
    async with await psycopg.AsyncConnection.connect(db_url(), autocommit=True, row_factory=dict_row) as conn:
        history = await load_history(conn, T0, EVAL_END)
        ms, revisions, scans, source_checks = [], [], 0, []
        t, resumed = EVAL_START + BUCKET, False
        check_at = {EVAL_START + timedelta(hours=h) for h in (3, 12, 24, 40, 96, 144)}
        while t <= EVAL_END:
            if OUTAGE[0] <= t < OUTAGE[1]:
                t += BUCKET
                continue
            span = BASELINE_SPAN if scans == 0 else SCAN_SPAN
            if t >= OUTAGE[1] and not resumed:
                span, resumed = BASELINE_SPAN, True
            # 复用生产完整扫描/事务合并，仅注入按时刻切窗的只读聚合快照。
            stats, events = await scan(conn, max(T0, t - span), t, history=history)
            # 恢复后又有更新或重开，由逐扫描的结束时间快照判断。
            current = await (await conn.execute("SELECT id,ended_at FROM alarm WHERE rule IS NOT NULL")).fetchall()
            if scans:
                for a in current:
                    if previous.get(a["id"]) is not None and previous[a["id"]] != a["ended_at"]:
                        revisions.append({"id": a["id"], "at": t.isoformat(), "before": previous[a["id"]], "after": a["ended_at"]})
            previous = {a["id"]: a["ended_at"] for a in current}
            ms.append(stats.ms)
            scans += 1
            if t in check_at:
                start = datetime.fromisoformat(stats.start)
                direct = [asdict(e) for e in detect(*await load_signals(conn, start, t))]
                cached = [asdict(e) for e in detect(*history.window(start, t))]
                assert equivalent(direct, cached), f"SQL/快照差异 {t}"
                source_checks.append(t.isoformat())
            if scans % 300 == 0:
                print(f"已完成 {scans} 次实际扫描，演示时钟 {t.isoformat()}", flush=True)
            t += BUCKET
        rows = await (await conn.execute("SELECT * FROM alarm WHERE rule IS NOT NULL ORDER BY ts,id")).fetchall()
        full = detect(*history.window(T0, EVAL_END))
        comparison = consistency(full, rows)
        for a in rows:
            a["class"] = classify(a, codes)
            a["plant"] = next(c for c, pid in codes.items() if pid == a["plant_id"])
        cases = case_results(rows, codes)
        tuning = tune_load(history, codes) if tune else []
        robustness = await resilience(conn)
    passed = not any(comparison[k] for k in ("missing", "extra", "field_diffs")) and not revisions
    return {"passed": passed, "cases": cases, "alarms": rows, "load_tuning": tuning,
            "consistency": {**comparison, "revisions": revisions, "scans": scans,
                            "scan_ms_median": round(statistics.median(ms), 1), "scan_ms_max": max(ms),
                            "sql_source_checks": source_checks}, "resilience": robustness,
            "timing_s": round(time.perf_counter() - t0, 2)}


def report(results, label, step):
    out = [f"# 告警因果回放：{label}", "", f"模拟器评测步长 {step:g} 秒；默认负荷参数 {asdict(DEFAULT_LOAD_CONFIG)}。",
           "每种子7天历史+6天评测，8个异常场景与1场阴性小雨；按5分钟推进演示时钟，实际调用生产scan/持久化，含6小时停机。",
           "基线对每个桶取其整点以前7天完整小时；不使用场景标签剔除训练数据。预载SQL聚合通过6个时刻与生产取数逐字段对照。",
           "检出延迟=实际首次写入扫描时刻−场景开始；缺失可能延缓确认。kg/h按同采样乘积聚合，浓度和负荷同时升高才预警。", "",
           "| 种子 | 场景 | 检出 | 首个规则 | 实际延迟(分钟) | 负荷预警较出水提前(分钟) |", "|---|---|---|---|---:|---:|"]
    for r in results:
        for c in r["cases"]:
            out.append(f"| {r['seed']} | {c['case']} | {c['detected']} | {c['first_rule'] or '—'} | {c['latency_min']} | {c['load_lead_vs_effluent_min']} |")
    out += ["", "| 种子 | 检出 | 误报事件 | 全量/增量 | 缺失/多出/字段差异 | 恢复改写 | 扫描次数 | 扫描中位/最大ms |", "|---|---|---:|---|---|---:|---:|---|"]
    for r in results:
        c = r["consistency"]
        false = sum(a["class"] in ("false", "negative") for a in r["alarms"])
        out.append(f"| {r['seed']} | {sum(c['detected'] for c in r['cases'] if not c['negative'])}/8 | {false} | {c['full']}/{c['incremental']} | "
                   f"{len(c['missing'])}/{len(c['extra'])}/{len(c['field_diffs'])} | {len(c['revisions'])} | {c['scans']} | {c['scan_ms_median']}/{c['scan_ms_max']} |")
    out += ["", "## 负荷参数对照", "", "只在60秒种子1/2比较参数，种子3作为保留验证；候选规则仍走同一因果基线与状态机。参数对照的延迟是条件确认时间，不含扫描停机。",
            "", "| 种子 | 负荷倍数 | 持续桶 | 冲击检出 | 延迟(分钟) | 无场景/阴性误报 | 雨期负荷事件 |", "|---|---:|---:|---|---|---:|---:|"]
    for r in results:
        for x in r["load_tuning"]:
            out.append(f"| {r['seed']} | {x['ratio']} | {x['on']} | {x['detected']}/{x['cases']} | {x['latencies']} | {x['false']} | {x['rain_alerts']} |")
    out += ["", "## 真实写入时序回归", ""]
    for r in results:
        out.append(f"- 种子{r['seed']}：{r['resilience']}。")
    out += ["", "## 限制", "", "- 全部来自模拟器。标签只用于评测归因；场景前后的6小时伴随窗口可能使其他异常不计为误报，不等同真实业务精确率。",
            "- 连续桶是连续有效判断桶，缺失不计数也不中断。迟到数据可能改写历史条件，detected_at保留实际首次发现时刻；普通补扫最近3小时，超窗迟到需手动补扫，已有开放事件会延长窗口。",
            "- 开放事件超过7天仍从开头重放，去掉coverage兜底；需保留原始历史，长事件增加计算开销。初次启动补扫仍为7天。",
            "- 全量/增量一致性只针对本次按时采样；迟到写入另做SQL回归。alarm.peak为REAL，一致性容差0.001，其余字段精确比较。",
            "- 60秒与5秒AR(1)噪声的相关时间不同，分开报告；三个相关污染物分别产生事件，不按事件数冒充独立场景数。", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--step", type=float, default=60)
    ap.add_argument("--tune", action="store_true")
    a = ap.parse_args()
    if not a.label or any(not (c.isalnum() or c in "-_") for c in a.label) or not 1 <= a.seeds <= 10 or a.step not in (5, 60):
        ap.error("label仅字母数字下划线连字符；seeds须1~10；step须5或60")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    target = out / a.label
    if target.with_suffix(".json").exists() or target.with_suffix(".md").exists():
        ap.error("结果标签已存在，请另选标签")
    results = []
    for seed in range(1, a.seeds + 1):
        t0 = time.perf_counter()
        codes = build(seed, a.step)
        print(f"seed {seed}: 造数完成 {time.perf_counter() - t0:.1f}s", flush=True)
        r = asyncio.run(replay(codes, tune=a.tune and seed <= 2 and a.step == 60))
        r.update(seed=seed, step_s=a.step, load_config=asdict(DEFAULT_LOAD_CONFIG))
        results.append(r)
        # 每个种子完成即保存，失败记录也保留，不用旧标签覆盖。
        target.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str))
        target.with_suffix(".md").write_text(report(results, a.label, a.step))
        print(f"seed {seed}: passed={r['passed']} 检出 {sum(c['detected'] for c in r['cases'] if not c['negative'])}/8，"
              f"误报 {sum(a['class'] in ('false','negative') for a in r['alarms'])}，扫描 {r['consistency']['scans']} 次", flush=True)
    if not all(r["passed"] for r in results):
        raise SystemExit(1)
    print(f"结果：{target}.md")


if __name__ == "__main__":
    main()

"""告警规则与事件合并的单元测试：纯函数，不需要数据库。"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from app.analytics.alarms import (
    Existing,
    baselines,
    bucket_floor,
    coverage,
    plan,
    replay_start,
)
from app.analytics.rules import (
    BUCKET,
    LOOKBACK,
    STOP,
    DeviceSignals,
    DoseModel,
    Episode,
    PlantSignals,
    device_episodes,
    fit_dose_model,
    fmt_vs,
    plant_episodes,
    runs,
    trend_state,
)

TZ = ZoneInfo("Asia/Shanghai")
T0 = datetime(2026, 11, 5, 8, 0, tzinfo=TZ)
T, F, N = True, False, None


def ts(n: int) -> list[datetime]:
    return [T0 + i * BUCKET for i in range(n)]


# ---------------------------------------------------------------- 连续计数

@pytest.mark.parametrize("conds, on, off, expected", [
    ([T, T, T, F, F, F], 2, 3, [(0, 2, 3)]),
    ([T, F, T, F, T, F], 2, 3, []),                       # 从未连续满足 on 个桶
    ([T, N, T, F, N, F, F], 2, 3, [(0, 2, 3)]),           # 无数据的桶不计入，也不打断
    ([T, T, F, T, F, F, F], 2, 3, [(0, 3, 4)]),           # 短暂回落不算恢复
    ([F, T, T, T], 2, 3, [(1, 3, None)]),                 # 到最后仍在持续
    ([T, T, STOP, T, T], 2, 3, [(0, 1, 2), (3, 4, None)]),  # 趋势预警升级为超标时立即结束
    ([T, T, F, F, F, T, T, F, F, F], 2, 3, [(0, 1, 2), (5, 6, 7)]),
])
def test_runs(conds, on, off, expected):
    assert runs(conds, on, off) == expected


# ---------------------------------------------------------------- 趋势预警

def test_trend_rising_toward_limit():
    vals = [3.0 + 0.1 * k for k in range(13)]           # 每 5 分钟 +0.1，当前 4.2，限值 5
    state, detail = trend_state(ts(13), vals, 12, 5.0)
    assert state is True
    assert detail["eta_min"] == pytest.approx(40, abs=0.5)
    assert detail["slope_per_h"] == pytest.approx(1.2, abs=0.01)
    assert detail["r2"] == pytest.approx(1.0)


@pytest.mark.parametrize("vals, i, expected", [
    ([4.0] * 13, 12, False),                           # 平稳不预警
    ([3.0 + 0.01 * k for k in range(13)], 12, False),  # 上升太慢，2 小时内不会超标
    ([1.0 + 0.05 * k for k in range(13)], 12, False),  # 远低于限值
    ([4.0 + 0.1 * k for k in range(13)], 12, STOP),    # 已经超标，交给超标规则
    ([3.0 + 0.1 * k for k in range(13)], 6, None),     # 历史不足 40 分钟，不判断
])
def test_trend_negative_cases(vals, i, expected):
    state, _ = trend_state(ts(len(vals)), vals, i, 5.0)
    assert state == expected


def test_trend_noisy_series_needs_fit_quality():
    rng = np.random.default_rng(0)
    vals = list(3.5 + rng.normal(0, 0.6, 13))           # 无趋势的大噪声：R² 低
    vals[-1] = 4.0
    state, _ = trend_state(ts(13), vals, 12, 5.0)
    assert state is not True


# ---------------------------------------------------------------- 药耗负荷模型

def _hourly(n: int, rng, outliers: int = 0) -> list[dict]:
    rows = []
    for k in range(n):
        q = 1000 + 300 * np.sin(k / 3.8)
        tn, cod, tp = 40 + 5 * np.sin(k / 2.1), 250 + 40 * np.cos(k / 5.0), 4 + 0.5 * np.sin(k / 1.7)
        carbon = 0.012 * q * tn - 0.0012 * q * cod + 0.01 * q
        pac = 0.02 * q * tp - 0.0001 * q * cod + 0.002 * q
        rows.append({"hour": T0 + timedelta(hours=k), "inflow_m3h": q, "tn_in": tn, "cod_in": cod, "tp_in": tp,
                     "carbon_dose_kgh": carbon * (1 + rng.normal(0, 0.01)), "pac_dose_kgh": pac})
    for k in range(outliers):           # 渗漏等异常时段：实际投加量翻倍
        rows[k * 7]["carbon_dose_kgh"] *= 2
    return rows


def test_fit_dose_model_recovers_coefficients_despite_outliers():
    rows = _hourly(72, np.random.default_rng(1), outliers=5)
    m = fit_dose_model("carbon_dose_kgh", rows)
    assert m is not None and m.features == ("tn_in", "cod_in")
    assert m.kept == pytest.approx(67 / 72, abs=0.02)   # 5 个离群小时被剔除
    assert m.r2 > 0.95
    # 三项特征都含进水量、相互共线，单个系数辨识不准；检验的是预测值
    for r in rows[1:7] + rows[8:14]:
        clean = 0.012 * r["inflow_m3h"] * r["tn_in"] - 0.0012 * r["inflow_m3h"] * r["cod_in"] + 0.01 * r["inflow_m3h"]
        assert m.predict(r["inflow_m3h"], r) == pytest.approx(clean, rel=0.03)
    q, x = 1000, {"tn_in": 40, "cod_in": 250}
    assert m.predict(None, x) is None and m.predict(q, {"tn_in": 40, "cod_in": None}) is None


def test_fit_dose_model_needs_a_day_of_data():
    assert fit_dose_model("pac_dose_kgh", _hourly(23, np.random.default_rng(2))) is None
    assert fit_dose_model("pac_dose_kgh", _hourly(24, np.random.default_rng(2))) is not None


def test_baselines_inflow_median_by_local_hour():
    rows = _hourly(48, np.random.default_rng(3))
    inflow, models = baselines(rows)
    h = rows[5]["hour"].astimezone(TZ).hour
    assert inflow[h] == pytest.approx(np.median([rows[5]["inflow_m3h"], rows[29]["inflow_m3h"]]))
    assert set(models) == {"carbon_dose_kgh", "pac_dose_kgh"}


CARBON = DoseModel("carbon_dose_kgh", ("tn_in", "cod_in"), (0.01, -0.001, 0.0), 1.0, 48, 1.0, 150.0)


def _dosing_plant(n: int, dose, q=1000.0, tn=40.0, median_pred=150.0) -> PlantSignals:
    t = ts(n)
    vals = {"inflow_m3h": [q] * n, "tn_in": [tn] * n, "cod_in": [250.0] * n, "carbon_dose_kgh": dose}
    model = DoseModel(CARBON.drug, CARBON.features, CARBON.coef, 1.0, 48, 1.0, median_pred)
    return PlantSignals(1, "P01", "测试厂", t, vals, inflow_baseline=dict.fromkeys(range(24), 1000.0),
                        dose_model={"carbon_dose_kgh": model})


def test_dosing_anomaly_detected_against_load_model():
    dose = [150.0] * 18 + [300.0] * 18                  # 后 90 分钟投加量翻倍，进水负荷不变
    eps = [e for e in plant_episodes(_dosing_plant(36, dose)) if e.rule == "dosing_anomaly"]
    assert len(eps) == 1
    e = eps[0]
    assert e.metric == "carbon_dose_kgh" and e.end is None and e.threshold == 1.3
    assert e.detail["expected_kgh"] == pytest.approx(150.0)
    assert e.detail["ratio"] > 1.3 and e.detail["model"]["terms"][0] == "进水量×进水总氮"


def test_dosing_not_evaluated_during_inflow_surge():
    dose = [450.0] * 36                                 # 预期 240，比值约 1.9
    eps = plant_episodes(_dosing_plant(36, dose, q=1600.0))
    assert [e.rule for e in eps] == ["inflow_surge"]


def test_dosing_not_evaluated_when_expected_load_is_tiny():
    # 预期 150，低于常态中位数 600 的 30%：如暴雨稀释，比值会被小分母放大
    eps = plant_episodes(_dosing_plant(36, [400.0] * 36, median_pred=600.0))
    assert not [e for e in eps if e.rule == "dosing_anomaly"]


# ---------------------------------------------------------------- 设备规则

def _device(kind="鼓风机", standby=False, vib=None, temp=None, running=None, n=12) -> DeviceSignals:
    return DeviceSignals(7, 1, "P01-BL-1", kind, standby, ts(n), vib or [3.0] * n, temp or [60.0] * n,
                         running or [True] * n)


def test_vibration_level_follows_peak():
    eps = device_episodes(_device(vib=[3.0, 5.0, 5.2, 5.5, 3.0, 3.0, 3.0, 3.0, 5.0, 7.5, 8.0, 3.0]))
    vib = [e for e in eps if e.rule == "device_vibration"]
    assert [(e.level, e.peak) for e in vib] == [("预警", 5.5), ("告警", 8.0)]


def test_stop_rule_skips_standby_and_sludge_pumps():
    running = [True] * 4 + [False] * 8
    assert [e.rule for e in device_episodes(_device(running=running))] == ["device_stop"]
    assert device_episodes(_device(running=running, standby=True)) == []
    assert device_episodes(_device(kind="污泥泵", running=running)) == []


# ---------------------------------------------------------------- 回看段只作历史

def test_lookback_buckets_do_not_start_episodes_or_set_peaks():
    vib = [8.0] * 4 + [5.0, 5.2] + [3.0] * 6
    d = _device(vib=vib)
    d.eval_from = d.ts[4]
    [e] = [e for e in device_episodes(d) if e.rule == "device_vibration"]
    assert (e.start, e.level, e.peak, e.end) == (d.ts[4], "预警", 5.2, d.ts[6])   # 回看段的 8.0 不算
    d.eval_from = d.ts[6]                               # 只有回看段满足条件：没有事件
    assert not [e for e in device_episodes(d) if e.rule == "device_vibration"]


def test_dosing_ignores_partial_windows_in_lookback():
    # 序列开头的 60 分钟滑动均值只有 9~11 个点。增量扫描若评估这一段，残缺窗口的值会混进峰值，与全量检测不一致
    s = _dosing_plant(36, [450.0] * 9 + [300.0] * 27)
    [naive] = [e for e in plant_episodes(s) if e.rule == "dosing_anomaly"]
    assert (naive.start, naive.peak) == (s.ts[8], 3.0)
    s.eval_from = s.ts[12]                              # 之后每个桶的窗口都是完整的 12 个点
    [e] = [e for e in plant_episodes(s) if e.rule == "dosing_anomaly"]
    assert e.start == s.ts[12]
    assert e.peak == pytest.approx((8 * 450 + 4 * 300) / 12 / 150, abs=1e-4)


# ---------------------------------------------------------------- 增量扫描：未恢复告警整段重放

def test_replay_start_moves_back_to_oldest_open_alarm_even_after_seven_days():
    now = T0 + timedelta(days=10)
    start = now - timedelta(hours=3)
    assert replay_start(start, now, []) == start
    assert replay_start(start, now, [now - timedelta(hours=1)]) == start      # 窗口内开始的不需要前移
    assert replay_start(start, now, [now - timedelta(hours=5), now - timedelta(hours=9)]) == now - timedelta(hours=9)
    assert replay_start(start, now, [now - timedelta(days=8)]) == now - timedelta(days=8)


def _window(d: DeviceSignals, start: datetime, end: datetime) -> DeviceSignals:
    """与 load_signals(start, end) 相同的切片：[start-60 分钟, end)，回看段只作历史。"""
    keep = [i for i, t in enumerate(d.ts) if start - LOOKBACK <= t < end]
    pick = lambda xs: [xs[i] for i in keep]   # noqa: E731
    return DeviceSignals(d.device_id, d.plant_id, d.code, d.kind, d.standby, pick(d.ts), pick(d.vibration),
                         pick(d.temp), pick(d.running), eval_from=start)


def _scan_like_monitor(d: DeviceSignals, replay: bool) -> tuple[list[Existing], list[tuple[str, datetime]]]:
    """每写满一个桶扫描一次最近 3 小时，用 plan() 合并；返回告警表与推送“已恢复”的时刻。"""
    rows: dict[int, Existing] = {}
    recovered = []
    for k in range(13, len(d.ts) + 1):
        now = d.ts[k - 1] + BUCKET
        start = now - timedelta(hours=3)
        if replay:
            start = replay_start(start, now, [a.ts for a in rows.values() if a.ended_at is None])
        win = _window(d, start, now)
        ch = plan(list(rows.values()), device_episodes(win), coverage([], [win]))
        for e in ch.inserts:
            rows[len(rows) + 1] = Existing(len(rows) + 1, e.plant_id, e.device_id or 0, e.rule, e.metric, e.start,
                                           e.end, e.level, e.peak, e.peak_at, {"last_seen": e.last_seen.isoformat()})
        for a, upd in ch.updates:
            for f, v in upd.items():
                setattr(rows[a.id], f, v)
        recovered += [(a.rule, now) for a in ch.recovered]
    return list(rows.values()), recovered


def test_open_alarm_is_replayed_across_long_data_gap():
    # 振动超标后风机跳停 3.5 小时（不运行时没有振动读数），之后重启、振动正常
    vib = [3.0] * 15 + [5.0] * 3 + [None] * 42 + [3.0] * 36
    running = [v is not None for v in vib]
    d = _device(vib=vib, temp=[60.0 if r else None for r in running], running=running, n=len(vib))
    [full] = [e for e in device_episodes(d) if e.rule == "device_vibration"]
    assert (full.start, full.end) == (d.ts[15], d.ts[60])          # 全量检测：无数据不打断，重启后才恢复

    rows, recovered = _scan_like_monitor(d, replay=True)
    [a] = [a for a in rows if a.rule == "device_vibration"]
    assert (a.ts, a.ended_at) == (full.start, full.end)
    assert [t for r, t in recovered if r == "device_vibration"] == [d.ts[63]]   # 连续 3 个正常桶写满后才推送恢复

    # 只看最近 3 小时会丢失状态，但不再靠 coverage 假定恢复；重放才可正常关闭。
    rows, recovered = _scan_like_monitor(d, replay=False)
    [a] = [a for a in rows if a.rule == "device_vibration"]
    assert a.ended_at is None and not [t for r, t in recovered if r == "device_vibration"]


@pytest.mark.parametrize("v, ref, digits, expected", [
    (0.5004, 0.5, 2, "0.5004"),     # 刚过限值：补位到能与限值区分
    (0.51, 0.5, 2, "0.51"),
    (0.9996, 1.0, 2, "0.9996"),     # 低于下限的 DO
    (75.04, 75, 1, "75.04"),
    (0.50001, 0.5, 2, "0.5"),       # 最多补到 4 位
    (None, 0.5, 2, "—"),
])
def test_fmt_vs_distinguishes_value_from_threshold(v, ref, digits, expected):
    assert fmt_vs(v, ref, digits) == expected


def test_temperature_alarm_message():
    eps = device_episodes(_device(temp=[60.0] * 4 + [78.0, 81.5, 82.0] + [60.0] * 5))
    (e,) = eps
    assert e.rule == "device_temp" and e.level == "告警" and e.peak == 82.0
    assert e.message == "P01-BL-1 轴承温度 78 ℃，超过关注值 75 ℃"


# ---------------------------------------------------------------- 合并

def _episode(rule="effluent_exceed", start=T0, end=None, level="告警", peak=0.8, metric="tp_out",
             last_seen=None) -> Episode:
    return Episode(rule, 1, metric, start, end, level, 0.6, 0.5, peak, start, last_seen or (end or start) - BUCKET,
                   "出水总磷超标")


def _existing(id=1, rule="effluent_exceed", ts=T0, ended_at=None, level="告警", peak=0.7, metric="tp_out",
              last_seen=None) -> Existing:
    return Existing(id, 1, 0, rule, metric, ts, ended_at, level, peak, ts,
                    {"last_seen": (last_seen or ts).isoformat()})


def test_plan_inserts_new_and_updates_overlapping():
    a = _existing(peak=0.7)
    later = _episode(start=T0 + timedelta(hours=5))                  # 不重叠：新事件
    same = _episode(start=T0 + BUCKET, end=T0 + timedelta(hours=1), peak=0.9)   # 截断窗口里的同一事件
    ch = plan([a], [same, later], {})
    assert ch.inserts == [later]
    ((orig, upd),) = ch.updates
    assert orig is a and upd["ended_at"] == T0 + timedelta(hours=1) and upd["peak"] == 0.9
    assert ch.recovered == [a]


def test_plan_prefers_latest_overlapping_alarm():
    old = _existing(id=1, ended_at=T0 + timedelta(hours=1))
    cur = _existing(id=2, ts=T0 + timedelta(hours=2))
    e = _episode(start=T0 + timedelta(minutes=30), peak=0.9)   # 窗口内看成一段持续事件
    ch = plan([old, cur], [e], {})
    assert [(a.id, upd) for a, upd in ch.updates] == [(2, {"peak": 0.9, "peak_at": e.peak_at})]   # 不重开旧告警
    assert ch.inserts == [] and ch.recovered == []


def test_plan_reopens_and_escalates():
    a = _existing(rule="device_vibration", metric="vibration_mms", ended_at=T0 + timedelta(minutes=30),
                  level="预警", peak=5.0)
    e = _episode(rule="device_vibration", metric="vibration_mms", start=T0, end=None, level="告警", peak=7.5)
    ((_, upd),) = plan([a], [e], {}).updates
    assert upd["ended_at"] is None and upd["level"] == "告警" and upd["peak"] == 7.5


def test_plan_never_moves_recovery_earlier():
    a = _existing(ended_at=T0 + timedelta(hours=2), last_seen=T0 + timedelta(hours=2) - BUCKET)
    e = _episode(start=T0 + timedelta(hours=1), end=T0 + timedelta(hours=1, minutes=30), peak=0.6,
                 last_seen=T0 + timedelta(hours=1))
    assert plan([a], [e], {}).updates == []


def test_plan_lower_do_is_worse():
    a = _existing(rule="do_low", metric="do_aerobic_1", peak=0.8)
    e = _episode(rule="do_low", metric="do_aerobic_1", peak=0.5)
    ((_, upd),) = plan([a], [e], {}).updates
    assert upd["peak"] == 0.5


def test_plan_requires_rule_conditions_instead_of_coverage_to_recover():
    seen = T0 + timedelta(minutes=40)
    a = _existing(last_seen=seen)
    assert plan([a], [], {(1, 0): seen + 2 * BUCKET}).updates == []
    ch = plan([a], [], {(1, 0): seen + 30 * BUCKET})
    assert ch.updates == [] and ch.recovered == []


def test_bucket_floor():
    t = datetime(2026, 11, 5, 9, 37, 42, 123, tzinfo=TZ)
    assert bucket_floor(t) == datetime(2026, 11, 5, 9, 35, tzinfo=TZ)

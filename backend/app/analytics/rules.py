"""告警规则：5 分钟信号序列 → 告警事件（episode）。

纯函数，不访问数据库；生产扫描（app/analytics/alarms.py）与离线评测（evals/alarms）共用。
一次持续异常只产生一个事件：连续 on 个桶满足条件才开始，连续 off 个桶不满足才算恢复，避免抖动；
无数据的桶不计数也不打断。eval_from 之前的桶只作滑动窗口和趋势拟合的历史，不判定。
限值来自 app/domain/metrics.py；预警阈值属于 Demo 运行管理口径，不是排放标准。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial
from zoneinfo import ZoneInfo

import numpy as np

from app.domain.metrics import DEVICE_METRICS, EFFLUENT_KEYS, LOAD_SOURCES, PLANT_METRICS

LOCAL_TZ = ZoneInfo("Asia/Shanghai")
BUCKET = timedelta(minutes=5)
LOOKBACK = timedelta(minutes=60)       # 趋势与药耗滑动窗口
TREND_HORIZON_MIN = 120                # 预计 2 小时内超标才预警
TREND_MIN_LEVEL = 0.5                  # 当前拟合值至少达到限值的一半
TREND_MIN_R2 = 0.5
WINDOW_MIN_POINTS = 9                  # 60 分钟窗口（12 个桶）至少 9 个有效桶
DO_LOW = 1.0
INFLOW_SURGE_RATIO = 1.5
# 药耗：实际投加量与「负荷模型」预期值之比。单耗（kg/m³）随进水碳氮比大幅波动，直接比单耗误报很多
DOSING_RATIO = {"carbon_dose_kgh": 1.3, "pac_dose_kgh": 1.3}
DOSE_FEATURES = {"carbon_dose_kgh": ("tn_in", "cod_in"),   # 反硝化碳源需求：进水 TN 决定需求，进水 COD 可抵扣
                 "pac_dose_kgh": ("tp_in", "cod_in")}      # 化学除磷：进水 TP，生物除磷效果受进水 COD 影响
DOSE_MIN_FIT_HOURS = 24
DOSE_MIN_PRED_FRAC = 0.3               # 预期投加量不到常态中位数的 30%（如暴雨稀释）时不评估
DOSE_MAX_INFLOW_RATIO = 1.3            # 进水超过同时段基线 1.3 倍时模型外推不可靠，不评估
LOAD_METRICS = {source: metric for metric, source in LOAD_SOURCES.items()}
LOAD_MIN_HOURS = 3                    # 同一北京时间钟点至少 3 个历史小时，历史不足不判断


@dataclass(frozen=True)
class LoadConfig:
    ratio: float = 1.5
    concentration_ratio: float = 1.25
    mad_sigma: float = 3.0
    on: int = 3
    off: int = 3


DEFAULT_LOAD_CONFIG = LoadConfig()
VIBRATION_WARN, VIBRATION_ALARM = 4.5, DEVICE_METRICS["vibration_mms"].limit
TEMP_WARN, TEMP_ALARM = 75.0, DEVICE_METRICS["temp_c"].limit
STOP_WATCH_KINDS = {"鼓风机", "提升泵", "回流泵", "加药泵"}   # 污泥泵按排泥周期启停，不判停机


@dataclass(frozen=True)
class Rule:
    id: str
    kind: str
    level: str
    on: int
    off: int


RULES = {r.id: r for r in [
    Rule("effluent_exceed", "水质超标", "告警", 2, 3),
    Rule("effluent_trend", "水质预警", "预警", 2, 3),
    Rule("do_low", "工艺异常", "告警", 2, 3),
    Rule("inflow_surge", "进水冲击", "预警", 3, 3),
    Rule("inflow_load", "进水负荷", "预警", DEFAULT_LOAD_CONFIG.on, DEFAULT_LOAD_CONFIG.off),
    Rule("dosing_anomaly", "药耗异常", "预警", 3, 6),
    Rule("device_vibration", "设备异常", "预警", 2, 3),
    Rule("device_temp", "设备异常", "预警", 2, 3),
    Rule("device_stop", "设备异常", "告警", 1, 1),
]}
LEVEL_RANK = {"提示": 0, "预警": 1, "告警": 2}

# 条件取值：True 满足；False 不满足；None 无数据（不计入连续计数）；STOP 立即结束（趋势预警已升级为超标）
STOP = "stop"


@dataclass(frozen=True)
class DoseModel:
    """投加量（kg/h）≈ Σ coef_k × 进水量 × 特征_k + coef_q × 进水量，用近 7 天小时数据拟合。"""
    drug: str
    features: tuple[str, ...]
    coef: tuple[float, ...]           # 与 features 一一对应，末项为进水量本身的系数
    r2: float
    hours: int
    kept: float                       # 剔除离群后参与拟合的比例
    median_pred: float

    def predict(self, q: float | None, x: dict[str, float | None]) -> float | None:
        if q is None or any(x.get(f) is None for f in self.features):
            return None
        return sum(c * q * x[f] for c, f in zip(self.coef, self.features)) + self.coef[-1] * q

    def describe(self) -> dict:
        terms = [f"进水量×{PLANT_METRICS[f].label}" for f in self.features] + ["进水量"]
        return {"terms": terms, "coef": [round(c, 6) for c in self.coef], "r2": round(self.r2, 4),
                "fit_hours": self.hours, "kept": round(self.kept, 3),
                "rule": "近7天小时数据最小二乘，按 3 倍 MAD 剔除离群后重拟合"}


def fit_dose_model(drug: str, rows: list[dict]) -> DoseModel | None:
    """rows：小时聚合，需含 inflow_m3h、药剂字段和特征字段。异常时段靠剔除离群压低影响。"""
    feats = DOSE_FEATURES[drug]
    data = [r for r in rows if r.get("inflow_m3h") and r.get(drug) is not None
            and all(r.get(f) is not None for f in feats)]
    if len(data) < DOSE_MIN_FIT_HOURS:
        return None
    X = np.array([[r["inflow_m3h"] * r[f] for f in feats] + [r["inflow_m3h"]] for r in data], dtype=float)
    y = np.array([r[drug] for r in data], dtype=float)
    floor = 0.03 * float(np.median(np.abs(y))) + 1e-9   # 残差极集中时 MAD 很小，设下限避免过度剔除
    keep = np.ones(len(y), dtype=bool)
    for _ in range(3):
        beta = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        resid = y - X @ beta
        r = resid[keep]
        scale = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), floor)
        new_keep = np.abs(resid) <= 3 * scale
        if new_keep.sum() < DOSE_MIN_FIT_HOURS or (new_keep == keep).all():
            break
        keep = new_keep
    beta = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
    pred = X @ beta
    yk = y[keep]
    ss_tot = float(((yk - yk.mean()) ** 2).sum())
    r2 = 1 - float(((yk - pred[keep]) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return DoseModel(drug, feats, tuple(float(b) for b in beta), r2, len(data), float(keep.mean()),
                     float(np.median(pred[keep])))


@dataclass(frozen=True)
class Reference:
    median: float
    mad: float
    hours: int

    def threshold(self, ratio: float, sigma: float) -> float:
        return max(self.median * ratio, self.median + sigma * 1.4826 * self.mad)


@dataclass
class Baseline:
    """as_of 之前的完整小时；同一扫描内不同小时的判断使用各自快照。"""
    as_of: datetime
    inflow: dict[int, float] = field(default_factory=dict)
    dose: dict[str, DoseModel] = field(default_factory=dict)
    reference: dict[tuple[int, str], Reference] = field(default_factory=dict)


@dataclass
class PlantSignals:
    plant_id: int
    code: str
    name: str
    ts: list[datetime]
    values: dict[str, list[float | None]]
    inflow_baseline: dict[int, float] = field(default_factory=dict)   # 北京时间小时 → 近 7 天同小时进水中位数
    dose_model: dict[str, DoseModel] = field(default_factory=dict)    # 药剂字段 → 负荷模型
    # 此前的桶只作历史（滑动均值、趋势拟合要用），不判定：否则回看段开头按残缺窗口算出的值会混进事件和峰值
    eval_from: datetime | None = None
    baselines: dict[datetime, Baseline] = field(default_factory=dict)

    def baseline_at(self, t: datetime) -> Baseline:
        anchor = t.replace(minute=0, second=0, microsecond=0)
        if self.baselines:
            return self.baselines.get(anchor, Baseline(anchor))
        # 纯函数调用方可显式传入已经冻结的基线；生产和回放总是使用逐小时快照。
        return Baseline(anchor, self.inflow_baseline, self.dose_model)


@dataclass
class DeviceSignals:
    device_id: int
    plant_id: int
    code: str
    kind: str
    standby: bool
    ts: list[datetime]
    vibration: list[float | None]       # 运行时段桶内最大值
    temp: list[float | None]
    running: list[bool | None]          # 桶内是否运行过
    eval_from: datetime | None = None   # 同 PlantSignals.eval_from


@dataclass
class Episode:
    rule: str
    plant_id: int
    metric: str
    start: datetime
    end: datetime | None                # None：评估到最后一个桶时仍在持续
    level: str
    value: float | None                 # 触发时的值
    threshold: float | None
    peak: float | None
    peak_at: datetime | None
    last_seen: datetime                 # 最后一个满足条件的桶
    message: str
    device_id: int | None = None
    detail: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return RULES[self.rule].kind

    @property
    def key(self) -> tuple:
        return self.plant_id, self.device_id or 0, self.rule, self.metric


def runs(conds: list, on: int, off: int) -> list[tuple[int, int, int | None]]:
    """返回 (开始桶, 最后满足条件的桶, 确认恢复的第一个桶或 None)。"""
    out, active = [], False
    streak_on = streak_off = 0
    start = run_start = last_true = off_start = 0
    for i, cond in enumerate(conds):
        if cond is None:
            continue
        if not active:
            if cond is True:
                if streak_on == 0:
                    run_start = i
                streak_on += 1
                last_true = i
                if streak_on >= on:
                    active, start, streak_off = True, run_start, 0
            else:
                streak_on = 0
            continue
        if cond is True:
            streak_off, last_true = 0, i
            continue
        if streak_off == 0:
            off_start = i
        streak_off += 1
        if cond == STOP or streak_off >= off:
            out.append((start, last_true, off_start))
            active, streak_on = False, 0
    if active:
        out.append((start, last_true, None))
    return out


def _worst(values: list[float | None], ts: list[datetime], lo: int, hi: int, lower_is_bad: bool):
    pairs = [(values[i], ts[i]) for i in range(lo, hi + 1) if values[i] is not None]
    if not pairs:
        return None, None
    return (min if lower_is_bad else max)(pairs, key=lambda p: p[0])


def _collect(rule: str, conds: list, ts: list[datetime], values: list[float | None], *, plant_id: int,
             metric: str, threshold: float | None, message: Callable[[int], str], unit: str,
             lower_is_bad: bool = False,
             device_id: int | None = None, detail: Callable[[int], dict] | None = None,
             level: Callable[[float | None], str] | None = None, eval_from: datetime | None = None,
             on: int | None = None, off: int | None = None) -> list[Episode]:
    r = RULES[rule]
    if eval_from is not None:
        conds = [None if t < eval_from else c for t, c in zip(ts, conds)]
    out = []
    for start, last, end in runs(conds, on or r.on, off or r.off):
        peak, peak_at = _worst(values, ts, start, last, lower_is_bad)
        confirmed = [i for i in range(start, last + 1) if conds[i] is True][(on or r.on) - 1]
        out.append(Episode(
            rule=rule, plant_id=plant_id, metric=metric, start=ts[start], end=None if end is None else ts[end],
            level=level(peak) if level else r.level, value=values[start], threshold=threshold,
            peak=None if peak is None else round(peak, 4), peak_at=peak_at, last_seen=ts[last],
            message=message(start), device_id=device_id,
            detail={"unit": unit, "algorithm_version": "causal-hourly-v1",
                    "condition_confirmed_at": (ts[confirmed] + BUCKET).isoformat(),
                    **(detail(start) if detail else {})}))
    return out


def fmt(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{v:.0f}"


def fmt_vs(v: float | None, ref: float, digits: int = 2) -> str:
    """与阈值对比的数值：舍入后与阈值相同则补位（写成“0.5004 超过限值 0.5”，而不是“0.5 超过限值 0.5”）。"""
    out = fmt(v, digits)
    while v is not None and digits < 4 and out == fmt(ref, 4):
        digits += 1
        out = fmt(v, digits)
    return out


# ---------------------------------------------------------------- 滑动窗口

def window(ts: list[datetime], values: list[float | None], i: int) -> list[tuple[float, float]]:
    """第 i 个桶及其前 60 分钟内的有效点，横坐标为相对第 i 个桶的分钟数（≤0）。"""
    pts = []
    j = i
    while j >= 0 and ts[i] - ts[j] < LOOKBACK:
        if values[j] is not None:
            pts.append(((ts[j] - ts[i]).total_seconds() / 60, values[j]))
        j -= 1
    return pts


def _recent_any(ts: list[datetime], flags: list[bool], i: int) -> bool:
    j = i
    while j >= 0 and ts[i] - ts[j] < LOOKBACK:
        if flags[j]:
            return True
        j -= 1
    return False


def rolling_mean(ts: list[datetime], values: list[float | None]) -> list[float | None]:
    out = []
    for i in range(len(ts)):
        pts = window(ts, values, i)
        out.append(sum(y for _, y in pts) / len(pts) if len(pts) >= WINDOW_MIN_POINTS else None)
    return out


def linear_fit(pts: list[tuple[float, float]]) -> tuple[float, float, float]:
    """最小二乘直线：返回 (x=0 处的拟合值, 每分钟斜率, R²)。"""
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        return my, 0.0, 0.0
    slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in pts)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return intercept, slope, r2


def trend_state(ts, values, i, limit) -> tuple[object, dict]:
    v = values[i]
    if v is None:
        return None, {}
    if v > limit:
        return STOP, {}
    if v < 0.8 * TREND_MIN_LEVEL * limit:   # 远低于预警起点，跳过拟合
        return False, {}
    pts = window(ts, values, i)
    if len(pts) < WINDOW_MIN_POINTS or -min(x for x, _ in pts) < 40:
        return None, {}
    fitted, slope, r2 = linear_fit(pts)
    if slope <= 0 or fitted < TREND_MIN_LEVEL * limit or r2 < TREND_MIN_R2:
        return False, {}
    eta = max(0.0, (limit - fitted) / slope)
    if eta > TREND_HORIZON_MIN:
        return False, {}
    return True, {"fitted": round(fitted, 4), "slope_per_h": round(slope * 60, 4), "r2": round(r2, 3),
                  "eta_min": round(eta, 1), "predicted_at": (ts[i] + timedelta(minutes=eta)).isoformat(),
                  "window_min": int(LOOKBACK.total_seconds() // 60), "points": len(pts)}


# ---------------------------------------------------------------- 厂站规则

def load_episodes(s: PlantSignals, cfg: LoadConfig = DEFAULT_LOAD_CONFIG) -> list[Episode]:
    """质量负荷与浓度同时升高才预警；雨天仅水量增加、浓度稀释不算浓度型冲击。"""
    out = []
    for concentration, metric in LOAD_METRICS.items():
        loads = s.values.get(metric, [None] * len(s.ts))
        conc = s.values.get(concentration, [None] * len(s.ts))
        conditions, details = [], []
        for i, t in enumerate(s.ts):
            b = s.baseline_at(t)
            hour = t.astimezone(LOCAL_TZ).hour
            ref, cr = b.reference.get((hour, metric)), b.reference.get((hour, concentration))
            if (loads[i] is None or conc[i] is None or ref is None or cr is None
                    or min(ref.hours, cr.hours) < LOAD_MIN_HOURS or min(ref.median, cr.median) <= 0):
                conditions.append(None)
                details.append({})
                continue
            limit = ref.threshold(cfg.ratio, cfg.mad_sigma)
            c_limit = cr.threshold(cfg.concentration_ratio, cfg.mad_sigma)
            conditions.append(loads[i] > limit and conc[i] > c_limit)
            q = (s.values.get("inflow_m3h") or [None] * len(s.ts))[i]
            base_q = b.inflow.get(hour)
            q_ratio = q / base_q if q is not None and base_q else None
            details.append({"source_metric": concentration, "concentration_mgl": round(conc[i], 4),
                            "inflow_m3h": q, "inflow_baseline_m3h": base_q,
                            "inflow_ratio": None if q_ratio is None else round(q_ratio, 4),
                            "pattern": ("浓度升高，水量基线不足" if q_ratio is None else
                                        "水量与浓度同时升高" if q_ratio > 1.3 else "浓度升高型负荷冲击"),
                            "baseline_kgh": round(ref.median, 4), "threshold_kgh": round(limit, 4),
                            "concentration_baseline_mgl": round(cr.median, 4),
                            "concentration_threshold_mgl": round(c_limit, 4),
                            "ratio": round(loads[i] / ref.median, 4), "baseline_hours": ref.hours,
                            "baseline_as_of": b.as_of.isoformat(), "load_ratio": cfg.ratio,
                            "concentration_ratio": cfg.concentration_ratio, "mad_sigma": cfg.mad_sigma,
                            "on_buckets": cfg.on, "off_buckets": cfg.off,
                            "aggregation": "同采样流量×浓度÷1000 后取5分钟均值，单位kg/h",
                            "baseline_rule": "判定桶所在整点之前7天同钟点完整小时中位数及MAD；至少3小时"})
        label = PLANT_METRICS[concentration].label
        eps = _collect(
            "inflow_load", conditions, s.ts, loads, plant_id=s.plant_id, metric=metric,
            threshold=None, unit="kg/h", eval_from=s.eval_from, on=cfg.on, off=cfg.off,
            detail=lambda i: details[i], message=lambda i: (
                f"{label}质量负荷 {fmt(loads[i])} kg/h，超过历史关注值 {fmt(details[i]['threshold_kgh'])} kg/h；"
                f"浓度 {fmt(conc[i])} mg/L 同时升高（{details[i]['pattern']}），请核查进水来源与仪表"))
        for e in eps:
            e.threshold = e.detail["threshold_kgh"]
        out += eps
    return out


def plant_episodes(s: PlantSignals, load_config: LoadConfig = DEFAULT_LOAD_CONFIG) -> list[Episode]:
    out: list[Episode] = []
    ts = s.ts
    collect = partial(_collect, eval_from=s.eval_from)
    for key in EFFLUENT_KEYS:
        m = PLANT_METRICS[key]
        vals = s.values.get(key) or [None] * len(ts)
        conds = [None if v is None else v > m.limit for v in vals]
        out += collect("effluent_exceed", conds, ts, vals, plant_id=s.plant_id, metric=key, threshold=m.limit,
                        unit=m.unit, message=lambda i, m=m, vals=vals: f"{m.label} {fmt_vs(vals[i], m.limit)} {m.unit}，超过限值 {fmt(m.limit)}")

        states = [trend_state(ts, vals, i, m.limit) for i in range(len(ts))]
        out += collect(
            "effluent_trend", [c for c, _ in states], ts, vals, plant_id=s.plant_id, metric=key, threshold=m.limit,
            unit=m.unit, detail=lambda i, states=states: states[i][1],
            message=lambda i, m=m, states=states: (
                f"{m.label}近 60 分钟每小时上升 {fmt(states[i][1]['slope_per_h'])} {m.unit}，"
                f"按线性趋势约 {fmt(states[i][1]['eta_min'], 0)} 分钟后超过限值 {fmt(m.limit)}"))

    do = s.values.get("do_aerobic_1") or [None] * len(ts)
    out += collect("do_low", [None if v is None else v < DO_LOW for v in do], ts, do, plant_id=s.plant_id,
                    metric="do_aerobic_1", threshold=DO_LOW, unit="mg/L", lower_is_bad=True,
                    message=lambda i: f"好氧区 DO {fmt_vs(do[i], DO_LOW)} mg/L，低于 {fmt(DO_LOW)} mg/L")

    q = s.values.get("inflow_m3h") or [None] * len(ts)
    contexts = [s.baseline_at(t) for t in ts]
    base_q = [b.inflow.get(t.astimezone(LOCAL_TZ).hour) for t, b in zip(ts, contexts)]
    if any(b is not None for b in base_q):
        ratio = [None if v is None or not b else v / b for v, b in zip(q, base_q)]
        surges = collect(
            "inflow_surge", [None if r is None else r > INFLOW_SURGE_RATIO for r in ratio], ts, q,
            plant_id=s.plant_id, metric="inflow_m3h", threshold=None, unit="m³/h",
            detail=lambda i: {"baseline": round(base_q[i], 1), "ratio": round(ratio[i], 3),
                              "surge_ratio": INFLOW_SURGE_RATIO, "baseline_as_of": contexts[i].as_of.isoformat(),
                              "baseline_rule": "判定桶所在整点之前7天同一小时进水量中位数"},
            message=lambda i: f"进水 {fmt(q[i], 0)} m³/h，为近 7 天同时段中位数 {fmt(base_q[i], 0)} 的 {fmt(ratio[i], 1)} 倍")
        for e in surges:   # 阈值随钟点变化，记录触发时段的值
            e.threshold = round(INFLOW_SURGE_RATIO * e.detail["baseline"], 1)
        out += surges

    # 进水冲击期间及其后 60 分钟（控制器积分回落）药耗模型外推不可靠
    surge = [v is not None and bool(b) and v > DOSE_MAX_INFLOW_RATIO * b for v, b in zip(q, base_q)]
    surge_recent = [_recent_any(ts, surge, i) for i in range(len(ts))]
    for key, limit_ratio in DOSING_RATIO.items():
        models = [b.dose.get(key) for b in contexts]
        dose = s.values.get(key)
        if not any(models) or dose is None:
            continue
        pred = [None if model is None else model.predict(
            q[i], {f: (s.values.get(f) or [None] * len(ts))[i] for f in model.features})
                for i, model in enumerate(models)]
        actual, expected = rolling_mean(ts, dose), rolling_mean(ts, pred)
        ratio = []
        for i in range(len(ts)):
            a, p = actual[i], expected[i]
            model = models[i]
            ratio.append(None if a is None or p is None or model is None or p <= 0
                         or p < DOSE_MIN_PRED_FRAC * model.median_pred or surge_recent[i] else a / p)
        label = PLANT_METRICS[key].label
        out += collect(
            "dosing_anomaly", [None if r is None else r > limit_ratio for r in ratio], ts, ratio,
            plant_id=s.plant_id, metric=key, threshold=limit_ratio, unit="倍",
            detail=lambda i, actual=actual, expected=expected, ratio=ratio, models=models, limit_ratio=limit_ratio: {
                "actual_kgh": round(actual[i], 3), "expected_kgh": round(expected[i], 3), "ratio": round(ratio[i], 3),
                "anomaly_ratio": limit_ratio, "window_min": 60, "model": models[i].describe(),
                "baseline_as_of": contexts[i].as_of.isoformat()},
            message=lambda i, actual=actual, expected=expected, ratio=ratio, label=label: (
                f"{label}近 60 分钟均值 {fmt(actual[i])} kg/h，为按进水负荷估算值 {fmt(expected[i])} kg/h 的 "
                f"{fmt(ratio[i])} 倍"))
    out += load_episodes(s, load_config)
    return out


# ---------------------------------------------------------------- 设备规则

def device_episodes(d: DeviceSignals) -> list[Episode]:
    out: list[Episode] = []
    ts = d.ts
    collect = partial(_collect, eval_from=d.eval_from)
    out += collect(
        "device_vibration", [None if v is None else v > VIBRATION_WARN for v in d.vibration], ts, d.vibration,
        plant_id=d.plant_id, device_id=d.device_id, metric="vibration_mms", threshold=VIBRATION_WARN, unit="mm/s",
        level=lambda peak: "告警" if peak is not None and peak > VIBRATION_ALARM else "预警",
        message=lambda i: f"{d.code} 振动 {fmt_vs(d.vibration[i], VIBRATION_WARN)} mm/s，超过关注值 {fmt(VIBRATION_WARN)}")
    out += collect(
        "device_temp", [None if v is None else v > TEMP_WARN for v in d.temp], ts, d.temp,
        plant_id=d.plant_id, device_id=d.device_id, metric="temp_c", threshold=TEMP_WARN, unit="℃",
        level=lambda peak: "告警" if peak is not None and peak > TEMP_ALARM else "预警",
        message=lambda i: f"{d.code} 轴承温度 {fmt_vs(d.temp[i], TEMP_WARN, 1)} ℃，超过关注值 {fmt(TEMP_WARN, 0)} ℃")
    if not d.standby and d.kind in STOP_WATCH_KINDS:
        stopped = [None if r is None else not r for r in d.running]
        out += [_drop_value(e) for e in collect(
            "device_stop", stopped, ts, [None] * len(ts), plant_id=d.plant_id, device_id=d.device_id,
            metric="running", threshold=None, unit="", message=lambda i: f"{d.code}（{d.kind}，非备用）停机")]
    return out


def _drop_value(e: Episode) -> Episode:
    e.value, e.peak, e.peak_at = 0.0, None, None
    return e


def detect(plants: list[PlantSignals], devices: list[DeviceSignals], *,
           load_config: LoadConfig = DEFAULT_LOAD_CONFIG) -> list[Episode]:
    out = [e for s in plants for e in plant_episodes(s, load_config)]
    out += [e for d in devices for e in device_episodes(d)]
    return sorted(out, key=lambda e: (e.start, e.plant_id, e.rule))


def worse_level(a: str, b: str) -> str:
    return a if LEVEL_RANK[a] >= LEVEL_RANK[b] else b

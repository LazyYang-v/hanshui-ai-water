"""智能曝气对照实验（M6）：同一厂站、同一组随机扰动下比较曝气策略的 DO 跟踪与鼓风机电耗。

策略
- fixed       固定风量（人工经验）：按旱天 PI 风量的高分位配风，不随负荷调整
- pi          PI 跟踪 DO 设定值（用含噪声的 DO 测量值；DO 控制常用 PI，测量噪声下微分项只放大噪声）
- ff_pi       前馈 + PI：按在线仪表测得的进水需氧负荷前馈风量，PI 修正残差
- ff_hw_pi    前馈 + PI，前馈用 Holt-Winters 预测的 h 分钟后负荷（DESIGN 6.4 的策略 C）

公平性
- 所有策略共用同一厂站模型、同一随机种子（进水、噪声完全相同，common random numbers）；
- 控制器只看到「测量值」：DO 探头一阶滞后 60 秒、0.04 mg/L 噪声、取上一步；进水流量 1% 噪声，
  COD/氨氮分析仪 5% 噪声、每 10 分钟出一个值；
- 执行机构（变频鼓风机 + 阀门）一阶滞后 3 分钟；
- 参数（PI 增益、前馈系数与滤波、预测参数、固定风量）只在校准种子的旱天上整定：在「风量每小时累计变化
  ≤ 0.5 倍设计风量」的动作预算内取 DO 误差最小，否则高增益会把测量噪声放大到鼓风机上；
- 评测用另一组种子和暴雨/冲击负荷场景。

局限：沿用模拟器的简化氧平衡，OUR 随进水负荷瞬时变化（没有水力停留带来的滞后），硝化按 Monod 简化；
结果只用于说明方法，不能当作真实厂的节能比例。
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .engine import air_temp, plant_config
from .model import DO_SAT, LOCAL_TZ, AerationObs, PlantConfig, PlantModel
from .scenarios import Scenario, ScenarioBook

PARAMS_PATH = Path(__file__).with_name("aeration_params.json")
STRATEGIES = ("fixed", "pi", "ff_pi", "ff_hw_pi")
LABELS = {"fixed": "固定曝气", "pi": "PI 反馈", "ff_pi": "前馈（实测负荷）+ PI", "ff_hw_pi": "前馈（负荷预测）+ PI"}
SCENARIOS = {
    "dry": "旱天",
    "storm": "暴雨：07:00 起 3 小时、峰值 18 mm/h",
    "shock": "冲击负荷：13:00 起 3 小时、进水浓度 ×1.5（鼓风机能力以内）",
    "overload": "超能力冲击：09:00 起 3 小时、进水浓度 ×1.8（叠加早高峰，所需风量超过运行鼓风机上限）",
}
EVAL_DAY = datetime(2026, 11, 5, tzinfo=LOCAL_TZ)
DT_S = 60.0
ACTUATOR_TAU_S = 180.0
SENSOR_TAU_S = 60.0
MOVE_BUDGET = 0.5        # 整定约束：风量每小时累计变化（设计风量倍数）
NH3N_LIMIT = 5.0

# 演示主厂：南郑中心污水处理厂（P01，5 万 m³/d，2 台运行鼓风机）
P01 = plant_config({"id": 1, "code": "P01", "kind": "中心厂", "process": "A2O", "capacity_m3d": 50000})


def design_load(cfg: PlantConfig) -> float:
    """设计工况的需氧负荷 Q·(0.4·COD + 4.57·NH3-N)，与模型里 ox_load 的口径一致。"""
    return cfg.q_design * (0.4 * cfg.cod_base + 4.57 * cfg.nh3n_base)


def kw_of(cfg: PlantConfig, air: float) -> float:
    return cfg.blower_kw_design * (air / cfg.air_design) ** 1.3 if air > 0 else 0.0


# ---------------------------------------------------------------- 在线仪表

class Instruments:
    """进水在线仪表：流量计 1% 白噪声；COD、氨氮分析仪 5% 白噪声，每 hold_min 分钟出一个值（期间保持）。"""

    def __init__(self, cfg: PlantConfig, rng: np.random.Generator, hold_min: float = 10.0):
        self.cfg, self.rng, self.hold_min = cfg, rng, hold_min
        self._next: datetime | None = None
        self._cod = cfg.cod_base
        self._nh3 = cfg.nh3n_base

    def load(self, obs: AerationObs) -> float:
        """测得的需氧负荷 / 设计负荷。"""
        n = self.rng.standard_normal(3)
        if self._next is None or obs.ts >= self._next:
            self._cod = obs.cod_in * (1 + 0.05 * n[1])
            self._nh3 = obs.nh3n_in * (1 + 0.05 * n[2])
            self._next = obs.ts + timedelta(minutes=self.hold_min)
        q = obs.inflow * (1 + 0.01 * n[0])
        return q * (0.4 * self._cod + 4.57 * self._nh3) / design_load(self.cfg)


# ---------------------------------------------------------------- 负荷预测

class HoltWinters:
    """乘法季节、无趋势的 Holt-Winters，5 分钟一槽、周期 1 天，在线更新。

    第一天只收集季节因子；之后每槽结束时更新：L ← α·y/S + (1−α)·L，S ← γ·y/L + (1−γ)·S。
    """

    def __init__(self, alpha: float, gamma: float, slot_min: int = 5):
        self.alpha, self.gamma, self.slot_min = alpha, gamma, slot_min
        self.period = 24 * 60 // slot_min
        self.season = np.ones(self.period)
        self.level: float | None = None
        self._first_day: list[float | None] = [None] * self.period
        self._slot: tuple[datetime, int] | None = None
        self._acc: list[float] = []
        self.last = 1.0

    def _slot_of(self, ts: datetime) -> int:
        t = ts.astimezone(LOCAL_TZ)
        return (t.hour * 60 + t.minute) // self.slot_min

    def update(self, ts: datetime, y: float) -> None:
        self.last = y
        key = (ts.astimezone(LOCAL_TZ).replace(minute=0, second=0, microsecond=0), self._slot_of(ts))
        if self._slot is not None and key != self._slot:
            self._finish(self._slot[1], float(np.mean(self._acc)))
            self._acc = []
        self._slot = key
        self._acc.append(y)

    def _finish(self, k: int, y: float) -> None:
        if self.level is None:
            self._first_day[k] = y
            if all(v is not None for v in self._first_day):
                values = np.array(self._first_day, dtype=float)
                self.level = float(values.mean())
                self.season = values / self.level
            return
        s = self.season[k]
        level = self.alpha * y / s + (1 - self.alpha) * self.level
        self.season[k] = self.gamma * y / level + (1 - self.gamma) * s
        self.level = level

    def forecast(self, ts: datetime, horizon_min: float) -> float:
        if self.level is None:
            return self.last
        return self.level * float(self.season[self._slot_of(ts + timedelta(minutes=horizon_min))])


# ---------------------------------------------------------------- 控制器

class FixedAir:
    def __init__(self, air_frac: float):
        self.air_frac = air_frac

    def command(self, obs: AerationObs) -> float:
        return self.air_frac * obs.air_design


class PI:
    """u = 前馈 + Kp·e + Ki·∫e，单位为设计风量的倍数；条件积分抗饱和。无前馈时偏置为设计风量。"""

    def __init__(self, setpoint: float, kp: float, ki: float,
                 feedforward: Callable[[AerationObs], float] | None = None):
        self.sp, self.kp, self.ki, self.ff = setpoint, kp, ki, feedforward
        self.integral = 0.0

    def command(self, obs: AerationObs) -> float:
        e = self.sp - obs.do_meas
        u_ff = self.ff(obs) if self.ff else 1.0
        cmd = (u_ff + self.kp * e + self.ki * self.integral) * obs.air_design
        if not ((cmd >= obs.air_max and e > 0) or (cmd <= obs.air_min and e < 0)):
            limit = 1.5 / self.ki if self.ki else 0.0       # 积分项最多 ±1.5 倍设计风量
            self.integral = max(-limit, min(limit, self.integral + e * obs.dt_h))
        return cmd


class LoadFeedforward:
    """前馈风量 = a + b·负荷（由校准日 PI 运行数据拟合），按氧传递推动力换算到其他设定值。

    稳态时 KLa·(DOsat − DO) = OUR，KLa ∝ 风量，所以同一负荷下风量 ∝ 1 / (DOsat − 设定值)。
    """

    def __init__(self, a: float, b: float, setpoint: float, instruments: Instruments,
                 forecaster: HoltWinters | None = None, horizon_min: float = 0.0, filter_min: float = 0.0):
        self.a, self.b, self.instruments = a, b, instruments
        self.scale = (DO_SAT - 2.0) / (DO_SAT - setpoint)
        self.forecaster, self.horizon_min = forecaster, horizon_min
        self.filter_min = filter_min
        self._filtered: float | None = None

    def __call__(self, obs: AerationObs) -> float:
        load = self.instruments.load(obs)
        if self.forecaster is not None:
            self.forecaster.update(obs.ts, load)
            load = self.forecaster.forecast(obs.ts, self.horizon_min)
        if self._filtered is None or not self.filter_min:
            self._filtered = load
        else:                     # 一阶滤波，避免分析仪噪声每 10 分钟一跳直接传到风量
            self._filtered += (load - self._filtered) * (1 - math.exp(-obs.dt_h * 60 / self.filter_min))
        return (self.a + self.b * self._filtered) * self.scale


@dataclass
class Params:
    """整定结果；evals/aeration/run.py calibrate 生成 simulator/aeration_params.json。"""
    fixed_air_frac: float = 1.3
    pi: dict = field(default_factory=lambda: {"kp": 0.25, "ki": 0.6})
    ff: dict = field(default_factory=lambda: {"a": 0.0, "b": 1.0})
    ff_pi: dict = field(default_factory=lambda: {"kp": 0.25, "ki": 0.6, "filter_min": 10.0})
    hw: dict = field(default_factory=lambda: {"alpha": 0.2, "gamma": 0.2, "horizon_min": 10.0, "filter_min": 10.0,
                                              "kp": 0.25, "ki": 0.6})
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = PARAMS_PATH) -> Params:
        return cls(**json.loads(path.read_text(encoding="utf-8"))) if path.exists() else cls()

    def save(self, path: Path = PARAMS_PATH) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def build_controller(strategy: str, params: Params, setpoint: float, cfg: PlantConfig, seed: int):
    instruments = Instruments(cfg, np.random.default_rng([seed, 999]))
    if strategy == "fixed":
        return FixedAir(params.fixed_air_frac)
    if strategy == "pi":
        return PI(setpoint, **params.pi)
    if strategy == "ff_pi":
        return PI(setpoint, params.ff_pi["kp"], params.ff_pi["ki"], feedforward=LoadFeedforward(
            **params.ff, setpoint=setpoint, instruments=instruments, filter_min=params.ff_pi["filter_min"]))
    if strategy == "ff_hw_pi":       # 前馈系数与 ff_pi 相同；PI 增益与预测参数单独整定
        hw = HoltWinters(params.hw["alpha"], params.hw["gamma"])
        return PI(setpoint, params.hw["kp"], params.hw["ki"], feedforward=LoadFeedforward(
            **params.ff, setpoint=setpoint, instruments=instruments, forecaster=hw,
            horizon_min=params.hw["horizon_min"], filter_min=params.hw["filter_min"]))
    raise ValueError(f"未知策略 {strategy!r}，可选：{', '.join(STRATEGIES)}")


# ---------------------------------------------------------------- 仿真

def scenario_book(scenario: str, cfg: PlantConfig, day: datetime = EVAL_DAY) -> ScenarioBook:
    book = ScenarioBook()
    if scenario == "storm":
        book.add(Scenario("storm", day + timedelta(hours=7), 180, params={"peak_mm_h": 18.0}))
    elif scenario == "shock":
        book.add(Scenario("shock_load", day + timedelta(hours=13), 180, cfg.code, params={"factor": 1.5}))
    elif scenario == "overload":
        book.add(Scenario("shock_load", day + timedelta(hours=9), 180, cfg.code, params={"factor": 1.8}))
    elif scenario != "dry":
        raise ValueError(f"未知场景 {scenario!r}，可选：{', '.join(SCENARIOS)}")
    return book


@dataclass
class Run:
    strategy: str
    scenario: str
    seed: int
    setpoint: float
    series: dict[str, list[float]]     # 评测日每步：ts（分钟）、do、do_meas、air_frac、kw、nh3n_out、load、inflow
    metrics: dict[str, float]


def simulate(strategy: str, *, params: Params | None = None, controller=None, scenario: str = "dry",
             seed: int = 101, setpoint: float = 2.0, warmup_days: int = 3, cfg: PlantConfig = P01,
             dt_s: float = DT_S, actuator_tau_s: float = ACTUATOR_TAU_S,
             sensor_tau_s: float = SENSOR_TAU_S) -> Run:
    """先运行 warmup_days 天（同一控制器，供预测器学习日内规律、状态进入稳态），再评测 1 天。"""
    params = params or Params.load()
    controller = controller or build_controller(strategy, params, setpoint, cfg, seed)
    model = PlantModel(cfg, np.random.default_rng([seed, cfg.id]), controller=controller,
                       actuator_tau_s=actuator_tau_s, sensor_tau_s=sensor_tau_s)
    book = scenario_book(scenario, cfg)
    start = EVAL_DAY - timedelta(days=warmup_days)
    steps_per_day = int(round(86400 / dt_s))
    keys = ("t_min", "do", "do_meas", "air_frac", "kw", "nh3n_out", "load", "inflow")
    series: dict[str, list[float]] = {k: [] for k in keys}
    for i in range((warmup_days + 1) * steps_per_day):
        ts = start + timedelta(seconds=i * dt_s)
        t_air = air_temp(ts)
        row = model.step(ts, dt_s, book.disturbance(cfg.code, ts), 16.0 + 0.15 * (t_air - 12.0))
        if ts < EVAL_DAY:
            continue
        s = model.s
        series["t_min"].append((ts - EVAL_DAY).total_seconds() / 60)
        series["do"].append(s.do1)
        series["do_meas"].append(s.do_meas)
        series["air_frac"].append(s.air / cfg.air_design)
        series["kw"].append(kw_of(cfg, s.air))
        series["nh3n_out"].append(s.nh3n_out)
        series["load"].append(row["inflow_m3h"] * (0.4 * row["cod_in"] + 4.57 * row["nh3n_in"]) / design_load(cfg))
        series["inflow"].append(row["inflow_m3h"])
    return Run(strategy, scenario, seed, setpoint, series, metrics(series, setpoint, dt_s))


def metrics(series: dict[str, list[float]], setpoint: float, dt_s: float = DT_S) -> dict[str, float]:
    do = np.array(series["do"])
    err = np.abs(do - setpoint)
    dt_h = dt_s / 3600
    air = np.array(series["air_frac"])
    nh3 = np.array(series["nh3n_out"])
    kwh = float(np.sum(series["kw"]) * dt_h)
    m3 = float(np.sum(series["inflow"]) * dt_h)
    hours = len(do) * dt_h
    return {
        "do_mae": round(float(err.mean()), 4),
        "do_p95_abs_err": round(float(np.percentile(err, 95)), 4),
        "do_low_pct": round(float(np.mean(do < setpoint - 0.5) * 100), 2),     # 低于设定值 0.5 以上：硝化风险
        "do_high_pct": round(float(np.mean(do > setpoint + 1.0) * 100), 2),    # 高于设定值 1.0 以上：过曝气
        "kwh": round(kwh, 1),
        "kwh_per_m3": round(kwh / m3, 4),
        "nh3n_mean": round(float(nh3.mean()), 3),
        "nh3n_max": round(float(nh3.max()), 3),
        "nh3n_over_limit_h": round(float(np.sum(nh3 > NH3N_LIMIT) * dt_h), 2),
        "air_moves_per_h": round(float(np.sum(np.abs(np.diff(air))) / hours), 4),  # 风量累计变化（设计风量倍数/小时）
    }


def downsample(series: dict[str, list[float]], every: int = 5) -> dict[str, list[float]]:
    """按 every 步取均值（60 s 步长 → 5 分钟），供接口与图表使用。"""
    out = {}
    for k, v in series.items():
        a = np.array(v, dtype=float)
        n = len(a) // every * every
        m = a[:n].reshape(-1, every)
        out[k] = [round(float(x), 4) for x in (m[:, 0] if k == "t_min" else m.mean(axis=1))]
    return out


def compare(scenario: str = "dry", seed: int = 101, setpoint: float = 2.0,
            strategies: tuple[str, ...] = STRATEGIES, params: Params | None = None) -> dict:
    """同一场景、种子下跑多个策略，返回指标与 5 分钟曲线；电耗同时给出相对固定曝气的变化。"""
    params = params or Params.load()
    runs = {s: simulate(s, params=params, scenario=scenario, seed=seed, setpoint=setpoint) for s in strategies}
    base = runs["fixed"].metrics["kwh"] if "fixed" in runs else None
    out = {"scenario": scenario, "scenario_label": SCENARIOS[scenario], "seed": seed, "setpoint": setpoint,
           "plant": P01.code, "step_s": DT_S, "actuator_tau_s": ACTUATOR_TAU_S, "sensor_tau_s": SENSOR_TAU_S,
           "notice": "仿真结果，仅用于说明方法；简化氧平衡模型，不代表真实厂的节能比例",
           "strategies": {}}
    common = downsample({k: runs[strategies[0]].series[k] for k in ("t_min", "load", "inflow")})
    out["series"] = common
    for s, r in runs.items():
        m = dict(r.metrics)
        if base:
            m["kwh_vs_fixed_pct"] = round((m["kwh"] / base - 1) * 100, 2)
        ds = downsample({k: r.series[k] for k in ("t_min", "do", "air_frac", "kw", "nh3n_out")})
        out["strategies"][s] = {"label": LABELS[s], "metrics": m,
                                "series": {k: ds[k] for k in ("do", "air_frac", "kw", "nh3n_out")}}
    return out

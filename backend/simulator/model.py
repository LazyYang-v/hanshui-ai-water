"""污水厂简化过程模型。

目标不是机理精确，而是让数据「行家看了不皱眉」：
- 进水量有日内双峰，下雨时入流入渗（I/I）抬高水量、稀释浓度；
- 溶解氧服从氧平衡  dDO/dt = KLa(风量)·(DOsat − DO) − OUR(负荷)，由 PI 控制风量；
  曝气对照实验（M6，simulator/aeration.py）可替换为外部控制器，并加执行机构滞后，默认路径不变；
- 硝化能力取决于 DO（Monod）、水力停留时间、污泥浓度和水温，出水氨氮一阶滞后；
- 旱天管道积泥，降雨初期被冲刷（初期冲刷），进水浓度短时升高；
- 反硝化受碳源约束，碳源投加 = 前馈 + PI（管道渗漏时投加量异常上升）；
- 除磷 = 生物除磷 + PAC 化学除磷，PAC 投加 = 前馈 + PI（加药泵堵塞时 TP 上升）；
- 大流量冲击会导致污泥流失（MLSS 下降），进一步削弱硝化。

单位：流量 m³/h，浓度 mg/L（= g/m³），投加量 kg/h，时间步长秒。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo

import numpy as np

LOCAL_TZ = ZoneInfo("Asia/Shanghai")

DO_SAT = 9.0          # 18℃ 左右的饱和溶解氧，mg/L
DO_SETPOINT = 2.0     # 好氧区 DO 设定值
TN_TARGET = 11.0      # 运行目标（标准 15，留余量）
TP_TARGET = 0.30      # 运行目标（标准 0.5，留余量）
ORGANIC_N_OUT = 2.0   # 出水有机氮
N_ASSIMILATION = 4.0  # 微生物同化去除的氮
ACETATE_COD = 0.78    # 乙酸钠 COD 当量，kgCOD/kg
COD_PER_N = 5.0       # 反硝化所需 COD/N
PAC_K = 0.083         # 化学除磷系数：tp = tp_bio · exp(−k · PAC mg/L)
MLSS_DESIGN = 3500.0
K_O_NIT = 1.0         # 硝化菌氧半饱和常数，mg/L
NIT_K = 3.4           # 出水氨氮 = 进水氨氮 · exp(−NIT_K · 硝化能力比)
NIT_CAPACITY = 2.4    # 使设计工况（16℃、DO 2）下硝化能力比约 1.2


@dataclass(frozen=True)
class PlantConfig:
    id: int
    code: str
    kind: str                # 中心厂 / 集镇厂
    process: str             # A2O / 氧化沟
    capacity_m3d: float
    ii_factor: float = 0.25  # 入流入渗系数：10 mm/h 降雨带来 ii_factor × 设计流量的额外水量
    cod_base: float = 260.0
    nh3n_base: float = 32.0
    tn_base: float = 42.0
    tp_base: float = 4.2
    blowers: int = 2         # 鼓风机台数（含备用）
    blowers_running: int = 1
    blower_capacity: float = 1.6  # 单台鼓风机最大风量 / 设计风量

    @property
    def q_design(self) -> float:
        return self.capacity_m3d / 24.0

    @property
    def air_design(self) -> float:
        return self.q_design * 6.0  # 气水比 6:1

    @property
    def blower_kw_design(self) -> float:
        return self.q_design * 0.10  # 鼓风机吨水电耗约 0.10 kWh/m³


@dataclass
class Disturbance:
    """由场景（scenarios.py）在每一步给出的扰动。"""
    rain_mm_h: float = 0.0
    blowers_lost: int = 0            # 故障跳停的鼓风机台数
    carbon_effective: float = 1.0    # 碳源实际进入生化池的比例（管道渗漏 < 1）
    pac_effective: float = 1.0       # PAC 实际投加比例（加药泵堵塞 < 1）
    load_factor: float = 1.0         # 进水浓度倍数（工业偷排等）


@dataclass(frozen=True)
class AerationObs:
    """每步交给外部曝气控制器的信息：上一步的 DO 测量值（含噪声）与本步进水真值（仪表噪声由控制器侧添加）。"""
    ts: datetime
    dt_h: float
    do_meas: float
    inflow: float
    cod_in: float
    nh3n_in: float
    air: float             # 当前实际风量（执行机构输出）
    air_min: float
    air_max: float
    air_design: float


class AerationController(Protocol):
    def command(self, obs: AerationObs) -> float: ...


@dataclass
class PlantState:
    q_rain: float = 0.0
    sediment: float = 1.0          # 管道积泥量（0~1）
    q_smooth: float = 0.0          # 经生化池缓冲后的流量
    n_load_smooth: float = 1.0     # 经生化池缓冲后的氨氮负荷比
    do1: float = DO_SETPOINT
    air: float = 0.0
    air_integral: float = 0.0
    nh3n_out: float = 1.2
    tn_out: float = TN_TARGET
    tp_out: float = TP_TARGET
    cod_out: float = 22.0
    ss_out: float = 6.0
    mlss: float = MLSS_DESIGN
    carbon_dose: float = 0.0
    carbon_integral: float = 0.0
    pac_dose: float = 0.0
    pac_integral: float = 0.0
    do_meas: float = DO_SETPOINT   # 上一步 DO 测量值，只在外部控制器路径使用
    do_probe: float = DO_SETPOINT  # DO 探头的滞后读数（不含噪声），只在外部控制器路径使用
    noise: dict[str, float] = field(default_factory=dict)


def _gauss(h: float, mu: float, sigma: float) -> float:
    d = min(abs(h - mu), 24 - abs(h - mu))  # 跨零点
    return math.exp(-(d * d) / (2 * sigma * sigma))


def _diurnal_raw(h: float) -> float:
    return 0.70 + 0.38 * _gauss(h, 8.5, 1.8) + 0.30 * _gauss(h, 20.0, 2.0) + 0.12 * _gauss(h, 13.0, 2.0)


_DIURNAL_MEAN = sum(_diurnal_raw(i / 10) for i in range(240)) / 240


def diurnal(hour: float) -> float:
    """日内流量系数，24 小时均值为 1。"""
    return _diurnal_raw(hour) / _DIURNAL_MEAN


def local_hour(ts: datetime) -> float:
    """厂站所在地的钟点（小数小时）。无时区的时间按 UTC 处理。"""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=ZoneInfo("UTC"))
    t = ts.astimezone(LOCAL_TZ)
    return t.hour + t.minute / 60 + t.second / 3600


def _lag(current: float, target: float, tau_h: float, dt_h: float) -> float:
    return current + (target - current) * (1 - math.exp(-dt_h / tau_h))


class PlantModel:
    def __init__(self, cfg: PlantConfig, rng: np.random.Generator, *,
                 controller: AerationController | None = None, actuator_tau_s: float = 0.0,
                 sensor_tau_s: float = 0.0):
        """controller 为 None 时使用内置 PI（演示与评测数据的生成路径，不能改动）。

        外部控制器路径可加执行机构滞后 actuator_tau_s 与 DO 探头滞后 sensor_tau_s（一阶）。"""
        self.cfg = cfg
        self.rng = rng
        self.controller, self.actuator_tau_s, self.sensor_tau_s = controller, actuator_tau_s, sensor_tau_s
        self.s = PlantState(air=cfg.air_design, q_smooth=cfg.q_design)

    # AR(1) 噪声：phi 越接近 1 越平滑
    def _ar(self, key: str, sigma: float, phi: float = 0.98) -> float:
        x = self.s.noise.get(key, 0.0)
        x = phi * x + math.sqrt(1 - phi * phi) * sigma * self.rng.standard_normal()
        self.s.noise[key] = x
        return x

    def _white(self, sigma: float) -> float:
        return sigma * self.rng.standard_normal()

    def step(self, ts: datetime, dt_s: float, d: Disturbance, water_temp: float = 18.0) -> dict:
        cfg, s = self.cfg, self.s
        dt_h = dt_s / 3600.0
        hour = local_hour(ts)
        qd = cfg.q_design

        # ---------- 进水 ----------
        q_dry = qd * diurnal(hour) * (1 + self._ar("q", 0.04))
        q_rain_target = qd * cfg.ii_factor * d.rain_mm_h / 10.0
        s.q_rain = _lag(s.q_rain, q_rain_target, 0.5, dt_h)
        q = min(q_dry + s.q_rain, 2.2 * qd)  # 超过 2.2 倍设计流量溢流
        dilution = q_dry / (q_dry + s.q_rain)

        # 初期冲刷：旱天 2 天积满，降雨时按雨水量冲刷
        flush_rate = 1.5 * (s.q_rain / qd) * s.sediment          # 1/h
        s.sediment = min(1.0, s.sediment + (1 / 48 - flush_rate) * dt_h)
        s.sediment = max(0.0, s.sediment)
        first_flush = 1 + 1.0 * flush_rate

        conc = (1 + self._ar("c", 0.06)) * d.load_factor * first_flush * dilution ** 0.9
        cod_in = cfg.cod_base * conc * (1 + self._ar("cod", 0.04))
        nh3n_in = cfg.nh3n_base * conc * (1 + self._ar("nh3", 0.04))
        tn_in = max(nh3n_in + 6, cfg.tn_base * conc * (1 + self._ar("tn", 0.03)))
        tp_in = cfg.tp_base * conc * (1 + self._ar("tp", 0.05))

        # ---------- 负荷与环境 ----------
        ox_design = qd * (0.4 * cfg.cod_base + 4.57 * cfg.nh3n_base)
        ox_load = q * (0.4 * cod_in + 4.57 * nh3n_in) / ox_design
        s.q_smooth = _lag(s.q_smooth, q, 2.0, dt_h)
        s.n_load_smooth = _lag(s.n_load_smooth, q * nh3n_in / (qd * cfg.nh3n_base), 2.0, dt_h)
        hrt = min(1.15, (qd / s.q_smooth) ** 0.8)
        f_temp = 1.07 ** (water_temp - 20)

        # 污泥流失：流量超过 1.3 倍设计值时 MLSS 下降，之后缓慢恢复
        ratio = q / qd
        if ratio > 1.3:
            s.mlss -= 600 * (ratio - 1.3) * dt_h
        else:
            s.mlss += (MLSS_DESIGN - s.mlss) * (1 - math.exp(-dt_h / 12))
        s.mlss = max(1500.0, s.mlss)
        f_mlss = s.mlss / MLSS_DESIGN

        # ---------- 曝气：PI 控制风量，氧平衡更新 DO ----------
        running = max(0, cfg.blowers_running - d.blowers_lost)
        air_max = running * cfg.blower_capacity * cfg.air_design
        air_min = 0.3 * cfg.air_design if running else 0.0
        if self.controller is None:
            err = DO_SETPOINT - s.do1
            air_cmd = cfg.air_design * (1.0 + 0.25 * err + 0.6 * s.air_integral)
            saturated = (air_cmd >= air_max and err > 0) or (air_cmd <= air_min and err < 0)
            if not saturated:  # 条件积分抗饱和
                s.air_integral = max(-3.0, min(3.0, s.air_integral + err * dt_h))
            s.air = max(air_min, min(air_max, air_cmd))
        else:
            air_cmd = self.controller.command(AerationObs(
                ts, dt_h, s.do_meas, q, cod_in, nh3n_in, s.air, air_min, air_max, cfg.air_design))
            air_cmd = max(air_min, min(air_max, air_cmd))
            # 鼓风机变频 + 阀门的一阶滞后；风机跳停时风量上限立即生效
            s.air = _lag(s.air, air_cmd, self.actuator_tau_s / 3600, dt_h) if self.actuator_tau_s else air_cmd
            s.air = max(air_min, min(air_max, s.air))

        our = 30.0 * ox_load * (0.6 + 0.4 * f_mlss)
        kla = 4.3 * s.air / cfg.air_design
        # 子步积分保证 dt=60s 时数值稳定
        n_sub = max(1, int(dt_s // 10))
        h_sub = dt_h / n_sub
        for _ in range(n_sub):
            s.do1 += (kla * (DO_SAT - s.do1) - our) * h_sub
            s.do1 = max(0.05, s.do1)
        do1 = s.do1 + self._white(0.04)
        do2 = s.do1 * 1.08 + 0.12 + self._white(0.05)
        if self.controller is not None:   # 复用同一个噪声样本，不额外消耗随机数
            s.do_probe = _lag(s.do_probe, s.do1, self.sensor_tau_s / 3600, dt_h) if self.sensor_tau_s else s.do1
            s.do_meas = max(0.0, s.do_probe + (do1 - s.do1))

        # ---------- 硝化：出水氨氮 ----------
        r = NIT_CAPACITY * f_temp * hrt * f_mlss * (s.do1 / (K_O_NIT + s.do1)) / max(s.n_load_smooth, 0.2)
        nh3_target = nh3n_in * math.exp(-NIT_K * r)
        s.nh3n_out = _lag(s.nh3n_out, nh3_target, 1.0, dt_h)
        nh3n_out = max(0.05, s.nh3n_out * (1 + self._white(0.03)))

        # ---------- 反硝化：碳源投加（前馈 + PI） ----------
        n_to_denit = max(0.0, tn_in - s.nh3n_out - ORGANIC_N_OUT - N_ASSIMILATION)   # g/m³
        no3_target = max(0.0, TN_TARGET - s.nh3n_out - ORGANIC_N_OUT)
        cod_need = q * max(0.0, n_to_denit - no3_target) * COD_PER_N / 1000            # kgCOD/h
        cod_avail_in = q * cod_in * 0.42 / 1000
        ff = max(0.0, cod_need - cod_avail_in) / ACETATE_COD
        carbon_max = qd * 120 / 1000  # 计量泵上限：折合 120 mg/L
        tn_err = (s.tn_out - TN_TARGET) / TN_TARGET
        carbon_cmd = ff * (1 + 0.5 * s.carbon_integral) + 0.002 * qd * s.carbon_integral
        if not ((carbon_cmd >= carbon_max and tn_err > 0) or (carbon_cmd <= 0 and tn_err < 0)):
            s.carbon_integral = max(-0.8, min(4.0, s.carbon_integral + 2.0 * tn_err * dt_h))
        s.carbon_dose = max(0.0, min(carbon_max, carbon_cmd))
        carbon_eff = s.carbon_dose * d.carbon_effective
        cod_total = cod_avail_in + carbon_eff * ACETATE_COD
        denit = min(q * n_to_denit * 0.92 / 1000, cod_total / COD_PER_N)               # kgN/h
        no3_out = max(0.0, n_to_denit - denit * 1000 / q)
        tn_target = s.nh3n_out + no3_out + ORGANIC_N_OUT
        s.tn_out = _lag(s.tn_out, tn_target, 1.5, dt_h)
        tn_out = max(nh3n_out + 1, s.tn_out * (1 + self._white(0.02)))

        # ---------- 除磷：PAC 投加（前馈 + PI） ----------
        bio_p = 0.75 * min(1.0, cod_in / cfg.cod_base + 0.2)
        tp_bio = tp_in * (1 - bio_p)
        pac_ff_mgl = math.log(max(tp_bio, TP_TARGET + 1e-3) / TP_TARGET) / PAC_K
        pac_max = 2.5 * 15 * qd / 1000
        tp_err = (s.tp_out - TP_TARGET) / TP_TARGET
        pac_cmd = pac_ff_mgl * (1 + 0.3 * s.pac_integral) * q / 1000
        if not ((pac_cmd >= pac_max and tp_err > 0) or (pac_cmd <= 0 and tp_err < 0)):
            s.pac_integral = max(-1.0, min(6.0, s.pac_integral + tp_err * dt_h))
        s.pac_dose = max(0.0, min(pac_max, pac_cmd))
        pac_eff_mgl = s.pac_dose * d.pac_effective * 1000 / q
        tp_target = tp_bio * math.exp(-PAC_K * pac_eff_mgl)
        s.tp_out = _lag(s.tp_out, tp_target, 1.0, dt_h)
        tp_out = max(0.02, s.tp_out * (1 + self._white(0.04)))

        # ---------- COD / SS ----------
        cod_target = 16 + 0.04 * cod_in * max(1.0, ox_load) / hrt
        s.cod_out = _lag(s.cod_out, cod_target, 1.0, dt_h)
        ss_target = 5.5 + 6.0 * max(0.0, ratio - 1.2)
        s.ss_out = _lag(s.ss_out, ss_target, 0.7, dt_h)

        blower_kw = cfg.blower_kw_design * (s.air / cfg.air_design) ** 1.3 if s.air > 0 else 0.0

        out = {
            "inflow_m3h": q,
            "cod_in": cod_in,
            "nh3n_in": nh3n_in,
            "tn_in": tn_in,
            "tp_in": tp_in,
            "cod_out": s.cod_out * (1 + self._white(0.03)),
            "nh3n_out": nh3n_out,
            "tn_out": tn_out,
            "tp_out": tp_out,
            "ss_out": max(1.0, s.ss_out * (1 + self._white(0.05))),
            "do_aerobic_1": max(0.0, do1),
            "do_aerobic_2": max(0.0, do2),
            "air_flow_m3h": s.air * (1 + self._white(0.01)),
            "blower_kw": blower_kw * (1 + self._white(0.015)),
            "carbon_dose_kgh": s.carbon_dose,
            "pac_dose_kgh": s.pac_dose,
            "mlss": s.mlss * (1 + self._white(0.01)),
            # 以下为内部量，供设备模型使用，不入库
            "_air_share": s.air / (running * cfg.blower_capacity * cfg.air_design) if running else 0.0,
            "_q_ratio": ratio,
        }
        return {k: float(v) for k, v in out.items()}

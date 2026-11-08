"""把厂站模型、设备、气象、场景组合成一个按时间步推进的仿真器。纯计算，不碰数据库。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime

import numpy as np

from .model import PlantConfig, PlantModel, PlantState, local_hour
from .scenarios import Scenario, ScenarioBook

AMPS_PER_KW = 1.79  # 380V、功率因数 0.85：I = P / (√3 · 0.38 · 0.85)

# 雨污混接程度（入流入渗系数）。P03 梁山镇管网老旧，是暴雨演示的主角
II_FACTOR = {"P01": 0.25, "P02": 0.30, "P03": 0.75, "P04": 0.35, "P05": 0.30, "P06": 0.25, "P07": 0.40}


@dataclass(frozen=True)
class DeviceConfig:
    id: int
    plant_id: int
    code: str
    kind: str
    rated_kw: float
    standby: bool

    @property
    def abbr(self) -> str:  # P01-BL-1 → BL-1
        return self.code.split("-", 1)[1]


def plant_config(row: dict) -> PlantConfig:
    """由 plant 表的一行构造模型参数。"""
    center = row["kind"] == "中心厂"
    return PlantConfig(
        id=row["id"],
        code=row["code"],
        kind=row["kind"],
        process=row["process"],
        capacity_m3d=row["capacity_m3d"],
        ii_factor=II_FACTOR.get(row["code"], 0.3),
        cod_base=280.0 if center else 240.0,
        nh3n_base=34.0 if center else 30.0,
        tn_base=45.0 if center else 40.0,
        tp_base=4.5 if center else 4.0,
        blowers=3 if center else 2,
        blowers_running=2 if center else 1,
        blower_capacity=0.85 if center else 1.6,
    )


@dataclass
class Snapshot:
    ts: datetime
    weather: dict
    plants: dict[int, dict]          # plant_id → plant_reading 行（不含 ts/plant_id）
    devices: list[dict]              # device_reading 行（含 device_id）
    device_status: dict[int, str] = field(default_factory=dict)  # 状态有变化的设备


def air_temp(ts: datetime) -> float:
    """11 月汉中气温：日均约 12℃，日较差 8℃，14 点最高。"""
    hour = local_hour(ts)
    return 12.0 + 4.0 * math.cos((hour - 14) / 24 * 2 * math.pi)


class Simulator:
    def __init__(self, plants: list[PlantConfig], devices: list[DeviceConfig], seed: int = 42):
        self.plants = {p.id: p for p in plants}
        self.models = {p.id: PlantModel(p, np.random.default_rng([seed, p.id])) for p in plants}
        self.devices = devices
        self.rng = np.random.default_rng([seed, 0])
        self.book = ScenarioBook()
        self._status: dict[int, str] = {d.id: ("备用" if d.standby else "运行") for d in devices}

    def dump_state(self) -> dict:
        """可 JSON 序列化的运行状态（水质与控制器状态、设备状态、场景），重启后由 load_state 恢复。"""
        return {
            "plants": {str(pid): asdict(m.s) for pid, m in self.models.items()},
            "device_status": {str(did): status for did, status in self._status.items()},
            "scenarios": [sc.to_dict() for sc in self.book.items],
        }

    def load_state(self, state: dict) -> None:
        """厂站、设备或状态字段与保存时不同的（台账或代码已更新）跳过，其余照常恢复。"""
        known = {f.name for f in fields(PlantState)}
        for pid, saved in state["plants"].items():
            if model := self.models.get(int(pid)):
                model.s = PlantState(**{k: v for k, v in saved.items() if k in known})
        for did, status in state["device_status"].items():
            if int(did) in self._status:
                self._status[int(did)] = status
        self.book.items = [Scenario.from_dict(d) for d in state["scenarios"]]

    def step(self, ts: datetime, dt_s: float) -> Snapshot:
        rain = self.book.rain(ts)
        t_air = air_temp(ts)
        water_temp = 16.0 + 0.15 * (t_air - 12.0)
        weather = {"rain_mm_h": float(rain), "temp_c": float(t_air + 0.3 * self.rng.standard_normal())}

        plant_rows: dict[int, dict] = {}
        for pid, model in self.models.items():
            d = self.book.disturbance(model.cfg.code, ts)
            plant_rows[pid] = model.step(ts, dt_s, d, water_temp)

        device_rows, changed = [], {}
        for dev in self.devices:
            row, status = self._device(dev, ts, plant_rows[dev.plant_id], t_air)
            device_rows.append(row)
            if self._status[dev.id] != status:
                self._status[dev.id] = status
                changed[dev.id] = status

        for row in plant_rows.values():
            row.pop("_air_share", None)
            row.pop("_q_ratio", None)
        return Snapshot(ts, weather, plant_rows, device_rows, changed)

    def _device(self, dev: DeviceConfig, ts: datetime, p: dict, t_air: float) -> tuple[dict, str]:
        cfg = self.plants[dev.plant_id]
        faults = self.book.device_faults(cfg.code, ts)
        fault = faults.get(dev.abbr, 0.0)
        n = self.rng.standard_normal(3)

        if dev.standby:
            return self._row(dev, ts, False, 0.0, 0.3 + 0.05 * abs(n[1]), t_air + 3 + 0.3 * n[2]), "备用"

        status = "运行"
        if dev.kind == "鼓风机":
            if fault >= 1.0:
                return self._row(dev, ts, False, 0.0, 0.4 + 0.05 * abs(n[1]), t_air + 25 + 0.5 * n[2]), "故障"
            tripped = sum(1 for k, v in faults.items() if k.startswith("BL-") and v >= 1.0)
            kw = p["blower_kw"] / max(1, cfg.blowers_running - tripped)
            load = p["_air_share"]
            vib = 2.2 + 0.15 * n[1] + 7.5 * fault ** 2
            temp = 52 + 14 * load + 0.6 * n[2] + 36 * fault ** 1.5
        elif dev.kind == "提升泵":
            kw = dev.rated_kw * min(1.0, 0.45 + 0.35 * p["_q_ratio"])
            vib, temp = 2.8 + 0.2 * n[1], 45 + 10 * p["_q_ratio"] + 0.5 * n[2]
        elif dev.kind == "污泥泵":
            if ts.minute >= 20:  # 每小时前 20 分钟排泥
                return self._row(dev, ts, False, 0.0, 0.3, t_air + 8), status
            kw, vib, temp = dev.rated_kw * 0.85, 3.1 + 0.2 * n[1], 55 + 0.5 * n[2]
        elif dev.kind == "加药泵":
            kw = dev.rated_kw * (0.55 + 0.4 * fault)        # 堵塞时负载升高
            vib, temp = 1.5 + 0.1 * n[1] + 2.5 * fault, 40 + 0.4 * n[2] + 12 * fault
        else:  # 回流泵
            kw, vib, temp = dev.rated_kw * 0.8, 2.5 + 0.15 * n[1], 48 + 0.5 * n[2]

        current = kw * AMPS_PER_KW * (1 + 0.01 * n[0])
        return self._row(dev, ts, True, current, max(0.1, vib), temp), status

    @staticmethod
    def _row(dev: DeviceConfig, ts: datetime, running: bool, current: float, vib: float, temp: float) -> dict:
        return {"ts": ts, "device_id": dev.id, "running": running,
                "current_a": round(float(current), 2), "vibration_mms": round(float(vib), 2),
                "temp_c": round(float(temp), 1)}

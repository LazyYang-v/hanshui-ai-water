"""指标目录：字段名 → 中文名、单位、标准限值。

也是 SQL 中可用列名的白名单，API 不会把未登记的列名拼进 SQL。
限值取 GB 18918-2002 一级 A；DO 和设备指标为运行管理阈值。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    unit: str
    group: str
    limit: float | None = None        # 超过即「告警」（DO 为低于下限）
    warn_ratio: float = 0.8           # 达到 limit × warn_ratio 即「预警」
    lower_is_bad: bool = False        # True：低于 limit 为异常（如 DO）

    def level(self, value: float | None) -> Literal["normal", "warn", "alarm"]:
        if value is None or self.limit is None:
            return "normal"
        if self.lower_is_bad:
            if value < self.limit:
                return "alarm"
            return "warn" if value < self.limit / self.warn_ratio else "normal"
        if value > self.limit:
            return "alarm"
        return "warn" if value > self.limit * self.warn_ratio else "normal"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("warn_ratio")
        return d

    @property
    def warn_at(self) -> float | None:
        """预警边界：越过即「预警」（DO 为低于该值）。"""
        if self.limit is None:
            return None
        return round(self.limit / self.warn_ratio if self.lower_is_bad else self.limit * self.warn_ratio, 4)

    def info(self) -> dict:
        """接口目录用：在 to_dict 基础上附预警边界（统计工具输出仍用 to_dict，不变）。"""
        return {**self.to_dict(), "warn_at": self.warn_at}


PLANT_METRICS: dict[str, Metric] = {m.key: m for m in [
    Metric("inflow_m3h", "进水流量", "m³/h", "进水"),
    Metric("cod_in", "进水 COD", "mg/L", "进水"),
    Metric("nh3n_in", "进水氨氮", "mg/L", "进水"),
    Metric("tn_in", "进水总氮", "mg/L", "进水"),
    Metric("tp_in", "进水总磷", "mg/L", "进水"),
    Metric("cod_out", "出水 COD", "mg/L", "出水", limit=50),
    Metric("nh3n_out", "出水氨氮", "mg/L", "出水", limit=5),
    Metric("tn_out", "出水总氮", "mg/L", "出水", limit=15),
    Metric("tp_out", "出水总磷", "mg/L", "出水", limit=0.5),
    Metric("ss_out", "出水 SS", "mg/L", "出水", limit=10),
    Metric("do_aerobic_1", "好氧区 DO（1#）", "mg/L", "工艺", limit=1.0, warn_ratio=0.67, lower_is_bad=True),
    Metric("do_aerobic_2", "好氧区 DO（2#）", "mg/L", "工艺"),
    Metric("air_flow_m3h", "曝气风量", "m³/h", "工艺"),
    Metric("mlss", "污泥浓度 MLSS", "mg/L", "工艺"),
    Metric("blower_kw", "鼓风机功率", "kW", "能耗药耗"),
    Metric("carbon_dose_kgh", "碳源投加量", "kg/h", "能耗药耗"),
    Metric("pac_dose_kgh", "PAC 投加量", "kg/h", "能耗药耗"),
]}

EFFLUENT_KEYS = [k for k, m in PLANT_METRICS.items() if m.group == "出水"]

# 派生指标独立登记，不属于 plant_reading 原始列白名单。
LOAD_SOURCES = {"cod_load_kgh": "cod_in", "tn_load_kgh": "tn_in", "nh3n_load_kgh": "nh3n_in"}
DERIVED_METRICS = {key: Metric(key, PLANT_METRICS[source].label + "质量负荷", "kg/h", "进水负荷")
                   for key, source in LOAD_SOURCES.items()}

DEVICE_METRICS: dict[str, Metric] = {m.key: m for m in [
    Metric("current_a", "运行电流", "A", "设备"),
    Metric("vibration_mms", "振动速度", "mm/s", "设备", limit=7.1, warn_ratio=4.5 / 7.1),
    Metric("temp_c", "轴承温度", "℃", "设备", limit=80, warn_ratio=0.85),
]}

LEVEL_ORDER = {"normal": 0, "warn": 1, "alarm": 2}


def worst(levels) -> Literal["normal", "warn", "alarm"]:
    return max(levels, key=LEVEL_ORDER.__getitem__, default="normal")

"""可注入的演示场景。

每个场景有起止时间，ScenarioBook 在每个时间步把所有活跃场景折算成
- 区域降雨强度（storm）
- 各厂站的 Disturbance（model.py）
- 各设备的故障程度（devices.py）
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

from .model import LOCAL_TZ, Disturbance

KINDS = {
    "storm":        "暴雨：区域降雨，雨污混接严重的厂站进水量激增、浓度被稀释，持续大流量会冲走污泥",
    "blower_fault": "鼓风机故障：1 号鼓风机振动和温度逐渐升高，随后跳停，曝气能力下降",
    "pipe_leak":    "碳源管道渗漏：碳源实际进入生化池的量减少，控制系统加大投加量，表现为药耗异常",
    "dosing_block": "PAC 加药泵堵塞：除磷剂实际投加量骤降，出水总磷上升",
    "shock_load":   "进水冲击负荷：进水浓度突增（如工业废水偷排）",
}

DEFAULT_DURATION_MIN = {
    "storm": 180,
    "blower_fault": 240,
    "pipe_leak": 600,
    "dosing_block": 360,
    "shock_load": 180,
}

BLOWER_RAMP_MIN = 30  # 鼓风机故障：振动/温度爬升到跳停的时长


@dataclass
class Scenario:
    kind: str
    start: datetime
    duration_min: float
    plant_code: str | None = None  # storm 为区域性，可为空
    params: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"未知场景 {self.kind!r}，可选：{', '.join(KINDS)}")
        if self.kind != "storm" and not self.plant_code:
            raise ValueError(f"场景 {self.kind} 需要指定厂站")

    @property
    def end(self) -> datetime:
        return self.start + timedelta(minutes=self.duration_min)

    def active(self, ts: datetime) -> bool:
        return self.start <= ts < self.end

    def elapsed_min(self, ts: datetime) -> float:
        return (ts - self.start).total_seconds() / 60

    def rain(self, ts: datetime) -> float:
        """降雨过程线：20 分钟爬升，保持，最后 40 分钟回落。"""
        peak = float(self.params.get("peak_mm_h", 18.0))
        t, dur = self.elapsed_min(ts), self.duration_min
        rise, fall = min(20.0, dur / 3), min(40.0, dur / 3)
        if t < rise:
            return peak * t / rise
        if t > dur - fall:
            return peak * max(0.0, (dur - t) / fall)
        return peak

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.astimezone(LOCAL_TZ).isoformat()
        d["end"] = self.end.astimezone(LOCAL_TZ).isoformat()
        d["description"] = KINDS[self.kind]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Scenario:
        """to_dict 的逆操作，end、description 由其余字段推出。"""
        return cls(d["kind"], datetime.fromisoformat(d["start"]), d["duration_min"], d["plant_code"], d["params"], d["id"])


class ScenarioBook:
    def __init__(self):
        self.items: list[Scenario] = []

    def add(self, sc: Scenario) -> Scenario:
        self.items.append(sc)
        return sc

    def clear(self, plant_code: str | None = None, kind: str | None = None, at: datetime | None = None) -> int:
        """结束匹配的场景。给定 at 时把结束时间截到 at（保留历史），否则直接删除。"""
        n = 0
        for sc in list(self.items):
            if plant_code and sc.plant_code != plant_code:
                continue
            if kind and sc.kind != kind:
                continue
            if at is None:
                self.items.remove(sc)
            elif sc.active(at):
                sc.duration_min = sc.elapsed_min(at)
            else:
                continue
            n += 1
        return n

    def active(self, ts: datetime) -> list[Scenario]:
        return [sc for sc in self.items if sc.active(ts)]

    def prune(self, ts: datetime) -> None:
        self.items = [sc for sc in self.items if sc.end > ts]

    def rain(self, ts: datetime) -> float:
        return max((sc.rain(ts) for sc in self.items if sc.kind == "storm" and sc.active(ts)), default=0.0)

    def disturbance(self, plant_code: str, ts: datetime) -> Disturbance:
        d = Disturbance(rain_mm_h=self.rain(ts))
        for sc in self.items:
            if sc.plant_code != plant_code or not sc.active(ts):
                continue
            if sc.kind == "blower_fault" and sc.elapsed_min(ts) >= BLOWER_RAMP_MIN:
                d.blowers_lost += 1
            elif sc.kind == "pipe_leak":
                d.carbon_effective *= float(sc.params.get("effective", 0.5))
            elif sc.kind == "dosing_block":
                d.pac_effective *= float(sc.params.get("effective", 0.2))
            elif sc.kind == "shock_load":
                d.load_factor *= float(sc.params.get("factor", 1.8))
        return d

    def device_faults(self, plant_code: str, ts: datetime) -> dict[str, float]:
        """设备缩写（如 BL-1）→ 故障程度 0~1；1 表示已跳停/堵死。"""
        out: dict[str, float] = {}
        for sc in self.items:
            if sc.plant_code != plant_code or not sc.active(ts):
                continue
            if sc.kind == "blower_fault":
                out["BL-1"] = min(1.0, sc.elapsed_min(ts) / BLOWER_RAMP_MIN)
            elif sc.kind == "dosing_block":
                out["DP-P"] = 1.0
        return out

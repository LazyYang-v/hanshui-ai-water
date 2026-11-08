"""API 响应模型（同时生成 OpenAPI 文档）。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["normal", "warn", "alarm", "offline"]
Level = Literal["normal", "warn", "alarm"]


class MetricInfo(BaseModel):
    key: str
    label: str
    unit: str
    group: str
    limit: float | None = Field(None, description="限值；lower_is_bad 时为下限")
    lower_is_bad: bool = False
    warn_at: float | None = Field(None, description="预警边界；越过即预警（lower_is_bad 时为低于该值）")


class Issue(BaseModel):
    metric: str | None
    label: str
    level: Level
    value: float | None
    limit: float | None


class PlantSummary(BaseModel):
    code: str
    name: str
    kind: str
    process: str
    capacity_m3d: int
    lng: float
    lat: float
    status: Status = Field(description="normal 正常 / warn 预警 / alarm 告警 / offline 数据中断")
    issues: list[Issue]
    latest_ts: datetime | None
    latest: dict[str, float] | None = Field(description="最新读数，键见 /api/metrics")


class PeriodStats(BaseModel):
    hours: float = Field(description="有数据的时长，小时")
    water_m3: int = Field(description="处理水量，m³")
    aeration_kwh: float = Field(description="鼓风机耗电，kWh")
    aeration_kwh_per_m3: float | None = Field(description="吨水曝气电耗，kWh/m³")
    carbon_kg: float = Field(description="碳源投加量，kg")
    pac_kg: float = Field(description="PAC 投加量，kg")
    compliance_rate: float | None = Field(description="出水达标率（5 分钟粒度，0~1）")


class Device(BaseModel):
    code: str
    kind: str
    model: str
    rated_kw: float
    installed_at: date
    status: str
    level: Level
    latest_ts: datetime | None
    running: bool | None
    current_a: float | None
    vibration_mms: float | None
    temp_c: float | None


class PlantDetail(PlantSummary):
    today: PeriodStats | None
    devices: list[Device]


class Series(BaseModel):
    entity: str
    bucket: str
    start: datetime
    end: datetime
    metrics: list[MetricInfo]
    ts: list[datetime]
    series: dict[str, list[float | None]]


class Weather(BaseModel):
    ts: datetime | None
    rain_mm_h: float | None
    temp_c: float | None


class HourlyPoint(BaseModel):
    ts: datetime
    inflow_m3h: float
    blower_kw: float
    rain_mm_h: float


class TodaySummary(BaseModel):
    water_m3: int
    aeration_kwh: float
    aeration_kwh_per_m3: float | None
    carbon_kg: float
    pac_kg: float
    compliance_rate: float | None


class Overview(BaseModel):
    now: datetime
    weather: Weather
    today: TodaySummary
    status_counts: dict[str, int]
    plants: list[PlantSummary]
    hourly: list[HourlyPoint] = Field(description="最近 24 小时全区逐小时数据")
    scenarios: list[dict] = Field(description="正在进行的演示场景")


class Clock(BaseModel):
    now: datetime = Field(description="演示时钟")
    wall: datetime = Field(description="墙上时间")
    ahead_s: float = Field(description="演示时钟领先墙上时间的秒数")
    speed: float


class ScenarioIn(BaseModel):
    kind: Literal["storm", "blower_fault", "pipe_leak", "dosing_block", "shock_load"]
    plant: str | None = Field(None, examples=["P03"])
    duration_min: float | None = Field(None, gt=0, le=7 * 24 * 60)
    params: dict[str, float] = Field(default_factory=dict, examples=[{"peak_mm_h": 20}])


class SpeedIn(BaseModel):
    value: float = Field(gt=0, le=600)


class CommandResult(BaseModel):
    message: str

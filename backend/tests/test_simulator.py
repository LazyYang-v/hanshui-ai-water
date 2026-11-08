"""模拟器标定测试：保证数据在正常工况下合规，在各场景下呈现预期的异常特征。"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from simulator.engine import DeviceConfig, Simulator, plant_config
from simulator.scenarios import Scenario, ScenarioBook

T0 = datetime(2026, 11, 1, tzinfo=ZoneInfo("Asia/Shanghai"))
DT = 60

PLANTS = [
    dict(id=1, code="P01", kind="中心厂", process="A2O", capacity_m3d=50000),
    dict(id=2, code="P02", kind="集镇厂", process="氧化沟", capacity_m3d=2000),
    dict(id=3, code="P03", kind="集镇厂", process="A2O", capacity_m3d=5000),
]
DEVICES = [
    DeviceConfig(21, 1, "P01-BL-1", "鼓风机", 110, False),
    DeviceConfig(22, 1, "P01-BL-2", "鼓风机", 110, False),
    DeviceConfig(23, 1, "P01-BL-3", "鼓风机", 110, True),
    DeviceConfig(49, 2, "P02-DP-P", "加药泵", 0.75, False),
]


def run(scenarios=(), hours=48, start=T0, seed=42):
    sim = Simulator([plant_config(p) for p in PLANTS], DEVICES, seed=seed)
    for sc in scenarios:
        sim.book.add(sc)
    out = []
    for i in range(int(hours * 3600 / DT)):
        out.append(sim.step(start + timedelta(seconds=i * DT), DT))
    return out


def series(snaps, plant_id, key, since=None, until=None):
    return np.array([s.plants[plant_id][key] for s in snaps
                     if (since is None or s.ts >= since) and (until is None or s.ts < until)])


def at(h):
    return T0 + timedelta(hours=h)


@pytest.fixture(scope="module")
def normal():
    return run(hours=72)


@pytest.mark.parametrize("pid", [1, 2, 3])
def test_normal_effluent_meets_grade_1a(normal, pid):
    warm = at(12)
    assert series(normal, pid, "nh3n_out", warm).max() < 3
    assert series(normal, pid, "tn_out", warm).max() < 15
    assert series(normal, pid, "tp_out", warm).max() < 0.5
    assert series(normal, pid, "cod_out", warm).max() < 50
    assert series(normal, pid, "ss_out", warm).max() < 10
    assert abs(series(normal, pid, "do_aerobic_1", warm).mean() - 2.0) < 0.2


def test_inflow_follows_design_capacity_and_diurnal(normal):
    q = series(normal, 1, "inflow_m3h", at(24), at(48))
    assert 0.9 < q.mean() / (50000 / 24) < 1.1
    hours = [s.ts.hour for s in normal if at(24) <= s.ts < at(48)]
    by_hour = {h: q[[i for i, x in enumerate(hours) if x == h]].mean() for h in range(24)}
    assert by_hour[9] > by_hour[4] * 1.25  # 早高峰明显高于凌晨（理论值约 1.5 倍，叠加随机扰动）


def test_storm_causes_nh3_exceedance_at_p03_then_recovers():
    storm = Scenario("storm", at(10), 180)
    snaps = run([storm], hours=36)
    nh3 = series(snaps, 3, "nh3n_out", at(10), at(18))
    assert nh3.max() > 5, "暴雨后 P03 出水氨氮应超标"
    assert series(snaps, 3, "nh3n_out", at(30), at(36)).max() < 3, "雨停后应恢复"
    assert series(snaps, 3, "mlss", at(10), at(18)).min() < 2800, "大流量应冲走污泥"
    # 中心厂雨污分流较好，不超标
    assert series(snaps, 1, "nh3n_out", at(10), at(18)).max() < 5


def test_dosing_block_raises_tp():
    snaps = run([Scenario("dosing_block", at(12), 360, "P02")], hours=20)
    assert series(snaps, 2, "tp_out", at(12), at(18)).max() > 0.5
    assert series(snaps, 1, "tp_out", at(12), at(18)).max() < 0.5


def test_pipe_leak_shows_as_carbon_dose_anomaly():
    base = run(hours=30)
    leak = run([Scenario("pipe_leak", at(12), 600, "P03")], hours=30)
    window = (at(16), at(22))
    ratio = series(leak, 3, "carbon_dose_kgh", *window).mean() / series(base, 3, "carbon_dose_kgh", *window).mean()
    assert ratio > 1.3, f"碳源投加应异常升高，实际为 {ratio:.2f} 倍"


def test_blower_fault_vibration_precedes_trip_and_do_drops():
    snaps = run([Scenario("blower_fault", at(12), 240, "P01")], hours=18)
    bl1 = [(s.ts, next(d for d in s.devices if d["device_id"] == 21)) for s in snaps]
    trip_ts = next(ts for ts, d in bl1 if not d["running"])
    pre = [d["vibration_mms"] for ts, d in bl1 if at(12) <= ts < trip_ts]
    assert max(pre) > 7.1, "跳停前振动应超过报警值"
    assert at(12) + timedelta(minutes=25) <= trip_ts <= at(12) + timedelta(minutes=35)
    assert any(s.device_status.get(21) == "故障" for s in snaps)
    assert series(snaps, 1, "do_aerobic_1", at(13), at(16)).mean() < 1.5


def test_deterministic_with_same_seed():
    a = run(hours=2, seed=7)
    b = run(hours=2, seed=7)
    assert a[-1].plants == b[-1].plants


def test_values_are_plain_floats():
    s = run(hours=0.1)[-1]
    assert all(type(v) is float for v in s.plants[1].values())
    assert all(not k.startswith("_") for k in s.plants[1])


def test_scenario_validation_and_clear():
    with pytest.raises(ValueError):
        Scenario("earthquake", T0, 10, "P01")
    with pytest.raises(ValueError):
        Scenario("pipe_leak", T0, 10)  # 需要厂站
    book = ScenarioBook()
    sc = book.add(Scenario("pipe_leak", T0, 600, "P03"))
    assert book.clear(plant_code="P03", at=at(1)) == 1
    assert sc.duration_min == 60 and not sc.active(at(2))

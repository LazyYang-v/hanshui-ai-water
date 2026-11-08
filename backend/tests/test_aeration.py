"""智能曝气仿真（M6）：控制器接口不影响原模拟器、对照实验的公平性、各组件的基本性质、接口。"""

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import aeration as aeration_api
from simulator.aeration import (
    EVAL_DAY,
    P01,
    PI,
    AerationObs,
    HoltWinters,
    Instruments,
    LoadFeedforward,
    Params,
    compare,
    downsample,
    metrics,
    simulate,
)
from simulator.model import DO_SAT, PlantModel
from simulator.scenarios import ScenarioBook

TZ = ZoneInfo("Asia/Shanghai")


def obs(**kw) -> AerationObs:
    base = dict(ts=EVAL_DAY, dt_h=1 / 60, do_meas=2.0, inflow=P01.q_design, cod_in=P01.cod_base,
                nh3n_in=P01.nh3n_base, air=P01.air_design, air_min=0.3 * P01.air_design,
                air_max=1.7 * P01.air_design, air_design=P01.air_design)
    return AerationObs(**{**base, **kw})


# ---------------------------------------------------------------- 原模拟器不受影响

def test_legacy_path_unchanged():
    """不传控制器时走原内置 PI，数值与加控制器接口之前逐位一致（此前用 3 厂 72 小时指纹核对过）。"""
    m = PlantModel(P01, np.random.default_rng(7))
    book = ScenarioBook()
    t0 = datetime(2026, 11, 1, tzinfo=TZ)
    do, air = 0.0, 0.0
    for i in range(360):
        ts = t0 + timedelta(minutes=i)
        m.step(ts, 60, book.disturbance("P01", ts))
        do += m.s.do1
        air += m.s.air
    assert do == pytest.approx(714.3531827310085, rel=1e-12)
    assert air == pytest.approx(3861559.756931784, rel=1e-12)


# ---------------------------------------------------------------- 公平性

def test_strategies_share_the_same_disturbances():
    p = Params.load()
    a = simulate("fixed", params=p, scenario="storm", seed=3, warmup_days=1)
    b = simulate("ff_hw_pi", params=p, scenario="storm", seed=3, warmup_days=1)
    assert a.series["inflow"] == b.series["inflow"]
    assert a.series["load"] == b.series["load"]
    assert a.series["air_frac"] != b.series["air_frac"]


def test_controller_sees_lagged_noisy_do_not_truth():
    class Spy:
        def __init__(self):
            self.seen = []

        def command(self, o):
            self.seen.append(o.do_meas)
            return o.air_design

    spy = Spy()
    run = simulate("fixed", controller=spy, seed=5, warmup_days=0)
    seen = np.array(spy.seen[1:])               # 第 k 步看到的是第 k-1 步末的测量
    true = np.array(run.series["do"][:-1])
    meas = np.array(run.series["do_meas"][:-1])
    assert np.allclose(seen, meas)
    assert 0.02 < np.std(meas - true) < 0.3


# ---------------------------------------------------------------- 组件

def test_holt_winters_learns_daily_pattern():
    hw = HoltWinters(alpha=0.2, gamma=0.2)
    t0 = datetime(2026, 11, 1, tzinfo=TZ)

    def y(ts):
        h = ts.hour + ts.minute / 60
        return 1 + 0.3 * math.sin(2 * math.pi * h / 24)

    for i in range(3 * 1440):
        ts = t0 + timedelta(minutes=i)
        hw.update(ts, y(ts))
    now = t0 + timedelta(days=3, hours=4)
    for h in (0, 30, 120):
        assert hw.forecast(now, h) == pytest.approx(y(now + timedelta(minutes=h)), rel=0.02)


def test_holt_winters_falls_back_to_last_value_before_first_day():
    hw = HoltWinters(0.3, 0.2)
    hw.update(EVAL_DAY, 1.23)
    assert hw.forecast(EVAL_DAY, 60) == 1.23


def test_pi_conditional_integration_stops_windup():
    pi = PI(2.0, kp=0.5, ki=4.8)
    for _ in range(600):                         # 设定值够不着，指令顶在上限
        pi.command(obs(do_meas=0.5, air=1.7 * P01.air_design))
    stuck = pi.integral
    assert stuck * pi.ki <= 1.5 + 1e-9
    pi.command(obs(do_meas=2.5))                  # 误差反号时立即开始回退
    assert pi.integral < stuck


def test_pi_without_feedforward_biases_at_design_air():
    assert PI(2.0, kp=0.5, ki=1.0).command(obs(do_meas=2.0)) == pytest.approx(P01.air_design)


def test_feedforward_scales_with_oxygen_driving_force():
    def ff(sp):
        return LoadFeedforward(0.1, 0.9, sp, Instruments(P01, np.random.default_rng(1)))(obs())

    assert ff(2.5) / ff(2.0) == pytest.approx((DO_SAT - 2.0) / (DO_SAT - 2.5))


def test_instruments_sample_and_hold():
    ins = Instruments(P01, np.random.default_rng(0), hold_min=10)
    vals = [ins.load(obs(ts=EVAL_DAY + timedelta(minutes=i), inflow=P01.q_design)) for i in range(25)]
    assert all(abs(v - 1) < 0.25 for v in vals)
    # 分析仪 10 分钟出一个值；流量计每步都有噪声，所以只比较分析仪换值处的跳变幅度
    jumps = np.abs(np.diff(vals))
    assert jumps[9] > np.median(jumps)


def test_metrics_on_known_series():
    n = 60
    series = {"do": [2.5] * n, "air_frac": [1.0] * (n // 2) + [1.2] * (n // 2), "kw": [100.0] * n,
              "nh3n_out": [6.0] * 30 + [1.0] * 30, "inflow": [1000.0] * n}
    m = metrics(series, setpoint=2.0, dt_s=60)
    assert m["do_mae"] == 0.5
    assert m["do_low_pct"] == 0 and m["do_high_pct"] == 0
    assert m["kwh"] == 100.0                     # 100 kW × 1 h
    assert m["kwh_per_m3"] == 0.1
    assert m["nh3n_over_limit_h"] == 0.5
    assert m["air_moves_per_h"] == pytest.approx(0.2)


def test_downsample_keeps_first_timestamp_and_means_values():
    ds = downsample({"t_min": [0, 1, 2, 3, 4, 5], "do": [1, 2, 3, 4, 5, 6]}, every=3)
    assert ds == {"t_min": [0.0, 3.0], "do": [2.0, 5.0]}


# ---------------------------------------------------------------- 对照结果

@pytest.fixture(scope="module")
def dry():
    return compare("dry", seed=101, setpoint=2.0)


def test_compare_shape_and_notice(dry):
    assert "仅用于说明方法" in dry["notice"]
    assert set(dry["strategies"]) == {"fixed", "pi", "ff_pi", "ff_hw_pi"}
    assert len(dry["series"]["t_min"]) == 288    # 24 h / 5 min
    for s in dry["strategies"].values():
        assert set(s["series"]) == {"do", "air_frac", "kw", "nh3n_out"}
        assert all(len(v) == 288 for v in s["series"].values())
    assert dry["strategies"]["fixed"]["metrics"]["kwh_vs_fixed_pct"] == 0


def test_feedback_tracks_setpoint_and_uses_less_energy_than_fixed(dry):
    m = {k: v["metrics"] for k, v in dry["strategies"].items()}
    for s in ("pi", "ff_pi", "ff_hw_pi"):
        assert m[s]["do_mae"] < 0.15
        assert m[s]["kwh"] < m["fixed"]["kwh"]
        assert m[s]["air_moves_per_h"] < 0.7     # 整定时的动作预算 0.5，评测种子允许一点余量
    assert m["fixed"]["do_high_pct"] > 30        # 按高峰配风，大部分时间过曝气


def test_higher_setpoint_costs_more_energy():
    lo = simulate("ff_pi", setpoint=1.5, seed=102).metrics
    hi = simulate("ff_pi", setpoint=2.5, seed=102).metrics
    assert hi["kwh"] > lo["kwh"]
    assert lo["do_mae"] < 0.15 and hi["do_mae"] < 0.15


def test_unknown_strategy_and_scenario_raise():
    with pytest.raises(ValueError):
        simulate("mpc")
    with pytest.raises(ValueError):
        simulate("pi", scenario="flood")


# ---------------------------------------------------------------- 接口

@pytest.fixture
def api():
    app = FastAPI()
    app.include_router(aeration_api.router)
    aeration_api._cached.cache_clear()
    return TestClient(app)


def test_api_options(api):
    r = api.get("/api/aeration/options")
    assert r.status_code == 200
    body = r.json()
    assert [s["id"] for s in body["strategies"]] == ["fixed", "pi", "ff_pi", "ff_hw_pi"]
    assert {s["id"] for s in body["scenarios"]} == {"dry", "storm", "shock", "overload"}
    assert "仅用于说明方法" in body["notice"]


def test_api_simulate_rounds_setpoint_and_caches(api):
    body = {"scenario": "shock", "setpoint": 2.04, "seed": 7, "strategies": ["pi", "fixed", "pi"]}
    r = api.post("/api/aeration/simulate", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["setpoint"] == 2.0
    assert list(out["strategies"]) == ["fixed", "pi"]
    api.post("/api/aeration/simulate", json={**body, "setpoint": 1.96})
    assert aeration_api._cached.cache_info().hits == 1


@pytest.mark.parametrize("body", [{"setpoint": 0.5}, {"setpoint": 3.5}, {"scenario": "flood"},
                                  {"strategies": []}, {"strategies": ["mpc"]}, {"seed": 0}])
def test_api_simulate_validates(api, body):
    assert api.post("/api/aeration/simulate", json=body).status_code == 422

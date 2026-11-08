"""智能曝气三策略对照（M6）：先在校准种子的旱天上整定参数，再在另一组种子、三种场景下评测。

    uv run python -m evals.aeration.run calibrate   # 写 simulator/aeration_params.json 与 results/calibration.json
    uv run python -m evals.aeration.run evaluate    # 写 results/eval.json，打印汇总表

整定（只用校准种子 1～5、旱天、设定值 2.0），每一步都在动作预算内（风量每小时累计变化 ≤ 0.5 倍设计风量）
取 DO 平均绝对误差最小；只看误差会选出最大增益，把 DO 测量噪声放大到鼓风机上：
  1. PI 增益：网格搜索；
  2. 前馈系数：用整定后的 PI 运行数据，10 分钟均值上回归「风量 ~ a + b·实测负荷」；
  3. 前馈 + PI 的增益与前馈滤波时间：网格搜索；
  4. Holt-Winters 的 α、γ、预测提前量、滤波时间与 PI 增益：网格搜索（各策略各自整定，互不沿用）；
  5. 固定风量：整定后 PI 在校准日风量的 95% 分位（按旱天高峰配风），另报告 50%/75% 分位作敏感性。
评测：种子 101～110 × 旱天/暴雨/冲击负荷/超能力冲击，所有策略共用同一组随机扰动；
另单独检验 Holt-Winters 的负荷预测精度（对比「沿用当前测量值」），把「预测准不准」和「预测对控制有没有用」分开。
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from simulator.model import diurnal

from simulator.aeration import (EVAL_DAY, MOVE_BUDGET, P01, PARAMS_PATH, SCENARIOS, STRATEGIES, PI, HoltWinters,
                                Instruments, Params, simulate)

RESULTS = Path(__file__).resolve().parent / "results"
CAL_SEEDS = (1, 2, 3, 4, 5)
EVAL_SEEDS = tuple(range(101, 111))
KP_GRID = (0.1, 0.25, 0.5, 1.0, 2.0)
KI_GRID = (0.3, 0.6, 1.2, 2.4, 4.8, 9.6)          # 1/h；积分时间 = Kp/Ki
FILTER_GRID = (0.0, 5.0, 10.0, 20.0)                # 前馈滤波时间，分钟


def pick(rows: list[dict]) -> dict:
    """动作预算内 DO 误差最小；没有满足预算的则取动作最小的。"""
    ok = [r for r in rows if r["air_moves_per_h"] <= MOVE_BUDGET]
    return min(ok, key=lambda r: r["do_mae"]) if ok else min(rows, key=lambda r: r["air_moves_per_h"])


def pareto(rows: list[dict]) -> list[dict]:
    """误差—动作的帕累托前沿：没有别的配置在两项上都不差、且至少一项更好。"""
    front = [r for r in rows if not any(o["do_mae"] <= r["do_mae"] and o["air_moves_per_h"] <= r["air_moves_per_h"]
                                        and (o["do_mae"], o["air_moves_per_h"]) != (r["do_mae"], r["air_moves_per_h"])
                                        for o in rows)]
    return sorted(front, key=lambda r: r["air_moves_per_h"])


def mean_metric(runs, key):
    return float(np.mean([r.metrics[key] for r in runs]))


def score(strategy: str, p: Params) -> dict:
    runs = [simulate(strategy, params=p, seed=s) for s in CAL_SEEDS]
    return {"do_mae": round(mean_metric(runs, "do_mae"), 4),
            "air_moves_per_h": round(mean_metric(runs, "air_moves_per_h"), 4), "kwh": round(mean_metric(runs, "kwh"), 1)}


def grid_pi(params: Params) -> tuple[dict, list]:
    rows = [{"kp": kp, "ki": ki, **score("pi", Params(**{**params.__dict__, "pi": {"kp": kp, "ki": ki}}))}
            for kp, ki in itertools.product(KP_GRID, KI_GRID)]
    best = pick(rows)
    return {"kp": best["kp"], "ki": best["ki"]}, rows


def grid_ff_pi(params: Params) -> tuple[dict, list]:
    rows = []
    for kp, ki, f in itertools.product(KP_GRID, KI_GRID, FILTER_GRID):
        cfg = {"kp": kp, "ki": ki, "filter_min": f}
        rows.append({**cfg, **score("ff_pi", Params(**{**params.__dict__, "ff_pi": cfg}))})
    best = pick(rows)
    return {k: best[k] for k in ("kp", "ki", "filter_min")}, rows


class LoggingPI(PI):
    """整定后的 PI，同时记录在线仪表测得的负荷与实际风量，用于拟合前馈。"""

    def __init__(self, *a, instruments: Instruments, **kw):
        super().__init__(*a, **kw)
        self.instruments, self.log = instruments, []

    def command(self, obs):
        self.log.append((obs.ts, self.instruments.load(obs), obs.air / obs.air_design))
        return super().command(obs)


def fit_feedforward(params: Params) -> tuple[dict, dict]:
    xs, ys, air_all = [], [], []
    for seed in CAL_SEEDS:
        ctl = LoggingPI(2.0, **params.pi, instruments=Instruments(P01, np.random.default_rng([seed, 999])))
        run = simulate("pi", params=params, controller=ctl, seed=seed)
        air_all += run.series["air_frac"]
        day = [(load, air) for ts, load, air in ctl.log if ts >= datetime.fromisoformat("2026-11-05T00:00:00+08:00")]
        arr = np.array(day)
        n = len(arr) // 10 * 10
        blocks = arr[:n].reshape(-1, 10, 2).mean(axis=1)       # 10 分钟均值，减小仪表噪声对斜率的衰减
        xs += list(blocks[:, 0])
        ys += list(blocks[:, 1])
    x, y = np.array(xs), np.array(ys)
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    r2 = 1 - float(np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    q = {f"p{p}": round(float(np.percentile(air_all, p)), 4) for p in (50, 75, 95)}
    return {"a": round(float(a), 4), "b": round(float(b), 4)}, {"r2": round(r2, 4), "n_blocks": len(x),
                                                                  "pi_air_quantiles": q}


HW_KEYS = ("alpha", "gamma", "horizon_min", "filter_min", "kp", "ki")


def grid_hw(params: Params) -> tuple[dict, list]:
    rows = []
    for kp, ki, alpha, h, f in itertools.product((0.25, 0.5, 1.0), (0.3, 1.2, 4.8), (0.2, 0.4),
                                                 (0.0, 10.0, 20.0), (0.0, 10.0)):
        cfg = {"alpha": alpha, "gamma": 0.2, "horizon_min": h, "filter_min": f, "kp": kp, "ki": ki}
        rows.append({**cfg, **score("ff_hw_pi", Params(**{**params.__dict__, "hw": cfg}))})
    best = pick(rows)
    return {k: best[k] for k in HW_KEYS}, rows


class ForecastProbe(PI):
    """按已整定的 PI 运行，同时用在线仪表测得的负荷训练 Holt-Winters，记录评测日每步的预测。"""

    def __init__(self, *a, instruments: Instruments, hw: HoltWinters, horizons: tuple[int, ...], **kw):
        super().__init__(*a, **kw)
        self.instruments, self.hw, self.horizons = instruments, hw, horizons
        self.rows: list[tuple] = []

    def command(self, obs):
        y = self.instruments.load(obs)
        self.hw.update(obs.ts, y)
        if obs.ts >= EVAL_DAY:
            self.rows.append((y, *(self.hw.forecast(obs.ts, h) for h in self.horizons)))
        return super().command(obs)


def forecast_check(params: Params, scenario: str = "dry", horizons: tuple[int, ...] = (10, 30, 60)) -> dict:
    """h 分钟后真实负荷的 MAPE（60 s 一步，h 分钟 = h 步）：
    - persistence：沿用当前测量值；
    - hw：Holt-Winters（只能从前 3 天带噪声的测量中学日内规律）；
    - oracle_shape：当前测量值 × 模拟器真实日变化曲线之比，即「日内规律完全已知」时的上限。
    """
    err = {h: {"persistence": [], "hw": [], "oracle_shape": []} for h in horizons}
    for seed in EVAL_SEEDS:
        ctl = ForecastProbe(2.0, **params.pi, instruments=Instruments(P01, np.random.default_rng([seed, 999])),
                            hw=HoltWinters(params.hw["alpha"], params.hw["gamma"]), horizons=horizons)
        run = simulate("pi", params=params, controller=ctl, scenario=scenario, seed=seed)
        truth = np.array(run.series["load"])
        rows = np.array(ctl.rows)
        hours = np.array(run.series["t_min"]) / 60
        for j, h in enumerate(horizons):
            t = truth[h:]
            shape = np.array([diurnal((x + h / 60) % 24) / diurnal(x) for x in hours])
            err[h]["hw"].append(float(np.mean(np.abs(rows[:-h, j + 1] / t - 1)) * 100))
            err[h]["persistence"].append(float(np.mean(np.abs(rows[:-h, 0] / t - 1)) * 100))
            err[h]["oracle_shape"].append(float(np.mean(np.abs((rows[:, 0] * shape)[:-h] / t - 1)) * 100))
    return {str(h): {k: round(float(np.mean(v)), 2) for k, v in e.items()} for h, e in err.items()}


def calibrate() -> None:
    t0 = time.perf_counter()
    params = Params()
    params.pi, pi_rows = grid_pi(params)
    print("pi", params.pi, flush=True)
    params.ff, fit = fit_feedforward(params)
    print("ff", params.ff, fit, flush=True)
    params.fixed_air_frac = fit["pi_air_quantiles"]["p95"]
    params.ff_pi, ff_rows = grid_ff_pi(params)
    print("ff_pi", params.ff_pi, flush=True)
    params.hw, hw_rows = grid_hw(params)
    print("hw", params.hw, flush=True)
    params.meta = {"calibrated_at": datetime.now().isoformat(timespec="seconds"), "seeds": list(CAL_SEEDS),
                   "scenario": "dry", "setpoint": 2.0, "plant": P01.code, "ff_fit": fit, "move_budget": MOVE_BUDGET,
                   "selected": {"pi": pick(pi_rows), "ff_pi": pick(ff_rows), "ff_hw_pi": pick(hw_rows)},
                   "fixed_rule": "整定后 PI 在校准日风量的 95% 分位"}
    params.save()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "calibration.json").write_text(json.dumps(
        {"params": params.__dict__, "pareto_pi": pareto(pi_rows), "pareto_ff_pi": pareto(ff_rows),
         "pareto_ff_hw_pi": pareto(hw_rows), "grid_pi": pi_rows, "grid_ff_pi": ff_rows, "grid_hw": hw_rows,
         "elapsed_s": round(time.perf_counter() - t0, 1)}, ensure_ascii=False, indent=1))
    print(PARAMS_PATH, round(time.perf_counter() - t0, 1), "s")


def summarize(values: list[float]) -> dict:
    a = np.array(values, dtype=float)
    return {"mean": round(float(a.mean()), 4), "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}


def evaluate() -> None:
    t0 = time.perf_counter()
    params = Params.load()
    if not PARAMS_PATH.exists():
        raise SystemExit("先运行 calibrate")
    q = params.meta["ff_fit"]["pi_air_quantiles"]
    variants = {s: (s, params) for s in STRATEGIES}
    for p in ("p50", "p75"):        # 固定风量敏感性：按更低的分位配风
        variants[f"fixed_{p}"] = ("fixed", Params(**{**params.__dict__, "fixed_air_frac": q[p]}))
    out = {"generated_at": datetime.now().isoformat(timespec="seconds"), "params": params.__dict__,
           "seeds": list(EVAL_SEEDS), "scenarios": {}}
    for sc in SCENARIOS:
        per = {name: [simulate(strategy, params=p, scenario=sc, seed=s) for s in EVAL_SEEDS]
               for name, (strategy, p) in variants.items()}
        res = {}
        for name, runs in per.items():
            keys = runs[0].metrics.keys()
            res[name] = {k: summarize([r.metrics[k] for r in runs]) for k in keys}
            # 与 PI 的配对差（同一种子），电耗用百分比
            res[name]["kwh_vs_pi_pct"] = summarize([(r.metrics["kwh"] / b.metrics["kwh"] - 1) * 100
                                                    for r, b in zip(runs, per["pi"])])
            res[name]["kwh_vs_fixed_pct"] = summarize([(r.metrics["kwh"] / b.metrics["kwh"] - 1) * 100
                                                       for r, b in zip(runs, per["fixed"])])
        out["scenarios"][sc] = res
        print(f"\n## {sc}  {SCENARIOS[sc]}")
        print("策略 | DO MAE | DO p95误差 | DO<设定-0.5 % | DO>设定+1 % | kWh/天 | 较固定 % | 较PI % | 氨氮均值/最大 | 超5mg/L h | 风量变化/h")
        for name, m in res.items():
            print(f"{name} | {m['do_mae']['mean']:.3f} | {m['do_p95_abs_err']['mean']:.3f} | {m['do_low_pct']['mean']:.2f} | "
                  f"{m['do_high_pct']['mean']:.1f} | {m['kwh']['mean']:.0f} | {m['kwh_vs_fixed_pct']['mean']:+.1f} "
                  f"[{m['kwh_vs_fixed_pct']['min']:+.1f},{m['kwh_vs_fixed_pct']['max']:+.1f}] | "
                  f"{m['kwh_vs_pi_pct']['mean']:+.2f} [{m['kwh_vs_pi_pct']['min']:+.2f},{m['kwh_vs_pi_pct']['max']:+.2f}] | "
                  f"{m['nh3n_mean']['mean']:.2f}/{m['nh3n_max']['max']:.2f} | {m['nh3n_over_limit_h']['mean']:.2f} | "
                  f"{m['air_moves_per_h']['mean']:.3f}", flush=True)
    # 设定值敏感性（旱天，前 5 个评测种子）：前馈按氧传递推动力换算是否成立
    sweep = {}
    for sp in (1.5, 2.0, 2.5):
        sweep[str(sp)] = {s: {k: summarize([simulate(s, params=params, seed=seed, setpoint=sp).metrics[k]
                                            for seed in EVAL_SEEDS[:5]])["mean"] for k in ("do_mae", "kwh")}
                          for s in ("pi", "ff_pi", "ff_hw_pi")}
    out["setpoint_sweep"] = sweep
    print("\nsetpoint sweep", json.dumps(sweep, ensure_ascii=False))
    out["forecast_mape_pct"] = {sc: forecast_check(params, sc) for sc in ("dry", "storm")}
    print("forecast MAPE %", json.dumps(out["forecast_mape_pct"], ensure_ascii=False))
    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(RESULTS / "eval.json", out["elapsed_s"], "s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["calibrate", "evaluate"])
    a = p.parse_args()
    calibrate() if a.step == "calibrate" else evaluate()


if __name__ == "__main__":
    main()

"""智能曝气仿真（M6）：同一场景、同一组随机扰动下对比固定曝气 / PI / 前馈 + PI / 预测前馈 + PI。

纯计算，不读写数据库；一次对比约 0.7 秒（4 策略 × 4 天、60 秒步长，M3 本机），放到线程里跑。
设定值按 0.1 取整后缓存，前端拖动滑块时重复的组合直接返回。
方法、整定与评测见 evals/aeration/results/report.md。
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from simulator.aeration import LABELS, SCENARIOS, STRATEGIES, Params, compare

router = APIRouter(prefix="/api/aeration", tags=["智能曝气"])

NOTICE = "仿真结果，仅用于说明方法；简化氧平衡模型，不代表真实厂的节能比例"
_slots = asyncio.Semaphore(2)          # 同时最多 2 个仿真占 CPU


class SimulateIn(BaseModel):
    scenario: Literal["dry", "storm", "shock", "overload"] = "dry"
    setpoint: float = Field(2.0, ge=1.0, le=3.0, description="DO 设定值 mg/L，按 0.1 取整")
    seed: int = Field(101, ge=1, le=1_000_000, description="随机种子；同一种子下各策略的进水与噪声完全相同")
    strategies: list[Literal["fixed", "pi", "ff_pi", "ff_hw_pi"]] = Field(default_factory=lambda: list(STRATEGIES))

    @field_validator("strategies")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("至少选择一个策略")
        return [s for s in STRATEGIES if s in v]      # 去重并按固定顺序


@lru_cache(maxsize=64)
def _cached(scenario: str, seed: int, setpoint: float, strategies: tuple[str, ...]) -> dict:
    return compare(scenario, seed, setpoint, strategies, Params.load())


@router.get("/options", summary="可选策略、场景与当前整定参数")
async def options() -> dict:
    p = Params.load()
    return {"strategies": [{"id": s, "label": LABELS[s]} for s in STRATEGIES],
            "scenarios": [{"id": k, "label": v} for k, v in SCENARIOS.items()],
            "setpoint": {"min": 1.0, "max": 3.0, "step": 0.1, "default": 2.0},
            "params": {k: getattr(p, k) for k in ("fixed_air_frac", "pi", "ff", "ff_pi", "hw")},
            "calibration": p.meta, "notice": NOTICE}


@router.post("/simulate", summary="运行一次对照仿真（评测日 24 小时，5 分钟曲线 + 指标）")
async def simulate(body: SimulateIn) -> dict:
    sp = round(body.setpoint, 1)
    async with _slots:
        return await asyncio.to_thread(_cached, body.scenario, body.seed, sp, tuple(body.strategies))

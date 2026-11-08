"""统一模型调用计时；旧录制没有总耗时时回退，并明确保留未知值。"""

import asyncio
from dataclasses import asdict, is_dataclass


def elapsed_ms(usage) -> float:
    total = getattr(usage, "total_latency_ms", None)
    if total is not None:
        return total
    return getattr(usage, "llm_ms", getattr(usage, "latency_ms", 0.0))


def call_usage(usage) -> dict:
    """只保存计量与尝试信息，不复制请求、回答或异常正文。"""
    failed = isinstance(usage, BaseException)
    return {
        "model": getattr(usage, "model", None),
        "provider": getattr(usage, "provider", None),
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "from_cache": bool(getattr(usage, "from_cache", False)),
        "finish_reason": getattr(usage, "finish_reason", None),
        "llm_ms": elapsed_ms(usage),
        "latency_ms": getattr(usage, "latency_ms", None),
        "total_latency_ms": getattr(usage, "total_latency_ms", None),
        "status": "cancelled" if isinstance(usage, asyncio.CancelledError) else "error" if failed else "success",
        "error_type": type(usage).__name__ if failed else None,
        "attempt_history": [asdict(a) if is_dataclass(a) else dict(a)
                            for a in getattr(usage, "attempt_history", [])],
    }

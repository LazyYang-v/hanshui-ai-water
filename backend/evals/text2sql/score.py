"""执行准确率（Execution Accuracy）判分。

本项目自定义的宽松执行匹配指标，不能与 Spider/BIRD 等基准的官方 EX 直接比较：
- 行数必须相同；金标的每一列都要在预测结果里找到取值一致的列（允许多出说明性的列、列序不同）
- 数值：相对误差 ≤1% 或绝对误差 ≤0.01（模型会 round，金标不 round）
- 时间：零点的时间戳视作日期（date_trunc('day') 与 ::date 等价）；带时区的统一到北京时间
- ordered=False 时忽略行序
"""

from __future__ import annotations

import itertools
import math
from datetime import datetime, time
from zoneinfo import ZoneInfo

LOCAL = ZoneInfo("Asia/Shanghai")
REL_TOL, ABS_TOL = 0.01, 0.01
MAX_COMBOS = 2000


def norm(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return s
        if dt.tzinfo is not None:
            dt = dt.astimezone(LOCAL)
        if dt.time() == time(0):
            return dt.date().isoformat()
        return dt.replace(tzinfo=None).isoformat(timespec="seconds")
    return s


def _is_clock(s: str) -> bool:
    return 5 <= len(s) <= 8 and s[2] == ":" and s.replace(":", "").isdigit()


def same(a, b, time_only: bool = False) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    if time_only and isinstance(a, str) and isinstance(b, str) and a != b:
        # 问「几点」时只答时刻（14:01）也算对：与金标时间戳的时刻部分比较
        for full, clock in ((a, b), (b, a)):
            if _is_clock(clock) and "T" in full:
                return full.split("T", 1)[1].startswith(clock)
    return a == b


def _key(v):
    if v is None:
        return (0, 0.0, "")
    if isinstance(v, float):
        return (1, v, "")
    return (2, 0.0, v)


def _seq_same(xs, ys, time_only: bool = False) -> bool:
    return len(xs) == len(ys) and all(same(a, b, time_only) for a, b in zip(xs, ys))


def _rows_same(g_rows, p_rows, ordered: bool, time_only: bool) -> bool:
    if not ordered:
        g_rows = sorted(g_rows, key=lambda r: [_key(v) for v in r])
        p_rows = sorted(p_rows, key=lambda r: [_key(v) for v in r])
    return all(_seq_same(a, b, time_only) for a, b in zip(g_rows, p_rows))


def match(gold_rows: list[list], pred_rows: list[list], ordered: bool = False, *, time_only: bool = False) -> bool:
    if len(gold_rows) != len(pred_rows):
        return False
    if not gold_rows:
        return True
    g = [[norm(v) for v in r] for r in gold_rows]
    p = [[norm(v) for v in r] for r in pred_rows]
    n_g, n_p = len(g[0]), len(p[0])
    if n_p < n_g:
        return False
    g_cols = [[r[j] for r in g] for j in range(n_g)]
    p_cols = [[r[k] for r in p] for k in range(n_p)]

    def col_ok(gc, pc):
        if ordered:
            return _seq_same(gc, pc, time_only)
        return _seq_same(sorted(gc, key=_key), sorted(pc, key=_key), time_only)

    candidates = [[k for k in range(n_p) if col_ok(g_cols[j], p_cols[k])] for j in range(n_g)]
    if any(not c for c in candidates):
        return False
    for combo in itertools.islice(itertools.product(*candidates), MAX_COMBOS):
        if len(set(combo)) < len(combo):
            continue
        if _rows_same(g, [[r[k] for k in combo] for r in p], ordered, time_only):
            return True
    return False

"""固定统计的语义回归；--noconftest 可运行全部纯函数和 JSON 协议测试。"""

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.ai.agent.models import Decision
from app.analytics.rules import BUCKET, LOCAL_TZ
from app.analytics.statistics import (
    StatisticsQuery,
    bounds,
    collect_statistics,
    make_result,
    summarize,
)

T0 = datetime(2026, 11, 5, tzinfo=LOCAL_TZ)
ALL_STATISTICS = [
    "mean", "instant_max", "instant_min", "bucket_mean_max", "bucket_mean_min",
    "exceed_hours", "longest_exceed_hours", "hours_with_exceed", "rank", "range", "top",
]


def query(**changes):
    return StatisticsQuery.model_validate({
        "metric": "nh3n_out", "start": T0, "end": T0 + timedelta(hours=1), **changes,
    })


def bucket(index, value, *, raw_max=None, raw_min=None, samples=1):
    return {"t": T0 + index * BUCKET, "mean": value,
            "raw_max": value if raw_max is None else raw_max,
            "raw_min": value if raw_min is None else raw_min, "samples": samples}


def plant(code, values, *, end=T0 + timedelta(hours=1), metric="nh3n_out"):
    return {"code": code, "name": f"测试厂{code}",
            "summary": summarize([bucket(i, v) for i, v in enumerate(values)], T0, end, metric)}


def fact(result, statistic, code="P01"):
    return next(f for f in result["facts"] if f["statistic"] == statistic and f["plant"] == code)


def test_agent_json_and_fastapi_decoded_json_accept_the_same_statistics_query():
    payload = {"plant": " p01 ", "metric": "nh3n_out", "start": T0.isoformat(),
               "end": (T0 + timedelta(hours=1)).isoformat(), "statistics": ALL_STATISTICS}
    plan = {"intent": "data", "resolved_question": "昨天各厂氨氮统计", "action": "tools",
            "calls": [{"tool": "plant_statistics", "question": "氨氮统计", "parameters": payload}],
            "reason": ""}
    agent_query = Decision.model_validate_json(json.dumps(plan)).calls[0].parameters
    app = FastAPI()

    @app.post("/statistics-contract")
    def accept(body: StatisticsQuery):
        return body.model_dump(mode="json")

    with TestClient(app) as client:
        response = client.post("/statistics-contract", json=payload)
        invalid = client.post("/statistics-contract", json={**payload, "sql": "SELECT 1"})
    assert response.status_code == 200, response.text
    assert response.json() == agent_query.model_dump(mode="json")
    assert response.json()["plant"] == "P01"
    assert invalid.status_code == 422


@pytest.mark.parametrize("change", [
    {"metric": "temp_c"}, {"metric": "nh3n_out); DROP TABLE plant; --"}, {"metric": 1},
    {"plant": "P08"}, {"plant": True}, {"sql": "SELECT 1"}, {"unexpected": None},
    {"statistics": []}, {"statistics": ["mean", "mean"]}, {"statistics": ["sum"]},
    {"statistics": "mean"}, {"rank_by": "instant_min"},
    {"start": "不是日期"}, {"start": "2026-11-05"}, {"start": 1793808000},
    {"start": True}, {"end": None}, {"end": T0}, {"end": T0 - BUCKET},
    {"end": T0 + timedelta(days=31, seconds=1)},
])
def test_query_rejects_invalid_or_unregistered_input(change):
    with pytest.raises(ValidationError):
        query(**change)


@pytest.mark.parametrize("statistics,rank_by", [
    (["exceed_hours"], "mean"), (["longest_exceed_hours"], "mean"),
    (["hours_with_exceed"], "mean"), (["rank"], "exceed_hours"),
])
def test_metric_without_threshold_cannot_produce_exceedance(statistics, rank_by):
    with pytest.raises(ValidationError, match="没有异常阈值"):
        query(metric="inflow_m3h", statistics=statistics, rank_by=rank_by)


def test_metric_without_threshold_can_report_measurements():
    q = query(metric="inflow_m3h", statistics=["mean", "instant_max"])
    result = make_result(q, T0, q.end, [plant("P01", [0.0] * 12, metric=q.metric)])
    assert fact(result, "mean")["value"] == 0
    assert fact(result, "mean")["unit"] == "m³/h"
    assert not result["warnings"]


def test_query_normalizes_naive_and_offset_times_and_accepts_31_days():
    q = query(start=T0.replace(tzinfo=None), end=(T0 + timedelta(days=31)).astimezone(UTC))
    assert q.start == T0 and q.start.tzinfo == LOCAL_TZ
    assert q.end - q.start == timedelta(days=31)
    assert q.end.tzinfo == LOCAL_TZ


def test_bounds_excludes_incomplete_edges_and_future_buckets():
    q = query(start=T0 + timedelta(minutes=2, seconds=1), end=T0 + timedelta(minutes=23))
    start, end = bounds(q, T0 + timedelta(minutes=17, seconds=59))
    assert (start, end) == (T0 + BUCKET, T0 + 3 * BUCKET)
    result = make_result(q, start, end, [
        {"code": "P01", "name": "测试厂", "summary": summarize(
            [bucket(0, 100), bucket(1, 4), bucket(2, 6), bucket(3, 1000)], start, end, q.metric)},
    ])
    assert result["plants"][0]["summary"]["instant_max"] == 6
    assert result["period"]["requested_end"] == q.end.isoformat()
    assert result["period"]["end"] == end.isoformat()
    assert any("完整5分钟桶" in warning for warning in result["warnings"])


def test_bounds_treats_naive_server_time_as_beijing_time():
    assert bounds(query(), (T0 + timedelta(minutes=17)).replace(tzinfo=None)) == (T0, T0 + 3 * BUCKET)


@pytest.mark.parametrize("start,end,now", [
    (T0 + timedelta(minutes=1), T0 + timedelta(minutes=4), T0 + timedelta(hours=1)),
    (T0, T0 + timedelta(minutes=4), T0 + timedelta(hours=1)),
    (T0 + timedelta(hours=1), T0 + timedelta(hours=2), T0),
    (T0, T0 + timedelta(hours=1), T0 + timedelta(minutes=4)),
])
def test_bounds_rejects_windows_without_a_complete_past_bucket(start, end, now):
    with pytest.raises(ValueError, match="没有已写满的5分钟桶"):
        bounds(query(start=start, end=end), now)


@pytest.mark.parametrize("include_null_row", [False, True])
def test_missing_buckets_interrupt_longest_streak_without_filling_zero(include_null_row):
    # 三段异常之间有正常值和缺失值；累计35分钟，最长连续20分钟。
    values = [6, 6, 4, 6, None, 6, 6, 6, 6]
    rows = [bucket(i, v) for i, v in enumerate(values) if v is not None or include_null_row]
    s = summarize(rows, T0, T0 + 9 * BUCKET, "nh3n_out")
    assert s["expected_buckets"] == 9
    assert s["observed_buckets"] == 8 and s["missing_buckets"] == 1
    assert s["coverage"] == pytest.approx(8 / 9)
    assert s["mean"] == 5.75
    assert s["exceed_hours"] == pytest.approx(35 / 60)
    assert s["longest_exceed_hours"] == pytest.approx(20 / 60)


@pytest.mark.parametrize("metric,values", [
    ("nh3n_out", [5.0, 5.00001]), ("do_aerobic_1", [1.0, 0.99999]),
])
def test_threshold_direction_and_equality_use_unrounded_values(metric, values):
    s = summarize([bucket(i, v) for i, v in enumerate(values)], T0, T0 + 2 * BUCKET, metric)
    assert s["exceed_hours"] == pytest.approx(5 / 60)
    assert s["longest_exceed_hours"] == pytest.approx(5 / 60)
    assert s["hours_with_exceed"] == 1


def test_absent_rows_and_all_null_buckets_remain_unknown():
    empty = summarize([], T0, T0 + timedelta(hours=1), "nh3n_out")
    nulls = summarize([bucket(i, None, samples=0) for i in range(12)], T0,
                      T0 + timedelta(hours=1), "nh3n_out")
    assert empty == nulls
    assert empty["coverage"] == 0 and empty["missing_buckets"] == 12
    assert all(empty[name] is None for name in ALL_STATISTICS[:8])
    result = make_result(query(statistics=ALL_STATISTICS), T0, T0 + timedelta(hours=1), [
        {"code": "P01", "name": "没有数据的厂", "summary": empty},
    ])
    assert result["facts"] == []
    assert any("不能解释为零" in warning for warning in result["warnings"])


def test_raw_extrema_and_bucket_mean_extrema_are_separate():
    rows = [bucket(0, 4, raw_min=0, raw_max=20, samples=10), bucket(1, 6), bucket(2, 4.5)]
    s = summarize(rows, T0, T0 + 3 * BUCKET, "nh3n_out")
    assert s["instant_max"] == 20 and s["bucket_mean_max"] == 6
    assert s["instant_min"] == 0 and s["bucket_mean_min"] == 4
    assert s["exceed_hours"] == pytest.approx(5 / 60)
    assert s["hours_with_exceed"] == 1


def test_hours_containing_raw_exceedance_do_not_imply_exceedance_duration():
    rows = [bucket(i, 4, raw_max=8 if i in {0, 11, 12} else 4) for i in range(24)]
    end = T0 + timedelta(hours=2)
    s = summarize(rows, T0, end, "nh3n_out")
    assert s["hours_with_exceed"] == 2
    assert s["exceed_hours"] == 0 and s["longest_exceed_hours"] == 0
    result = make_result(query(end=end, statistics=["hours_with_exceed"]), T0, end,
                         [{"code": "P01", "name": "测试厂", "summary": s}])
    statement = fact(result, "hours_with_exceed")
    assert "不能据此当作连续2小时" in statement["text"]
    assert statement["unit"] == "个小时桶"


def test_do_raw_dip_uses_the_lower_threshold_without_claiming_a_bad_bucket():
    s = summarize([bucket(0, 1.2, raw_min=0.4, raw_max=1.9)], T0, T0 + BUCKET, "do_aerobic_1")
    assert s["hours_with_exceed"] == 1 and s["exceed_hours"] == 0
    assert s["instant_min"] == 0.4 and s["bucket_mean_min"] == 1.2
    result = make_result(query(metric="do_aerobic_1", end=T0 + BUCKET,
                               statistics=["hours_with_exceed", "exceed_hours"]),
                         T0, T0 + BUCKET, [{"code": "P01", "name": "测试厂", "summary": s}])
    assert "低于下限" in fact(result, "hours_with_exceed")["text"]
    assert fact(result, "exceed_hours")["value"] == 0


def test_period_mean_weights_buckets_equally_despite_different_sampling_densities():
    end = T0 + 2 * BUCKET
    s = summarize([bucket(0, 2, samples=100), bucket(1, 8, samples=1)], T0, end, "nh3n_out")
    assert s["mean"] == 5 and s["coverage"] == 1
    result = make_result(query(end=end, statistics=["mean"]), T0, end, [
        {"code": "P01", "name": "测试厂", "summary": s},
    ])
    assert "等权" in fact(result, "mean")["aggregation"]
    assert "采样数" in fact(result, "mean")["aggregation"]


def test_rows_outside_half_open_window_do_not_change_summary():
    s = summarize([bucket(-1, 999), bucket(0, 4), bucket(1, 6), bucket(2, 999)],
                  T0, T0 + 2 * BUCKET, "nh3n_out")
    assert s["observed_buckets"] == 2 and s["coverage"] == 1
    assert s["mean"] == 5 and s["instant_max"] == 6


def test_four_decimal_dense_ranking_and_ties_have_consistent_text_and_values():
    plants = [plant("P01", [123.45671] * 12), plant("P02", [123.45672] * 12),
              plant("P03", [123.45661] * 12)]
    result = make_result(query(statistics=["rank", "range", "top"]), T0, T0 + timedelta(hours=1), plants)
    assert [fact(result, "rank", code)["value"] for code in ["P01", "P02", "P03"]] == [1, 1, 2]
    assert "并列第1名" in fact(result, "rank")["text"]
    assert "123.4567" in fact(result, "rank")["text"]
    assert fact(result, "rank")["unit"] == "名"
    assert set(fact(result, "rank")["coverage"]) == {"P01", "P02", "P03"}
    assert fact(result, "range", None)["value"] == {"min": 123.4566, "max": 123.4567}
    highest = fact(result, "top", None)
    assert "并列最高" in highest["text"] and "P01" in highest["text"] and "P02" in highest["text"]
    assert "P03" not in highest["text"]


@pytest.mark.parametrize("plants", [
    [], [plant("P01", [6] * 12)],
    [plant("P01", [6] * 12), plant("P02", [7] * 11)],
    [plant("P01", [6] * 11), plant("P02", [7] * 11)],
    [plant("P01", [6] * 12), plant("P02", [])],
])
def test_cross_plant_facts_require_complete_coverage_and_at_least_two_plants(plants):
    result = make_result(query(statistics=["mean", "rank", "range", "top"]),
                         T0, T0 + timedelta(hours=1), plants)
    assert not any(f["statistic"] in {"rank", "range", "top"} for f in result["facts"])
    assert any("比较对象不足" in warning for warning in result["warnings"])


@pytest.mark.parametrize("rank_by,highest,lowest,p01_rank,winner", [
    ("mean", 6, 5, 1, "P01"), ("instant_max", 100, 5, 2, "P02"),
    ("bucket_mean_max", 7, 5, 2, "P02"), ("exceed_hours", 0.1667, 0, 1, "P01"),
])
def test_cross_plant_comparison_uses_the_requested_aggregation(rank_by, highest, lowest, p01_rank, winner):
    end = T0 + 2 * BUCKET
    rows = {"P01": [bucket(0, 6, raw_max=8), bucket(1, 6)],
            "P02": [bucket(0, 4, raw_max=100), bucket(1, 7)],
            "P03": [bucket(0, 5), bucket(1, 5)]}
    plants = [{"code": code, "name": code, "summary": summarize(data, T0, end, "nh3n_out")}
              for code, data in rows.items()]
    result = make_result(query(end=end, statistics=["rank", "range", "top"], rank_by=rank_by), T0, end, plants)
    assert fact(result, "rank")["value"] == p01_rank
    assert fact(result, "range", None)["value"] == {"min": lowest, "max": highest}
    assert fact(result, "top", None)["value"] == highest
    assert f"最高为{winner}" in fact(result, "top", None)["text"]
    assert fact(result, "top", None)["unit"] == ("小时" if rank_by == "exceed_hours" else "mg/L")
    assert set(fact(result, "range", None)["coverage"]) == set(rows)


def test_missing_comparison_values_cannot_produce_a_partial_ranking():
    plants = [plant("P01", [6] * 12), plant("P02", [7] * 12)]
    plants[1]["summary"]["mean"] = None
    result = make_result(query(statistics=["rank", "range", "top"]), T0, T0 + timedelta(hours=1), plants)
    assert not result["facts"]
    assert any("比较对象不足" in warning for warning in result["warnings"])


def test_selected_plant_still_compares_against_the_full_eligible_population():
    plants = [plant("P01", [6] * 12), plant("P02", [7] * 12), plant("P03", [4] * 12)]
    result = make_result(query(plant="P01", statistics=["rank", "top"]),
                         T0, T0 + timedelta(hours=1), plants)
    assert [p["code"] for p in result["plants"]] == ["P01"]
    assert fact(result, "rank")["value"] == 2
    assert fact(result, "top", None)["value"] == 7
    assert "P02" in fact(result, "top", None)["text"]


@pytest.mark.parametrize("statistic,unit", [
    ("mean", "mg/L"), ("instant_max", "mg/L"), ("instant_min", "mg/L"),
    ("bucket_mean_max", "mg/L"), ("bucket_mean_min", "mg/L"),
    ("exceed_hours", "小时"), ("longest_exceed_hours", "小时"), ("hours_with_exceed", "个小时桶"),
])
def test_server_facts_include_aggregation_units_and_coverage(statistic, unit):
    result = make_result(query(statistics=[statistic]), T0, T0 + timedelta(hours=1),
                         [plant("P01", [6] * 11)])
    statement = fact(result, statistic)
    assert statement["unit"] == unit and statement["aggregation"]
    assert statement["coverage"]["P01"] == {
        "expected_buckets": 12, "observed_buckets": 11, "missing_buckets": 1, "coverage": 11 / 12,
    }
    assert "有效11/12" in statement["text"]
    assert "缺失处中断连续计数" in statement["text"]


def test_duration_text_preserves_exact_minutes_and_bucket_count():
    end = T0 + 13 * BUCKET
    result = make_result(query(end=end, statistics=["exceed_hours", "longest_exceed_hours"]),
                         T0, end, [plant("P01", [6] * 13, end=end)])
    for statistic in ["exceed_hours", "longest_exceed_hours"]:
        statement = fact(result, statistic)
        assert statement["value"] == pytest.approx(65 / 60)
        assert "65分钟" in statement["text"] and "13个5分钟桶" in statement["text"]
        assert "1.0833小时" in statement["text"]
        assert "阈值5 mg/L" in statement["text"]


async def test_database_statistics_contract(request):
    # 纯函数命令不加载 conftest，也不创建测试库；由主线程统一执行这一项。
    if request.config.getoption("noconftest"):
        pytest.skip("数据库集成由主线程统一运行")
    import psycopg

    db_url = request.getfixturevalue("db_url")
    end = T0 + timedelta(hours=12)  # 与现有夹具 END 相同；此刻之前已有两天的分钟采样。
    q = query(end=end, statistics=ALL_STATISTICS)
    async with await psycopg.AsyncConnection.connect(db_url) as conn:
        async with conn.transaction():
            await conn.execute("SET TRANSACTION READ ONLY")
            result = await collect_statistics(conn, q, end)
    assert len(result["plants"]) == 7
    assert not result["warnings"]
    assert len(result["facts"]) == 7 * 9 + 2
    for p in result["plants"]:
        s = p["summary"]
        assert s["expected_buckets"] == s["observed_buckets"] == 144
        assert s["coverage"] == 1 and s["missing_buckets"] == 0
        assert s["instant_min"] <= s["bucket_mean_min"] <= s["mean"] <= s["bucket_mean_max"] <= s["instant_max"]
        assert 0 <= s["longest_exceed_hours"] <= s["exceed_hours"] <= 12

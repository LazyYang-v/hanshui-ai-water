"""历史可见性、负荷单位以及真实扫描合并/恢复回归。"""

from dataclasses import asdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg
import pytest

from app.analytics.alarms import SignalHistory, load_signals, scan
from app.analytics.rules import BUCKET, LOAD_METRICS, detect, load_episodes

TZ = ZoneInfo("Asia/Shanghai")
T0 = datetime(2028, 1, 8, 8, tzinfo=TZ)


def history(*, loads=None, concentrations=None, hourly_days=7):
    hourly = []
    for k in range(hourly_days * 24):
        t = T0 - timedelta(days=hourly_days) + timedelta(hours=k)
        row = {"plant_id": 1, "hour": t, "inflow_m3h": 1000.0,
               "cod_in": 100.0, "tn_in": 20.0, "nh3n_in": 10.0}
        row.update({m: row[c] for c, m in LOAD_METRICS.items()})
        hourly.append(row)
    loads = loads or [180.0] * 3 + [100.0] * 3
    concentrations = concentrations or loads
    rows = [{"plant_id": 1, "t": T0 + i * BUCKET, "inflow_m3h": 1000.0,
             "cod_in": c, "cod_load_kgh": v} for i, (v, c) in enumerate(zip(loads, concentrations))]
    return SignalHistory([{"id": 1, "code": "P01", "name": "测试厂"}], [], rows, hourly, [])


def test_future_hours_cannot_change_past_decisions():
    src = history()
    end = T0 + 6 * BUCKET
    before = [asdict(e) for e in detect(*src.window(T0, end))]
    # 已预载未来的大流量/高浓度/异常药耗，历史判断也不可使用它们。
    src.hourly += [{"plant_id": 1, "hour": T0 + timedelta(hours=k), "inflow_m3h": 99999.0,
                    "cod_in": 99999.0, "cod_load_kgh": 99999.0} for k in range(100)]
    future = SignalHistory(src.plants, [], src.rows, src.hourly, [])
    assert [asdict(e) for e in detect(*future.window(T0, end))] == before
    assert before[0]["detail"]["baseline_as_of"] == T0.isoformat()


def test_load_warning_is_not_effluent_exceedance():
    src = history()
    [plant], _ = src.window(T0, T0 + 6 * BUCKET)
    [e] = load_episodes(plant)
    assert (e.metric, e.kind, e.value, e.threshold, e.peak) == (
        "cod_load_kgh", "进水负荷", 180.0, 150.0, 180.0)
    assert e.start == T0 and e.end == T0 + 3 * BUCKET
    assert e.detail["unit"] == "kg/h"
    assert e.detail["condition_confirmed_at"] == (T0 + 3 * BUCKET).isoformat()


@pytest.mark.parametrize("loads,conc", [([180.0] * 6, [60.0] * 6), ([100.0] * 6, [180.0] * 6)])
def test_rain_dilution_and_concentration_without_load_do_not_alert(loads, conc):
    src = history(loads=loads, concentrations=conc)
    [plant], _ = src.window(T0, T0 + 6 * BUCKET)
    assert load_episodes(plant) == []


def test_load_history_must_cover_three_separate_hours_of_same_clock_hour():
    src = history(hourly_days=2)
    [plant], _ = src.window(T0, T0 + 6 * BUCKET)
    assert load_episodes(plant) == []


def test_missing_values_do_not_count_as_normal_or_make_confirmation_earlier():
    src = history(loads=[180.0, None, 180.0, None, 180.0, 100.0, None, 100.0, 100.0],
                  concentrations=[180.0] * 5 + [100.0] * 4)
    [plant], _ = src.window(T0, T0 + 9 * BUCKET)
    [e] = load_episodes(plant)
    assert e.detail["condition_confirmed_at"] == (T0 + 5 * BUCKET).isoformat()
    assert e.end == T0 + 5 * BUCKET


@pytest.mark.asyncio
async def test_real_scan_late_report_and_missing_recovery(db_url):
    async with await psycopg.AsyncConnection.connect(db_url, autocommit=True) as conn:
        async with conn.transaction(force_rollback=True):
            await conn.execute("INSERT INTO plant_reading(ts,plant_id,do_aerobic_1) VALUES (%s,1,0.5)", (T0,))
            _, events = await scan(conn, T0, T0 + 2 * BUCKET)
            assert not [e for e in events if e["alarm"]["ts"] == T0.isoformat()]
            # 第二个异常桶迟到 25 分钟；事件开始仍是 T0，实际首次检出须记到上报后的扫描。
            await conn.execute("INSERT INTO plant_reading(ts,plant_id,do_aerobic_1) VALUES (%s,1,0.5)", (T0 + BUCKET,))
            _, events = await scan(conn, T0, T0 + 6 * BUCKET)
            [new] = [e["alarm"] for e in events if e["event"] == "new" and e["alarm"]["rule"] == "do_low"]
            assert new["detail"]["detected_at"] == (T0 + 6 * BUCKET).isoformat()
            assert new["detail"]["condition_confirmed_at"] == (T0 + 2 * BUCKET).isoformat()
            # 其他列有数据但 DO 缺失，不能靠 coverage 判恢复。
            await conn.execute("INSERT INTO plant_reading(ts,plant_id,inflow_m3h) VALUES (%s,1,100)", (T0 + 6 * BUCKET,))
            _, events = await scan(conn, T0, T0 + 7 * BUCKET)
            assert not [e for e in events if e["event"] == "recovered"]
            for i in range(3):
                await conn.execute("INSERT INTO plant_reading(ts,plant_id,do_aerobic_1) VALUES (%s,1,2)",
                                   (T0 + (7 + i) * BUCKET,))
            _, events = await scan(conn, T0, T0 + 10 * BUCKET)
            [recovered] = [e["alarm"] for e in events if e["event"] == "recovered" and e["alarm"]["id"] == new["id"]]
            assert recovered["ended_at"] == (T0 + 7 * BUCKET).isoformat()
            stats, events = await scan(conn, T0, T0 + 10 * BUCKET)
            assert stats.inserted == stats.updated == stats.recovered == 0 and events == []


@pytest.mark.asyncio
async def test_open_alarm_older_than_seven_days_replays_until_real_recovery(db_url):
    async with await psycopg.AsyncConnection.connect(db_url, autocommit=True) as conn:
        async with conn.transaction(force_rollback=True):
            for i in range(2):
                await conn.execute("INSERT INTO device_reading(ts,device_id,running,vibration_mms) VALUES (%s,21,true,5)",
                                   (T0 + i * BUCKET,))
            await scan(conn, T0, T0 + 2 * BUCKET)
            later = T0 + timedelta(days=9)
            await conn.execute("INSERT INTO device_reading(ts,device_id,running) VALUES (%s,21,false)", (later,))
            stats, events = await scan(conn, later - timedelta(hours=3), later + BUCKET)
            assert stats.start == T0.isoformat() and stats.recovered == 0
            for i in range(1, 4):
                await conn.execute("INSERT INTO device_reading(ts,device_id,running,vibration_mms) VALUES (%s,21,true,3)",
                                   (later + i * BUCKET,))
            _, events = await scan(conn, later - timedelta(hours=3), later + 4 * BUCKET)
            [vib] = [e["alarm"] for e in events if e["event"] == "recovered" and e["alarm"]["rule"] == "device_vibration"]
            assert vib["ts"] == T0.isoformat() and vib["ended_at"] == (later + BUCKET).isoformat()


@pytest.mark.asyncio
async def test_sql_load_uses_mean_of_paired_products_and_local_row_factory(db_url):
    async with await psycopg.AsyncConnection.connect(db_url, autocommit=True) as conn:
        async with conn.transaction(force_rollback=True):
            await conn.execute("INSERT INTO plant_reading(ts,plant_id,inflow_m3h,cod_in) VALUES (%s,1,100,1000),(%s,1,1000,100)",
                               (T0, T0 + timedelta(minutes=1)))
            plants, _ = await load_signals(conn, T0, T0 + BUCKET)
            p = next(p for p in plants if p.plant_id == 1)
            assert p.values["cod_load_kgh"] == [100.0]  # 均值的乘积为302.5，口径错误
            assert (await (await conn.execute("SELECT 42")).fetchone()) == (42,)

"""检查点随读数同事务落库，重启时按停机时长决定恢复还是预热。"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg
import pytest

from simulator import runner
from simulator.scenarios import Scenario

TS = datetime(2026, 11, 20, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


@pytest.fixture
def conn(db_url):
    with psycopg.connect(db_url, autocommit=True) as c:
        c.execute("DELETE FROM sim_checkpoint")
        yield c
        c.execute("DELETE FROM sim_checkpoint")
        c.execute("DELETE FROM plant_reading WHERE ts >= %s", (TS,))
        c.execute("DELETE FROM device_reading WHERE ts >= %s", (TS,))
        c.execute("DELETE FROM weather WHERE ts >= %s", (TS,))


def _saved(conn):
    sim, _ = runner.load_simulator(conn, seed=1)
    sim.book.add(Scenario("shock_load", TS - timedelta(minutes=30), 180, "P03"))
    pid = next(iter(sim.models))
    sim.models[pid].s.nh3n_out = 6.5
    writer = runner.Writer(conn)
    writer.add(sim.step(TS, 5))
    writer.checkpoint = (TS, sim.dump_state())
    writer.flush()
    return pid, sim.models[pid].s


def test_restart_restores_state_and_active_scenario_from_database(conn):
    pid, state = _saved(conn)

    restarted, _ = runner.load_simulator(conn, seed=2)
    assert runner.restore_checkpoint(conn, restarted, TS + timedelta(minutes=2)) is True

    assert restarted.models[pid].s == state
    assert [sc.kind for sc in restarted.book.active(TS + timedelta(minutes=2))] == ["shock_load"]


def test_checkpoint_is_overwritten_not_appended(conn):
    _saved(conn)
    writer = runner.Writer(conn)
    writer.checkpoint = (TS + timedelta(seconds=5), {"plants": {}, "device_status": {}, "scenarios": []})
    writer.flush()

    assert conn.execute("SELECT count(*), max(ts) FROM sim_checkpoint").fetchone() == (1, TS + timedelta(seconds=5))


def test_stale_checkpoint_is_not_restored(conn):
    pid, _ = _saved(conn)

    restarted, _ = runner.load_simulator(conn, seed=2)
    fresh = restarted.models[pid].s

    assert runner.restore_checkpoint(conn, restarted, TS + runner.CHECKPOINT_MAX_GAP + timedelta(seconds=1)) is False
    assert restarted.models[pid].s is fresh and restarted.book.items == []


def test_readonly_role_cannot_see_checkpoint(conn):
    assert conn.execute("SELECT has_table_privilege('text2sql_ro', 'sim_checkpoint', 'SELECT')").fetchone()[0] is False

"""SQL 安全校验与改写：纯单元测试。"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.ai.text2sql.guard import SQLRejected, guard

TABLES = {"plant", "plant_reading", "plant_reading_1h", "device", "device_reading", "weather"}
NOW = datetime(2026, 11, 5, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def g(sql, **kw):
    return guard(sql, TABLES, NOW, **kw)


@pytest.mark.parametrize("sql", [
    "DELETE FROM plant",
    "UPDATE device SET status = '故障'",
    "INSERT INTO plant(id) VALUES (9)",
    "DROP TABLE plant",
    "TRUNCATE plant_reading",
    "CREATE TABLE x AS SELECT 1",
    "SELECT * INTO backup FROM plant",
    "SET statement_timeout = 0",
    "COPY plant TO '/tmp/x'",
    "GRANT ALL ON plant TO public",
])
def test_rejects_non_select(sql):
    with pytest.raises(SQLRejected):
        g(sql)


def test_rejects_multiple_statements():
    with pytest.raises(SQLRejected, match="一条"):
        g("SELECT 1; DELETE FROM plant")


def test_rejects_writes_hidden_in_cte():
    with pytest.raises(SQLRejected):
        g("WITH d AS (DELETE FROM plant RETURNING *) SELECT * FROM d")


@pytest.mark.parametrize("sql", [
    "SELECT * FROM alarm",
    "SELECT * FROM pg_catalog.pg_user",
    "SELECT usename FROM pg_user",
    "SELECT * FROM information_schema.tables",
    "SELECT * FROM plant p JOIN secret s ON s.id = p.id",
    "WITH alarm AS (SELECT 1) SELECT * FROM public.alarm",
    "WITH pg_user AS (SELECT * FROM pg_user) SELECT * FROM pg_user",
    "SELECT * FROM other.public.plant",
])
def test_rejects_tables_outside_whitelist(sql):
    with pytest.raises(SQLRejected):
        g(sql)


@pytest.mark.parametrize("sql", [
    "SELECT pg_sleep(10)",
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT set_config('statement_timeout', '0', false)",
    "SELECT current_setting('data_directory')",
    "SELECT * FROM dblink('host=x', 'select 1') AS t(a int)",
    "SELECT version()",
])
def test_rejects_dangerous_functions(sql):
    with pytest.raises(SQLRejected):
        g(sql)


def test_allows_cte_and_union():
    out = g("WITH t AS (SELECT id FROM plant) SELECT id FROM t UNION ALL SELECT id FROM device")
    assert "LIMIT 200" in out


def test_rewrites_now_to_demo_clock():
    out = g("SELECT * FROM plant_reading WHERE ts > now() - interval '1 hour' AND ts::date = CURRENT_DATE "
            "AND ts <= CURRENT_TIMESTAMP")
    assert "now()" not in out.lower() and "current_" not in out.lower()
    assert "'2026-11-05T12:00:00+08:00'" in out
    assert "CAST('2026-11-05' AS DATE)" in out


def test_limit_added_and_capped():
    assert g("SELECT * FROM plant").endswith("LIMIT 200")
    assert g("SELECT * FROM plant LIMIT 5").endswith("LIMIT 5")
    assert g("SELECT * FROM plant LIMIT 100000").endswith("LIMIT 200")
    assert g("SELECT * FROM plant", max_rows=10).endswith("LIMIT 10")


def test_syntax_error_and_empty():
    with pytest.raises(SQLRejected):
        g("SELEC * FRM plant")
    with pytest.raises(SQLRejected):
        g("  ;  ")


def test_trailing_semicolon_ok():
    assert g("SELECT name FROM plant;").startswith("SELECT name FROM plant")

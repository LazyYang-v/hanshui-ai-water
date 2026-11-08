"""SQL 安全校验与改写（数据库只读账号之外的第二道防线）。

- 只允许单条 SELECT（含 WITH / UNION）
- 只允许访问白名单中的表和视图；函数走白名单，未列入的（table_to_xml、pg_*、dblink 等）一律拒绝
- 把 now() / CURRENT_TIMESTAMP / CURRENT_DATE / LOCALTIMESTAMP 等改写为演示时钟（加速演示时它领先墙上时间）
- 补 LIMIT，并给过大的 LIMIT 封顶
"""

from __future__ import annotations

from datetime import datetime

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

MAX_ROWS = 200
TIME_FUNCS = {"now", "statement_timestamp", "transaction_timestamp", "clock_timestamp", "localtimestamp"}
# sqlglot 不认识的函数都解析成 Anonymous，这里是它们的白名单：只放数值、字符串、时间计算和 TimescaleDB 分桶。
# 名单外的正常函数被拒后，模型会收到报错并自修复，代价是一次重试；反过来放行系统函数的代价更大。
ALLOWED_ANONYMOUS = TIME_FUNCS | {
    "time_bucket", "time_bucket_gapfill", "locf", "interpolate", "first", "last",
    "make_date", "make_time", "make_timestamp", "make_timestamptz", "make_interval", "to_date", "to_timestamp",
    "to_char", "to_number", "date_part", "date_bin", "age", "justify_days", "justify_hours", "justify_interval",
    "isfinite", "trunc", "ceiling", "cbrt", "degrees", "radians", "div", "gcd", "lcm", "scale", "sign", "width_bucket",
    "mode", "percent_rank", "cume_dist", "dense_rank", "rank", "ntile", "lead", "first_value", "last_value", "nth_value",
    "stddev", "stddev_pop", "stddev_samp", "variance", "var_pop", "var_samp", "corr", "covar_pop", "covar_samp",
    "regr_slope", "regr_intercept", "regr_r2", "percentile_cont", "percentile_disc", "bool_and", "bool_or", "every",
    "string_agg", "array_agg", "array_length", "array_to_string", "cardinality",
    "length", "char_length", "lower", "upper", "initcap", "trim", "ltrim", "rtrim", "btrim", "lpad", "rpad", "left",
    "right", "substr", "substring", "replace", "translate", "split_part", "position", "strpos", "concat", "concat_ws",
    "format", "repeat", "reverse", "starts_with", "regexp_replace", "regexp_match", "regexp_like",
}
# sqlglot 把这些系统信息函数解析成专门的节点（而不是 Anonymous），按类型拦截
SYSTEM_INFO_NODES = tuple(getattr(exp, n) for n in (
    "CurrentVersion", "CurrentUser", "CurrentDatabase", "CurrentSchema", "CurrentRole", "SessionUser",
    "CurrentCatalog") if hasattr(exp, n))
WRITE_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter, exp.Command,
               exp.TruncateTable, exp.Grant, exp.Copy, exp.Into, exp.Lock, exp.Set, exp.Transaction)


class SQLRejected(ValueError):
    pass


def guard(sql: str, allowed_tables: set[str], now: datetime, max_rows: int = MAX_ROWS) -> str:
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise SQLRejected("SQL 为空")
    try:
        trees = [t for t in sqlglot.parse(sql, read="postgres") if t is not None]
    except sqlglot.errors.ParseError as e:
        raise SQLRejected(f"SQL 语法错误：{str(e).splitlines()[0]}") from e
    if len(trees) != 1:
        raise SQLRejected("只允许一条语句")
    tree = trees[0]
    if not isinstance(tree, exp.Query):
        raise SQLRejected(f"只允许 SELECT 查询，收到 {type(tree).__name__}")
    if node := tree.find(*WRITE_NODES):
        raise SQLRejected(f"不允许的操作：{type(node).__name__}")

    # 逐层解析物理表与 CTE 引用，防止用同名 CTE 掩盖真实的非白名单表。
    for scope in traverse_scope(tree):
        for table in scope.sources.values():
            if not isinstance(table, exp.Table):
                continue  # 已解析为 CTE / 派生表作用域，内部物理表会单独校验
            if isinstance(table.this, exp.Func):
                continue  # FROM generate_series(...) 这类表函数不是物理表，由下方函数校验把关
            if table.catalog or (table.db and table.db != "public"):
                raise SQLRejected("只允许访问当前数据库的 public schema")
            if table.name not in allowed_tables:
                raise SQLRejected(f"不允许访问表 {table.name}，可用：{', '.join(sorted(allowed_tables))}")

    if node := tree.find(*SYSTEM_INFO_NODES):
        raise SQLRejected(f"不允许调用函数 {node.sql_name().lower()}")
    for fn in tree.find_all(exp.Anonymous):
        if (name := fn.name.lower()) not in ALLOWED_ANONYMOUS:
            raise SQLRejected(f"不允许调用函数 {name}")

    now_lit = exp.cast(exp.Literal.string(now.isoformat()), exp.DataType.build("timestamptz"))
    today_lit = exp.cast(exp.Literal.string(now.date().isoformat()), exp.DataType.build("date"))
    # 不带时区的“当前时间”按演示时钟的本地钟面值改写
    local_lit = exp.cast(exp.Literal.string(now.replace(tzinfo=None).isoformat()), exp.DataType.build("timestamp"))
    time_lit = exp.cast(exp.Literal.string(now.time().replace(tzinfo=None).isoformat()), exp.DataType.build("time"))

    def rewrite(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.CurrentTimestamp):
            return now_lit.copy()
        if isinstance(node, exp.CurrentDate):
            return today_lit.copy()
        if isinstance(node, exp.Localtimestamp):
            return local_lit.copy()
        if isinstance(node, exp.Localtime | exp.CurrentTime):
            return time_lit.copy()
        if isinstance(node, exp.Anonymous) and node.name.lower() in TIME_FUNCS:
            return now_lit.copy()
        return node

    tree = tree.transform(rewrite)

    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(max_rows)
    else:
        try:
            n = int(limit.expression.this) if isinstance(limit.expression, exp.Literal) else None
        except (TypeError, ValueError):
            n = None
        if n is None or n > max_rows:
            tree = tree.limit(max_rows)
    return tree.sql(dialect="postgres")

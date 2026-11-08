"""SQL 安全校验与改写（数据库只读账号之外的第二道防线）。

- 只允许单条 SELECT（含 WITH / UNION）
- 只允许访问白名单中的表和视图；禁止系统函数（pg_*、dblink、set_config 等）
- 把 now() / CURRENT_TIMESTAMP / CURRENT_DATE 改写为演示时钟（加速演示时它领先墙上时间）
- 补 LIMIT，并给过大的 LIMIT 封顶
"""

from __future__ import annotations

from datetime import datetime

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

MAX_ROWS = 200
TIME_FUNCS = {"now", "statement_timestamp", "transaction_timestamp", "clock_timestamp", "localtimestamp"}
BLOCKED_FUNCS = {"dblink", "set_config", "current_setting", "lo_import", "lo_export", "copy", "query_to_xml",
                 "version", "txid_current", "inet_server_addr"}
# sqlglot 把这些系统信息函数解析成专门的节点（而不是 Anonymous），按类型拦截
SYSTEM_INFO_NODES = tuple(getattr(exp, n) for n in (
    "CurrentVersion", "CurrentUser", "CurrentDatabase", "CurrentSchema", "CurrentRole", "SessionUser",
    "CurrentCatalog") if hasattr(exp, n))
WRITE_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter, exp.Command,
               exp.TruncateTable, exp.Grant, exp.Copy, exp.Into, exp.Lock, exp.Set, exp.Transaction)


class SQLRejected(ValueError):
    pass


def _func_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return node.name.lower()
    return node.sql_name().lower()


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
            if table.catalog or (table.db and table.db != "public"):
                raise SQLRejected("只允许访问当前数据库的 public schema")
            if table.name not in allowed_tables:
                raise SQLRejected(f"不允许访问表 {table.name}，可用：{', '.join(sorted(allowed_tables))}")

    if node := tree.find(*SYSTEM_INFO_NODES):
        raise SQLRejected(f"不允许调用函数 {node.sql_name().lower()}")
    for fn in tree.find_all(exp.Func):
        name = _func_name(fn)
        if name.startswith("pg_") or name in BLOCKED_FUNCS:
            raise SQLRejected(f"不允许调用函数 {name}")

    now_lit = exp.cast(exp.Literal.string(now.isoformat()), exp.DataType.build("timestamptz"))
    today_lit = exp.cast(exp.Literal.string(now.date().isoformat()), exp.DataType.build("date"))

    def rewrite(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.CurrentTimestamp):
            return now_lit.copy()
        if isinstance(node, exp.CurrentDate):
            return today_lit.copy()
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

"""独立数据集与真实HTTP API工厂；导入不连库、不加载模型。

后端目录执行（本模块不会自行运行这些命令）::

    uv run python -m evals.agent.diagnosis --build-only
    uv run uvicorn evals.agent.diagnosis_data:create_eval_app --factory --host 127.0.0.1 --port 8011
    uv run python -m evals.agent.diagnosis --url http://127.0.0.1:8011 --label flash-20261107-a

已存在的独立库需要显式 --build-only --rebuild；仅可能重建 shuiwu_diagnosis_eval。
数据库主机/管理员来自 EVAL_ADMIN_DATABASE_URL（默认本地compose），只读账号固定
text2sql_ro，密码来自 TEXT2SQL_RO_PASSWORD。模型配置照常读项目根.env，不打印key。
API使用进程内fakeredis固定时钟、关闭alarmmonitor，dense遵循本地RAG配置；请用新端口。
22题是22次Agent HTTP请求，规划/Text2SQL/综合/修复会使底层Flash请求多于22次。
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from evals.agent.diagnosis_score import UNITS, invariant_failures, same_time, same_value

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVAL_DB = 'shuiwu_diagnosis_eval'
RO_ROLE = 'text2sql_ro'
SPEC_PATH = HERE / 'diagnosis.yaml'


class EvalError(ValueError):
    """只包含可展示诊断信息，不包含连接串或模型密钥。"""


def load_spec():
    spec = yaml.safe_load(SPEC_PATH.read_text())
    ids = [c['id'] for c in spec['cases']]
    if not 20 <= len(ids) <= 24 or len(set(ids)) != len(ids):
        raise EvalError('诊断集必须有20~24个不同题目')
    seen = set()
    for case in spec['cases']:
        if case.get('after') and (case['after'] not in seen or not case.get('session')):
            raise EvalError(f'{case["id"]}的会话前序不合法')
        for snap in case.get('snapshots', []):
            if not set(snap['metrics']) <= UNITS.keys():
                raise EvalError('存在未登记的参考指标')
        seen.add(case['id'])
    return spec


def spec_digest():
    return hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()


def connection_urls():
    """不接受任意评测目标库，避免误用DATABASE_URL/READONLY_DATABASE_URL。"""
    from dotenv import load_dotenv
    from psycopg.conninfo import make_conninfo

    load_dotenv(ROOT.parent / '.env', override=False)
    source = os.getenv('EVAL_ADMIN_DATABASE_URL', 'postgresql://shuiwu:shuiwu@localhost:15432/postgres')
    admin = make_conninfo(source, dbname='postgres', connect_timeout='5')
    target = make_conninfo(source, dbname=EVAL_DB, connect_timeout='5')
    readonly = make_conninfo(source, dbname=EVAL_DB, user=RO_ROLE,
                             password=os.getenv('TEXT2SQL_RO_PASSWORD', 'readonly'), connect_timeout='5')
    return admin, target, readonly


def _json_cell(value):
    from datetime import date
    from decimal import Decimal

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def data_signature(conn):
    """仅查询独立库；用于识别造数未完成或服务器接错数据集。"""
    signature = {}
    for table in ('plant_reading', 'device_reading', 'weather'):
        row = conn.execute(f'SELECT count(*), min(ts), max(ts) FROM {table}').fetchone()
        signature[table] = [_json_cell(v) for v in row]
    return signature


def build_dataset(*, rebuild=False):
    """显式入口；仅本函数建库，纯评分测试不会调用它。"""
    import psycopg
    from psycopg import sql
    from psycopg.types.json import Jsonb

    from simulator import runner
    from simulator.scenarios import Scenario

    spec = load_spec()
    admin_url, target_url, readonly_url = connection_urls()
    with psycopg.connect(admin_url, autocommit=True) as admin:
        exists = admin.execute('SELECT 1 FROM pg_database WHERE datname=%s', (EVAL_DB,)).fetchone()
        if exists and not rebuild:
            raise EvalError('独立库已存在；如需重建，仅可显式使用 --build-only --rebuild')
        if not admin.execute('SELECT 1 FROM pg_roles WHERE rolname=%s', (RO_ROLE,)).fetchone():
            raise EvalError('集群缺少text2sql_ro角色；请先完成项目数据库初始化，本脚本不改共享角色密码')
        if exists:
            admin.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(EVAL_DB)))
        admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(EVAL_DB)))
    with psycopg.connect(target_url, autocommit=True) as conn:
        conn.execute("SET TIME ZONE 'Asia/Shanghai'")
        for name in ('001_schema.sql', '002_seed.sql', '005_ops.sql'):
            conn.execute((ROOT / 'db' / 'init' / name).read_text())
        # 不执行003/004：避免ALTER共享只读角色和密码；只授予本独立库的SELECT。
        sim, _ = runner.load_simulator(conn, seed=spec['seed'])
        for i, scenario in enumerate(spec['scenarios']):
            sim.book.add(Scenario(scenario['kind'], datetime.fromisoformat(scenario['start']),
                                  scenario['minutes'], scenario.get('plant'), scenario.get('params', {}),
                                  id=f'diagnosis-{i}'))
        clock = datetime.fromisoformat(spec['clock'])
        t, writer, count = clock - timedelta(days=spec['history_days']), runner.Writer(conn), 0
        while t <= clock:
            writer.add(sim.step(t, spec['step_seconds']))
            t += timedelta(seconds=spec['step_seconds'])
            count += 1
            if count % 1000 == 0:
                writer.flush()
        writer.flush()
        for gap in spec['gaps']:
            params = (gap['plant'], gap['start'], gap['end'])
            conn.execute('DELETE FROM plant_reading r USING plant p WHERE p.id=r.plant_id AND p.code=%s AND r.ts>=%s AND r.ts<%s', params)
            conn.execute('DELETE FROM device_reading r USING device d, plant p WHERE d.id=r.device_id AND p.id=d.plant_id AND p.code=%s AND r.ts>=%s AND r.ts<%s', params)
        for gap in spec['nulls']:
            assignments = sql.SQL(', ').join(sql.SQL('{}=NULL').format(sql.Identifier(m)) for m in gap['metrics'])
            conn.execute(sql.SQL('UPDATE plant_reading r SET {} FROM plant p WHERE p.id=r.plant_id AND p.code=%s AND r.ts>=%s AND r.ts<%s').format(assignments),
                         (gap['plant'], gap['start'], gap['end']))
        runner.refresh_aggregates(conn)
        # 元数据置于非public schema，Text2SQL看不到场景标签和答案。
        conn.execute('CREATE SCHEMA diagnosis_meta')
        conn.execute('CREATE TABLE diagnosis_meta.manifest (id int PRIMARY KEY CHECK(id=1), payload jsonb NOT NULL)')
        conn.execute(sql.SQL('GRANT CONNECT ON DATABASE {} TO {}').format(sql.Identifier(EVAL_DB), sql.Identifier(RO_ROLE)))
        conn.execute(sql.SQL('GRANT USAGE ON SCHEMA public TO {}').format(sql.Identifier(RO_ROLE)))
        conn.execute(sql.SQL('GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}').format(sql.Identifier(RO_ROLE)))
        # 写入完成标记之前核实预先声明的异常/正常范围，失败则不标成已建成。
        for case in spec['cases']:
            reference = database_reference(case, conn=conn)
            failures = invariant_failures(case, reference)
            if failures:
                raise EvalError(f'{case["id"]}的数据集不变量失败：{failures}')
            if case['mode'] == 'missing' and reference['valid_samples'] != 0:
                raise EvalError(f'{case["id"]}的缺失数据不变量失败')
        manifest = {'version': spec['version'], 'spec_sha256': spec_digest(), 'clock': spec['clock'],
                    'seed': spec['seed'], 'history_days': spec['history_days'], 'step_seconds': spec['step_seconds'],
                    'case_count': len(spec['cases']), 'signature': data_signature(conn)}
        conn.execute('INSERT INTO diagnosis_meta.manifest VALUES (1,%s)', (Jsonb(manifest),))
    with psycopg.connect(readonly_url, autocommit=True) as conn:
        check_readonly_identity(conn.execute('SELECT current_database(), current_user, current_setting(\'transaction_read_only\'), has_table_privilege(current_user, \'plant_reading\', \'INSERT,UPDATE,DELETE\')').fetchone())
    return manifest


def check_readonly_identity(row):
    if tuple(row[:2]) != (EVAL_DB, RO_ROLE) or row[2] != 'on' or row[3]:
        raise EvalError('只读连接必须指向独立库、使用text2sql_ro且无写入权限')


def database_reference(case, *, conn=None):
    """按固定实体和时刻直接读取金标，绝不从被评测模型的SQL/输出反推金标。"""
    if conn is None:
        import psycopg
        with psycopg.connect(connection_urls()[2], autocommit=True) as opened:
            opened.execute("SET TIME ZONE 'Asia/Shanghai'")
            return database_reference(case, conn=opened)
    from psycopg import sql
    from psycopg.rows import dict_row

    with conn.cursor(row_factory=dict_row) as cur:
        plants = cur.execute('SELECT id,code,name FROM plant ORDER BY code').fetchall()
        reference = {'plants': {p['code']: dict(p) for p in plants}, 'snapshots': []}
        for spec in case.get('snapshots', []):
            entity, metrics = spec['entity'], spec['metrics']
            columns = sql.SQL(',').join(sql.SQL('r.{}').format(sql.Identifier(m)) for m in metrics)
            if entity == 'weather':
                query = sql.SQL('SELECT {} FROM weather r WHERE r.ts=%s').format(columns)
                rows = cur.execute(query, (spec['at'],)).fetchall()
                device_id = None
            elif '-' in entity:
                query = sql.SQL('SELECT d.id AS device_id, {} FROM device_reading r JOIN device d ON d.id=r.device_id WHERE d.code=%s AND r.ts=%s').format(columns)
                rows = cur.execute(query, (entity, spec['at'])).fetchall()
                device_id = rows[0]['device_id'] if rows else None
            else:
                query = sql.SQL('SELECT {} FROM plant_reading r JOIN plant p ON p.id=r.plant_id WHERE p.code=%s AND r.ts=%s').format(columns)
                rows = cur.execute(query, (entity, spec['at'])).fetchall()
                device_id = None
            if len(rows) != 1:
                raise EvalError(f'{case["id"]}缺少唯一原始参考采样')
            reference['snapshots'].append({'entity': entity, 'at': spec['at'], 'device_id': device_id,
                                          'values': {m: _json_cell(rows[0][m]) for m in metrics},
                                          'units': {m: UNITS[m] for m in metrics}})
        if 'statistics' in case:
            query = case['statistics']
            metric = query['metric']
            if metric not in UNITS:
                raise EvalError('统计参考指标未登记')
            query_sql = sql.SQL('SELECT p.code,r.ts,r.{} AS value FROM plant_reading r JOIN plant p ON p.id=r.plant_id WHERE p.code=%s AND r.ts>=%s AND r.ts<%s ORDER BY r.ts').format(sql.Identifier(metric))
            rows = cur.execute(query_sql, (query['plant'], query['start'], query['end'])).fetchall()
            reference['statistics_raw'] = [{k: _json_cell(v) for k, v in r.items()} for r in rows]
            reference['valid_samples'] = sum(r['value'] is not None for r in rows)
    return reference


def independent_statistics(query, rows):
    """另一种实现从原始采样构建参考，不调用生产summarize/make_result。

    本集仅完整5分钟边界、单厂、氨氮阈值5mg/L；不把此函数当通用统计产品。
    """
    start, end = datetime.fromisoformat(query['start']), datetime.fromisoformat(query['end'])
    if any(t.second or t.microsecond or t.minute % 5 for t in (start, end)):
        raise EvalError('本评测参考只支持完整5分钟边界')
    if query['metric'] != 'nh3n_out':
        raise EvalError('本评测独立统计参考只声明氨氮5mg/L阈值')
    buckets, values = {}, []
    for row in rows:
        at = datetime.fromisoformat(row['ts']) if isinstance(row['ts'], str) else row['ts']
        value = row['value']
        if row['code'] != query['plant'] or not start <= at < end:
            raise EvalError('独立统计参考出现厂站或时段越界')
        if value is not None:
            bucket = int((at - start).total_seconds()) // 300
            buckets.setdefault(bucket, []).append(float(value))
            values.append((at, float(value)))
    expected = int((end - start).total_seconds()) // 300
    means = {i: sum(v) / len(v) for i, v in buckets.items()}
    flags = [i in means and means[i] > 5 for i in range(expected)]
    streak = longest = 0
    for flag in flags:
        streak = streak + 1 if flag else 0
        longest = max(streak, longest)
    bad_hours = {(at.astimezone(start.tzinfo).date(), at.astimezone(start.tzinfo).hour)
                 for at, value in values if value > 5}
    return {'values': {
        'mean': round(sum(means.values()) / len(means), 4) if means else None,
        'instant_max': round(max(v for _, v in values), 4) if values else None,
        'bucket_mean_max': round(max(means.values()), 4) if means else None,
        'exceed_hours': sum(flags) / 12 if means else None,
        'longest_exceed_hours': longest / 12 if means else None,
        'hours_with_exceed': len(bad_hours) if means else None},
        'coverage': {'expected_buckets': expected, 'observed_buckets': len(means),
                     'missing_buckets': expected - len(means), 'coverage': len(means) / expected}}


def verify_statistics_reference(query, server, raw):
    """固定HTTP统计事实还要通过独立原始数据、单位、覆盖、ID与边界校验。"""
    independent = independent_statistics(query, raw)
    period = server.get('period', {})
    failures = []
    if (not same_time(period.get('start'), query['start']) or not same_time(period.get('end'), query['end'])
            or not same_time(period.get('requested_start'), query['start'])
            or not same_time(period.get('requested_end'), query['end'])
            or period.get('timezone') != 'Asia/Shanghai'):
        failures.append('reference_statistics_period')
    if server.get('metric', {}).get('key') != query['metric'] or server.get('metric', {}).get('unit') != 'mg/L':
        failures.append('reference_statistics_metric_unit')
    facts = {f['id']: f for f in server.get('facts', [])}
    expected_ids = {f'{query["plant"]}.{query["metric"]}.{stat}' for stat in query['statistics']
                    if independent['values'][stat] is not None}
    if facts.keys() != expected_ids:
        failures.append('reference_statistics_ids')
    for stat in query['statistics']:
        fid = f'{query["plant"]}.{query["metric"]}.{stat}'
        if fid not in facts:
            continue
        fact, want = facts[fid], independent['values'][stat]
        unit = '小时' if stat in {'exceed_hours', 'longest_exceed_hours'} else '个小时桶' if stat == 'hours_with_exceed' else 'mg/L'
        if (fact.get('statistic') != stat or fact.get('metric') != query['metric'] or fact.get('plant') != query['plant']
                or fact.get('unit') != unit or not same_value(fact.get('value'), want)
                or not same_value(fact.get('coverage'), {query['plant']: independent['coverage']})):
            failures.append(f'reference_statistics_fact:{fid}')
    if not expected_ids and not server.get('warnings'):
        failures.append('reference_missing_warning')
    if [p.get('code') for p in server.get('plants', [])] != [query['plant']]:
        failures.append('reference_statistics_plant')
    return failures


def verify_sql_outputs(result):
    """以独立库只读账号重放被引用SQL，核对返回表；仍另外核对固定实体/时刻金标。"""
    import psycopg

    from app.ai.text2sql.guard import guard
    from app.ai.text2sql.service import _cell

    failures = []
    with psycopg.connect(connection_urls()[2], autocommit=True) as conn:
        for call in result.get('tool_calls', []):
            if call.get('tool') != 'sql_query' or call.get('status') not in {'ok', 'empty'}:
                continue
            out = call.get('output', {})
            if not out.get('sql'):
                failures.append(f'sql_missing:{call.get("id")}')
                continue
            safe = guard(out['sql'], {'plant', 'device', 'plant_reading', 'plant_reading_1h', 'device_reading', 'weather',
                                      'alarm', 'work_order', 'report'}, datetime.fromisoformat(result['now']))
            with conn.transaction():
                conn.execute('SET TRANSACTION READ ONLY')
                conn.execute("SET LOCAL TIME ZONE 'Asia/Shanghai'")
                conn.execute("SET LOCAL statement_timeout='5s'")
                with conn.cursor() as cur:
                    cur.execute(safe)
                    columns = [d.name for d in cur.description or []]
                    rows = [[_cell(v) for v in row] for row in cur.fetchmany(200)]
            if columns != out.get('columns') or not same_value(rows, out.get('rows')):
                failures.append(f'sql_database_replay:{call.get("id")}')
    return failures


def create_eval_app():
    """供uvicorn --factory加载；真实网关，fakeredis隔离，无演示库/Redis接触。"""
    from contextlib import asynccontextmanager
    from dataclasses import replace

    import fakeredis

    from app.config import Settings
    from app.main import create_app

    spec = load_spec()
    _, target, readonly = connection_urls()
    if ([v.strip() for v in os.getenv('LLM_PROVIDERS', 'deepseek').split(',') if v.strip()] != ['deepseek']
            or os.getenv('DEEPSEEK_MODEL', 'deepseek-flash') != 'deepseek-flash'
            or os.getenv('LLM_MODE', 'live') != 'live'):
        raise EvalError('评测服务器必须唯一启用deepseek/deepseek-flash且LLM_MODE=live')
    settings = replace(Settings(), database_url=target, readonly_database_url=readonly, alarm_monitor_enabled=False)
    redis = fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer(), decode_responses=True)
    app = create_app(settings, redis=redis)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        await redis.set('sim:clock', spec['clock'])
        await redis.set('sim:speed', '0')
        async with original_lifespan(app):
            yield
    app.router.lifespan_context = lifespan

    @app.get('/__diagnosis__/identity', include_in_schema=False)
    async def identity():
        from psycopg.rows import tuple_row
        # 业务连接可能返回dict；身份检查使用局部游标，不依赖或改变共享连接格式。
        async with app.state.pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
            primary = await (await cur.execute('SELECT current_database()')).fetchone()
            manifest = await (await cur.execute('SELECT payload FROM diagnosis_meta.manifest WHERE id=1')).fetchone()
        async with app.state.ro_pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
            ro = await (await cur.execute("SELECT current_database(),current_user,current_setting('transaction_read_only'),has_table_privilege(current_user,'plant_reading','INSERT,UPDATE,DELETE')")).fetchone()
            check_readonly_identity(ro)
        return {'database': primary[0], 'readonly_database': ro[0], 'readonly_user': ro[1],
                'readonly': ro[2] == 'on' and not ro[3], 'redis_isolated': True,
                'alarm_monitor_enabled': settings.alarm_monitor_enabled,
                'clock': await redis.get('sim:clock'), 'manifest': manifest[0]}
    return app

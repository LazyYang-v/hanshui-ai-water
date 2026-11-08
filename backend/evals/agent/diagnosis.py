"""独立端到端诊断评测入口；默认完整执行22题，不使用LLM judge。

    uv run python -m evals.agent.diagnosis --describe
    uv run python -m evals.agent.diagnosis --build-only
    uv run uvicorn evals.agent.diagnosis_data:create_eval_app --factory --host 127.0.0.1 --port 8011
    uv run python -m evals.agent.diagnosis --url http://127.0.0.1:8011 --label flash-a
    uv run python -m evals.agent.diagnosis --label flash-b --ids blower-followup-latest --limit 1

--limit先限制选题，再自动补齐同会话前序，实际请求数可能比limit多。
每题保存真实原始响应后再评分，异常逐题记录并继续；新label目录独占，不覆盖历史结果。
人工核查未完成的题只标needs_review，不宣称诊断语义通过。退出码1表示有自动检查失败。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx

from evals.agent.diagnosis_data import (
    EVAL_DB,
    HERE,
    RO_ROLE,
    EvalError,
    build_dataset,
    connection_urls,
    data_signature,
    database_reference,
    load_spec,
    spec_digest,
    verify_sql_outputs,
    verify_statistics_reference,
)
from evals.agent.diagnosis_score import compact, same_time, score_case
from evals.http import alogin


def select_cases(spec, ids=None, limit=None):
    cases = spec['cases']
    available = {c['id']: c for c in cases}
    if ids and set(ids) - available.keys():
        raise EvalError('不存在的题号：' + ','.join(sorted(set(ids) - available.keys())))
    requested = [c for c in cases if not ids or c['id'] in ids]
    if limit is not None:
        if limit < 1:
            raise EvalError('limit必须为正整数')
        requested = requested[:limit]
    selected = {c['id'] for c in requested}
    for case in requested:
        while case.get('after'):
            selected.add(case['after'])
            case = available[case['after']]
    return [c for c in cases if c['id'] in selected]


def reserve_output(label, directory=None):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', label):
        raise EvalError('label只允许字母数字下划线和连字符')
    parent = directory or HERE / 'results'
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / f'diagnosis-{label}'
    try:
        target.mkdir()  # 原子占位；启动失败也不允许复用该标签。
    except FileExistsError as exc:
        raise EvalError('结果标签已存在，请换新label，不覆盖已有结果') from exc
    return target


def save_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def safe_exception(exc):
    if isinstance(exc, EvalError):
        return {'type': type(exc).__name__, 'message': str(exc)}
    if isinstance(exc, httpx.HTTPStatusError):
        return {'type': type(exc).__name__, 'status_code': exc.response.status_code,
                'path': exc.request.url.path}
    # 网络/数据库异常可能含连接串或认证信息，仅保留类型。
    return {'type': type(exc).__name__, 'message': '本题失败；原始响应如已收到则保存在本题文件中'}


async def get_json(client, path):
    response = await client.get(path)
    response.raise_for_status()
    return response.json()


async def assert_flash_gateway(client):
    gateway = await get_json(client, '/api/ai/gateway')
    providers = gateway.get('providers', {})
    if (not gateway.get('available') or gateway.get('mode') != 'live' or set(providers) != {'deepseek'}
            or providers['deepseek'].get('model') != 'deepseek-flash'):
        raise EvalError('真实调用被阻止：API必须唯一启用deepseek/deepseek-flash，且mode=live')
    return gateway


def verify_local_dataset(manifest):
    import psycopg
    with psycopg.connect(connection_urls()[2], autocommit=True) as conn:
        conn.execute("SET TIME ZONE 'Asia/Shanghai'")
        if conn.execute('SELECT current_database()').fetchone()[0] != EVAL_DB:
            raise EvalError('参考连接未指向独立诊断库')
        if data_signature(conn) != manifest.get('signature'):
            raise EvalError('独立诊断库范围/采样数与完成标记不符，请先重新造数')


async def preflight(client, spec):
    # 在任何付费Agent请求前，先核对主库、只读库、进程内Redis、固定时钟与数据版本。
    identity = await get_json(client, '/__diagnosis__/identity')
    manifest = identity.get('manifest', {})
    if (identity.get('database') != EVAL_DB or identity.get('readonly_database') != EVAL_DB
            or identity.get('readonly_user') != RO_ROLE or identity.get('readonly') is not True
            or identity.get('redis_isolated') is not True or identity.get('alarm_monitor_enabled') is not False
            or not same_time(identity.get('clock'), spec['clock'])
            or manifest.get('spec_sha256') != spec_digest() or manifest.get('version') != spec['version']
            or manifest.get('case_count') != len(spec['cases'])):
        raise EvalError('API身份/时钟/数据版本不符，请使用独立diagnosis API工厂')
    await asyncio.to_thread(verify_local_dataset, manifest)
    if not same_time((await get_json(client, '/api/clock')).get('now'), spec['clock']):
        raise EvalError('HTTP演示时钟不符')
    catalog = await get_json(client, '/api/ai/knowledge')
    if not catalog.get('dense_available'):
        raise EvalError('独立API的本地RAG dense尚不可用，请先准备本地模型缓存')
    gateway = await assert_flash_gateway(client)
    return {'identity': identity, 'gateway': gateway, 'dense_available': True}


async def verify_knowledge_originals(client, result):
    for cite in result.get('citations', []):
        if cite.get('kind') != 'knowledge':
            continue
        path = cite.get('url', '')
        parsed = urlsplit(path)
        if parsed.scheme or parsed.netloc or not path.startswith('/api/'):
            return False
        original = await get_json(client, path)
        quote = compact(cite.get('quote', ''))
        if len(quote) < 8 or quote not in compact(original.get('text', '')):
            return False
    return True


async def run_cases(client, cases, spec, target, report):
    sessions, runs = {}, {}
    for case in cases:
        started = time.perf_counter()
        record = {'id': case['id'], 'family': case['family'], 'mode': case['mode'],
                  'question': case['question'], 'status': 'pending', 'agent_http_requests': 0}
        path = target / f'{case["id"]}.json'
        save_json(path, record)
        try:
            reference = await asyncio.to_thread(database_reference, case)
            if 'statistics' in case:
                response = await client.post('/api/ai/statistics', json=case['statistics'])
                response.raise_for_status()
                reference['statistics'] = response.json()
                failures = verify_statistics_reference(case['statistics'], reference['statistics'], reference.pop('statistics_raw'))
                reference['statistics_database_verified'] = not failures
                if failures:
                    record['reference'] = reference
                    raise EvalError('独立数据库统计金标核验失败：' + ','.join(failures))
            record['reference'] = reference
            session = case.get('session')
            if case.get('after') and session not in sessions:
                raise EvalError('前序未返回可用会话，本题无法以同会话执行')
            payload = {'question': case['question']}
            if session in sessions:
                payload['session_id'] = sessions[session]
            record['request'] = payload
            await assert_flash_gateway(client)  # 每题付费调用前再次确认唯一Flash。
            record['agent_http_requests'] = 1
            save_json(path, record)
            response = await client.post('/api/ai/chat', json=payload)
            # HTTP失败体也保留，不伪造Agent模型结果。
            record['http_status'] = response.status_code
            if response.is_error:
                record['http_error_body'] = response.text[:2000]
                save_json(path, record)
            response.raise_for_status()
            try:
                result = response.json()
            except ValueError:
                record['http_error_body'] = response.text[:2000]
                save_json(path, record)
                raise
            record['result'] = result
            save_json(path, record)  # 在任何后续评分、SQL重放、知识原文HTTP之前保存真实原始结果。
            session_verified = True
            if session:
                sid = result.get('session_id')
                if not isinstance(sid, str) or not re.fullmatch(r'[0-9a-f]{32}', sid):
                    session_verified = False
                else:
                    session_verified = not (session in sessions and sessions[session] != sid)
                    sessions[session] = sid
                    runs.setdefault(session, []).append(result.get('run_id'))
                    stored = await get_json(client, f'/api/ai/chat/sessions/{sid}')
                    session_verified &= [t.get('run_id') for t in stored.get('turns', [])] == runs[session][-6:]
                    record['stored_session_run_ids'] = [t.get('run_id') for t in stored.get('turns', [])]
            verified = await verify_knowledge_originals(client, result)
            score = score_case(case, result, reference, clock=spec['clock'],
                               knowledge_verified=verified, session_verified=session_verified)
            replay_failures = await asyncio.to_thread(verify_sql_outputs, result)
            if replay_failures:
                score['failures'].extend(replay_failures)
                score.update(mechanical_pass=False, semantic_pass=False, status='fail')
            record.update(score=score, status=score['status'])
        except Exception as exc:
            record.update(status='fail', exception=safe_exception(exc))
            record.setdefault('score', {'mechanical_pass': False, 'semantic_pass': False,
                                        'status': 'fail', 'failures': ['case_exception'],
                                        'manual_review': [{'status': 'pending', 'item': x} for x in case.get('manual', [])]})
        record['elapsed_ms'] = round((time.perf_counter() - started) * 1000, 1)
        save_json(path, record)
        report['cases'].append({'id': case['id'], 'status': record['status'], 'file': path.name,
                                'mechanical_pass': record['score']['mechanical_pass'],
                                'semantic_pass': record['score']['semantic_pass'],
                                'agent_http_requests': record['agent_http_requests'],
                                'failures': record['score']['failures'],
                                'exception': record.get('exception')})
        report['summary'] = summarize(report['cases'])
        save_json(target / 'report.json', report)
        print(json.dumps({'id': case['id'], 'status': record['status'], 'ms': record['elapsed_ms'],
                          'failures': record['score']['failures'], 'exception': record.get('exception')},
                         ensure_ascii=False), flush=True)
    return report


def summarize(cases):
    return {'completed': len(cases), 'agent_http_requests': sum(c['agent_http_requests'] for c in cases),
            'mechanical_pass': sum(c['mechanical_pass'] for c in cases),
            'semantic_pass': sum(c['semantic_pass'] is True for c in cases),
            'needs_review': sum(c['status'] == 'needs_review' for c in cases),
            'failed': sum(c['status'] == 'fail' for c in cases)}


async def evaluate(args, spec, cases, target):
    report = {'label': args.label, 'started_at': datetime.now(UTC).isoformat(),
              'clock': spec['clock'], 'version': spec['version'], 'spec_sha256': spec_digest(),
              'database': EVAL_DB, 'selected_ids': [c['id'] for c in cases],
              'request_count_note': '题数是Agent HTTP请求数；底层Flash调用因规划、Text2SQL与重试会更多',
              'semantic_policy': '人工项未审核则semantic_pass=null；mechanical_pass不等于诊断准确率',
              'cases': []}
    save_json(target / 'report.json', report)
    try:
        async with httpx.AsyncClient(base_url=args.url, timeout=240, trust_env=False) as client:
            await alogin(client)
            report['preflight'] = await preflight(client, spec)
            save_json(target / 'report.json', report)
            await run_cases(client, cases, spec, target, report)
    except Exception as exc:
        report['startup_exception'] = safe_exception(exc)
    finally:
        report['finished_at'] = datetime.now(UTC).isoformat()
        report['summary'] = summarize(report['cases'])
        save_json(target / 'report.json', report)
    return not report.get('startup_exception') and report['summary']['failed'] == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--build-only', action='store_true', help='只建独立库，不发送模型请求')
    parser.add_argument('--rebuild', action='store_true', help='仅配合build-only，显式重建独立库')
    parser.add_argument('--describe', action='store_true', help='只展示固定时钟、题目与启动步骤')
    parser.add_argument('--url', default='http://127.0.0.1:8011')
    parser.add_argument('--label')
    parser.add_argument('--ids', help='逗号分隔题号；自动补齐同会话前序')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    spec = load_spec()
    if args.rebuild and not args.build_only:
        parser.error('--rebuild只能与--build-only一起使用')
    if args.describe:
        print(json.dumps({'database': EVAL_DB, 'clock': spec['clock'], 'case_count': len(spec['cases']),
                          'history_days': spec['history_days'], 'seed': spec['seed'],
                          'ids': [c['id'] for c in spec['cases']], 'instructions': __doc__}, ensure_ascii=False, indent=2))
        return
    if args.build_only:
        manifest = build_dataset(rebuild=args.rebuild)
        print(json.dumps({'built': EVAL_DB, **manifest}, ensure_ascii=False, indent=2))
        return
    if not args.label:
        parser.error('真实评测必须指定未占用的--label')
    try:
        url = urlsplit(args.url)
        if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise EvalError('url必须是无认证信息、无查询参数的HTTP API基址')
        cases = select_cases(spec, args.ids.split(',') if args.ids else None, args.limit)
        target = reserve_output(args.label)
        success = asyncio.run(evaluate(args, spec, cases, target))
    except EvalError as exc:
        parser.error(str(exc))
    print(f'逐题原始响应与评分：{target}/report.json；人工项目须另行核查', flush=True)
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""固定只读统计：先声明口径，再计算事实；累计时长与连续时长分别计算。

5 分钟完整桶至少有一个非空值即视为有数据；缺失桶不记零，且打断连续异常。
原始采样极值和桶均值极值分别保留。排序按显示精度（4位小数）比较，采用密集排名。
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Literal, get_args

from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.analytics.rules import BUCKET, LOCAL_TZ
from app.domain.metrics import PLANT_METRICS

Statistic = Literal['mean', 'instant_max', 'instant_min', 'bucket_mean_max', 'bucket_mean_min',
                    'exceed_hours', 'longest_exceed_hours', 'hours_with_exceed', 'rank', 'range', 'top']
RANK_LABELS = {'mean': '5分钟均值的时段平均值', 'instant_max': '原始采样最大值',
               'bucket_mean_max': '5分钟均值最大值', 'exceed_hours': '5分钟均值累计异常时长'}


class StatisticsQuery(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True)
    plant: str | None = Field(None, description='厂站编码；null 表示所有厂站')
    metric: str = Field(description='PLANT_METRICS 中的指标字段名')
    # FastAPI 先把 JSON 解码为 Python dict，日期字段需允许 ISO 字符串；其余字段保持严格。
    start: datetime = Field(strict=False)
    end: datetime = Field(strict=False)
    statistics: list[Statistic] = Field(default_factory=lambda: [
        'exceed_hours', 'longest_exceed_hours', 'instant_max', 'bucket_mean_max'],
        min_length=1, max_length=len(get_args(Statistic)))
    rank_by: Literal['mean', 'instant_max', 'bucket_mean_max', 'exceed_hours'] = 'mean'

    @field_validator('start', 'end', mode='before')
    @classmethod
    def datetime_input(cls, value):
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            value = value.strip()
            if len(value) > 10 and value[10] in {'T', 't', ' '}:
                return value
        raise ValueError('时间须为包含时分的ISO日期时间，不能使用日期或数字时间戳')

    @field_validator('metric')
    @classmethod
    def known_metric(cls, value):
        if value not in PLANT_METRICS:
            raise ValueError('统计指标必须来自指标目录')
        return value

    @field_validator('plant')
    @classmethod
    def known_code(cls, value):
        if value is not None:
            value = value.upper()
            if value not in {f'P{i:02}' for i in range(1, 8)}:
                raise ValueError('厂站编码须为P01至P07或null')
        return value

    @model_validator(mode='after')
    def valid_period(self):
        self.start = self.start.replace(tzinfo=LOCAL_TZ) if self.start.tzinfo is None else self.start.astimezone(LOCAL_TZ)
        self.end = self.end.replace(tzinfo=LOCAL_TZ) if self.end.tzinfo is None else self.end.astimezone(LOCAL_TZ)
        if not timedelta(0) < self.end - self.start <= timedelta(days=31):
            raise ValueError('统计时段须大于0且不超过31天')
        if len(set(self.statistics)) != len(self.statistics):
            raise ValueError('statistics 不得重复')
        if (set(self.statistics) & {'exceed_hours', 'longest_exceed_hours', 'hours_with_exceed'} or
                self.rank_by == 'exceed_hours') and PLANT_METRICS[self.metric].limit is None:
            raise ValueError('该指标没有异常阈值，不能统计超标时长；请显式选择mean/instant_max等统计项')
        return self


def bounds(query: StatisticsQuery, now: datetime) -> tuple[datetime, datetime]:
    now = now.replace(tzinfo=LOCAL_TZ) if now.tzinfo is None else now.astimezone(LOCAL_TZ)

    def floor(t):
        t = t.astimezone(LOCAL_TZ)
        return t - timedelta(minutes=t.minute % 5, seconds=t.second, microseconds=t.microsecond)
    start = floor(query.start)
    if start < query.start:
        start += BUCKET
    end = floor(min(query.end, now))
    if start >= end:
        raise ValueError('所选时段没有已写满的5分钟桶；未来或未完成的桶不能作为统计证据')
    return start, end


def summarize(rows: list[dict], start: datetime, end: datetime, metric: str) -> dict:
    """rows 含 t/mean/raw_max/raw_min/samples；不补零，缺失处切断连续异常。"""
    points = {r['t']: r for r in rows if r.get('mean') is not None and start <= r['t'] < end}
    expected = int((end - start) / BUCKET)
    meta = PLANT_METRICS[metric]
    longest = streak = bad_count = 0
    bad_hours = set()
    t = start
    while t < end:
        row = points.get(t)
        bad = row is not None and meta.limit is not None and (
            row['mean'] < meta.limit if meta.lower_is_bad else row['mean'] > meta.limit)
        streak = streak + 1 if bad else 0
        longest = max(longest, streak)
        bad_count += bool(bad)
        if row is not None and meta.limit is not None and (
            row['raw_min'] < meta.limit if meta.lower_is_bad else row['raw_max'] > meta.limit):
            bad_hours.add(t.replace(minute=0, second=0, microsecond=0))
        t += BUCKET
    values = [r['mean'] for r in points.values()]
    return {'expected_buckets': expected, 'observed_buckets': len(points),
            'missing_buckets': expected - len(points), 'coverage': len(points) / expected,
            'mean': None if not values else round(sum(values) / len(values), 4),
            'instant_max': None if not values else round(max(r['raw_max'] for r in points.values()), 4),
            'instant_min': None if not values else round(min(r['raw_min'] for r in points.values()), 4),
            'bucket_mean_max': None if not values else round(max(values), 4),
            'bucket_mean_min': None if not values else round(min(values), 4),
            'exceed_hours': bad_count / 12 if meta.limit is not None and values else None,
            'longest_exceed_hours': longest / 12 if meta.limit is not None and values else None,
            'hours_with_exceed': len(bad_hours) if meta.limit is not None and values else None}


def make_result(query: StatisticsQuery, start: datetime, end: datetime, plants: list[dict]) -> dict:
    """plants 含 code/name/summary。事实文本、口径、单位和覆盖均由服务端生成。"""
    meta = PLANT_METRICS[query.metric]
    period = f'{start:%Y-%m-%d %H:%M}至{end:%Y-%m-%d %H:%M}（北京时间）'
    direction = '低于下限' if meta.lower_is_bad else '超标'
    facts, warnings = [], []
    if start != query.start or end != query.end:
        warnings.append(f'仅统计{period}内的完整5分钟桶；边界未满5分钟及未来数据未纳入统计')
    selected = [p for p in plants if query.plant is None or p['code'] == query.plant]
    rank_values = {p['code']: round(p['summary'][query.rank_by], 4)
                   for p in plants if p['summary'][query.rank_by] is not None}
    comparable = (len(plants) >= 2 and len(rank_values) == len(plants)
                  and all(p['summary']['coverage'] == 1 for p in plants))
    ranking = list(rank_values.values())
    ranks = {v: i + 1 for i, v in enumerate(sorted(set(ranking), reverse=True))}
    ties = Counter(ranking)
    rank_unit = '小时' if query.rank_by == 'exceed_hours' else meta.unit
    label = RANK_LABELS[query.rank_by]
    coverage = {p['code']: {k: p['summary'][k] for k in (
        'expected_buckets', 'observed_buckets', 'missing_buckets', 'coverage')} for p in plants}
    titles = {'mean': '5分钟均值的时段平均值', 'instant_max': '原始采样瞬时最大值',
              'instant_min': '原始采样瞬时最小值', 'bucket_mean_max': '5分钟均值最大值',
              'bucket_mean_min': '5分钟均值最小值'}
    aggregations = {
        'mean': '完整5分钟桶内对有效采样求均值，再对有效桶等权平均，不按各桶采样数加权',
        'instant_max': '统计时段内完整5分钟桶中，有效原始采样值的最大值',
        'instant_min': '统计时段内完整5分钟桶中，有效原始采样值的最小值',
        'bucket_mean_max': '完整5分钟桶内先对有效采样求均值，再取有效桶均值的最大值',
        'bucket_mean_min': '完整5分钟桶内先对有效采样求均值，再取有效桶均值的最小值',
        'exceed_hours': f'5分钟桶均值{direction}即累计5分钟；缺失不补零，不推断桶内持续异常',
        'longest_exceed_hours': f'5分钟桶均值{direction}的最长连续桶数乘5分钟；正常或缺失桶中断，不推断桶内持续异常',
        'hours_with_exceed': f'统计时段内曾有原始采样{direction}的北京时间小时桶数，不代表持续异常小时数',
    }
    comparison = f'同一时段覆盖完整的至少两座厂站，按{label}显示到小数点后4位比较'
    aggregations.update(rank=comparison + '，降序密集排名', range=comparison + '，取最小和最大值',
                        top=comparison + '，取最高值及所有并列厂站')

    def number(value):
        # :g 的默认6位有效数字会把123.4567和123.4566显示成同一个值。
        return f'{round(value, 4):.4f}'.rstrip('0').rstrip('.') if round(value, 4) else '0'

    def add(fid, statistic, value, text, plant=None, note='', subject=''):
        unit = ('名' if statistic == 'rank' else '个小时桶' if statistic == 'hours_with_exceed'
                else '小时' if statistic in {'exceed_hours', 'longest_exceed_hours'}
                else rank_unit if statistic in {'range', 'top'} else meta.unit)
        covered_codes = [plant] if plant is not None and statistic != 'rank' else coverage
        # note：覆盖说明（已含在 text 末尾）；多条事实合成一句时相同的说明只写一次
        # subject：句首主语（厂站+指标），同一陈述里连续同主语的事实只写第一次
        facts.append({'id': fid, 'statistic': statistic, 'value': value, 'text': text + note, 'note': note, 'subject': subject,
                      'metric': query.metric, 'plant': plant, 'unit': unit,
                      'aggregation': aggregations[statistic],
                      'coverage': {code: coverage[code] for code in covered_codes}})

    for plant in selected:
        s, code = plant['summary'], plant['code']
        who = f"{plant['name']}（{code}）{meta.label}"
        if not s['observed_buckets']:
            warnings.append(f'{code}没有有效数据，不能解释为零、达标或无超标')
            continue
        suffix = (f"；有效{s['observed_buckets']}/{s['expected_buckets']}个5分钟桶" +
                  ('，缺失处中断连续计数，时长仅代表已观测数据' if s['missing_buckets'] else ''))
        for stat in query.statistics:
            v = s.get(stat)
            text = None
            if stat in titles:
                text = f'{who}的{titles[stat]}为{number(v)} {meta.unit}'
                if stat == 'mean':
                    text += '（有效5分钟桶等权，不按采样数加权）'
            elif stat in {'exceed_hours', 'longest_exceed_hours'}:
                title = '累计' if stat == 'exceed_hours' else '已观测的最长连续'
                count = round(v * 12)
                text = (f'{who}按5分钟均值统计，{title}{direction}时长为{number(v)}小时'
                        f'（{count * 5}分钟，{count}个5分钟桶；阈值{number(meta.limit)} {meta.unit}），'
                        '不代表桶内持续观测')
            elif stat == 'hours_with_exceed':
                text = f'{who}有{v}个小时桶内曾出现{direction}的原始采样值，不能据此当作连续{v}小时{direction}'
            elif stat == 'rank' and comparable:
                v = rank_values[code]
                text = (f'{who}在覆盖完整的{len(plants)}座厂站中，按{label}降序'
                        f"{'并列' if ties[v] > 1 else ''}第{ranks[v]}名（密集排名，本厂值{number(v)} {rank_unit}）")
                v = ranks[v]
            if text:
                add(f'{code}.{query.metric}.{stat}', stat, v, text, code, suffix, who)
    if set(query.statistics) & {'rank', 'range', 'top'}:
        if not comparable:
            warnings.append('厂站数据缺失或比较对象不足，不能生成跨厂排名、范围或最高值结论')
        else:
            low, high = min(ranking), max(ranking)
            leaders = [p['name'] + f"（{p['code']}）" for p in plants if rank_values[p['code']] == high]
            expected = int((end - start) / BUCKET)
            suffix = f'；各厂均有{expected}/{expected}个有效5分钟桶'
            if 'range' in query.statistics:
                add(f'plants.{query.metric}.{query.rank_by}.range', 'range', {'min': low, 'max': high},
                    f'{len(plants)}座覆盖完整的厂站中，{meta.label}的{label}范围为'
                    f'{number(low)}至{number(high)} {rank_unit}', note=suffix)
            if 'top' in query.statistics:
                add(f'plants.{query.metric}.{query.rank_by}.top', 'top', high,
                    f"{meta.label}按{label}比较，{'并列' if len(leaders) > 1 else ''}最高为{'、'.join(leaders)}，"
                    f'值为{number(high)} {rank_unit}', note=suffix)
    return {'period': {'requested_start': query.start.isoformat(), 'requested_end': query.end.isoformat(),
                       'start': start.isoformat(), 'end': end.isoformat(), 'timezone': 'Asia/Shanghai'},
            'metric': meta.to_dict(), 'bucket_minutes': 5, 'ranking_precision': 4,
            'coverage_policy': '完整5分钟桶内至少一个有效采样即计一个有效桶；缺失不补零且打断连续异常；不保证桶内持续观测',
            'facts': facts, 'warnings': warnings, 'plants': selected, 'period_text': period}


async def collect_statistics(conn, query: StatisticsQuery, now: datetime) -> dict:
    start, end = bounds(query, now)
    statement = sql.SQL('''
        WITH b AS (
            SELECT plant_id, time_bucket('5 minutes', ts) AS t,
                   avg({metric})::float8 AS mean, max({metric})::float8 AS raw_max,
                   min({metric})::float8 AS raw_min, count({metric}) AS samples
            FROM plant_reading WHERE ts >= %(start)s AND ts < %(end)s
            GROUP BY 1, 2
        )
        SELECT p.code, p.name, b.t, b.mean, b.raw_max, b.raw_min, b.samples
        FROM plant p LEFT JOIN b ON b.plant_id = p.id ORDER BY p.code, b.t
    ''').format(metric=sql.Identifier(query.metric))
    # 只修改当前游标的行格式；连接归还池后仍会被Text2SQL复用。
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(statement, {'start': start, 'end': end})
        rows = await cur.fetchall()
    grouped = {}
    for row in rows:
        grouped.setdefault(row['code'], {'code': row['code'], 'name': row['name'], 'rows': []})['rows'].append(row)
    plants = [{'code': p['code'], 'name': p['name'], 'summary': summarize(p['rows'], start, end, query.metric)}
              for p in grouped.values()]
    if query.plant is not None and not any(p['code'] == query.plant for p in plants):
        raise ValueError('厂站不存在')
    result = make_result(query, start, end, plants)
    result['sql'] = statement.as_string(conn)
    result['params'] = {'start': start.isoformat(), 'end': end.isoformat()}
    return result


class PlantStatistics:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def query(self, query: StatisticsQuery, now: datetime) -> dict:
        async with self.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute('SET TRANSACTION READ ONLY')
                await conn.execute("SET LOCAL statement_timeout = '5s'")
                return await collect_statistics(conn, query, now)

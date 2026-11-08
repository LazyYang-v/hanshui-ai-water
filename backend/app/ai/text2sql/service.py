"""Text2SQL：问题 → (schema 裁剪 + few-shot) → LLM 生成 SQL → 校验改写 → 只读执行 → 出错自修复。"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import sqlglot
import yaml
from psycopg.rows import tuple_row
from psycopg_pool import AsyncConnectionPool
from sqlglot import exp

from app.ai.gateway import LLMError, LLMGateway
from app.ai.text2sql.guard import MAX_ROWS, SQLRejected, guard
from app.ai.text2sql.schema import SchemaInfo
from app.ai.usage import call_usage, elapsed_ms

EXAMPLES_PATH = Path(__file__).with_name("examples.yaml")
# 问题明确要“最接近/前后N分钟/最近一条采样”时才允许按时间距离选点；“最近7天”不算。
NEAREST_ALLOWED = re.compile(r"最接近|就近|附近|前后\s*\d*\s*分钟|最近(?:的)?(?:一条|一次|一个)?(?:采样|读数|记录|数据点|时刻)")


class SQLConvention(ValueError):
    """SQL 安全但不符合查数约定，交给模型修复。"""


def nearest_sample_violation(sql: str, question: str) -> str | None:
    """拦截 ORDER BY ABS(ts - 目标时刻) 的选点写法：指定时刻缺数时它会静默返回相邻采样。

    只识别这一种已观察到的写法，不声称覆盖所有近似取点方式。
    """
    if NEAREST_ALLOWED.search(question):
        return None
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.ParseError:
        return None
    for order in tree.find_all(exp.Order):
        for func in order.find_all(exp.Abs):
            if any(c.name.lower() == "ts" for sub in func.find_all(exp.Sub) for c in sub.find_all(exp.Column)):
                return ("问题指定了时刻，原始采样须用 ts 等值或 IN 筛选，不能按时间距离选最近点；"
                        "没有该时刻采样就返回空结果")
    return None


@dataclass(frozen=True)
class Text2SQLConfig:
    """评测时用于消融对比的开关。"""
    use_fewshot: bool = True
    fewshot_k: int = 3
    prune_schema: bool = True
    max_repairs: int = 2
    max_tokens: int = 4000          # Flash 推理和 JSON 输出共用此预算，与评测保持一致
    # Agent 的工具上限是 60 秒：剩余时间不够再完整生成一次时停止修复，返回明确错误而不是被取消
    time_budget_s: float | None = 50.0
    min_attempt_s: float = 12.0


TRUNCATED = ("上次输出在 token 上限处被截断（推理过长），没有得到 JSON。"
             "请不要展开推理，直接输出 JSON；子问题项目多时，用一条按小时聚合的 SELECT 覆盖即可。")


# ---------------------------------------------------------------- few-shot 检索（字符 bigram BM25）

def _tokens(text: str) -> list[str]:
    s = "".join(ch.lower() for ch in text if not ch.isspace() and ch not in "，。？！、,.?!")
    return [s[i:i + 2] for i in range(len(s) - 1)] or [s]


class ExampleIndex:
    def __init__(self, examples: list[dict], k1: float = 1.5, b: float = 0.75):
        self.examples = examples
        self.docs = [Counter(_tokens(e["q"])) for e in examples]
        self.avgdl = sum(sum(d.values()) for d in self.docs) / max(1, len(self.docs))
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    @classmethod
    def load(cls, path: Path = EXAMPLES_PATH) -> ExampleIndex:
        return cls(yaml.safe_load(path.read_text()))

    def top(self, question: str, k: int) -> list[dict]:
        q = _tokens(question)
        scores = []
        for i, d in enumerate(self.docs):
            dl = sum(d.values())
            s = sum(self.idf.get(t, 0) * d[t] * (self.k1 + 1) / (d[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                    for t in q if t in d)
            scores.append((s, i))
        scores.sort(reverse=True)
        return [self.examples[i] for s, i in scores[:k] if s > 0]


# ---------------------------------------------------------------- 提示词

SYSTEM = """你是水务行业的数据分析师，负责把用户的问题转换成 PostgreSQL（TimescaleDB）查询。

## 规则
1. 只写一条 SELECT 语句（可以用 WITH 子查询），不要写任何修改数据的语句。
2. 当前时间一律用 now()，系统会替换为演示时钟（当前为 {now}）。会话时区为 Asia/Shanghai，
   「今天」= date_trunc('day', now())，「昨天」= 前一个自然日，「本周」= date_trunc('week', now())，
   「最近 N 天/小时」= now() - interval 'N days/hours'。
3. 按小时及以上粒度统计或跨天统计时，优先查 plant_reading_1h；查最新值、分钟级数据时查 plant_reading。
   原始采样即使比较不同日期，也必须查原始plant_reading/device_reading/weather。指定精确时刻时用原始ts等值/IN筛选，
   不擅自用前后若干分钟的最近点代替；没有该时刻采样就返回空结果。只有明确要求最近采样时才按时间距离选点。
   返回的采样时间必须来自实际读数表的ts，不能把查询目标时间常量当作采样时间。
4. 处理水量（m³）= sum(plant_reading_1h.inflow_m3h)；鼓风机耗电量（kWh）= sum(plant_reading_1h.blower_kw)；
   药剂投加量（kg）= sum(plant_reading_1h.carbon_dose_kgh 或 pac_dose_kgh)；吨水电耗 = 耗电量 / 处理水量。
5. 超标判断用Demo固定演示阈值：COD>50、氨氮>5、总氮>15、总磷>0.5、SS>10（mg/L），不能据此声称真实厂站排放合规。
   在 plant_reading_1h 中判断氨氮、总磷是否超标用 nh3n_out_max、tp_out_max。
6. 厂站按名称提问时用 plant.name LIKE '%关键字%' 匹配；结果中厂站显示 plant.name。
7. 列名可用中文别名，明细查询加LIMIT。原始读数保留数据库原值，不再round或聚合；统计展示值才保留合理小数。
   多个原始指标优先直接SELECT成宽表，每个指标独立一列，同一采样一行，保留实际ts和厂站code；涉及设备时另保留设备code。
   简单原始查数优先直接JOIN原始表，不要为转成“指标名/数值”长表而重复扫描、UNION或嵌套CTE。
8. 如果问题与数据库无关或数据库里没有相关数据，sql 返回空字符串，在 explanation 中说明原因。

## 数据库结构
{schema}
{examples}
## 输出
只输出 JSON：{{"sql": "SQL 语句", "explanation": "一句话说明查询思路"}}"""


def build_messages(question: str, schema: SchemaInfo, examples: list[dict], cfg: Text2SQLConfig,
                   now: datetime) -> list[dict]:
    tables = schema.select(question) if cfg.prune_schema else list(schema.tables)
    ex_text = ""
    if examples:
        ex_text = "\n## 示例\n" + "\n".join(f"问：{e['q']}\nSQL：\n{e['sql'].strip()}\n" for e in examples)
    system = SYSTEM.format(now=now.strftime("%Y-%m-%d %H:%M:%S%z"), schema=schema.render(tables), examples=ex_text)
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


# ---------------------------------------------------------------- 执行

def _cell(v):
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return v


async def execute_readonly(pool: AsyncConnectionPool, sql: str) -> tuple[list[str], list[list], bool]:
    async with pool.connection() as conn:
        async with conn.transaction():  # 连接池须为 autocommit，这里才是真正的事务而非 savepoint
            await conn.execute("SET TRANSACTION READ ONLY")
            await conn.execute("SET LOCAL TIME ZONE 'Asia/Shanghai'")
            await conn.execute("SET LOCAL statement_timeout = '5s'")
            # 与其他只读工具共用连接池时，结果格式不能依赖连接上次被谁使用。
            async with conn.cursor(row_factory=tuple_row) as cur:
                await cur.execute(sql)
                columns = [d.name for d in cur.description or []]
                rows = await cur.fetchmany(MAX_ROWS + 1)
    truncated = len(rows) > MAX_ROWS
    return columns, [[_cell(v) for v in r] for r in rows[:MAX_ROWS]], truncated


# ---------------------------------------------------------------- 主流程

@dataclass
class Attempt:
    sql: str | None
    error: str | None = None


@dataclass
class Text2SQLResult:
    question: str
    ok: bool
    refused: bool = False          # 仅模型明确返回空 SQL 才算拒答，不能把解析错误当拒答
    provider: str | None = None
    model: str | None = None
    sql: str | None = None
    explanation: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    truncated: bool = False
    error: str | None = None
    attempts: list[Attempt] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_ms: float = 0.0
    generation_calls: list[dict] = field(default_factory=list)
    total_ms: float = 0.0
    examples_used: list[str] = field(default_factory=list)
    tables_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Text2SQL:
    def __init__(self, gateway: LLMGateway, ro_pool: AsyncConnectionPool, schema: SchemaInfo,
                 examples: ExampleIndex | None = None, cfg: Text2SQLConfig = Text2SQLConfig()):
        self.gateway, self.pool, self.schema = gateway, ro_pool, schema
        self.examples = examples or ExampleIndex.load()
        self.cfg = cfg

    async def ask(self, question: str, now: datetime) -> Text2SQLResult:
        t0 = time.perf_counter()
        cfg = self.cfg
        examples = self.examples.top(question, cfg.fewshot_k) if cfg.use_fewshot else []
        messages = build_messages(question, self.schema, examples, cfg, now)
        res = Text2SQLResult(question, ok=False, examples_used=[e["q"] for e in examples],
                             tables_used=self.schema.select(question) if cfg.prune_schema else list(self.schema.tables))

        thinking: bool | None = None            # 厂商默认（Flash 开启思考）；被截断后的修复关闭思考
        for k in range(cfg.max_repairs + 1):
            if k and cfg.time_budget_s is not None and \
                    time.perf_counter() - t0 > cfg.time_budget_s - cfg.min_attempt_s:
                res.error = f"查数生成耗时过长，已停止修复（上次问题：{res.error}）"
                break
            try:
                r = await self.gateway.chat(messages, temperature=0.0, json_mode=True, max_tokens=cfg.max_tokens,
                                            thinking=thinking)
            except LLMError as e:
                res.llm_ms += elapsed_ms(e)
                res.generation_calls.append(call_usage(e))
                res.error = "大模型调用失败，请稍后重试"
                break
            res.provider, res.model = r.provider, r.model
            res.prompt_tokens += r.prompt_tokens
            res.completion_tokens += r.completion_tokens
            res.llm_ms += elapsed_ms(r)
            res.generation_calls.append(call_usage(r))

            truncated = r.finish_reason == "length"
            try:
                if truncated:
                    raise ValueError("truncated")
                out = json.loads(r.content)
                if (not isinstance(out, dict) or not isinstance(out.get("sql"), str)
                        or not isinstance(out.get("explanation"), str)):
                    raise ValueError("sql 和 explanation 必须为字符串")
                raw_sql = out["sql"].strip()
                res.explanation = out["explanation"]
            except (ValueError, TypeError):
                raw_sql, err = None, (TRUNCATED if truncated else "输出不是合法 JSON 对象（必须含字符串 sql 和 explanation）")
                if truncated:
                    thinking = False
            else:
                err = None
            if raw_sql == "" and err is None:  # 模型明确判断无法回答
                res.refused, res.sql = True, None
                res.attempts.append(Attempt(sql=None))
                res.error = res.explanation or "无法用现有数据回答"
                break

            if err is None:
                try:
                    safe_sql = guard(raw_sql, self.schema.names, now)
                    violation = nearest_sample_violation(safe_sql, question)
                    if violation:
                        raise SQLConvention(violation)
                    res.columns, res.rows, res.truncated = await execute_readonly(self.pool, safe_sql)
                    res.sql, res.ok, res.error = safe_sql, True, None
                    res.attempts.append(Attempt(sql=safe_sql))
                    break
                except SQLRejected as e:
                    err = f"SQL 未通过安全校验：{e}"
                except SQLConvention as e:
                    err = f"SQL 不符合查数约定：{e}"
                except Exception as e:  # noqa: BLE001 数据库报错交给模型修复
                    err = f"执行报错：{str(e).splitlines()[0]}"
            res.attempts.append(Attempt(sql=raw_sql, error=err))
            res.sql, res.error = raw_sql, err
            messages = messages + [
                {"role": "assistant", "content": r.content},
                {"role": "user", "content": f"{err}\n请修正后重新输出 JSON。"},
            ]

        res.total_ms = round((time.perf_counter() - t0) * 1000, 1)
        res.llm_ms = round(res.llm_ms, 1)
        return res

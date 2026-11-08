"""运行日报/周报：SQL 汇总事实 → Flash 写叙述（数字只能写占位符）→ 校验 → Markdown。

叙述中的每个数字都来自事实表：模型只能写 {{事实id}}，由服务端替换成带单位的值。
占位符以外出现阿拉伯数字（厂站、设备编码除外）或“三次”“两座”这类中文计数即判失败，
带着错误修复一次，仍失败则用模板叙述，保证报告里没有模型自己写的数字。
数字都是真的，跨厂站的关系却可能写错（如拿两个厂站的值当范围端点，漏掉了更低的第三个），
所以另给服务端算好的 plants.* 汇总事实，并校验范围、“均为”、最高/最低和“低于其他厂站”几种写法（check_relations）。
表格由服务端按同一批事实生成，不经过模型。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.ai.agent.service import error_detail
from app.ai.gateway import LLMError, LLMGateway
from app.ai.usage import call_usage, elapsed_ms
from app.deps import LOCAL_TZ
from app.domain.metrics import EFFLUENT_KEYS, PLANT_METRICS
from app.services import queries

log = logging.getLogger(__name__)

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.]+)\s*\}\}")
# 允许直接写的带数字名称：厂站与设备编码（P06、P06-BL-1）、工艺名 A2O
CODE = re.compile(r"P0\d(?:-[A-Z]{1,3}(?:-[A-Z0-9]{1,3})?)?|A2/?O")
ARABIC = re.compile(r"[0-9０-９]")
CN_COUNT = re.compile(r"[零〇一二两三四五六七八九十百千万]+\s*(?:个|次|座|台|起|项|条|张|家|处|小时|天|吨|立方|倍)|百分之")
IDIOM = re.compile(r"一起|一次性|同一|之一|逐一|一一")    # 含数字字但不表示数量的常用词，检查前去掉
MAX_EVENTS = 20
MAX_PARAGRAPH = 300
MAX_SAVED_OUTPUT = 8000            # 失败尝试保存的原始输出上限（字符）
LEVEL_ORDER = {"告警": 0, "预警": 1, "提示": 2}
LEVEL_SLUG = {"告警": "alarm", "预警": "warning", "提示": "info"}   # 事实 id 只用 ASCII，与 PLACEHOLDER 一致
SECTIONS = [("overview", "一、运行概况"), ("events", "二、异常事件与处置"), ("consumption", "三、能耗药耗"),
            ("attention", "四、关注事项")]
DO_LOW = 1.0

REPORT_SYSTEM = """你是汉水智控的运行报告撰写助手，根据用户JSON中的facts、events、work_orders写简短、客观的中文叙述。
1. 任何数字（数量、时间、日期、百分比、浓度、水量、电耗、药量、编号）都不能直接写，只能写占位符{{事实id}}，
   服务端会替换为带单位的数值，所以占位符后面不要再写单位；id必须原样出现在输入中。
   数量替换后是“7 座”“19 起”这样的数加量词，后面照常接名词，如“{{plants.n}}厂站”“{{alarms.total}}告警事件”。
   告警、工单编号也一样，写{{A44.id}}、{{W2.id}}，不要写“#44”。厂站编码（如P06）和设备编码（如P06-BL-1）可以直接写。
2. 占位符以外不得出现阿拉伯数字，也不要用中文数字表示数量或时长（如“三次”“两座”“五小时”）。
3. 只陈述输入中的内容：不推测原因，不给操作参数，不编造处置结果；未关闭的告警、工单如实写为未关闭。
   告警的ended表示数据已恢复，status表示人工处理状态，两者不要混淆。所有运行数据均为模拟数据。
4. 跨厂站的范围、最高最低和“均为”类概括只能用plants.*汇总事实，如“各厂站吨水曝气电耗在
   {{plants.kwh_per_m3.min}}至{{plants.kwh_per_m3.max}}之间”；不要拿单个厂站的值当范围端点或代表其他厂站。
5. events只写最重要的几条，同一厂站的连锁告警合并叙述；没有告警时events给空数组。
6. 只输出JSON，只含这四个字段：{"overview":["…"],"events":["…"],"consumption":["…"],"attention":["…"]}。
   overview 1至3段，events 0至4段，consumption 1至2段，attention 0至3段，每段不超过150字。"""


class NoData(ValueError):
    """统计时段内没有运行数据。"""


class Narrative(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    overview: list[str] = Field(min_length=1, max_length=3)
    events: list[str] = Field(default_factory=list, max_length=4)
    consumption: list[str] = Field(min_length=1, max_length=2)
    attention: list[str] = Field(default_factory=list, max_length=3)


@dataclass
class Fact:
    id: str
    label: str                 # 送给模型，不含阿拉伯数字（模型可能照抄标签）
    text: str                  # 替换进叙述的文本（含单位）
    value: float | int | str | None = None
    llm: bool = True           # False：只入库备查，不送给模型（如远低于限值的指标最高值）


def _num(v: float | None, digits: int = 1, unit: str = "") -> str:
    if v is None:
        return "无数据"
    return f"{v:,.{digits}f} {unit}".strip()


def _pct(v: float | None) -> str:
    return "无数据" if v is None else f"{v * 100:.2f}%"


def _t(ts: datetime | None) -> str:
    return "" if ts is None else ts.astimezone(LOCAL_TZ).strftime("%m-%d %H:%M")


def _cell(v) -> str:
    return "" if v is None else str(v).replace("|", "\\|").replace("\n", " ")


def _duration(minutes: float) -> str:
    return _num(minutes / 60, 1, "小时") if minutes >= 90 else f"{minutes:.0f} 分钟"


# 各厂站可横向比较的汇总指标：事实 id 后缀 → (period_stats 字段, 标签, 格式化)
PLANT_TOTALS = {
    "water_m3": ("water_m3", "处理水量", lambda v: _num(v, 0, "m³")),
    "compliance": ("compliance_rate", "出水达标率", _pct),
    "kwh_per_m3": ("aeration_kwh_per_m3", "吨水曝气电耗", lambda v: _num(v, 3, "kWh/m³")),
    "carbon_kg": ("carbon_kg", "碳源用量", lambda v: _num(v, 1, "kg")),
    "pac_kg": ("pac_kg", "PAC用量", lambda v: _num(v, 1, "kg")),
}


def period_for(kind: str, day: date, now: datetime) -> tuple[datetime, datetime]:
    """日报：当天 0 点至次日 0 点；周报：截至当天的 7 天。结束时间不超过演示时钟。"""
    end = datetime(day.year, day.month, day.day, tzinfo=LOCAL_TZ) + timedelta(days=1)
    start = end - timedelta(days=1 if kind == "日报" else 7)
    return start, min(end, now)


# ---------------------------------------------------------------- 事实

async def collect_facts(conn: AsyncConnection, kind: str, start: datetime, end: datetime, now: datetime) -> dict:
    window = {"start": start, "end": end}
    # 告警与工单的上界：截至此刻的报告要包含此刻。演示时钟按步推进，同一步内新建/关闭的工单
    # 与报告拿到同一个时间戳，按 [start, end) 会漏掉它们（时间戳精度为微秒，+1µs 即 ≤ end）
    hi = end + timedelta(microseconds=1) if end >= now else end
    records = {**window, "hi": hi}
    plants = await (await conn.execute("SELECT id, code, name FROM plant ORDER BY id")).fetchall()
    stats = await queries.period_stats(conn, start, end)
    if not stats:
        raise NoData("统计时段内没有运行数据")
    total = queries.summarize(stats)
    facts: list[Fact] = [
        Fact("period.start", "统计开始", start.strftime("%Y-%m-%d %H:%M"), start.isoformat()),
        Fact("period.end", "统计结束", end.strftime("%Y-%m-%d %H:%M"), end.isoformat()),
        Fact("period.hours", "统计时长", _num((end - start).total_seconds() / 3600, 1, "小时")),
        Fact("plants.n", "厂站数", f"{len(plants)} 座", len(plants)),
        Fact("total.water_m3", "全区处理水量", _num(total["water_m3"], 0, "m³"), total["water_m3"]),
        Fact("total.compliance", "全区出水达标率（按运行时长加权）", _pct(total["compliance_rate"]), total["compliance_rate"]),
        Fact("total.aeration_kwh", "全区曝气电耗", _num(total["aeration_kwh"], 0, "kWh"), total["aeration_kwh"]),
        Fact("total.kwh_per_m3", "全区吨水曝气电耗", _num(total["aeration_kwh_per_m3"], 3, "kWh/m³"),
             total["aeration_kwh_per_m3"]),
        Fact("total.carbon_kg", "全区碳源用量", _num(total["carbon_kg"], 1, "kg"), total["carbon_kg"]),
        Fact("total.pac_kg", "全区PAC用量", _num(total["pac_kg"], 1, "kg"), total["pac_kg"]),
    ]
    by_plant = {s["plant_id"]: s for s in stats}

    # 出水：5 分钟均值的最高值与超标时长；DO 偏低时长
    aggs = sql.SQL(", ").join(
        sql.SQL("max({k}) AS {mx}, count(*) FILTER (WHERE {k} > {lim}) AS {n}").format(
            k=sql.Identifier(k), mx=sql.Identifier(k + "_max"), n=sql.Identifier(k + "_n"),
            lim=sql.Literal(PLANT_METRICS[k].limit)) for k in EFFLUENT_KEYS)
    avgs = sql.SQL(", ").join(sql.SQL("avg({k}) AS {k}").format(k=sql.Identifier(k)) for k in EFFLUENT_KEYS)
    effluent = {r["plant_id"]: r for r in await (await conn.execute(sql.SQL("""
        WITH b AS (
            SELECT plant_id, time_bucket('5 minutes', ts) AS t, {avgs}, avg(do_aerobic_1) AS do1
            FROM plant_reading WHERE ts >= %(start)s AND ts < %(end)s GROUP BY 1, 2
        )
        SELECT plant_id, {aggs}, min(do1) AS do_min, count(*) FILTER (WHERE do1 < {do_low}) AS do_low_n
        FROM b GROUP BY 1""").format(avgs=avgs, aggs=aggs, do_low=sql.Literal(DO_LOW)), window)).fetchall()}

    plant_rows = []
    for p in plants:
        c, s, e = p["code"], by_plant.get(p["id"]), effluent.get(p["id"], {})
        row = {"code": c, "name": p["name"]}
        if s:
            facts += [Fact(f"{c}.{key}", f"{p['name']}{label}", show(s[col]), s[col])
                      for key, (col, label, show) in PLANT_TOTALS.items()]
            row.update(water=s["water_m3"], compliance=s["compliance_rate"], kwh=s["aeration_kwh_per_m3"],
                       carbon=s["carbon_kg"], pac=s["pac_kg"])
        exceed = []
        for key in EFFLUENT_KEYS:
            m = PLANT_METRICS[key]
            mx, n = e.get(key + "_max"), e.get(key + "_n") or 0
            near = mx is not None and mx > m.limit * m.warn_ratio
            facts.append(Fact(f"{c}.{key}.max", f"{p['name']}{m.label}最高值（时段均值）", _num(mx, 2, m.unit), mx,
                              llm=near))
            if n:
                hours = n * 5 / 60
                facts.append(Fact(f"{c}.{key}.exceed_h", f"{p['name']}{m.label}超标时长", _num(hours, 1, "小时"), hours))
                exceed.append(f"{m.label}超标 {hours:.1f} 小时（最高 {mx:.2f} {m.unit}）")
        if e.get("do_low_n"):
            hours = e["do_low_n"] * 5 / 60
            facts += [Fact(f"{c}.do_low_h", f"{p['name']}好氧区DO低于下限的时长", _num(hours, 1, "小时"), hours),
                      Fact(f"{c}.do_min", f"{p['name']}好氧区DO最低值（时段均值）", _num(e["do_min"], 2, "mg/L"),
                           e["do_min"])]
            exceed.append(f"DO 低于 {DO_LOW:g} mg/L {hours:.1f} 小时")
        row["exceed"] = "；".join(exceed) or "无"
        plant_rows.append(row)

    # 跨厂站的最低/最高值（标签里写出是哪些厂站），供模型写范围与最值
    for key, (col, label, show) in PLANT_TOTALS.items():
        vals = [(p["name"], by_plant[p["id"]][col]) for p in plants
                if p["id"] in by_plant and by_plant[p["id"]][col] is not None]
        if len(vals) < 2:
            continue
        for side, pick, word in (("min", min, "最低"), ("max", max, "最高")):
            v = pick(x for _, x in vals)
            names = "、".join(n for n, x in vals if show(x) == show(v))
            facts.append(Fact(f"plants.{key}.{side}", f"各厂站{label}{word}值（{names}）", show(v), v))
    full = sum(1 for s in stats if _pct(s["compliance_rate"]) == _pct(1.0))
    facts.append(Fact("plants.compliance.full_n", "出水全时段达标（达标率为满分）的厂站数", f"{full} 座", full))

    rain = await (await conn.execute("""
        WITH b AS (
            SELECT time_bucket('5 minutes', ts) AS t, avg(rain_mm_h) AS rain, max(rain_mm_h) AS peak
            FROM weather WHERE ts >= %(start)s AND ts < %(end)s GROUP BY 1
        )
        SELECT max(peak) AS peak, count(*) FILTER (WHERE rain > 0.5) AS n FROM b""", window)).fetchone()
    if rain and rain["peak"] is not None:
        wet = rain["peak"] > 0.5   # 没下雨时不送给模型，免得写出“最大降雨强度 0.0 mm/h”
        facts += [Fact("rain.peak", "最大降雨强度", _num(rain["peak"], 1, "mm/h"), rain["peak"], llm=wet),
                  Fact("rain.hours", "降雨时长（雨强超过半毫米每小时）", _num(rain["n"] * 5 / 60, 1, "小时"),
                       rain["n"] * 5 / 60, llm=wet)]

    # 告警：与时段相交的事件（含时段前开始、仍在持续的）
    alarms = await (await conn.execute("""
        SELECT a.id, a.ts, a.ended_at, a.level, a.kind, a.message, a.status, a.peak, a.metric,
               p.code AS plant_code, p.name AS plant_name, d.code AS device_code
        FROM alarm a JOIN plant p ON p.id = a.plant_id LEFT JOIN device d ON d.id = a.device_id
        WHERE a.ts < %(hi)s AND (a.ended_at IS NULL OR a.ended_at >= %(start)s)""", records)).fetchall()
    alarms.sort(key=lambda a: (LEVEL_ORDER.get(a["level"], 9), a["ts"]))
    counts: dict[str, int] = {}
    for a in alarms:
        counts[a["level"]] = counts.get(a["level"], 0) + 1
    unclosed = sum(1 for a in alarms if a["status"] != "已关闭")
    facts.append(Fact("alarms.total", "时段内告警事件数（含预警）", f"{len(alarms)} 起", len(alarms)))
    facts += [Fact(f"alarms.level_{slug}", f"{lv}级事件数", f"{counts.get(lv, 0)} 起", counts.get(lv, 0))
              for lv, slug in LEVEL_SLUG.items()]
    facts.append(Fact("alarms.unclosed", "处理状态未关闭的告警（截至生成时）", f"{unclosed} 起", unclosed))
    events = []
    for a in alarms[:MAX_EVENTS]:
        aid = f"A{a['id']}"
        dur = ((a["ended_at"] or end) - a["ts"]).total_seconds() / 60
        facts += [Fact(f"{aid}.id", "告警编号", f"#{a['id']}", a["id"]),
                  Fact(f"{aid}.start", "开始时间", _t(a["ts"]), a["ts"].isoformat()),
                  Fact(f"{aid}.duration", "持续时长（未恢复的计至统计结束）", _duration(dur), round(dur, 1)),
                  Fact(f"{aid}.message", "触发时的告警说明", a["message"], a["message"])]
        if a["ended_at"]:
            facts.append(Fact(f"{aid}.end", "数据恢复时间", _t(a["ended_at"]), a["ended_at"].isoformat()))
        events.append({"id": aid, "plant": f"{a['plant_code']} {a['plant_name']}", "device": a["device_code"],
                       "level": a["level"], "kind": a["kind"], "status": a["status"], "ended": a["ended_at"] is not None,
                       "facts": [f"{aid}.{k}" for k in ("id", "start", "end", "duration", "message")
                                 if a["ended_at"] or k != "end"]})

    # 工单：时段内新建、关闭，以及统计结束时仍未关闭的
    orders = await (await conn.execute("""
        SELECT w.id, w.alarm_id, w.title, w.assignee, w.status, w.created_at, w.started_at, w.closed_at,
               p.code AS plant_code
        FROM work_order w LEFT JOIN plant p ON p.id = w.plant_id
        WHERE w.created_at < %(hi)s AND (w.closed_at IS NULL OR w.closed_at >= %(start)s) ORDER BY w.id""",
        records)).fetchall()

    def before_end(t: datetime | None) -> bool:
        return t is not None and t < hi

    created = sum(1 for o in orders if o["created_at"] >= start)
    closed = sum(1 for o in orders if before_end(o["closed_at"]))
    still_open = len(orders) - closed
    facts += [Fact("orders.created", "时段内新建工单", f"{created} 张", created),
              Fact("orders.closed", "时段内关闭工单", f"{closed} 张", closed),
              Fact("orders.open", "统计结束时未关闭的工单", f"{still_open} 张", still_open)]
    order_items = []
    for o in orders:
        wid = f"W{o['id']}"
        ids = [f"{wid}.id", f"{wid}.title", f"{wid}.assignee"]
        facts += [Fact(f"{wid}.id", "工单编号", f"#{o['id']}", o["id"]),
                  Fact(f"{wid}.title", "工单标题", o["title"], o["title"]),       # 用户可编辑，可能含数字
                  Fact(f"{wid}.assignee", "负责人", o["assignee"] or "未指派", o["assignee"])]
        if o["alarm_id"]:
            facts.append(Fact(f"{wid}.alarm", "关联告警编号", f"#{o['alarm_id']}", o["alarm_id"]))
            ids.append(f"{wid}.alarm")
        # 统计结束时的状态（不是当前状态）：之后才开始/关闭的工单在本时段内按当时的状态算
        status = "已关闭" if before_end(o["closed_at"]) else "处理中" if before_end(o["started_at"]) else "待处理"
        order_items.append({"id": wid, "plant": o["plant_code"], "facts": ids, "status": status})

    devices = []
    if end >= now - timedelta(minutes=10):   # 台账状态是当前值，只有截至现在的报告才写
        devices = await (await conn.execute("""
            SELECT d.code, d.kind, d.status, p.code AS plant_code FROM device d JOIN plant p ON p.id = d.plant_id
            WHERE d.status IN ('故障', '检修') ORDER BY d.code""")).fetchall()

    return {"kind": kind, "start": start, "end": end, "facts": {f.id: f for f in facts}, "plants": plant_rows,
            "alarms": alarms, "events": events, "orders": orders, "order_items": order_items, "devices": devices}


# ---------------------------------------------------------------- 叙述与校验

PLANT_FACT = r"(?:P0\d|plants)\.[A-Za-z0-9_.]+"
RANGE = re.compile(r"\{\{\s*(" + PLANT_FACT + r")\s*\}\}\s*(?:至|到|~|～|-|–|—|－)\s*\{\{\s*(" + PLANT_FACT + r")\s*\}\}")
ALL_SAME = re.compile(r"(?:均|都|皆|全部|全)(?:为|是|在|达到|达|保持|维持)\s*\{\{\s*(" + PLANT_FACT + r")\s*\}\}")
SUPERLATIVE = re.compile(r"(最高|最低|最大|最小|最多|最少)[^。；！？{}]{0,12}\{\{\s*(P0\d)\.([A-Za-z0-9_]+)\s*\}\}")
COMPARE = re.compile(r"\{\{\s*(P0\d)\.([A-Za-z0-9_]+)\s*\}\}[^。；！？{}]{0,6}(低于|高于|少于|多于|小于|大于)(?:其他|其余|其它|各)")
EXCEPT = re.compile(r"除了?([^。；，{}]{1,30}?)(?:以外|之外|外)")
LOWER_WORDS = {"最低", "最小", "最少", "低于", "少于", "小于"}


def _unit(text: str) -> str:
    """事实文本末尾的单位（“6 座”→“座”，“89.24%”→“%”）；末段含数字的（时间、编号）不算。"""
    unit = text.rsplit(" ", 1)[-1] if " " in text else ("%" if text.endswith("%") else "")
    return "" if ARABIC.search(unit) else unit


def _metric_of(fid: str) -> str | None:
    """P06.kwh_per_m3 → kwh_per_m3；plants.kwh_per_m3.min → kwh_per_m3；其他汇总事实 → None。"""
    head, _, rest = fid.partition(".")
    if head != "plants":
        return rest
    metric, _, side = rest.rpartition(".")
    return metric if side in ("min", "max") else None


def _by_plant(facts: dict[str, Fact], metric: str) -> dict[str, Fact]:
    """各厂站同一指标的事实：厂站编码 → 事实（含不送给模型的）。"""
    pat = re.compile(r"P0\d\." + re.escape(metric))
    return {k.split(".", 1)[0]: f for k, f in facts.items() if pat.fullmatch(k) and isinstance(f.value, (int, float))}


def _alias(name: str) -> str:
    return re.sub(r"(污水处理厂|污水处理站|污水厂|处理厂|处理站)$", "", name)


def check_relations(para: str, facts: dict[str, Fact], names: dict[str, str]) -> list[str]:
    """跨厂站关系：范围的两端须是各厂站的最低和最高值；“均为某值”须对所有厂站成立，或把不同的厂站写进同一句的
    “除……外”；“最高/最低”后紧跟的厂站须确实是最值；“低于/高于其他厂站”须对其余每座厂站成立。
    按显示文本比较（四舍五入后相同即视为相同）。只能拦住这几种句式，其余写法靠提示词约束。"""
    errors = []

    def label(metric: str) -> str:
        return PLANT_TOTALS[metric][1] if metric in PLANT_TOTALS else metric

    def who(codes) -> str:
        return "、".join(names.get(c, c) for c in codes)

    for m in RANGE.finditer(para):
        a, b = m.groups()
        metric = _metric_of(a)
        group = _by_plant(facts, metric) if metric and metric == _metric_of(b) else {}
        if len(group) < 2 or a not in facts or b not in facts:
            continue
        lo = min(group.values(), key=lambda f: f.value).text
        hi = max(group.values(), key=lambda f: f.value).text
        if {facts[a].text, facts[b].text} != {lo, hi}:
            hint = (f"请写{{{{plants.{metric}.min}}}}至{{{{plants.{metric}.max}}}}" if f"plants.{metric}.min" in facts
                    else "请逐个写出")
            errors.append(f"范围“{m.group(0)}”的两端不是各厂站{label(metric)}的最低值和最高值，{hint}")

    for m in ALL_SAME.finditer(para):
        fid = m.group(1)
        metric = _metric_of(fid)
        group = _by_plant(facts, metric) if metric and fid in facts else {}
        differ = [c for c, f in group.items() if f.text != facts[fid].text] if len(group) >= 2 else []
        if not differ:
            continue
        head = para[max(para.rfind("。", 0, m.start()), para.rfind("；", 0, m.start())) + 1: m.start()]
        excluded = "".join(EXCEPT.findall(head))
        missing = [c for c in differ if not any(x and x in excluded for x in (c, names.get(c), _alias(names.get(c, ""))))]
        if missing:
            hint = "如{{plants.compliance.full_n}}" if metric == "compliance" else "如plants.*的最低值、最高值"
            errors.append(f"“{m.group(0)}”与事实不符：{who(missing)}的{label(metric)}不同。概括多个厂站请用汇总事实"
                          f"（{hint}），或逐个写出")

    for m in SUPERLATIVE.finditer(para):
        word, code, metric = m.groups()
        group = _by_plant(facts, metric) if metric in PLANT_TOTALS else {}
        if len(group) < 2 or code not in group:
            continue
        side = "min" if word in LOWER_WORDS else "max"
        best = (min if side == "min" else max)(group.values(), key=lambda f: f.value).text
        if group[code].text != best:
            errors.append(f"“{m.group(0)}”：各厂站{label(metric)}{word}的是{who(c for c, f in group.items() if f.text == best)}，"
                          f"不是{names.get(code, code)}；最值请用{{{{plants.{metric}.{side}}}}}")

    for m in COMPARE.finditer(para):
        code, metric, word = m.groups()
        group = _by_plant(facts, metric) if metric in PLANT_TOTALS else {}
        if len(group) < 2 or code not in group:
            continue
        me, lower = group[code], word in LOWER_WORDS
        bad = [c for c, f in group.items() if c != code and (f.text == me.text or (f.value < me.value) == lower)]
        if bad:
            errors.append(f"“{m.group(0)}”与事实不符：{who(bad)}的{label(metric)}并不{'更高' if lower else '更低'}，"
                          "请删去这一比较或改用plants.*的最低值、最高值")
    return errors


def check_narrative(n: Narrative, facts: dict[str, Fact], names: dict[str, str] | None = None) -> list[str]:
    """names：厂站编码 → 名称，用于识别“除……外”里写的厂站。"""
    errors = []
    for key, _ in SECTIONS:
        for i, para in enumerate(getattr(n, key)):
            if not para:
                errors.append(f"{key}[{i}] 是空段落")
            unknown = [m for m in PLACEHOLDER.findall(para) if m not in facts]
            if unknown:
                errors.append(f"{key}[{i}] 使用了不存在的事实id：{', '.join(unknown[:5])}")
            bare = PLACEHOLDER.sub("", para)
            if "{" in bare or "}" in bare:
                errors.append(f"{key}[{i}] 有格式不对的占位符（只能是{{{{事实id}}}}，id 原样取自输入）")
            rest = CODE.sub("", bare)
            if (m := ARABIC.search(rest)) is not None:
                errors.append(f"{key}[{i}] 占位符以外出现数字：“{rest[max(0, m.start() - 8): m.end() + 8]}”")
            if (m := CN_COUNT.search(IDIOM.sub("", rest))) is not None:
                errors.append(f"{key}[{i}] 用中文数字写了数量：“{m.group(0)}”，请改用占位符")
            if len(para) > MAX_PARAGRAPH:
                errors.append(f"{key}[{i}] 超过{MAX_PARAGRAPH}字")
            for m in PLACEHOLDER.finditer(para):
                unit = _unit(facts[m.group(1)].text) if m.group(1) in facts else ""
                if unit and para[m.end():].lstrip().startswith(unit):
                    errors.append(f"{key}[{i}] {{{{{m.group(1)}}}}}后面又写了单位“{unit}”，替换后的值已带单位")
            errors += [f"{key}[{i}] {e}" for e in check_relations(para, facts, names or {})]
    return errors


# 模型偶尔照抄编号文本（“关联工单#2”）而不写占位符：编号与输入事实完全一致时由服务端换成占位符
LITERAL_ID = re.compile(r"(?<![0-9A-Za-z#])#\s*(\d{1,7})(?![\d.])")
ID_KIND = {"工单": "W", "告警": "A"}


def _id_placeholder(prefix: str, number: str, facts: dict[str, Fact]) -> str | None:
    """本句前文最近的“工单/告警”决定编号种类；没写种类时，编号文本须在输入里只对应一种含义。"""
    text = f"#{number}"
    cands = [k for k, f in facts.items() if f.text == text and k.endswith((".id", ".alarm"))]
    meaning = {k: ("W" if k.startswith("W") and k.endswith(".id") else "A") for k in cands}
    hits = [w for w in ID_KIND if w in prefix]
    if hits:
        want = ID_KIND[max(hits, key=prefix.rfind)]
        cands = sorted((k for k in cands if meaning[k] == want), key=lambda k: not k.endswith(".id"))
        return cands[0] if cands else None
    return cands[0] if len(set(meaning.values())) == 1 else None


def _swap(m: re.Match, para: str, facts: dict[str, Fact], notes: list[str], key: str) -> str:
    prefix = re.split(r"[，。；、,;]", para[: m.start()])[-1]
    fid = _id_placeholder(prefix, m.group(1), facts)
    if fid is None:
        return m.group(0)
    notes.append(f"{key}：“{m.group(0)}”改为{{{{{fid}}}}}")
    return "{{" + fid + "}}"


def parse_narrative(content: str, facts: dict[str, Fact]) -> tuple[Narrative, list[str]]:
    """解析模型输出。容错只做两件不改变含义的事：丢弃多余的顶层字段、把与输入事实完全一致的编号换成占位符；
    所做替换记入 attempts 备查。其余问题仍交给 Narrative 与 check_narrative 校验，失败则重试。"""
    raw = json.loads(content)
    notes: list[str] = []
    if isinstance(raw, dict):
        extra = [k for k in raw if k not in Narrative.model_fields]
        if extra:
            notes.append(f"忽略多余字段：{', '.join(map(str, extra[:5]))}")
            raw = {k: v for k, v in raw.items() if k in Narrative.model_fields}
        for key, value in raw.items():
            if isinstance(value, list):
                raw[key] = [_drop_units(LITERAL_ID.sub(lambda m, p=p, key=key: _swap(m, p, facts, notes, key), p),
                                        facts, notes, key) if isinstance(p, str) else p for p in value]
    return Narrative.model_validate(raw), notes


def _drop_units(para: str, facts: dict[str, Fact], notes: list[str], key: str) -> str:
    """占位符后紧跟的单位与该事实自带的单位完全相同时删去（“{{plants.compliance.full_n}}座” → 替换后不会变成“5 座座”）。"""
    def repl(m: re.Match) -> str:
        holder, fid, rest = m.groups()
        unit = _unit(facts[fid].text) if fid in facts else ""
        if unit and rest.lstrip().startswith(unit):
            notes.append(f"{key}：删去{{{{{fid}}}}}后重复的单位“{unit}”")
            return holder + rest.lstrip()[len(unit):]
        return m.group(0)
    return re.sub(r"(\{\{\s*([A-Za-z0-9_.]+)\s*\}\})((?:(?!\{\{).)*)", repl, para)


def fill(text: str, facts: dict[str, Fact]) -> str:
    """替换占位符。只检查叙述自身的文字：事实值（如用户填写的工单标题）里有花括号照原样输出。"""
    bare = PLACEHOLDER.sub("", text)
    if "{" in bare or "}" in bare:
        raise ValueError(f"叙述里有未识别的占位符：{text[:80]}")
    return PLACEHOLDER.sub(lambda m: facts[m.group(1)].text, text)


def template_narrative(data: dict) -> Narrative:
    f = data["facts"]
    overview = ["统计时段 {{period.start}} 至 {{period.end}}，{{plants.n}}厂站合计处理水量 {{total.water_m3}}，"
                "出水达标率 {{total.compliance}}。"]
    if "rain.peak" in f and (f["rain.peak"].value or 0) > 0.5:
        overview.append("时段内最大降雨强度 {{rain.peak}}，降雨时长约 {{rain.hours}}。")
    overview.append("时段内告警事件 {{alarms.total}}，其中告警级 {{alarms.level_alarm}}、预警级 {{alarms.level_warning}}。"
                    if f["alarms.total"].value else "时段内没有告警事件。")
    events = []
    for ev in data["events"][:4]:
        state = "数据已恢复" if ev["ended"] else "数据仍未恢复"
        device = f"{ev['device']} " if ev["device"] else ""
        events.append(f"告警{{{{{ev['id']}.id}}}}：{ev['plant']} {device}{ev['kind']}（{ev['level']}），"
                      f"开始于 {{{{{ev['id']}.start}}}}，{{{{{ev['id']}.message}}}}；{state}，处理状态为{ev['status']}。")
    consumption = ["全区曝气电耗 {{total.aeration_kwh}}，吨水曝气电耗 {{total.kwh_per_m3}}；"
                   "碳源用量 {{total.carbon_kg}}，PAC 用量 {{total.pac_kg}}。"]
    if "plants.kwh_per_m3.min" in f:
        consumption.append("各厂站吨水曝气电耗在 {{plants.kwh_per_m3.min}} 至 {{plants.kwh_per_m3.max}} 之间。")
    attention = []
    if f["alarms.unclosed"].value or f["orders.open"].value:
        attention.append("处理状态未关闭的告警 {{alarms.unclosed}}，统计结束时未关闭的工单 {{orders.open}}。")
    if data["devices"]:
        attention.append("当前台账为故障或检修的设备：" + "、".join(f"{d['code']}（{d['status']}）" for d in data["devices"])
                         + "。")
    return Narrative(overview=overview, events=events, consumption=consumption, attention=attention)


def llm_input(data: dict) -> dict:
    facts = data["facts"]

    def expand(ids: list[str]) -> list[dict]:
        return [{"id": i, "label": facts[i].label, "text": facts[i].text} for i in ids]

    return {"report": data["kind"],
            "facts": [{"id": f.id, "label": f.label, "text": f.text} for f in facts.values()
                      if f.llm and not f.id.startswith(("A", "W"))],
            "events": [{**ev, "facts": expand(ev["facts"])} for ev in data["events"]],
            "work_orders": [{**o, "facts": expand(o["facts"])} for o in data["order_items"]],
            "devices_in_fault": [dict(d) for d in data["devices"]]}


# ---------------------------------------------------------------- Markdown

def render(data: dict, narrative: Narrative, meta: dict) -> str:
    f = data["facts"]
    last_day = data["end"] - timedelta(seconds=1)
    title = f"汉水智控运行{data['kind']}（{data['start']:%Y-%m-%d}" + (
        "）" if data["kind"] == "日报" else f" 至 {last_day:%Y-%m-%d}）")
    source = "模型撰写，数字由服务端按统计结果填入" if meta["narrative"] == "llm" else "模板生成"
    out = [f"# {title}", "",
           f"> 统计时段：{f['period.start'].text} 至 {f['period.end'].text}（演示时钟，北京时间）。"
           f"运行数据由模拟器生成，仅用于演示。叙述：{source}；表格由服务端按同一批统计结果生成。",
           ">",
           "> 口径：达标率要求每个 5 分钟桶内的原始采样点出水五项全部达标，按有效桶时长加权；"
           "超标时长与最高值按 5 分钟均值统计。两者不能互相换算（短时尖峰会拉低达标率，但不一定使均值超标）；"
           "缺失数据不计为达标。", ""]
    tables = {
        "overview": ["| 厂站 | 处理水量 (m³) | 出水达标率 | 吨水曝气电耗 (kWh/m³) | 超标情况（5 分钟均值） |",
                     "|---|---:|---:|---:|---|"] + [
            f"| {r['code']} {r['name']} | {_num(r.get('water'), 0)} | {_pct(r.get('compliance'))} | "
            f"{_num(r.get('kwh'), 3)} | {r['exceed']} |" for r in data["plants"]],
        "events": ["| 告警 | 厂站/设备 | 类型 | 级别 | 开始 | 数据恢复 | 处理状态 | 说明 |",
                   "|---|---|---|---|---|---|---|---|"] + [
            f"| #{a['id']} | {a['plant_code']}{' ' + a['device_code'] if a['device_code'] else ''} | {a['kind']} | "
            f"{a['level']} | {_t(a['ts'])} | {_t(a['ended_at']) or '未恢复'} | {a['status']} | {_cell(a['message'])} |"
            for a in data["alarms"][:MAX_EVENTS]] if data["alarms"] else [],
        "consumption": ["| 厂站 | 碳源 (kg) | PAC (kg) |", "|---|---:|---:|"] + [
            f"| {r['code']} {r['name']} | {_num(r.get('carbon'), 1)} | {_num(r.get('pac'), 1)} |" for r in data["plants"]],
        "attention": ["| 工单 | 厂站 | 当前状态 | 负责人 | 标题 |", "|---|---|---|---|---|"] + [
            f"| #{o['id']} | {o['plant_code'] or ''} | {o['status']} | {_cell(o['assignee'])} | {_cell(o['title'])} |"
            for o in data["orders"]] if data["orders"] else [],
    }
    for key, heading in SECTIONS:
        out += [f"## {heading}", ""]
        out += [fill(p, f) + "\n" for p in getattr(narrative, key)]
        if tables[key]:
            out += tables[key] + [""]
        if key == "events" and len(data["alarms"]) > MAX_EVENTS:
            out.append(f"> 表中只列出最重要的 {MAX_EVENTS} 起，共 {len(data['alarms'])} 起。\n")
    usage = (f"（{meta['model']}，{meta['prompt_tokens']}+{meta['completion_tokens']} tokens）"
             if meta.get("model") else "")
    out += ["---", f"生成时间 {meta['generated_at'][:16].replace('T', ' ')}（演示时钟）；叙述来源 {meta['narrative']}{usage}。"]
    return "\n".join(out)


class Reporter:
    def __init__(self, gateway: LLMGateway | None, max_tokens: int = 3000):
        self.gateway, self.max_tokens = gateway, max_tokens

    async def narrate(self, data: dict) -> tuple[Narrative, dict]:
        meta: dict = {"narrative": "template", "attempts": [], "model": None, "prompt_tokens": 0,
                      "completion_tokens": 0, "llm_ms": 0.0, "generation_calls": []}
        if self.gateway is None or not getattr(self.gateway, "available", True):
            meta["attempts"].append({"errors": ["模型未配置"]})
            return template_narrative(data), meta
        messages = [{"role": "system", "content": REPORT_SYSTEM},
                    {"role": "user", "content": json.dumps(llm_input(data), ensure_ascii=False, default=str)}]
        for _ in range(2):
            try:
                out = await self.gateway.chat(messages, json_mode=True, temperature=0.0, max_tokens=self.max_tokens,
                                              thinking=False)
            except LLMError as exc:
                meta["llm_ms"] = round(meta["llm_ms"] + elapsed_ms(exc), 1)
                meta["generation_calls"].append(call_usage(exc))
                meta["attempts"].append({"errors": ["模型调用失败，请稍后重试"]})
                break
            meta["model"] = out.model
            meta["prompt_tokens"] += out.prompt_tokens
            meta["completion_tokens"] += out.completion_tokens
            meta["llm_ms"] = round(meta["llm_ms"] + elapsed_ms(out), 1)
            meta["generation_calls"].append(call_usage(out))
            try:
                if out.finish_reason == "length":
                    raise ValueError("输出达到token上限，请缩短每段文字")
                narrative, fixes = parse_narrative(out.content, data["facts"])
                errors = check_narrative(narrative, data["facts"], {r["code"]: r["name"] for r in data["plants"]})
            except (ValidationError, ValueError) as exc:
                errors, fixes = [error_detail(exc)], []
            attempt = {"errors": errors, **({"fixes": fixes} if fixes else {})}
            if errors:
                # 保存失败那次的原始输出（截断），之后可用 evals/report/replay.py 按当前校验规则重放
                attempt.update(output=out.content[:MAX_SAVED_OUTPUT], finish_reason=out.finish_reason)
            meta["attempts"].append(attempt)
            if not errors:
                meta["narrative"] = "llm"
                return narrative, meta
            messages += [{"role": "assistant", "content": out.content},
                         {"role": "user", "content": "校验失败：" + "；".join(errors[:6]) +
                          "。所有数字都必须写成输入中存在的{{id}}占位符，请重新输出完整JSON。"}]
        return template_narrative(data), meta

    async def generate(self, pool: AsyncConnectionPool, kind: str, start: datetime, end: datetime,
                       now: datetime) -> dict:
        """取数和写库各用一次连接，调用模型期间不占用连接。"""
        t0 = time.perf_counter()
        async with pool.connection() as conn:
            conn.row_factory = dict_row
            data = await collect_facts(conn, kind, start, end, now)
        narrative, meta = await self.narrate(data)
        meta.update(generated_at=now.isoformat(), total_ms=round((time.perf_counter() - t0) * 1000, 1),
                    narrative_json=narrative.model_dump())
        content = render(data, narrative, meta)
        facts = {k: asdict(v) for k, v in data["facts"].items()}

        def dumps(o) -> str:
            return json.dumps(o, ensure_ascii=False, default=str)

        async with pool.connection() as conn:
            conn.row_factory = dict_row
            row = await (await conn.execute("""
                INSERT INTO report (kind, period_start, period_end, content_md, created_at, facts, meta)
                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (kind, start, end, content, now, Jsonb(facts, dumps), Jsonb(meta, dumps)))).fetchone()
        return {"id": row["id"], "kind": kind, "period_start": start, "period_end": end, "created_at": now,
                "content_md": content, "meta": meta}

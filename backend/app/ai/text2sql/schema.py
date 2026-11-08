"""从数据库注释生成给大模型看的 schema 说明，并按问题裁剪相关表。"""

from __future__ import annotations

from dataclasses import dataclass, field

from psycopg import AsyncConnection
from psycopg.rows import dict_row

# Text2SQL 可见的表。alarm / work_order / report 在 M4 才有数据，先不暴露，免得模型查空表
VISIBLE_TABLES = ["plant", "plant_reading", "plant_reading_1h", "device", "device_reading", "weather"]

# 裁剪用的关键词（命中任一即纳入）；plant 总是纳入
TABLE_KEYWORDS: dict[str, list[str]] = {
    "plant_reading": ["水质", "进水", "出水", "氨氮", "COD", "cod", "总氮", "总磷", "TN", "TP", "SS", "悬浮物",
                      "DO", "溶解氧", "曝气", "风量", "电耗", "能耗", "耗电", "电量", "药耗", "碳源", "PAC", "除磷剂",
                      "加药", "投加", "水量", "处理量", "流量", "MLSS", "污泥", "超标", "达标", "最新", "当前", "实时",
                      "现在"],
    "plant_reading_1h": ["小时", "每天", "每日", "日均", "平均", "趋势", "最近", "本周", "上周", "过去", "天",
                         "统计", "合计", "总共", "累计", "排名", "最高", "最低", "超标", "达标", "电耗", "能耗",
                         "耗电", "药耗", "处理量", "水量"],
    "device": ["设备", "鼓风机", "风机", "泵", "台账", "型号", "功率", "故障", "备用", "检修", "安装"],
    # 「运行了多少分钟」这类问题要看 running 字段（评测 D06 曾因漏召回答错）
    "device_reading": ["振动", "温度", "电流", "轴承", "运行", "运转", "开机", "停机", "跳停", "故障", "时长"],
    "weather": ["雨", "降雨", "天气", "气温", "气象", "暴雨"],
}


@dataclass
class Column:
    name: str
    type: str
    comment: str | None


@dataclass
class Table:
    name: str
    kind: str          # table / view
    comment: str | None
    columns: list[Column] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"-- {self.comment}" if self.comment else "", f"{self.name} ("]
        for c in self.columns:
            lines.append(f"  {c.name} {c.type}" + (f"  -- {c.comment}" if c.comment else ""))
        lines.append(")")
        return "\n".join(x for x in lines if x)


@dataclass
class SchemaInfo:
    tables: dict[str, Table]
    plants: list[dict]

    @property
    def names(self) -> set[str]:
        return set(self.tables)

    def select(self, question: str | None) -> list[str]:
        """按关键词裁剪；question 为 None 或没命中任何表时返回全部。"""
        if question is None:
            return list(self.tables)
        q = question.lower()
        hit = {t for t, kws in TABLE_KEYWORDS.items() if any(k.lower() in q for k in kws)}
        if "device_reading" in hit:
            hit.add("device")
        if not hit:
            return list(self.tables)
        return [t for t in self.tables if t == "plant" or t in hit]

    def render(self, tables: list[str]) -> str:
        parts = [self.tables[t].render() for t in tables]
        plant_lines = "\n".join(
            f"  {p['id']} | {p['code']} | {p['name']} | {p['kind']} | {p['process']} | {p['capacity_m3d']}"
            for p in self.plants)
        parts.append("-- 厂站清单（id | code | name | kind | process | capacity_m3d）\n" + plant_lines)
        if "device" in tables:
            parts.append("-- 设备编码：{厂站编码}-{类型}-{序号}。BL=鼓风机，LP=提升泵，RP=回流泵，SP=污泥泵，"
                         "DP-C=碳源加药泵，DP-P=PAC 加药泵。例：P01-BL-1")
        return "\n\n".join(parts)


async def load_schema(conn: AsyncConnection) -> SchemaInfo:
    q = """
        SELECT c.relname AS table_name, c.relkind, obj_description(c.oid, 'pg_class') AS table_comment,
               a.attname AS column_name, format_type(a.atttypid, a.atttypmod) AS column_type,
               col_description(c.oid, a.attnum) AS column_comment
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
        WHERE n.nspname = 'public' AND c.relname = ANY(%s)
        ORDER BY c.relname, a.attnum
    """
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(q, (VISIBLE_TABLES,))
        rows = await cur.fetchall()
        await cur.execute("SELECT id, code, name, kind, process, capacity_m3d FROM plant ORDER BY id")
        plants = await cur.fetchall()
    tables: dict[str, Table] = {}
    for r in rows:
        t = tables.setdefault(r["table_name"], Table(r["table_name"], "view" if r["relkind"] == "v" else "table",
                                                     r["table_comment"]))
        t.columns.append(Column(r["column_name"], r["column_type"], r["column_comment"]))
    ordered = {name: tables[name] for name in VISIBLE_TABLES if name in tables}
    return SchemaInfo(ordered, plants)

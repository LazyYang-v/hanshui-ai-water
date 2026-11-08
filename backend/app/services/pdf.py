"""报告 Markdown → PDF。

只覆盖 `app.ai.reporter.render` 产出的语法：一二三级标题、引用说明、段落、管道表格、分隔线。
用 reportlab 内置的 Adobe 中文字体 STSong-Light（CID 字体，不嵌入字体文件），镜像无需安装系统字体；
代价是阅读器需支持 Adobe-GB1 字体替换（Acrobat、Chrome、macOS 预览均可）。
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT = "STSong-Light"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm
BODY_W = PAGE_W - 2 * MARGIN
ACCENT = colors.HexColor("#1f5fbf")
MUTED = colors.HexColor("#5b6b80")
GRID = colors.HexColor("#c9d4e3")


# ---------------------------------------------------------------- 解析

@dataclass
class Block:
    kind: str                                   # h1 / h2 / h3 / quote / para / table / hr
    text: str = ""
    rows: list[list[str]] = field(default_factory=list)
    align: list[str] = field(default_factory=list)   # 每列 left / right


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse(md: str) -> list[Block]:
    blocks: list[Block] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if m := re.match(r"^(#{1,3})\s+(.*)$", line):
            blocks.append(Block(f"h{len(m.group(1))}", m.group(2).strip()))
            i += 1
        elif re.match(r"^-{3,}$", line.strip()):
            blocks.append(Block("hr"))
            i += 1
        elif line.startswith(">"):
            parts: list[str] = []
            while i < len(lines) and lines[i].startswith(">"):
                parts.append(lines[i][1:].strip())
                i += 1
            # 引用内的空行（单独一个 >）分段
            for chunk in "\n".join(parts).split("\n\n"):
                if chunk.strip():
                    blocks.append(Block("quote", " ".join(chunk.split("\n"))))
        elif line.lstrip().startswith("|"):
            raw: list[str] = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                raw.append(lines[i])
                i += 1
            rows = [_cells(r) for r in raw]
            align: list[str] = []
            if len(rows) > 1 and all(re.match(r"^:?-+:?$", c) for c in rows[1]):
                align = ["right" if c.endswith(":") and not c.startswith(":") else "left" for c in rows[1]]
                rows = [rows[0]] + rows[2:]
            n = max(len(r) for r in rows)
            rows = [r + [""] * (n - len(r)) for r in rows]
            blocks.append(Block("table", rows=rows, align=align + ["left"] * (n - len(align))))
        else:
            parts = []
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,3}\s|>|\||-{3,}$)", lines[i].strip()):
                parts.append(lines[i].strip())
                i += 1
            blocks.append(Block("para", "".join(parts) if _cjk_end(parts) else " ".join(parts)))
    return blocks


def _cjk_end(parts: list[str]) -> bool:
    """中文换行续接时不补空格。"""
    return all(p and ord(p[-1]) > 0x2E80 for p in parts[:-1])


# ---------------------------------------------------------------- 排版

def _style(name: str, size: float, **kw) -> ParagraphStyle:
    return ParagraphStyle(name, fontName=FONT, fontSize=size, leading=size * 1.55, wordWrap="CJK", **kw)


STYLES = {
    "h1": _style("h1", 18, textColor=colors.HexColor("#0b2447"), spaceAfter=8),
    "h2": _style("h2", 13.5, textColor=ACCENT, spaceBefore=10, spaceAfter=5),
    "h3": _style("h3", 11.5, textColor=colors.HexColor("#0b2447"), spaceBefore=6, spaceAfter=3),
    "para": _style("para", 10, spaceAfter=5),
    "quote": _style("quote", 8.8, textColor=MUTED, backColor=colors.HexColor("#f4f6fa"),
                    borderPadding=(5, 7, 5, 7), leftIndent=7, rightIndent=7, spaceBefore=3, spaceAfter=8),
    "cell": _style("cell", 8.2, alignment=TA_LEFT),
    "cell_r": _style("cell_r", 8.2, alignment=TA_RIGHT),
    "head": _style("head", 8.2, textColor=MUTED),
}


LATIN = "Helvetica"
# STSong-Light 缺少这些字符（会显示成 ◆ / ▲），换成可用写法
SUBST = {"·": "，", "–": "-", "—": "-", "\u00a0": " "}


def _inline(text: str) -> str:
    """转义并按字符分段：西文/数字用 Helvetica（CID 字体的西文字形过窄），² ³ 排成上标。

    报告里没有强调和代码样式，出现时去掉标记即可。
    """
    t = re.sub(r"\*\*(.+?)\*\*|`(.+?)`", lambda m: m.group(1) or m.group(2), text)
    t = "".join(SUBST.get(ch, ch) for ch in t)
    out: list[str] = []
    for run in re.findall(r"[\x20-\x7e]+|[²³]|[^\x20-\x7e²³]+", t):
        if run in ("²", "³"):
            out.append(f'<super><font name="{LATIN}">{"2" if run == "²" else "3"}</font></super>')
        elif run[0] <= "\x7e":
            out.append(f'<font name="{LATIN}">{escape(run)}</font>')
        else:
            out.append(escape(run))
    return "".join(out)


def _col_widths(rows: list[list[str]]) -> list[float]:
    """按各列最长文本分配宽度：短列给够、长列换行，总宽等于版心。"""
    def width(s: str) -> float:
        return sum(2 if ord(ch) > 0x2E80 else 1.1 for ch in s)
    need = [max(width(r[c]) for r in rows) for c in range(len(rows[0]))]
    unit = 8.2 * 0.5                      # 半个字宽
    want = [max(4, w) * unit + 8 for w in need]
    total = sum(want)
    if total <= BODY_W:
        # 窄表不硬撑满版心，最多放宽 30%，避免数字列被推到最右
        k = min(BODY_W / total, 1.3)
        return [w * k for w in want]
    # 超宽：短列保留所需宽度，剩余宽度按需求比例分给长列
    short = [w if w <= BODY_W / len(want) else None for w in want]
    rest = BODY_W - sum(w for w in short if w)
    long_need = sum(w for w, s in zip(want, short) if s is None)
    return [s if s is not None else w / long_need * rest for w, s in zip(want, short)]


def _table(b: Block) -> Table:
    data = [[Paragraph(_inline(c), STYLES["head"]) for c in b.rows[0]]]
    for r in b.rows[1:]:
        data.append([Paragraph(_inline(c), STYLES["cell_r" if b.align[k] == "right" else "cell"]) for k, c in enumerate(r)])
    t = Table(data, colWidths=_col_widths(b.rows), repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1fb")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def render_pdf(md: str, *, title: str = "汉水智控运行报告", footer: str = "汉水智控：运行数据由模拟器生成，仅用于演示") -> bytes:
    story = []
    blocks = parse(md)
    for k, b in enumerate(blocks):
        if b.kind == "table":
            story.append(_table(b))
            story.append(Spacer(1, 6))
        elif b.kind == "hr":
            story.append(HRFlowable(width="100%", thickness=0.5, color=GRID, spaceBefore=6, spaceAfter=6))
        elif b.kind in ("h2", "h3") and k + 1 < len(blocks):
            # 标题不落在页尾：和下一块放在一起
            nxt = blocks[k + 1]
            follow = Paragraph(_inline(nxt.text), STYLES[nxt.kind]) if nxt.kind in STYLES and nxt.kind != "table" else None
            story.append(KeepTogether([Paragraph(_inline(b.text), STYLES[b.kind])] + ([follow] if follow else [])))
            if follow:
                blocks[k + 1] = Block("skip")
        elif b.kind in STYLES:
            story.append(Paragraph(_inline(b.text), STYLES[b.kind]))

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 9 * mm, footer)
        canvas.drawRightString(PAGE_W - MARGIN, 9 * mm, f"第 {doc.page} 页")   # 页码数字少，CID 西文字形可接受
        canvas.setStrokeColor(GRID)
        canvas.line(MARGIN, 12 * mm, PAGE_W - MARGIN, 12 * mm)
        canvas.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=14 * mm,
                            bottomMargin=17 * mm, title=title, author="汉水智控 Demo", subject=title)
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return buf.getvalue()

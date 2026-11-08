"""报告 PDF 排版：解析覆盖报告用到的 Markdown 语法，内联转换处理 CID 字体缺字。"""

import re

from app.services.pdf import _inline, parse, render_pdf

MD = """# 汉水智控运行日报（2026-11-07）

> 统计时段：2026-11-07 00:00 至 2026-11-08 00:00。
>
> 口径：达标率按 5 分钟桶统计。

## 一、运行概况

全区处理水量 69,203 m³，
出水达标率 99.45%。

| 厂站 | 处理水量 (m³) | 超标情况 |
|---|---:|---|
| P04 高台镇 | 2,994 | 出水氨氮超标 0.8 小时 |

---
"""


def test_parse_blocks():
    blocks = parse(MD)
    assert [b.kind for b in blocks] == ["h1", "quote", "quote", "h2", "para", "table", "hr"]
    assert blocks[2].text.startswith("口径")
    assert blocks[4].text == "全区处理水量 69,203 m³，出水达标率 99.45%。"   # 中文续行不补空格
    table = blocks[5]
    assert table.rows[0] == ["厂站", "处理水量 (m³)", "超标情况"]
    assert table.rows[1][1] == "2,994" and table.align == ["left", "right", "left"]


def test_inline_escapes_and_replaces_missing_glyphs():
    out = _inline("**水量** 1 m³ · <b>&")
    assert "**" not in out and "&lt;b&gt;&amp;" in out
    assert "³" not in out and "<super>" in out and "·" not in out
    assert '<font name="Helvetica"> 1 m</font>' in out


def test_render_pdf():
    data = render_pdf(MD + "\n".join(f"段落 {i}，" * 40 for i in range(40)), title="日报")
    assert data.startswith(b"%PDF")
    assert len(re.findall(rb"/Type /Page\b(?!s)", data)) >= 2      # 长文正常分页

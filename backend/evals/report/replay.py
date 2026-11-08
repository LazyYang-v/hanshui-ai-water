"""按当前解析与校验规则重放已保存的日报失败尝试，不调用模型。

报告 meta.attempts 里失败的那次保存了模型原始输出（2026-11-08 起），事实保存在 report.facts。
用法（backend 目录）：uv run python -m evals.report.replay [--ids 6 7] [--json out.json]
"""

from __future__ import annotations

import argparse
import json

import psycopg
from psycopg.rows import dict_row
from pydantic import ValidationError

from app.ai.agent.service import error_detail
from app.ai.reporter import Fact, check_narrative, parse_narrative
from app.config import Settings


def replay(output: str, facts: dict[str, Fact], names: dict[str, str]) -> dict:
    try:
        narrative, fixes = parse_narrative(output, facts)
        errors = check_narrative(narrative, facts, names)
    except (ValidationError, ValueError) as exc:
        errors, fixes = [error_detail(exc)], []
    return {"errors": errors, "fixes": fixes}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", type=int, nargs="*", help="报告编号；不填则重放全部保存了原始输出的报告")
    ap.add_argument("--json", help="结果另存为 JSON")
    args = ap.parse_args()
    with psycopg.connect(Settings().database_url, row_factory=dict_row) as conn:
        names = {r["code"]: r["name"] for r in conn.execute("SELECT code, name FROM plant")}
        where, params = ("WHERE id = ANY(%s)", (args.ids,)) if args.ids else ("", ())
        rows = conn.execute(f"SELECT id, facts, meta FROM report {where} ORDER BY id", params).fetchall()
    results = []
    for r in rows:
        facts = {k: Fact(v["id"], v["label"], v["text"], v.get("value"), v.get("llm", True))
                 for k, v in (r["facts"] or {}).items()}
        for i, at in enumerate((r["meta"] or {}).get("attempts", [])):
            if not at.get("output"):
                continue
            now = replay(at["output"], facts, names)
            results.append({"report": r["id"], "attempt": i, "then": at["errors"], **now, "passes_now": not now["errors"]})
            mark = "现在通过" if not now["errors"] else f"仍失败：{'；'.join(now['errors'][:2])}"
            print(f"#{r['id']} 第{i + 1}次  当时：{'；'.join(at['errors'][:2])}\n    {mark}"
                  + (f"\n    容错：{'；'.join(now['fixes'])}" if now["fixes"] else ""))
    print(f"共 {len(results)} 次保存了原文的失败尝试，按当前规则通过 {sum(r['passes_now'] for r in results)} 次")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

"""诊断评测的人工核查：导出核查表，再汇总人工结论得到语义通过数。

export：为一次评测目录生成 review.md（问题、回答、引用摘要、要核查的要点）和 review.json（待填结论）。
score ：读 review.json，semantic_pass = 自动检查通过 且 所有核查要点都判 pass；未填完的题不计入。
用法（backend 目录）：
  uv run python -m evals.agent.review export evals/agent/results/diagnosis-flash-agent-rules-v5
  # 在 review.json 里把每个要点的 "verdict" 填成 "pass" / "fail"，可写 "note"
  uv run python -m evals.agent.review score  evals/agent/results/diagnosis-flash-agent-rules-v5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cite(c: dict) -> str:
    if c["kind"] == "data":
        return f"[{c['number']}] {c['call_id']} 第{c['row'] + 1}行 {c.get('column_name')} = {c.get('value')}"
    if c["kind"] == "statistic":
        return f"[{c['number']}] 统计 {c.get('fact_id')} = {c.get('value')} {c.get('unit') or ''}"
    quote = (c.get("quote") or "").replace("\n", " ")
    return f"[{c['number']}] 知识《{c.get('title')}》{c.get('heading') or ''}：{quote[:80]}"


def export(run: Path) -> None:
    report = json.loads((run / "report.json").read_text())
    md = [f"# 人工核查表：{run.name}", "",
          "每题先看“要核查的要点”，再对照回答与引用判断；在 review.json 里把每个要点填 pass / fail。",
          "自动检查只覆盖数值、单位、时间、引用结构，没有判断中文语义。", ""]
    todo = {}
    for c in report["cases"]:
        if c["status"] != "needs_review" and not (c["status"] == "fail" and c["mechanical_pass"] is False):
            continue
        rec = json.loads((run / c["file"]).read_text())
        res, score = rec.get("result") or {}, rec["score"]
        items = [m["item"] for m in score.get("manual_review", [])]
        md += [f"## {c['id']}（{rec.get('mode')}，自动检查：{'通过' if score['mechanical_pass'] else '未通过'}）", "",
               f"**问题：** {rec['question']}", ""]
        if score.get("failures"):
            md += [f"**自动检查失败项：** {', '.join(score['failures'])}（该题已判不通过，人工项仅供参考）", ""]
        md += ["**回答：**", "", *[f"> {line}" if line else ">" for line in (res.get("answer") or res.get("reason") or "（无）").splitlines()], ""]
        if res.get("citations"):
            md += ["<details><summary>引用（" + str(len(res["citations"])) + " 处）</summary>", "",
                   *[f"- {_cite(x)}" for x in res["citations"]], "", "</details>", ""]
        md += ["**要核查的要点：**", "", *[f"{i + 1}. [ ] {it}" for i, it in enumerate(items)], ""]
        if score["status"] == "needs_review":
            todo[c["id"]] = [{"item": it, "verdict": None, "note": ""} for it in items]
    (run / "review.md").write_text("\n".join(md))
    path = run / "review.json"
    if path.exists():
        old = json.loads(path.read_text())                  # 已填的结论不覆盖
        todo = {k: old.get(k, v) for k, v in todo.items()}
    path.write_text(json.dumps(todo, ensure_ascii=False, indent=1))
    print(f"{len(todo)} 题待核查 → {run / 'review.md'}；结论填在 {path}")


def score(run: Path) -> None:
    report = json.loads((run / "report.json").read_text())
    verdicts = json.loads((run / "review.json").read_text())
    rows, pending = [], []
    for c in report["cases"]:
        if c["status"] == "pass":
            rows.append((c["id"], True, "自动"))
        elif c["status"] == "fail":
            rows.append((c["id"], False, "自动检查失败"))
        else:
            v = [x["verdict"] for x in verdicts.get(c["id"], [])]
            if not v or None in v:
                pending.append(c["id"])
                continue
            rows.append((c["id"], all(x == "pass" for x in v), "人工"))
    ok = sum(p for _, p, _ in rows)
    summary = {"cases": len(report["cases"]), "reviewed_or_auto": len(rows), "semantic_pass": ok,
               "pending": pending, "rows": [{"id": i, "pass": p, "by": b} for i, p, b in rows]}
    (run / "review-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"语义通过 {ok}/{len(rows)}（共 {len(report['cases'])} 题，待核查 {len(pending)} 题）")
    for i, p, b in rows:
        if not p:
            print(f"  不通过：{i}（{b}）")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["export", "score"])
    ap.add_argument("run", type=Path)
    a = ap.parse_args()
    (export if a.cmd == "export" else score)(a.run)


if __name__ == "__main__":
    main()

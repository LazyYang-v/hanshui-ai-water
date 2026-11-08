"""视觉巡检 HTTP 验收：真实模型 + 内置样例，演示告警的开启、升级、保持、恢复、再开启与工单草稿。

在独立测试库上跑（不污染演示库）；用 Redis 里的演示时钟控制时间，结果写 evals/vision/results/api-smoke.md。

    DATABASE_URL=postgresql://shuiwu:shuiwu@localhost:15432/shuiwu_test REDIS_URL=redis://localhost:16379/5 \\
      ALARM_MONITOR_ENABLED=false RAG_DENSE_ENABLED=false uv run uvicorn app.main:app --port 8012
    REDIS_URL=redis://localhost:16379/5 uv run python -m evals.vision.api_smoke \\
      --reset-db postgresql://shuiwu:shuiwu@localhost:15432/shuiwu_test
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import psycopg
import redis

from evals.http import login

RESULTS = Path(__file__).resolve().parent / "results"
T0 = datetime.fromisoformat("2026-11-05T09:00:00+08:00")


def brief(r: dict) -> dict:
    """接口返回里与验收相关的字段。"""
    return {"source": r["source"], "camera": r["camera"], "ts": r.get("ts"),
            "detections": [f"{d['cls']} {d['score']:.2f}" for d in r["detections"]],
            "events": r["events"], "alarm_note": r["alarm_note"],
            "timings_ms": {k: r["timings_ms"][k] for k in ("decode", "infer", "total")}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8012")
    p.add_argument("--reset-db", help="先清空该库的视觉告警与巡检记录（只允许 shuiwu_test）")
    a = p.parse_args()
    if a.reset_db:
        if not a.reset_db.rstrip("/").endswith("/shuiwu_test"):
            raise SystemExit("--reset-db 只允许测试库 shuiwu_test")
        with psycopg.connect(a.reset_db) as conn:
            conn.execute("DELETE FROM work_order WHERE alarm_id IN (SELECT id FROM alarm WHERE rule LIKE 'vision_%')")
            conn.execute("DELETE FROM alarm WHERE rule LIKE 'vision_%'")
            conn.execute("TRUNCATE vision_inspection")
    db = os.environ["REDIS_URL"].rstrip("/").rsplit("/", 1)[-1]
    if not db.isdigit() or db == "0":
        raise SystemExit("请用独立的 Redis 库（如 redis://localhost:16379/5），避免改动演示时钟")
    rds = redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    c = httpx.Client(base_url=a.url, timeout=60)
    login(c)
    log: list[tuple[str, object]] = []

    def step(title: str, method: str, path: str, *, minutes: float | None = None, brief_it=True, **kw):
        if minutes is not None:
            rds.set("sim:clock", (T0 + timedelta(minutes=minutes)).isoformat())
        r = c.request(method, path, **kw)
        body = r.json()
        out = brief(body) if brief_it and r.status_code == 200 else body
        log.append((f"{title}  `{method} {path}{'?' + str(httpx.QueryParams(kw.get('params'))) if kw.get('params') else ''}` → {r.status_code}", out))
        return r.status_code, body

    _, card = step("模型卡", "GET", "/api/vision/model", brief_it=False)
    log[-1] = (log[-1][0], {k: card[k] for k in ("name", "classes", "imgsz", "conf", "experiment", "model_id", "serving",
                                                  "samples")} | {"real_val": card["metrics"]["real_val"]["floating_debris"]})
    _, s = step("样例清单", "GET", "/api/vision/samples", brief_it=False)
    log[-1] = (log[-1][0], [f"{x['name']}  {x['camera']}  {x['note']}" for x in s["items"]])

    riv = {"plant": "P01", "camera": "CAM-RIV-01"}
    step("① 2 个漂浮物 → 新告警（提示）", "POST", "/api/vision/samples/real_nsw_00119.jpg/inspect", minutes=0, params=riv)
    step("② 6 个漂浮物 → 升级为预警", "POST", "/api/vision/samples/real_hobart_00754.jpg/inspect", minutes=5, params=riv)
    step("③ 只有低分框（0.25～0.45 之间）→ 保持，不恢复也不升级", "POST",
         "/api/vision/samples/real_indonesia_00057.jpg/inspect", minutes=10, params=riv)
    step("④ 无检出 → 恢复", "POST", "/api/vision/samples/synth_negative_glare.jpg/inspect", minutes=15, params=riv)
    step("⑤ 再次检出 → 开新事件（不并入已恢复的事件）", "POST", "/api/vision/samples/real_nsw_00119.jpg/inspect",
         minutes=20, params=riv)
    step("⑥ 管廊渗漏 → 预警", "POST", "/api/vision/samples/synth_leak_streak.jpg/inspect", minutes=21,
         params={"plant": "P01"})
    step("⑦ 难负样本（锈迹、管道阴影）→ 渗漏恢复", "POST", "/api/vision/samples/synth_negative_rust.jpg/inspect",
         minutes=22, params={"plant": "P01"})
    img = (Path(__file__).resolve().parents[2] / "models/vision/samples/synth_clarifier.jpg").read_bytes()
    step("⑧ 上传图片、不指定厂站 → 只记录不告警", "POST", "/api/vision/inspect", minutes=23,
         content=img, headers={"Content-Type": "image/jpeg"})
    step("⑨ 非图片 → 415", "POST", "/api/vision/inspect", content=b"hello", headers={"Content-Type": "text/plain"},
         brief_it=False)

    _, alarms = step("告警列表（视觉巡检）", "GET", "/api/alarms", params={"kind": "视觉巡检"}, brief_it=False)
    items = alarms["items"]
    log[-1] = (log[-1][0], [{k: x.get(k) for k in ("id", "rule", "level", "metric", "ts", "ended_at", "value",
                                                    "message")} for x in items])
    debris = next(x for x in items if x["rule"] == "vision_debris")
    _, detail = step("告警详情", "GET", f"/api/alarms/{debris['id']}", brief_it=False)
    d = detail.get("alarm", detail)
    log[-1] = (log[-1][0], {"detail": {k: v for k, v in (d.get("detail") or {}).items() if k != "boxes"}})
    _, draft = step("工单草稿（模板步骤）", "POST", f"/api/alarms/{debris['id']}/work-order-draft", brief_it=False)
    log[-1] = (log[-1][0], {k: draft.get(k) for k in ("title", "steps", "source")})
    step("AI 处置建议 → 422（知识库无对应规程，不调用模型）", "POST", f"/api/alarms/{debris['id']}/advice",
         brief_it=False)
    _, insp = step("巡检记录", "GET", "/api/vision/inspections", params={"limit": 3}, brief_it=False)
    log[-1] = (log[-1][0], [{k: x[k] for k in ("id", "ts", "plant", "camera", "source", "alarm_ids")} for x in insp["items"]])

    lines = ["# 视觉巡检 HTTP 验收记录", "",
             f"生成于 {datetime.now().isoformat(timespec='seconds')}；独立测试库 `shuiwu_test`，演示时钟从 {T0.isoformat()} 起由脚本设置；"
             f"模型 `{card['model_id']}`（{card['experiment']}）。由 `evals/vision/api_smoke.py` 生成。", ""]
    for title, body in log:
        lines += [f"### {title}", "", "```json", json.dumps(body, ensure_ascii=False, indent=1), "```", ""]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "api-smoke.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

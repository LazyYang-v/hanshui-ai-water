"""启动真实Uvicorn/HTTP服务验收负荷告警。独立库+进程内Redis，不调用模型。

uv run python -m evals.alarms.smoke --label load-api-final
默认自动选择空闲本地端口；不改变演示库、演示时钟或演示Redis频道。
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import psycopg

from app.analytics.rules import BUCKET, LOCAL_TZ
from evals.http import login

HERE = Path(__file__).parent
DB_NAME = "shuiwu_alarm_smoke"
T0 = datetime(2026, 11, 7, 8, tzinfo=LOCAL_TZ)
END = T0 + 6 * BUCKET


SMOKE_PASSWORD = "smoke-admin"   # 临时进程专用，验收结束即销毁


def create_smoke_app():
    import fakeredis

    from app.config import Settings
    from app.main import create_app
    server = fakeredis.FakeServer()
    fake = fakeredis.FakeRedis(server=server, decode_responses=True)
    fake.set("sim:clock", END.isoformat())
    return create_app(Settings(alarm_monitor_enabled=False, rag_dense_enabled=False, admin_password=SMOKE_PASSWORD),
                      redis=fakeredis.FakeAsyncRedis(server=server, decode_responses=True))


def build(url):
    admin_url = os.getenv("EVAL_ADMIN_DATABASE_URL", "postgresql://shuiwu:shuiwu@localhost:15432/shuiwu")
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {DB_NAME} WITH (FORCE)")
        conn.execute(f"CREATE DATABASE {DB_NAME}")
    with psycopg.connect(url, autocommit=True) as conn:
        for f in sorted((HERE.parents[1] / "db/init").glob("*.sql")):
            conn.execute(f.read_text())
        rows = []
        for i in range(168):
            rows.append((T0 - timedelta(days=7) + timedelta(hours=i), 1000, 100, 20, 10))
        for i in range(6):
            factor = 1.8 if i < 3 else 1.0
            rows.append((T0 + i * BUCKET, 1000, 100 * factor, 20 * factor, 10 * factor))
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO plant_reading(ts,plant_id,inflow_m3h,cod_in,tn_in,nh3n_in) "
                            "VALUES (%s,1,%s,%s,%s,%s)", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--port", type=int, default=0, help="0为自动选择空闲端口")
    a = ap.parse_args()
    if not a.label or any(not (c.isalnum() or c in "-_") for c in a.label):
        ap.error("label仅字母数字下划线连字符")
    target = HERE / "results" / f"api-{a.label}.json"
    if target.exists():
        ap.error("结果标签已存在")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", a.port))
        a.port = sock.getsockname()[1]
    admin_url = os.getenv("EVAL_ADMIN_DATABASE_URL", "postgresql://shuiwu:shuiwu@localhost:15432/shuiwu")
    url = admin_url.rsplit("/", 1)[0] + "/" + DB_NAME
    build(url)
    env = dict(os.environ, DATABASE_URL=url, READONLY_DATABASE_URL=url, DEEPSEEK_API_KEY="", QWEN_API_KEY="",
               LLM_PROVIDERS="", RAG_DENSE_ENABLED="false", ALARM_MONITOR_ENABLED="false")
    results = {"database": DB_NAME, "clock": END.isoformat(), "checks": {}, "responses": {}}
    with tempfile.TemporaryFile(mode="w+") as log:
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "evals.alarms.smoke:create_smoke_app", "--factory",
                                 "--host", "127.0.0.1", "--port", str(a.port)], env=env, cwd=HERE.parents[1],
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{a.port}", timeout=30) as client:
                deadline = time.monotonic() + 30
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError("验收API进程提前结束")
                    try:
                        if client.get("/api/health").status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError("验收API启动超时")
                    time.sleep(0.1)
                login(client, SMOKE_PASSWORD)

                def get(path, **kwargs):
                    r = client.get(path, **kwargs)
                    r.raise_for_status()
                    return r.json()

                def post(path, **kwargs):
                    r = client.post(path, **kwargs)
                    r.raise_for_status()
                    return r.json()

                checks = results["checks"]
                checks["nine_rules"] = len(get("/api/alarms/monitor")["rules"]) == 9
                checks["three_derived_metrics"] = len(get("/api/metrics")["derived"]) == 3
                first = post("/api/alarms/scan", params={"hours": 3})
                checks["three_load_episodes"] = first["inserted"] == 3
                alarms = get("/api/alarms", params={"kind": "进水负荷"})["items"]
                checks["load_units_and_values"] = (len(alarms) == 3 and
                    {r["metric"]: r["value"] for r in alarms} == {"cod_load_kgh": 180, "tn_load_kgh": 36, "nh3n_load_kgh": 18}
                    and all(r["unit"] == "kg/h" and r["detail"]["pattern"] == "浓度升高型负荷冲击" for r in alarms))
                checks["three_normal_buckets_recover"] = all(r["ended_at"] == (T0 + 3 * BUCKET).isoformat() for r in alarms)
                checks["actual_detection_time"] = all(r["detail"]["detected_at"] == END.isoformat() for r in alarms)
                drafts = [post(f"/api/alarms/{r['id']}/work-order-draft") for r in alarms]
                checks["drafts_explain_load_verification"] = all(any("kg/h" in s["text"] for s in d["steps"]) for d in drafts)
                checks["no_work_order_written"] = get("/api/work-orders")["total"] == 0
                again = post("/api/alarms/scan", params={"hours": 3})
                checks["idempotent_scan"] = again["inserted"] == again["updated"] == 0
                results["responses"] = {"first_scan": first, "alarms": alarms, "drafts": drafts, "again": again}
                results["passed"] = all(checks.values())
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    target.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    assert results["passed"], results["checks"]
    print(f"真实HTTP验收 {len(results['checks'])}/{len(results['checks'])} 通过：{target}")


if __name__ == "__main__":
    main()

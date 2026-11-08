import json
import threading
from datetime import timedelta

import pytest

from tests.conftest import END

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def plant(client, code):
    return next(p for p in client.get("/api/plants").json() if p["code"] == code)


# ---------------------------------------------------------------- 基础

def test_health(client):
    body = client.get("/api/health").json()
    assert body["db"] == "ok" and body["redis"] == "ok"
    assert body["simulator"] == "not running"


def test_clock_uses_sim_clock(client):
    assert client.get("/api/clock").json()["now"] == END.isoformat()


def test_metrics_catalog_has_limits(client):
    plant_metrics = {m["key"]: m for m in client.get("/api/metrics").json()["plant"]}
    assert plant_metrics["nh3n_out"]["limit"] == 5
    assert plant_metrics["do_aerobic_1"]["lower_is_bad"] is True
    assert plant_metrics["do_aerobic_1"]["warn_at"] == round(1.0 / 0.67, 4)
    assert plant_metrics["nh3n_out"]["warn_at"] == 4.0 and plant_metrics["inflow_m3h"]["warn_at"] is None


# ---------------------------------------------------------------- 厂站状态

def test_plant_status_reflects_scenarios(client):
    plants = {p["code"]: p for p in client.get("/api/plants").json()}
    assert len(plants) == 7

    p02 = plants["P02"]
    assert p02["status"] == "alarm"
    assert any(i["metric"] == "tp_out" and i["level"] == "alarm" for i in p02["issues"])

    p06 = plants["P06"]
    assert p06["status"] == "alarm"
    assert any("设备故障" in i["label"] for i in p06["issues"])
    assert any(i["metric"] == "do_aerobic_1" for i in p06["issues"])

    assert plants["P01"]["status"] == "normal"
    assert plants["P01"]["latest_ts"] == END.isoformat()


def test_plant_detail_lists_devices_and_today(client):
    body = client.get("/api/plants/p06").json()  # 编码大小写不敏感
    devices = {d["code"]: d for d in body["devices"]}
    assert devices["P06-BL-1"]["status"] == "故障" and devices["P06-BL-1"]["level"] == "alarm"
    assert devices["P06-BL-1"]["running"] is False
    assert devices["P06-BL-2"]["status"] == "备用"
    assert body["today"]["hours"] == pytest.approx(12, abs=0.2)
    assert body["today"]["compliance_rate"] < 1  # 鼓风机跳停后氨氮超标


def test_unknown_plant_404(client):
    assert client.get("/api/plants/P99").status_code == 404
    assert client.get("/api/plants/P99/readings").status_code == 404


# ---------------------------------------------------------------- 总览

def test_overview(client):
    body = client.get("/api/overview").json()
    assert sum(body["status_counts"].values()) == 7
    assert body["status_counts"]["alarm"] >= 2
    assert len(body["hourly"]) == 24
    # 半天、全区设计规模 6.9 万 m³/d：处理量应在 1/3 到 2/3 天之间
    assert 69000 / 3 < body["today"]["water_m3"] < 69000 * 2 / 3
    assert 0.07 < body["today"]["aeration_kwh_per_m3"] < 0.13
    assert body["today"]["compliance_rate"] < 1
    assert body["weather"]["ts"] == END.isoformat()


# ---------------------------------------------------------------- 时间序列

def test_readings_auto_bucket_last_24h(client):
    body = client.get("/api/plants/P03/readings").json()
    assert body["bucket"] == "5m"
    assert 285 <= len(body["ts"]) <= 289
    assert set(body["series"]) == {"cod_out", "nh3n_out", "tn_out", "tp_out", "ss_out"}
    assert body["metrics"][1]["limit"] == 5


def test_readings_naive_time_is_beijing_and_explicit_bucket(client):
    body = client.get("/api/plants/P01/readings", params={
        "metrics": "inflow_m3h,do_aerobic_1", "start": "2026-11-05T00:00", "end": "2026-11-05T06:00", "bucket": "1h",
    }).json()
    assert body["ts"][0] == "2026-11-05T00:00:00+08:00"
    assert len(body["ts"]) == 6
    assert all(1.5 < v < 2.5 for v in body["series"]["do_aerobic_1"])


def test_readings_show_tp_rising_after_dosing_block(client):
    body = client.get("/api/plants/P02/readings", params={
        "metrics": "tp_out", "start": (END - timedelta(hours=6)).isoformat(), "end": END.isoformat(), "bucket": "1h",
    }).json()
    tp = body["series"]["tp_out"]
    assert tp[0] < 0.4 and tp[-1] > 0.5


@pytest.mark.parametrize("params, fragment", [
    ({"metrics": "nh3n_out;drop table plant"}, "未知指标"),
    ({"start": "2026-11-05T10:00", "end": "2026-11-05T09:00"}, "早于"),
    ({"start": "2026-09-01T00:00", "end": "2026-11-05T00:00"}, "31 天"),
    ({"bucket": "7m"}, "bucket"),
])
def test_readings_validation(client, params, fragment):
    r = client.get("/api/plants/P01/readings", params=params)
    assert r.status_code == 422
    assert fragment in r.json()["detail"]


def test_device_readings_show_vibration_before_trip(client):
    body = client.get("/api/devices/P06-BL-1/readings", params={
        "start": (END - timedelta(hours=2)).isoformat(), "end": (END - timedelta(hours=1)).isoformat(), "bucket": "1m",
    }).json()
    vib, cur = body["series"]["vibration_mms"], body["series"]["current_a"]
    assert max(vib) > 7.1
    assert cur[0] > 0 and cur[-1] == 0


# ---------------------------------------------------------------- 演示控制

def test_demo_without_simulator_returns_503(client):
    r = client.post("/api/demo/scenarios", json={"kind": "storm"})
    assert r.status_code == 503


def test_demo_validates_body(client):
    assert client.post("/api/demo/scenarios", json={"kind": "earthquake"}).status_code == 422
    assert client.put("/api/demo/speed", json={"value": 0}).status_code == 422


def fake_simulator(sync_redis, ok=True):
    """模拟器替身：先回一条别人的回复，再回本请求的回复，验证 API 按 id 匹配。"""
    ps = sync_redis.pubsub(ignore_subscribe_messages=True)
    ps.subscribe("sim:control")
    ps.get_message(timeout=0.1)  # 读掉订阅确认，订阅才生效（与 CLI 中的做法相同）
    ready = threading.Event()

    def loop():
        ready.set()
        while True:
            msg = ps.get_message(timeout=5)
            if msg is None:
                return
            cmd = json.loads(msg["data"])
            sync_redis.publish("sim:events", json.dumps({"id": "someone-else", "ok": True, "message": "别人的"}))
            sync_redis.publish("sim:events", json.dumps(
                {"id": cmd["id"], "ok": ok, "message": f"收到 {cmd['action']} {cmd.get('kind') or ''}".strip()},
                ensure_ascii=False))
            return

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    ready.wait()
    return t


def test_demo_command_matches_reply_by_id(client, sync_redis):
    t = fake_simulator(sync_redis)
    r = client.post("/api/demo/scenarios", json={"kind": "blower_fault", "plant": "p01"})
    t.join()
    assert r.status_code == 200
    assert r.json()["message"] == "收到 inject blower_fault"


def test_demo_command_error_is_400(client, sync_redis):
    t = fake_simulator(sync_redis, ok=False)
    r = client.put("/api/demo/speed", json={"value": 60})
    t.join()
    assert r.status_code == 400


# ---------------------------------------------------------------- WebSocket

def test_websocket_sends_latest_then_ticks(client, sync_redis):
    sync_redis.set("realtime:latest", json.dumps({"ts": "latest"}))
    with client.websocket_connect("/ws/realtime") as ws:
        assert json.loads(ws.receive_text()) == {"ts": "latest"}
        # 等订阅生效后再发布
        for _ in range(50):
            if sync_redis.pubsub_numsub("realtime:tick")[0][1] > 0:
                break
            threading.Event().wait(0.05)
        sync_redis.publish("realtime:tick", json.dumps({"ts": "tick-1"}))
        assert json.loads(ws.receive_text()) == {"ts": "tick-1"}

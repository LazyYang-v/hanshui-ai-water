"""视觉巡检：推理前后处理（纯函数）、告警合并规则（测试库 + 假检测器）、接口限制；有真实模型时做一次冒烟。"""

import io
import json
from pathlib import Path

import numpy as np
import psycopg
import pytest
from PIL import Image

from app.vision import detector as det
from app.vision.alarms import VISION_RULES, decide, level_of
from app.vision.detector import Detection, Result, letterbox, nms, postprocess
from app.vision.service import VisionService

MODEL = Path(__file__).resolve().parents[1] / "models/vision/inspect.onnx"


def jpeg(w=320, h=180, color=(90, 110, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------- 前后处理

def test_letterbox_scales_and_centers():
    x, r, (left, top) = letterbox(np.zeros((540, 960, 3), np.uint8), 640)
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
    assert r == pytest.approx(640 / 960) and (left, top) == (0, 140)
    assert x[0, :, 0, 0] == pytest.approx([114 / 255] * 3)          # 填充区
    assert x[0, :, 320, 320] == pytest.approx([0, 0, 0])            # 图像区


def test_nms_keeps_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], np.float32)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2]


def raw_output(rows, nc=2, anchors=8):
    """rows: [(cx, cy, w, h, [各类分数])]，按 YOLO 导出格式 (1, 4+nc, anchors) 排布。"""
    out = np.zeros((1, 4 + nc, anchors), np.float32)
    for i, (cx, cy, w, h, scores) in enumerate(rows):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4:, i] = scores
    return out


def test_postprocess_maps_back_and_separates_classes():
    # 960x540 原图 → r=2/3，上填充 140。两个同位置不同类的框都保留（按类 NMS），低分框被滤掉
    raw = raw_output([(320, 320, 60, 30, [0.9, 0.0]), (320, 320, 60, 30, [0.0, 0.8]), (100, 200, 10, 10, [0.1, 0.0])])
    dets = postprocess(raw, ["floating_debris", "leak"], 640 / 960, (0, 140), (960, 540), conf=0.25, iou=0.5)
    assert [d.cls for d in dets] == ["floating_debris", "leak"]
    x1, y1, x2, y2 = dets[0].box
    assert (x1, y1, x2, y2) == pytest.approx((435, 247.5, 525, 292.5))


def test_postprocess_clips_to_image():
    raw = raw_output([(5, 145, 40, 40, [0.9, 0.0])])
    (d,) = postprocess(raw, ["floating_debris", "leak"], 640 / 960, (0, 140), (960, 540), conf=0.25, iou=0.5)
    assert d.box[0] == 0 and d.box[1] == 0


def test_decode_rejects_non_images_and_oversized(monkeypatch):
    with pytest.raises(det.ImageError, match="无法识别"):
        det.decode(b"not an image")
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "GIF")
    with pytest.raises(det.ImageError, match="不支持"):
        det.decode(buf.getvalue())
    monkeypatch.setattr(det, "MAX_PIXELS", 1000)
    with pytest.raises(det.ImageError, match="像素过多"):
        det.decode(jpeg(100, 100))


def test_decode_applies_exif_orientation():
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6                                                  # 顺时针 90°
    Image.new("RGB", (40, 20)).save(buf, "JPEG", exif=exif)
    assert det.decode(buf.getvalue()).size == (20, 40)


# ---------------------------------------------------------------- 告警判定

def d(cls, score):
    return Detection(cls, score, (0, 0, 10, 10))


def test_decide_uses_hysteresis():
    rule = VISION_RULES["floating_debris"]
    assert decide(rule, [d("floating_debris", 0.6), d("floating_debris", 0.3)])[0] == "hit"
    assert len(decide(rule, [d("floating_debris", 0.6), d("floating_debris", 0.5)])[1]) == 2
    assert decide(rule, [d("floating_debris", 0.35)])[0] == "hold"     # 介于恢复与告警阈值之间
    assert decide(rule, [d("floating_debris", 0.1), d("leak", 0.9)])[0] == "clear"
    assert decide(rule, [])[0] == "clear"


def test_level_depends_on_count_for_debris_only():
    assert level_of(VISION_RULES["floating_debris"], 1) == "提示"
    assert level_of(VISION_RULES["floating_debris"], 3) == "预警"
    assert level_of(VISION_RULES["leak"], 1) == "预警"


# ---------------------------------------------------------------- 评测指标（evals/vision/run.py）

def test_eval_ap_and_matching():
    from evals.vision.run import average_precision, match
    gt = [np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])]
    preds = [(0.9, np.array([0, 0, 10, 10])), (0.8, np.array([1, 0, 10, 10])), (0.7, np.array([20, 20, 30, 30]))]
    used: set[int] = set()
    assert match(preds, gt, 0.5, used) == [True, False, True] and used == {0, 1}   # 重复框是 FP
    assert average_precision([(0.9, True), (0.7, True)], 2) == pytest.approx(1.0)
    assert average_precision([(0.9, False), (0.7, True)], 2) == pytest.approx(51 / 101 * 0.5)
    assert average_precision([], 3) == 0.0 and np.isnan(average_precision([], 0))


def test_eval_breakdown_by_region_and_size():
    from evals.vision.run import breakdown
    # 1280×720 原图，640 输入里缩小一半：40px 宽的框 → 20px，200px → 100px
    items = [{"size": (1280, 720), "meta": {"region": "NSW"},
              "labels": [[0, 100 / 1280, 100 / 720, 40 / 1280, 40 / 720], [0, 600 / 1280, 300 / 720, 200 / 1280, 200 / 720]]}]
    dets = [[Detection("floating_debris", 0.9, (500, 200, 700, 400)),
             Detection("floating_debris", 0.9, (1000, 600, 1100, 700)),
             Detection("floating_debris", 0.2, (80, 80, 120, 120))]]      # 低于告警阈值，不算命中
    b = breakdown(items, dets)
    assert b["by_region"]["NSW"] == {"gt": 2, "precision": 0.5, "recall": 0.5}
    assert b["recall_by_size"]["16-32px"] == {"gt": 1, "recall": 0.0}
    assert b["recall_by_size"][">=64px"] == {"gt": 1, "recall": 1.0}


# ---------------------------------------------------------------- 接口（测试库）

class FakeDetector:
    """按脚本返回检测结果；推理接口之外的行为与真实 Detector 相同。"""

    model_sha256 = "f" * 64

    def __init__(self):
        self.script: list[list[Detection]] = []

    def detect_bytes(self, data, conf=None):
        im = det.decode(data)
        dets = self.script.pop(0) if self.script else []
        return Result(im.width, im.height, dets, {"infer": 1.0}, "a" * 64)

    def info(self):
        return {"classes": ["floating_debris", "leak"], "imgsz": 640, "model_sha256": self.model_sha256}


@pytest.fixture
def vision(client, db_url, tmp_path):
    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute("TRUNCATE alarm, work_order, vision_inspection RESTART IDENTITY CASCADE")
    (tmp_path / "pool.jpg").write_bytes(jpeg())
    (tmp_path / "samples.json").write_text(json.dumps([
        {"name": "pool.jpg", "scene": "pool", "source": "synthetic", "camera": "CAM-AER-01"},
        {"name": "../escape.jpg", "scene": "pool", "source": "synthetic", "camera": "X"}]))
    fake = FakeDetector()
    client.app.state.vision = VisionService(fake, concurrency=2, threads=1, samples_dir=tmp_path)
    return client, fake


def inspect(client, **params):
    r = client.post("/api/vision/samples/pool.jpg/inspect", params={"plant": "P01", **params})
    assert r.status_code == 200, r.text
    return r.json()


def vision_alarms(client):
    return client.get("/api/alarms", params={"kind": "视觉巡检", "limit": 50}).json()["items"]


def test_samples_listing_rejects_unsafe_names(vision):
    client, _ = vision
    names = [s["name"] for s in client.get("/api/vision/samples").json()["items"]]
    assert names == ["pool.jpg"]
    assert client.get("/api/vision/samples/pool.jpg").status_code == 200
    assert client.get("/api/vision/samples/escape.jpg").status_code == 404


def test_detection_opens_merges_and_recovers_alarm(vision):
    client, fake = vision
    fake.script = [[d("floating_debris", 0.8)],
                   [d("floating_debris", 0.9), d("floating_debris", 0.7), d("floating_debris", 0.6)],
                   [d("floating_debris", 0.3)],                                    # 滞回区：不恢复
                   []]
    first = inspect(client)
    assert [e["event"] for e in first["events"]] == ["new"] and first["inspection_id"] == 1
    (a,) = vision_alarms(client)
    assert (a["rule"], a["level"], a["metric_label"], a["unit"], a["detail"]["camera"]) == \
        ("vision_debris", "提示", "漂浮物", "置信度", "CAM-AER-01")
    assert a["active"] and a["device"] is None

    second = inspect(client)
    assert [e["event"] for e in second["events"]] == ["escalated"]
    (a,) = vision_alarms(client)
    assert a["level"] == "预警" and a["detail"]["hits"] == 2 and a["detail"]["count"] == 3
    assert a["peak"] == pytest.approx(0.9) and len(a["detail"]["images"]) == 2

    assert inspect(client)["events"] == []
    assert vision_alarms(client)[0]["active"]

    last = inspect(client)
    assert [e["event"] for e in last["events"]] == ["recovered"]
    (a,) = vision_alarms(client)
    assert not a["active"]
    rows = client.get("/api/vision/inspections", params={"camera": "CAM-AER-01"}).json()["items"]
    assert len(rows) == 4 and rows[0]["alarm_ids"] == [a["id"]]


def test_cameras_and_classes_are_separate_events(vision):
    client, fake = vision
    fake.script = [[d("floating_debris", 0.8), d("leak", 0.9)], [d("floating_debris", 0.8)]]
    inspect(client)
    inspect(client, camera="CAM-SEC-02")
    alarms = vision_alarms(client)
    assert sorted((a["rule"], a["detail"]["camera"]) for a in alarms) == [
        ("vision_debris", "CAM-AER-01"), ("vision_debris", "CAM-SEC-02"), ("vision_leak", "CAM-AER-01")]


def test_closed_alarm_is_not_merged(vision):
    client, fake = vision
    fake.script = [[d("leak", 0.9)], [d("leak", 0.9)]]
    inspect(client)
    (a,) = vision_alarms(client)
    assert client.post(f"/api/alarms/{a['id']}/close", json={"note": "现场确认为误报"}).status_code == 200
    inspect(client)
    alarms = sorted(vision_alarms(client), key=lambda x: x["id"])
    assert len(alarms) == 2 and not alarms[0]["active"] and alarms[1]["active"]


def test_without_plant_only_records(vision):
    client, fake = vision
    fake.script = [[d("leak", 0.9)]]
    r = client.post("/api/vision/samples/pool.jpg/inspect")
    assert r.status_code == 200
    body = r.json()
    assert body["events"] == [] and "不写告警" in body["alarm_note"] and body["inspection_id"]
    assert vision_alarms(client) == []


def test_upload_limits(vision):
    client, fake = vision
    url = "/api/vision/inspect"
    assert client.post(url, content=b"x", headers={"Content-Type": "text/plain"}).status_code == 415
    assert client.post(url, content=b"junk", headers={"Content-Type": "image/jpeg"}).status_code == 422
    assert client.post(url, content=b"", headers={"Content-Type": "image/jpeg"}).status_code == 422
    big = b"\0" * (10 * 1024 * 1024 + 1)
    assert client.post(url, content=big, headers={"Content-Type": "image/jpeg"}).status_code == 413
    assert client.post(url, params={"plant": "NOPE", "camera": "C1"}, content=jpeg(),
                       headers={"Content-Type": "image/jpeg"}).status_code == 404
    assert client.post(url, params={"camera": "bad camera!"}, content=jpeg(),
                       headers={"Content-Type": "image/jpeg"}).status_code == 422
    fake.script = [[d("floating_debris", 0.8)]]
    r = client.post(url, params={"plant": "P01", "camera": "CAM-UP-01"}, content=jpeg(640, 360),
                    headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 200 and r.json()["width"] == 640 and r.json()["events"][0]["event"] == "new"
    assert "queue" in r.json()["timings_ms"]


def test_vision_alarm_has_template_draft_and_no_ai_advice(vision):
    client, fake = vision
    fake.script = [[d("leak", 0.9)]]
    inspect(client)
    (a,) = vision_alarms(client)
    assert client.post(f"/api/alarms/{a['id']}/advice").status_code == 422
    draft = client.post(f"/api/alarms/{a['id']}/work-order-draft").json()
    assert draft["source"]["kind"] == "template" and "渗漏" in draft["steps"][0]["text"]


def test_signal_scan_ignores_vision_alarms(vision, db_url):
    client, fake = vision
    fake.script = [[d("leak", 0.9)]]
    inspect(client)
    with psycopg.connect(db_url, autocommit=True) as conn:            # 让它成为扫描窗口之前就开着的事件
        conn.execute("UPDATE alarm SET ts = ts - interval '2 days'")
    (a,) = vision_alarms(client)
    r = client.post("/api/alarms/scan", params={"hours": 3})
    assert r.status_code == 200, r.text
    assert r.json()["start"].startswith("2026-11-05T09:00")           # 起点没有被视觉告警拉回两天前
    (after,) = vision_alarms(client)
    assert after["active"] and after["updated_at"] == a["updated_at"]


def test_model_endpoint_unavailable_without_model(client):
    client.app.state.vision = None
    assert client.get("/api/vision/model").status_code == 503


# ---------------------------------------------------------------- 真实模型冒烟

@pytest.mark.skipif(not MODEL.exists(), reason="未导出模型")
def test_real_model_runs_on_samples():
    svc = VisionService.load(MODEL, concurrency=1, threads=1, samples_dir=MODEL.parent / "samples")
    info = svc.info()
    assert info["classes"] == ["floating_debris", "leak"] and svc.samples
    for name, s in svc.samples.items():
        res = svc.detector.detect_bytes((svc.samples_dir / name).read_bytes())
        assert res.width > 0 and all(0 <= x.score <= 1 for x in res.detections)
        if s.note.startswith("expect:"):
            assert {x.cls for x in res.detections if x.score >= 0.25} >= set(s.note[7:].split(",")), name

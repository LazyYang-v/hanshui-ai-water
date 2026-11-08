"""生成 API 内置样例：models/vision/samples/*.jpg + samples.json。

合成样例固定挑选（覆盖三类水池、两种渗漏形态、两张难负样本）；真实样例从 FloatWaste val 子集里
按地区各挑一张「部署模型检得最好」的图，另挑一张检得最差的作为已知失败案例，避免样例只展示成功。
note 写 "expect:类别" 的条件是部署模型确实在 0.25 阈值下检出了这些类别（tests/test_vision.py 冒烟测试会校验）。

    uv run python vision/samples.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.vision.detector import Detector  # noqa: E402

DATA = ROOT / "data/vision"
OUT = ROOT / "models/vision/samples"
MODEL = ROOT / "models/vision/inspect.onnx"
NAMES = ["floating_debris", "leak"]
POOL_LABEL = {"aeration": "曝气池", "clarifier": "二沉池", "channel": "渠道", "outfall": "出水口"}
POOL_CAMERA = {"aeration": "CAM-AER-01", "clarifier": "CAM-CLR-01", "channel": "CAM-CHN-01", "outfall": "CAM-OUT-01"}
SYNTH = [  # (测试集文件, 样例名, 摄像头, 说明)
    ("test_00046.jpg", "synth_aeration.jpg", None, None),
    ("test_00009.jpg", "synth_clarifier.jpg", None, None),
    ("test_00023.jpg", "synth_outfall.jpg", None, None),
    ("test_00014.jpg", "synth_leak_streak.jpg", "CAM-GAL-01", "管廊渗漏：管壁水痕 + 地面积水"),
    ("test_00025.jpg", "synth_leak_puddle.jpg", "CAM-GAL-02", "管廊渗漏：法兰下方积水"),
    ("test_00000.jpg", "synth_negative_rust.jpg", "CAM-GAL-01", "难负样本：锈迹、管道阴影，无渗漏"),
    ("test_00027.jpg", "synth_negative_glare.jpg", "CAM-GAL-02", "难负样本：油漆、污渍、地面反光，无渗漏"),
]
ALARM_CONF = 0.45


def labels(path: Path, w: int, h: int) -> np.ndarray:
    rows = [list(map(float, ln.split())) for ln in path.read_text().splitlines() if ln.strip()] if path.exists() else []
    out = []
    for c, cx, cy, bw, bh in rows:
        out.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out).reshape(-1, 5)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), 2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], 1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def f1(det, gt: np.ndarray) -> tuple[float, int, int, int]:
    """贪心匹配（IoU≥0.5）后的 F1、TP、FP、FN；只用于挑样例。"""
    pred = sorted([d for d in det if d.score >= ALARM_CONF], key=lambda d: -d.score)
    used, tp = set(), 0
    if len(gt) and pred:
        m = iou(np.array([d.box for d in pred]), gt[:, 1:])
        for i in range(len(pred)):
            j = next((j for j in np.argsort(-m[i]) if m[i, j] >= 0.5 and j not in used), None)
            if j is not None:
                used.add(j)
                tp += 1
    fp, fn = len(pred) - tp, len(gt) - tp
    return (2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0), tp, fp, fn


def expect(det, gt_classes: set[str]) -> str | None:
    found = {d.cls for d in det if d.score >= 0.25}
    if gt_classes and gt_classes <= found:
        return "expect:" + ",".join(sorted(gt_classes))
    return None


def main() -> None:
    det = Detector.load(MODEL, threads=4)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.jpg"):
        old.unlink()
    synth = {it["file"]: it for it in json.loads((DATA / "synth/manifest.json").read_text())["splits"]["test"]["items"]}
    samples = []
    for src, name, camera, desc in SYNTH:
        it = synth[src]
        path = DATA / "synth/images/test" / src
        res = det.detect_bytes(path.read_bytes())
        gt = labels(DATA / "synth/labels/test" / src.replace(".jpg", ".txt"), res.width, res.height)
        gt_cls = {NAMES[int(c)] for c in gt[:, 0]}
        if it["scene"] == "pool":
            camera = POOL_CAMERA[it["pool"]]
            desc = f"合成{POOL_LABEL[it['pool']]}池面：{len(gt)} 个漂浮物，干扰物 {', '.join(sorted(set(it['distractors'])))}"
        score, tp, fp, fn = f1(res.detections, gt)
        note = expect(res.detections, gt_cls) or ("" if gt_cls else "expect-none")
        if note == "expect-none":
            note = "" if any(d.score >= 0.25 for d in res.detections) else "期望无检出"
        shutil.copyfile(path, OUT / name)
        samples.append({"name": name, "scene": "pool" if it["scene"] == "pool" else "gallery", "source": "synthetic",
                        "camera": camera, "note": note, "description": desc, "attribution": None,
                        "check": {"gt": len(gt), "tp": tp, "fp": fp, "fn": fn}})
        print(name, note, tp, fp, fn)

    fw = json.loads((DATA / "floatwaste/manifest.json").read_text())
    scored = []
    for it in fw["splits"]["val"]["items"]:
        if not 2 <= it["boxes"] <= 10:
            continue
        path = DATA / "floatwaste/images/val" / it["file"]
        res = det.detect_bytes(path.read_bytes())
        gt = labels(DATA / "floatwaste/labels/val" / it["file"].replace(".jpg", ".txt"), res.width, res.height)
        scored.append((f1(res.detections, gt), it, res, path))
    picks = []
    for region in ("Hobart", "NSW", "Indonesia"):
        cands = [s for s in scored if s[1]["region"] == region]
        if cands:
            picks.append((max(cands, key=lambda s: (s[0][0], s[0][1])), "good"))
    picks.append((min(scored, key=lambda s: (s[0][0], -s[0][3])), "bad"))
    src = fw["source"]
    for (score, it, res, path), kind in picks:
        name = f"real_{it['region'].lower()}_{it['id']:05d}.jpg"
        _, tp, fp, fn = score
        if kind == "good":
            desc = f"FloatWaste 真实河道/排水沟（{it['region']}），{it['boxes']} 个标注漂浮物，部署模型检出 {tp}"
            note = expect(res.detections, {"floating_debris"}) or ""
        else:
            desc = (f"已知失败案例：FloatWaste（{it['region']}）{it['boxes']} 个标注漂浮物，阈值 {ALARM_CONF} 下"
                    f"检出 {tp}、误检 {fp}、漏检 {fn}；说明合成→真实的域差")
            note = ""
        shutil.copyfile(path, OUT / name)
        samples.append({"name": name, "scene": "river", "source": "floatwaste", "camera": "CAM-RIV-01",
                        "note": note, "description": desc,
                        "attribution": f"{src['dataset']}（{src['license']}，DOI {src['doi']}），原图 {it['source']}，"
                                       f"已缩放至长边 1280",
                        "check": {"gt": it["boxes"], "tp": tp, "fp": fp, "fn": fn}})
        print(name, kind, note, tp, fp, fn)
    (OUT / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

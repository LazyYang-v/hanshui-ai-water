"""视觉巡检评测：服务端推理链路的精度、PyTorch↔ONNX 一致性、延迟与并发压测。

    uv run python -m evals.vision.run pipeline                      # 用 app/vision 的 ONNX 链路算 AP / 运行点指标
    uv run --group vision python -m evals.vision.run parity         # 同一批图 PyTorch 与 ONNX 检测结果对比
    uv run python -m evals.vision.run latency                       # 进程内单张延迟（不同算子线程数）
    uv run python -m evals.vision.run http --url http://127.0.0.1:8012   # HTTP 并发 1/4/8/16

pipeline 评的是线上实际跑的代码（letterbox + onnxruntime + numpy NMS），不是 Ultralytics 自带的 val；
两者预处理插值不同，数字会有小差异，parity 给出差多少。AP 按 COCO 101 点插值，IoU 0.5 与 0.5:0.95。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from app.vision.alarms import VISION_RULES
from app.vision.detector import Detection, Detector, decode, letterbox, postprocess
from evals.http import alogin

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/vision"
MODEL = ROOT / "models/vision/inspect.onnx"
RESULTS = Path(__file__).resolve().parent / "results"
SETS = {"synth_test": DATA / "synth", "real_val": DATA / "floatwaste"}
SPLIT = {"synth_test": "test", "real_val": "val"}
CLASSES = ["floating_debris", "leak"]


# ---------------------------------------------------------------- 数据

def load_set(name: str) -> list[dict]:
    root, split = SETS[name], SPLIT[name]
    meta = {}
    manifest = root / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        meta = {it["file"]: it for it in m["splits"].get(split, {}).get("items", [])}
    items = []
    for img in sorted((root / "images" / split).glob("*.jpg")):
        lbl = root / "labels" / split / f"{img.stem}.txt"
        rows = [list(map(float, ln.split())) for ln in lbl.read_text().splitlines() if ln.strip()] if lbl.exists() else []
        items.append({"path": img, "labels": rows, "meta": meta.get(img.name, {})})
    return items


def to_xyxy(rows: list[list[float]], w: int, h: int) -> list[tuple[int, np.ndarray]]:
    return [(int(c), np.array([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
            for c, x, y, bw, bh in rows]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    def area(z):
        return (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


# ---------------------------------------------------------------- 指标

def match(preds: list[tuple[float, np.ndarray]], gts: list[np.ndarray], thr: float,
          used: set[int] | None = None) -> list[bool]:
    """单张图单类别：按分数降序贪心匹配，返回每个预测是否为 TP（与 preds 同序）；传入 used 时回填被匹配的标注下标。"""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][0])
    used = set() if used is None else used
    tp = [False] * len(preds)
    if not gts:
        return tp
    ious = iou_matrix(np.array([p[1] for p in preds]).reshape(-1, 4), np.array(gts).reshape(-1, 4))
    for i in order:
        best, bj = thr, -1
        for j in range(len(gts)):
            if j not in used and ious[i, j] >= best:
                best, bj = ious[i, j], j
        if bj >= 0:
            used.add(bj)
            tp[i] = True
    return tp


def average_precision(scored: list[tuple[float, bool]], n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if not scored:
        return 0.0
    scored = sorted(scored, key=lambda x: -x[0])
    tp = np.cumsum([s[1] for s in scored])
    fp = np.cumsum([not s[1] for s in scored])
    recall, precision = tp / n_gt, tp / np.maximum(tp + fp, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def evaluate(items: list[dict], dets: list[list[Detection]]) -> dict:
    out = {}
    thresholds = np.arange(0.5, 0.96, 0.05)
    for c, cls in enumerate(CLASSES):
        gts_all = []
        preds_all = []
        for it, ds in zip(items, dets):
            w, h = it["size"]
            gts_all.append([b for k, b in to_xyxy(it["labels"], w, h) if k == c])
            preds_all.append([(d.score, np.array(d.box)) for d in ds if d.cls == cls])
        n_gt = sum(len(g) for g in gts_all)
        if n_gt == 0:
            # 该集合没有这一类（如真实集里没有渗漏）：只统计误报
            conf = VISION_RULES[cls].alarm_conf
            fp_imgs = sum(1 for p in preds_all if any(s >= conf for s, _ in p))
            out[cls] = {"gt": 0, "false_alarm_images": fp_imgs, "images": len(items), "alarm_conf": conf}
            continue
        aps = []
        for thr in thresholds:
            scored = []
            for p, g in zip(preds_all, gts_all):
                scored += list(zip([s for s, _ in p], match(p, g, thr)))
            aps.append(average_precision(scored, n_gt))
        # 运行点：告警阈值下的 P / R（IoU 0.5），以及图片级触发率
        conf = VISION_RULES[cls].alarm_conf
        tp = fp = 0
        img_pos = img_pos_hit = img_neg = img_neg_hit = 0
        for p, g in zip(preds_all, gts_all):
            kept = [x for x in p if x[0] >= conf]
            flags = match(kept, g, 0.5)
            tp += sum(flags)
            fp += len(flags) - sum(flags)
            if g:
                img_pos += 1
                img_pos_hit += bool(kept)
            else:
                img_neg += 1
                img_neg_hit += bool(kept)
        out[cls] = {
            "gt": n_gt, "ap50": aps[0], "ap50_95": float(np.mean(aps)),
            "alarm_conf": conf, "precision": tp / max(tp + fp, 1), "recall": tp / n_gt,
            "image_hit_rate": img_pos_hit / max(img_pos, 1), "images_with_target": img_pos,
            "image_false_alarm_rate": img_neg_hit / max(img_neg, 1), "images_without_target": img_neg,
        }
    return out


def slices(items: list[dict], dets: list[list[Detection]]) -> dict:
    """合成集按场景与干扰物切片：无目标图上出现告警级检测的比例，看模型被哪类干扰骗到。"""
    groups = defaultdict(lambda: [0, 0])
    for it, ds in zip(items, dets):
        m = it["meta"]
        if not m or it["labels"]:
            continue
        alarm = any(d.score >= VISION_RULES[d.cls].alarm_conf for d in ds)
        for key in [f"scene={m['scene']}"] + [f"distractor={x}" for x in set(m.get("distractors", []))]:
            groups[key][0] += 1
            groups[key][1] += alarm
    return {k: {"negative_images": n, "false_alarm_images": a} for k, (n, a) in sorted(groups.items())}


SIZE_BUCKETS = [(0, 16, "<16px"), (16, 32, "16-32px"), (32, 64, "32-64px"), (64, 1e9, ">=64px")]


def breakdown(items: list[dict], dets: list[list[Detection]], cls: str = "floating_debris") -> dict:
    """真实集分解：按地区的告警阈值 P/R，按目标尺寸（模型 640 输入里的 √面积）的召回，用来解释域差。"""
    c, conf = CLASSES.index(cls), VISION_RULES[cls].alarm_conf
    region = defaultdict(lambda: [0, 0, 0])          # tp, fp, gt
    size = defaultdict(lambda: [0, 0])               # 命中, gt
    for it, ds in zip(items, dets):
        w, h = it["size"]
        scale = 640 / max(w, h)
        gts = [b for k, b in to_xyxy(it["labels"], w, h) if k == c]
        kept = [(d.score, np.array(d.box)) for d in ds if d.cls == cls and d.score >= conf]
        used: set[int] = set()
        flags = match(kept, gts, 0.5, used)
        r = region[it["meta"].get("region", "?")]
        r[0] += sum(flags)
        r[1] += len(flags) - sum(flags)
        r[2] += len(gts)
        for j, b in enumerate(gts):
            side = float(np.sqrt(max(b[2] - b[0], 0) * max(b[3] - b[1], 0))) * scale
            label = next(lab for lo, hi, lab in SIZE_BUCKETS if lo <= side < hi)
            size[label][0] += j in used
            size[label][1] += 1
    return {
        "alarm_conf": conf,
        "by_region": {k: {"gt": g, "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(g, 1), 4)}
                      for k, (tp, fp, g) in sorted(region.items())},
        "recall_by_size": {lab: {"gt": size[lab][1], "recall": round(size[lab][0] / max(size[lab][1], 1), 4)}
                           for _, _, lab in SIZE_BUCKETS},
    }


def run_pipeline(args) -> dict:
    det = Detector.load(Path(args.model), threads=os.cpu_count() or 4)
    report = {"model": str(Path(args.model).resolve().relative_to(ROOT)), "model_sha256": det.model_sha256,
              "imgsz": det.imgsz, "generated_at": datetime.now().isoformat(timespec="seconds"), "sets": {}}
    for name in SETS:
        items = load_set(name)
        if not items:
            continue
        dets, infer = [], []
        for it in items:
            res = det.detect_bytes(it["path"].read_bytes(), conf=0.001)
            it["size"] = (res.width, res.height)
            dets.append(res.detections)
            infer.append(res.timings_ms["infer"])
        entry = {"images": len(items), "boxes": sum(len(it["labels"]) for it in items), **evaluate(items, dets)}
        if name == "synth_test":
            entry["negative_slices"] = slices(items, dets)
        else:
            entry["breakdown"] = breakdown(items, dets)
        report["sets"][name] = entry
        print(name, json.dumps({k: v for k, v in entry.items() if k != "negative_slices"}, ensure_ascii=False))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"pipeline-{args.label}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


# ---------------------------------------------------------------- PyTorch ↔ ONNX

def _match(a: list[Detection], b: list[Detection]) -> tuple[int, list[float]]:
    """同类、IoU≥0.9 的一对一匹配数与分数差。"""
    matched, diff = 0, []
    for cls in CLASSES:
        x = [d for d in a if d.cls == cls]
        y = [d for d in b if d.cls == cls]
        if not x or not y:
            continue
        ious = iou_matrix(np.array([d.box for d in x]), np.array([d.box for d in y]))
        used = set()
        for i in np.argsort([-d.score for d in x]):
            j = next((int(j) for j in np.argsort(-ious[i]) if ious[i, j] >= 0.9 and int(j) not in used), None)
            if j is not None:
                used.add(j)
                matched += 1
                diff.append(abs(x[i].score - y[j].score))
    return matched, diff


def run_parity(args) -> dict:
    """两层对照：
    tensor    同一个 letterbox 后的输入张量分别送 PyTorch 模型和 ONNX，检验导出本身（原始输出最大绝对误差、后处理结果）；
    end2end   部署链路（Pillow 缩放、640×640 方形 letterbox）对照 Ultralytics predict（OpenCV 缩放、.pt 默认按步长
              取最小填充的矩形输入），二者预处理不同，差异反映的是预处理而不是导出。
    """
    import torch
    from ultralytics import YOLO
    det = Detector.load(Path(args.model), threads=4)
    yolo = YOLO(str(args.weights))
    net = yolo.model.float().eval()
    rows = []
    for name in SETS:
        for it in load_set(name)[:args.n]:
            data = it["path"].read_bytes()
            img = np.asarray(decode(data))
            x, r, pad = letterbox(img, det.imgsz)
            raw_onnx = det.session.run(None, {det.input: x})[0]
            with torch.no_grad():
                out = net(torch.from_numpy(x))
            raw_pt = (out[0] if isinstance(out, (tuple, list)) else out).numpy()
            size = (img.shape[1], img.shape[0])
            a = postprocess(raw_onnx, det.classes, r, pad, size, 0.25, det.iou)
            b = postprocess(raw_pt, det.classes, r, pad, size, 0.25, det.iou)
            t_match, t_diff = _match(a, b)
            pt = yolo.predict(str(it["path"]), imgsz=det.imgsz, conf=0.25, iou=det.iou, device="cpu", verbose=False)[0]
            c = [Detection(CLASSES[int(k)], float(sc), tuple(map(float, bx)))
                 for bx, sc, k in zip(pt.boxes.xyxy.numpy(), pt.boxes.conf.numpy(), pt.boxes.cls.numpy())]
            e_match, e_diff = _match(a, c)
            rows.append({"set": name, "file": it["path"].name,
                         "tensor": {"max_abs_raw": float(np.abs(raw_onnx - raw_pt).max()), "onnx": len(a),
                                    "torch": len(b), "matched_iou90": t_match, "max_score_diff": max(t_diff, default=0.0)},
                         "end2end": {"onnx": len(a), "ultralytics": len(c), "matched_iou90": e_match,
                                     "max_score_diff": max(e_diff, default=0.0)}})
    summary = {}
    for layer, ref in (("tensor", "torch"), ("end2end", "ultralytics")):
        for name in SETS:
            rs = [x[layer] for x in rows if x["set"] == name]
            s = {"images": len(rs), "onnx_detections": sum(x["onnx"] for x in rs),
                 f"{ref}_detections": sum(x[ref] for x in rs), "matched_iou90": sum(x["matched_iou90"] for x in rs),
                 "max_score_diff": round(max(x["max_score_diff"] for x in rs), 4),
                 "p95_score_diff": round(pct([x["max_score_diff"] for x in rs], 95), 4)}
            s["match_rate"] = round(s["matched_iou90"] / max(s[f"{ref}_detections"], 1), 4)
            if layer == "tensor":
                s["max_abs_raw"] = float(max(x["max_abs_raw"] for x in rs))
            summary.setdefault(layer, {})[name] = s
    summary["note"] = run_parity.__doc__.strip()
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"parity-{args.label}.json").write_text(json.dumps({"model_sha256": det.model_sha256, "summary": summary,
                                                                  "rows": rows}, ensure_ascii=False, indent=1))
    return summary


# ---------------------------------------------------------------- 延迟与压测

def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))]


def machine() -> dict:
    info = {"platform": platform.platform(), "cpu_count": os.cpu_count(), "load_avg_1m": round(os.getloadavg()[0], 1)}
    try:
        info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                     text=True, timeout=5).stdout.strip()
    except Exception:
        pass
    return info


def sample_image(args) -> bytes:
    return Path(args.image).read_bytes() if args.image else sorted((DATA / "synth/images/test").glob("*.jpg"))[0].read_bytes()


def run_latency(args) -> dict:
    data = sample_image(args)
    out = {"machine_before": machine(), "runs": args.n, "threads": {}}
    for threads in (1, 2, 4):
        det = Detector.load(Path(args.model), threads=threads)
        for _ in range(10):
            det.detect_bytes(data)
        t = defaultdict(list)
        for _ in range(args.n):
            r = det.detect_bytes(data)
            for k, v in r.timings_ms.items():
                t[k].append(v)
            t["total"].append(sum(r.timings_ms.values()))
        out["threads"][threads] = {k: {"p50": round(pct(v, 50), 2), "p95": round(pct(v, 95), 2)} for k, v in t.items()}
        print(threads, json.dumps(out["threads"][threads]))
    out["machine_after"] = machine()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"latency-{args.label}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


async def _http(args) -> dict:
    import httpx
    data = sample_image(args)
    out = {"url": args.url, "requests_per_level": args.n, "machine_before": machine(), "levels": {}}
    async with httpx.AsyncClient(base_url=args.url, timeout=60) as client:
        await alogin(client)
        info = (await client.get("/api/vision/model")).json()
        out["serving"] = info.get("serving")
        params = {"alarm": "false"}
        for _ in range(10):
            await client.post("/api/vision/inspect", params=params, content=data, headers={"Content-Type": "image/jpeg"})
        for c in args.levels:
            lat, server, queue, errors = [], [], [], 0
            sem = asyncio.Semaphore(c)

            async def one():
                nonlocal errors
                async with sem:
                    t0 = time.perf_counter()
                    r = await client.post("/api/vision/inspect", params=params, content=data,
                                          headers={"Content-Type": "image/jpeg"})
                    lat.append((time.perf_counter() - t0) * 1000)
                    if r.status_code != 200:
                        errors += 1
                        return
                    tm = r.json()["timings_ms"]
                    server.append(tm["total"])
                    queue.append(tm["queue"])

            t0 = time.perf_counter()
            await asyncio.gather(*(one() for _ in range(args.n)))
            wall = time.perf_counter() - t0
            out["levels"][c] = {"p50_ms": round(pct(lat, 50), 1), "p95_ms": round(pct(lat, 95), 1),
                                "throughput_rps": round(args.n / wall, 1), "errors": errors,
                                "server_total_p50_ms": round(pct(server, 50), 1) if server else None,
                                "queue_p50_ms": round(pct(queue, 50), 1) if queue else None,
                                "queue_p95_ms": round(pct(queue, 95), 1) if queue else None}
            print(c, out["levels"][c], flush=True)
    out["machine_after"] = machine()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"http-{args.label}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["pipeline", "parity", "latency", "http"])
    p.add_argument("--model", default=str(MODEL))
    p.add_argument("--weights", default=str(DATA / "runs/e2_mixed/weights/best.pt"))
    p.add_argument("--label", default="v1")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--image")
    p.add_argument("--url", default="http://127.0.0.1:8012")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8, 16])
    a = p.parse_args(argv)
    if a.cmd == "pipeline":
        run_pipeline(a)
    elif a.cmd == "parity":
        run_parity(a)
    elif a.cmd == "latency":
        run_latency(a)
    else:
        asyncio.run(_http(a))


if __name__ == "__main__":
    main()

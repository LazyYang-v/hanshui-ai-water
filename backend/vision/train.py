"""训练、评估与导出（Ultralytics YOLO，AGPL-3.0；只在 vision 依赖组里，API 镜像不装）。

两组实验，用同一个起点（yolo11n COCO 预训练权重）和同样的超参：
  E1 synth   只用合成数据训练
  E2 mixed   合成 + FloatWaste 真实 train 子集
都在「合成 test」和「FloatWaste 真实 val 子集」上评估：前者是同分布成绩，后者衡量真实域上的漂浮物检测；
渗漏没有真实数据，真实 val 只有 floating_debris 一类。

    uv run --group vision python vision/train.py train  --exp e1_synth
    uv run --group vision python vision/train.py eval   --exp e1_synth        # Ultralytics 自带 val
    uv run --group vision python vision/train.py export --exp e1_synth        # → runs/e1_synth/weights/best.onnx
    uv run python -m evals.vision.run pipeline --model data/vision/runs/e1_synth/weights/best.onnx --label e1_synth
    uv run --group vision python vision/train.py export --exp e2_mixed --deploy   # → models/vision/inspect.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/vision"
RUNS = DATA / "runs"
RESULTS = ROOT / "evals/vision/results"
MODEL_DIR = ROOT / "models/vision"
NAMES = {0: "floating_debris", 1: "leak"}
EXPERIMENTS = {
    "e1_synth": [DATA / "synth/images/train"],
    "e2_mixed": [DATA / "synth/images/train", DATA / "floatwaste/images/train"],
}
EVALS = {
    "synth_test": DATA / "synth/images/test",
    "real_val": DATA / "floatwaste/images/val",
}


def write_yaml(path: Path, train: list[Path], val: Path) -> Path:
    lines = ["path: " + str(DATA), "train:"] + [f"  - {p}" for p in train] + [f"val: {val}", "names:"]
    lines += [f"  {k}: {v}" for k, v in NAMES.items()]
    path.write_text("\n".join(lines) + "\n")
    return path


def train(exp: str, epochs: int, imgsz: int, batch: int, device: str, seed: int) -> None:
    from ultralytics import YOLO
    cfg = write_yaml(DATA / f"{exp}.yaml", EXPERIMENTS[exp], DATA / "synth/images/val")
    model = YOLO("yolo11n.pt")
    t0 = time.time()
    model.train(data=str(cfg), epochs=epochs, imgsz=imgsz, batch=batch, device=device, seed=seed,
                workers=4, project=str(RUNS), name=exp, exist_ok=True, patience=0, plots=False,
                close_mosaic=5, cache="ram", verbose=False)
    (RUNS / exp / "train_meta.json").write_text(json.dumps({
        "exp": exp, "epochs": epochs, "imgsz": imgsz, "batch": batch, "device": device, "seed": seed,
        "train_dirs": [str(p.relative_to(ROOT)) for p in EXPERIMENTS[exp]],
        "minutes": round((time.time() - t0) / 60, 1), "machine": platform.platform(),
    }, ensure_ascii=False, indent=1))


def _metrics(m) -> dict:
    out = {"all": {"precision": float(m.box.mp), "recall": float(m.box.mr),
                   "map50": float(m.box.map50), "map50_95": float(m.box.map)}}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        out[NAMES[int(c)]] = {"precision": float(p), "recall": float(r), "map50": float(ap50), "map50_95": float(ap)}
    return out


def evaluate(exp: str, imgsz: int, device: str, weights: Path | None = None) -> dict:
    from ultralytics import YOLO
    weights = weights or RUNS / exp / "weights/best.pt"
    model = YOLO(str(weights))
    report = {"exp": exp, "weights": str(weights.relative_to(ROOT)), "imgsz": imgsz, "sets": {}}
    for name, d in EVALS.items():
        cfg = write_yaml(DATA / f"eval_{name}.yaml", [d], d)
        m = model.val(data=str(cfg), imgsz=imgsz, batch=16, device=device, conf=0.001, iou=0.6,
                      plots=False, verbose=False, project=str(RUNS / exp), name=f"val_{name}", exist_ok=True)
        report["sets"][name] = {"images": len(list(d.glob("*.jpg"))), **_metrics(m)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{exp}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return report


LICENSE = {
    "framework": "Ultralytics YOLO（AGPL-3.0）。作品展示与学习可用；闭源商用分发须开源整个作品或购买 Ultralytics 商业授权",
    "weights": "由 Ultralytics yolo11n.pt（COCO 预训练）微调，权重同受 AGPL-3.0 约束",
    "data": "合成数据由本仓库 vision/synth.py 生成；真实漂浮物图片来自 FloatWaste（OliviaWuYJ/sc-floatwaste，"
            "CC BY 4.0，DOI 10.5281/zenodo.21934286），子集经缩放与类别合并",
}
LIMITATIONS = [
    "渗漏（leak）只有程序合成数据，没有真实管廊图片验证，真实场景效果未知",
    "漂浮物真实验证集是 FloatWaste 的河道/排水沟图片，不是污水厂池面；厂内摄像头需另行采集标注",
    "合成 test 与训练同一生成器，指标偏乐观，只说明模型学会了生成器定义的视觉线索",
]


def card(exp: str, imgsz: int) -> dict:
    meta_path = RUNS / exp / "train_meta.json"
    return {"name": f"shuiwu-inspect-yolo11n-{exp}", "task": "detect", "classes": list(NAMES.values()),
            "class_labels": {"floating_debris": "漂浮物", "leak": "渗漏"}, "imgsz": imgsz, "conf": 0.25, "iou": 0.5,
            "experiment": exp, "training": json.loads(meta_path.read_text()) if meta_path.exists() else None,
            "license": LICENSE, "limitations": LIMITATIONS}


def export(exp: str, imgsz: int, deploy: bool) -> Path:
    """导出到训练目录（best.onnx + 同名模型卡）。

    deploy 不重新导出（Ultralytics 会把导出时间写进元数据，字节每次不同），而是复制已评测过的 best.onnx，
    并要求 pipeline 报告里的 sha256 与之一致，保证部署的就是评测过的那个文件。
    """
    out = RUNS / exp / "weights/best.onnx"
    info = card(exp, imgsz)
    if not deploy:
        from ultralytics import YOLO
        best = RUNS / exp / "weights/best.pt"
        out = Path(YOLO(str(best)).export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False, nms=False))
        out.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
        print(out, out.stat().st_size)
        return out
    pipeline = RESULTS / f"pipeline-{exp}.json"
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    rep = json.loads(pipeline.read_text()) if pipeline.exists() else {}
    if rep.get("model_sha256") != sha:
        raise SystemExit(f"{pipeline.name} 不是对当前 {out.name}（sha256 {sha[:12]}）做的评测，先重跑 pipeline 再部署")
    info["metrics"] = {name: {k: v for k, v in s.items() if k not in ("negative_slices", "breakdown")} for name, s in rep["sets"].items()}
    info["metrics_source"] = str(pipeline.relative_to(ROOT))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODEL_DIR / "inspect.onnx"
    shutil.copyfile(out, dst)
    dst.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(dst, dst.stat().st_size)
    return dst


def main(argv: list[str] | None = None) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    os.chdir(DATA)              # 预训练权重下载到 data/vision/，不落在仓库目录
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "eval", "export"])
    p.add_argument("--exp", choices=list(EXPERIMENTS), required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deploy", action="store_true", help="export 时复制到 models/vision/ 供 API 使用")
    a = p.parse_args(argv)
    if a.cmd == "train":
        train(a.exp, a.epochs, a.imgsz, a.batch, a.device, a.seed)
    elif a.cmd == "eval":
        evaluate(a.exp, a.imgsz, a.device)
    else:
        export(a.exp, a.imgsz, a.deploy)


if __name__ == "__main__":
    sys.exit(main())

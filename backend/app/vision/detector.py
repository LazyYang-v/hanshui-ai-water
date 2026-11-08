"""ONNX 目标检测推理：解码 → letterbox → onnxruntime（CPU）→ 按类 NMS → 还原到原图坐标。

只依赖 onnxruntime、numpy、Pillow，API 镜像不装 PyTorch / Ultralytics。
模型是 YOLO11 导出的静态 ONNX（不含 NMS），输出 (1, 4+类别数, 锚点数)：cx, cy, w, h + 各类分数。
预处理与 Ultralytics 的 LetterBox 保持一致（等比缩放、居中填充 114 灰），导出后用同一批图核对过两者结果。
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 40_000_000           # 解码前按头信息拒绝超大图，防解压炸弹
FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
PAD = 114
MAX_WH = 7680                     # 按类 NMS 的坐标偏移量


class ImageError(ValueError):
    """图片无法解码或超出限制。"""


@dataclass(frozen=True)
class Detection:
    cls: str
    score: float
    box: tuple[float, float, float, float]     # 原图像素 x1, y1, x2, y2

    def out(self) -> dict:
        return {"cls": self.cls, "score": round(self.score, 4), "box": [round(v, 1) for v in self.box]}


@dataclass
class Result:
    width: int
    height: int
    detections: list[Detection]
    timings_ms: dict[str, float] = field(default_factory=dict)
    sha256: str = ""

    def out(self) -> dict:
        return {"width": self.width, "height": self.height, "sha256": self.sha256,
                "detections": [d.out() for d in self.detections], "timings_ms": self.timings_ms}


def decode(data: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError) as e:
        raise ImageError("无法识别的图片格式") from e
    if im.format not in FORMATS:
        raise ImageError(f"不支持的图片格式 {im.format}，仅支持 {'/'.join(sorted(FORMATS))}")
    if im.width * im.height > MAX_PIXELS:
        raise ImageError(f"图片像素过多（{im.width}×{im.height}），上限 {MAX_PIXELS // 1_000_000} MP")
    try:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        raise ImageError(f"图片解码失败：{e}") from e


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """HxWx3 uint8 → 1x3xSxS float32；返回缩放比与左上填充 (left, top)。"""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    if (nw, nh) != (w, h):
        img = np.asarray(Image.fromarray(img).resize((nw, nh), Image.Resampling.BILINEAR))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    left, top = round(dw - 0.1), round(dh - 0.1)
    canvas = np.full((size, size, 3), PAD, np.uint8)
    canvas[top:top + nh, left:left + nw] = img
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255
    return np.ascontiguousarray(x), r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """经典贪心 NMS，返回保留下标（按分数降序）。"""
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1, yy1 = np.maximum(x1[i], x1[rest]), np.maximum(y1[i], y1[rest])
        xx2, yy2 = np.minimum(x2[i], x2[rest]), np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ious = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[ious <= iou]
    return np.asarray(keep, dtype=np.int64)


def postprocess(raw: np.ndarray, classes: list[str], r: float, pad: tuple[int, int], size: tuple[int, int],
                conf: float, iou: float, max_det: int = 100) -> list[Detection]:
    pred = raw[0].T                                   # (锚点数, 4+类别数)
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(1)
    scores = scores_all[np.arange(len(pred)), cls]
    m = scores >= conf
    if not m.any():
        return []
    xywh, cls, scores = pred[m, :4], cls[m], scores[m]
    boxes = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
    keep = nms(boxes + (cls * MAX_WH)[:, None], scores, iou)[:max_det]
    left, top = pad
    w, h = size
    out = []
    for i in keep:
        x1, y1, x2, y2 = (boxes[i] - [left, top, left, top]) / r
        out.append(Detection(classes[int(cls[i])], float(scores[i]),
                             (float(np.clip(x1, 0, w)), float(np.clip(y1, 0, h)),
                              float(np.clip(x2, 0, w)), float(np.clip(y2, 0, h)))))
    return out


class Detector:
    def __init__(self, session: ort.InferenceSession, card: dict, model_sha256: str):
        self.session, self.card, self.model_sha256 = session, card, model_sha256
        self.input = session.get_inputs()[0].name
        self.imgsz = int(card["imgsz"])
        self.classes = list(card["classes"])
        self.conf, self.iou = float(card.get("conf", 0.25)), float(card.get("iou", 0.5))
        shape = session.get_outputs()[0].shape
        if len(shape) != 3 or shape[1] != 4 + len(self.classes):
            raise ValueError(f"模型输出 {shape} 与模型卡类别数 {len(self.classes)} 不符")

    @classmethod
    def load(cls, model_path: Path, threads: int = 1) -> Detector:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        data = model_path.read_bytes()
        session = ort.InferenceSession(data, opts, providers=["CPUExecutionProvider"])
        card = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(session, card, hashlib.sha256(data).hexdigest())

    def detect_array(self, img: np.ndarray, conf: float | None = None) -> tuple[list[Detection], dict]:
        t0 = time.perf_counter()
        x, r, pad = letterbox(img, self.imgsz)
        t1 = time.perf_counter()
        raw = self.session.run(None, {self.input: x})[0]
        t2 = time.perf_counter()
        dets = postprocess(raw, self.classes, r, pad, (img.shape[1], img.shape[0]),
                           self.conf if conf is None else conf, self.iou)
        t3 = time.perf_counter()
        return dets, {"preprocess": (t1 - t0) * 1000, "infer": (t2 - t1) * 1000, "postprocess": (t3 - t2) * 1000}

    def detect_bytes(self, data: bytes, conf: float | None = None) -> Result:
        t0 = time.perf_counter()
        im = decode(data)
        arr = np.asarray(im)
        t1 = time.perf_counter()
        dets, timings = self.detect_array(arr, conf)
        timings = {"decode": (t1 - t0) * 1000, **timings}
        return Result(im.width, im.height, dets, {k: round(v, 2) for k, v in timings.items()},
                      hashlib.sha256(data).hexdigest())

    def info(self) -> dict:
        return {**self.card, "model_sha256": self.model_sha256,
                "runtime": {"onnxruntime": ort.__version__, "providers": self.session.get_providers()}}

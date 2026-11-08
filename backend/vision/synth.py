"""程序合成巡检图片：池面漂浮物（floating_debris）与管廊渗漏（leak）。

没有真实的厂内摄像头数据，漂浮物另有公开数据集做真实域验证（vision/floatwaste.py），
渗漏只有合成数据，指标只能说明模型学会了这里定义的视觉线索，不代表真实管廊的效果。

池面场景：曝气池（褐色、泡沫、气泡羽流）、二沉池（暗绿、平静）、渠道、出水口；池壁、栏杆、
倒影、反光、波纹、落叶和摄像头叠加的时间水印都是不标注的干扰。目标：瓶、袋、泡沫块、树枝、罐、杯、餐盒。
管廊场景：墙面、地面、管道、法兰、支架、灯光。渗漏 = 法兰下方的湿痕、滴痕、地面积水：
表面变暗、边缘更深、积水平滑并有灯光倒影。困难负样本：管道阴影（边缘锐利、无反光）、
锈迹（偏橙）、地面污渍（斑驳、哑光）、干涸水印（偏白）、补漆块、干燥地面的灯光反光。

    uv run --group vision python vision/synth.py                 # train 2000 / val 300 / test 300
    uv run --group vision python vision/synth.py --preview 12    # 只出几张带框预览（依赖 OpenCV）
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 960, 540
CLASSES = ["floating_debris", "leak"]
SPLITS = {"train": 2000, "val": 300, "test": 300}
SPLIT_SEED = {"train": 1, "val": 2, "test": 3}     # 各划分用不同随机流，不共享任何一张图
POOL_SHARE = 0.55            # 其余为管廊
NEGATIVE_SHARE = 0.10        # 无目标图（干扰物照常出现）

POOLS = {
    "aeration": {"base": (118, 98, 66), "var": 30, "foam": 0.8, "bubbles": 0.9, "ripple": 0.9, "scum": 0.1},
    "clarifier": {"base": (62, 78, 68), "var": 18, "foam": 0.1, "bubbles": 0.0, "ripple": 0.4, "scum": 0.5},
    "channel": {"base": (92, 88, 78), "var": 24, "foam": 0.3, "bubbles": 0.2, "ripple": 0.7, "scum": 0.2},
    "outfall": {"base": (55, 66, 78), "var": 22, "foam": 0.1, "bubbles": 0.0, "ripple": 0.6, "scum": 0.2},
}
DEBRIS = ["bottle", "bag", "foam", "branch", "can", "cup", "container"]
PIPE_COLORS = [(52, 92, 150), (60, 118, 82), (128, 130, 128), (165, 45, 40), (120, 92, 70), (190, 170, 60)]


# ---------------------------------------------------------------- 基础工具

def noise(rng, h, w, cell, octaves=4, persistence=0.5):
    """多倍频值噪声，[0,1] 附近。"""
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        c = max(cell / 2 ** o, 2)
        g = Image.fromarray(rng.random((int(h / c) + 2, int(w / c) + 2), dtype=np.float32))
        out += amp * np.asarray(g.resize((w, h), Image.Resampling.BICUBIC), np.float32)
        tot += amp
        amp *= persistence
    return out / tot


def blur(a: np.ndarray, r: float) -> np.ndarray:
    if r <= 0:
        return a
    return cv2.GaussianBlur(np.ascontiguousarray(a, np.float32), (0, 0), r)


def blur_rgb(img: np.ndarray, r: float) -> np.ndarray:
    return blur(img, r) if r > 0 else img


def poly_mask(polys, h=H, w=W, blur_r=0.0) -> np.ndarray:
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for p in polys:
        d.polygon([(float(x), float(y)) for x, y in p], fill=255)
    a = np.asarray(m, np.float32) / 255
    return blur(a, blur_r)


def blob(rng, cx, cy, rx, ry, n=16, jitter=0.35, rot=0.0):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-0.15, 0.15, n)
    r = 1 - jitter * rng.random(n)
    r = (r + np.roll(r, 1) + np.roll(r, -1)) / 3        # 平滑，避免星形
    x, y = rx * r * np.cos(ang), ry * r * np.sin(ang)
    c, s = math.cos(rot), math.sin(rot)
    return [(cx + c * a - s * b, cy + s * a + c * b) for a, b in zip(x, y)]


def gauss_spot(cx, cy, sx, sy, h=H, w=W):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)


def box_of(mask: np.ndarray, thr: float):
    ys, xs = np.nonzero(mask > thr)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def jitter_color(rng, c, s=14, alpha=None):
    out = tuple(int(np.clip(v + rng.normal(0, s), 0, 255)) for v in c)
    return out + ((alpha,) if alpha is not None else ())


# ---------------------------------------------------------------- 漂浮物

def render_debris(rng, kind: str, L: float) -> Image.Image:
    """返回 RGBA，长边约 L 像素（先 3 倍超采样再缩小抗锯齿）。"""
    S = 3
    Lp = max(int(L * S), 12)
    if kind == "bottle":
        wd = max(int(Lp * rng.uniform(0.26, 0.34)), 4)
        col = [(205, 218, 224, 150), (60, 140, 80, 205), (85, 130, 200, 185), (110, 70, 30, 215)][rng.integers(4)]
        im = Image.new("RGBA", (Lp, wd), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        body, neck = int(Lp * 0.72), max(int(wd * 0.42), 2)
        d.rounded_rectangle([0, 0, body, wd - 1], radius=max(wd // 3, 1), fill=jitter_color(rng, col[:3], 10, col[3]))
        d.rectangle([body - 2, (wd - neck) // 2, Lp - int(Lp * 0.06), (wd + neck) // 2], fill=col)
        d.rectangle([Lp - int(Lp * 0.07), (wd - neck) // 2 - 1, Lp - 1, (wd + neck) // 2 + 1],
                    fill=jitter_color(rng, (rng.integers(256), rng.integers(256), rng.integers(256)), 0, 255))
        if rng.random() < 0.6:
            d.rectangle([int(body * 0.25), 0, int(body * 0.65), wd - 1],
                        fill=jitter_color(rng, (rng.integers(256), rng.integers(256), rng.integers(256)), 0, 235))
        d.line([(wd // 3, wd // 4), (body - wd // 3, wd // 4)], fill=(255, 255, 255, 170), width=max(1, wd // 8))
    elif kind == "bag":
        h = int(Lp * rng.uniform(0.55, 1.0))
        col = [(232, 232, 226), (70, 110, 185), (32, 32, 32), (190, 50, 45), (220, 210, 120)][rng.integers(5)]
        im = Image.new("RGBA", (Lp, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.polygon(blob(rng, Lp / 2, h / 2, Lp / 2 - 1, h / 2 - 1, n=14, jitter=0.45),
                  fill=jitter_color(rng, col, 10, int(rng.uniform(170, 235))))
        for _ in range(rng.integers(3, 8)):
            p = [(rng.uniform(0.2, 0.8) * Lp, rng.uniform(0.2, 0.8) * h) for _ in range(2)]
            shade = 255 if rng.random() < 0.5 else 0
            d.line(p, fill=(shade, shade, shade, 70), width=max(1, S))
        im = im.filter(ImageFilter.GaussianBlur(S * 0.7))
    elif kind == "foam":
        h = max(int(Lp * rng.uniform(0.4, 0.8)), 6)
        im = Image.new("RGBA", (Lp, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        pts = [(0, 0), (Lp / 2, 0), (Lp - 1, 0), (Lp - 1, h / 2), (Lp - 1, h - 1), (Lp / 2, h - 1), (0, h - 1), (0, h / 2)]
        pts = [(min(max(x + rng.uniform(-0.08, 0.08) * Lp, 0), Lp - 1), min(max(y + rng.uniform(-0.1, 0.1) * h, 0), h - 1))
               for x, y in pts]
        g = int(rng.uniform(228, 250))
        d.polygon(pts, fill=(g, g, g - 4, 255))
        d.polygon(pts[4:7] + [(pts[6][0], pts[6][1] - h * 0.18), (pts[4][0], pts[4][1] - h * 0.18)],
                  fill=(g - 35, g - 35, g - 38, 255))
        a = np.asarray(im, np.float32)
        a[..., :3] -= rng.random(a.shape[:2])[..., None] * 18
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    elif kind == "branch":
        h = int(Lp * 0.5)
        im = Image.new("RGBA", (Lp, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        col = jitter_color(rng, (92, 66, 40), 15, 255)
        n = int(rng.integers(3, 7))
        pts = [(Lp * i / n, h / 2 + rng.uniform(-0.12, 0.12) * h) for i in range(n + 1)]
        wmain = max(int(Lp * rng.uniform(0.04, 0.08)), 2)
        d.line(pts, fill=col, width=wmain, joint="curve")
        for _ in range(rng.integers(1, 4)):
            x0, y0 = pts[rng.integers(1, n)]
            ang = rng.uniform(-1.2, 1.2) + (0 if rng.random() < 0.5 else math.pi)
            ln = Lp * rng.uniform(0.15, 0.35)
            d.line([(x0, y0), (x0 + ln * math.cos(ang) * 0.6, y0 + ln * math.sin(ang) * 0.6)],
                   fill=col, width=max(wmain * 2 // 3, 1))
    elif kind == "can":
        h = max(int(Lp * 0.55), 6)
        base = np.array([(185, 40, 40), (190, 192, 196), (40, 90, 170), (40, 140, 70), (230, 200, 60)][rng.integers(5)], np.float32)
        v = np.linspace(-1, 1, h)[:, None]
        shade = 0.55 + 0.6 * np.exp(-((v + 0.3) / 0.35) ** 2)
        rgb = np.clip(base * shade[..., None] * np.ones((1, Lp, 1)), 0, 255)
        alpha = np.full((h, Lp, 1), 255, np.float32)
        im = Image.fromarray(np.concatenate([rgb, alpha], -1).astype(np.uint8))
        d = ImageDraw.Draw(im)
        d.ellipse([Lp - h * 0.35, 0, Lp - 1, h - 1], fill=(205, 205, 205, 255))
        if rng.random() < 0.5:
            d.rectangle([int(Lp * 0.3), 0, int(Lp * 0.45), h - 1], fill=jitter_color(rng, (240, 240, 240), 10, 255))
    elif kind == "cup":
        h = max(int(Lp * 0.75), 6)
        col = [(240, 238, 230), (150, 105, 70), (200, 50, 45), (225, 210, 180)][rng.integers(4)]
        im = Image.new("RGBA", (Lp, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.polygon([(0, h * 0.22), (Lp * 0.9, 0), (Lp * 0.9, h - 1), (0, h * 0.78)], fill=jitter_color(rng, col, 8, 255))
        d.ellipse([Lp * 0.8, 0, Lp - 1, h - 1], fill=jitter_color(rng, col, 8, 255))
        d.ellipse([Lp * 0.84, h * 0.12, Lp - 3, h * 0.88], fill=(70, 60, 50, 255))
    else:  # container
        h = max(int(Lp * rng.uniform(0.55, 0.75)), 6)
        col = [(238, 238, 235), (30, 30, 32), (200, 210, 215)][rng.integers(3)]
        a = 255 if col != (200, 210, 215) else 160
        im = Image.new("RGBA", (Lp, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([0, 0, Lp - 1, h - 1], radius=max(h // 8, 1), fill=jitter_color(rng, col, 6, a))
        inset = max(int(h * 0.12), 2)
        rim = tuple(min(255, c + 30) if c < 128 else c - 35 for c in col)
        d.rounded_rectangle([inset, inset, Lp - 1 - inset, h - 1 - inset], radius=max(h // 10, 1),
                            outline=rim + (a,), width=max(S, 2))
    w, h = im.size
    im = im.resize((max(round(w / S), 2), max(round(h / S), 2)), Image.Resampling.LANCZOS)
    return im.rotate(float(rng.uniform(0, 360)), resample=Image.Resampling.BICUBIC, expand=True)


def composite(img, obj: Image.Image, x: int, y: int, *, tint=None, tint_amt=0.0, submerge=None, shadow=0.0):
    """把 RGBA 贴到 img（float HxWx3）的 (x, y)；返回贴上后可见的 alpha 全图（用于取框）。"""
    a = np.asarray(obj, np.float32) / 255
    rgb, alpha = a[..., :3] * 255, a[..., 3].copy()
    h, w = alpha.shape
    if submerge is not None:      # 半浸没：沿某个方向渐隐
        ang, floor = submerge
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = ((xx - w / 2) * math.cos(ang) + (yy - h / 2) * math.sin(ang)) / (max(w, h) / 2) * 0.5 + 0.5
        alpha *= np.clip(1 - t * (1 - floor), floor, 1)
    if tint is not None:
        rgb = rgb * (1 - tint_amt) + np.asarray(tint, np.float32) * tint_amt
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return None
    sl = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    if shadow:
        sh = blur(alpha, 2.0)
        dy, dx = 3, 2
        sy0, sx0, sy1, sx1 = max(y + dy, 0), max(x + dx, 0), min(y + dy + h, H), min(x + dx + w, W)
        if sy1 > sy0 and sx1 > sx0:
            img[sy0:sy1, sx0:sx1] *= (1 - shadow * sh[sy0 - y - dy:sy1 - y - dy, sx0 - x - dx:sx1 - x - dx])[..., None]
    reg = img[y0:y1, x0:x1]
    al = alpha[sl][..., None]
    img[y0:y1, x0:x1] = reg * (1 - al) + rgb[sl] * al
    vis = np.zeros((H, W), np.float32)
    vis[y0:y1, x0:x1] = alpha[sl]
    return vis


def water_scene(rng, n_targets: int):
    kind = list(POOLS)[rng.integers(len(POOLS))]
    cfg = POOLS[kind]
    base = np.asarray(cfg["base"], np.float32) + rng.normal(0, 12, 3)
    img = base + (noise(rng, H, W, 220, 4) - 0.5)[..., None] * cfg["var"] * 2
    t = np.linspace(1, 0, H, dtype=np.float32)[:, None, None]
    img += t ** 2 * rng.uniform(0, 45)                       # 远处天空反射更亮
    distractors = []
    if rng.random() < cfg["ripple"]:
        yy = np.arange(H, dtype=np.float32)[:, None]
        xx = np.arange(W, dtype=np.float32)[None, :]
        freq = rng.uniform(0.08, 0.2) * (1 + 2.5 * (1 - yy / H))
        phase = yy * freq + noise(rng, H, W, 120, 3) * rng.uniform(6, 18) + xx * rng.uniform(-0.01, 0.01)
        lines = blur(np.clip((np.sin(phase) - 0.6) / 0.4, 0, 1), rng.uniform(0.8, 2.0))
        img += (lines * rng.uniform(8, 24))[..., None]
        distractors.append("ripple")
    if rng.random() < cfg["foam"]:
        m = np.clip((noise(rng, H, W, 70, 4) - rng.uniform(0.56, 0.68)) / 0.08, 0, 1)
        m *= 0.6 + 0.4 * noise(rng, H, W, 8, 2)
        img = img * (1 - m[..., None]) + np.array([214, 208, 192], np.float32) * m[..., None]
        distractors.append("foam_patch")
    if rng.random() < cfg["bubbles"]:
        layer = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(layer)
        for _ in range(rng.integers(1, 4)):
            cx, cy = rng.uniform(0, W), rng.uniform(H * 0.25, H)
            for _ in range(rng.integers(80, 350)):
                bx, by, r = cx + rng.normal(0, 60), cy + rng.normal(0, 30), rng.uniform(0.8, 3.5)
                d.ellipse([bx - r, by - r, bx + r, by + r], outline=int(rng.uniform(120, 230)))
        b = np.asarray(layer, np.float32) / 255
        img = img + (235 - img) * b[..., None] * 0.7
        distractors.append("bubbles")
    if rng.random() < cfg["scum"]:
        m = np.clip((noise(rng, H, W, 90, 3) - 0.6) / 0.1, 0, 1) * 0.5
        img = img * (1 - m[..., None]) + np.array([88, 96, 52], np.float32) * m[..., None]
        distractors.append("scum")
    for _ in range(rng.integers(0, 4)):                       # 反光
        g = gauss_spot(rng.uniform(0, W), rng.uniform(0, H * 0.7), rng.uniform(20, 120), rng.uniform(4, 18))
        img += g[..., None] * rng.uniform(50, 140)
        distractors.append("glare")
    leaves = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(leaves)
    n_leaves = int(rng.integers(0, 45))
    for _ in range(n_leaves):                                 # 落叶等自然漂浮物，不标注
        x, y, ln, ang = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(2, 7), rng.uniform(0, np.pi)
        col = jitter_color(rng, [(110, 80, 35), (70, 100, 40), (170, 140, 60)][rng.integers(3)], 15, 230)
        d.line([(x, y), (x + ln * math.cos(ang), y + ln * math.sin(ang))], fill=col, width=int(rng.integers(1, 3)))
    if n_leaves:
        composite(img, leaves, 0, 0)
        distractors.append("leaves")

    top = np.zeros(W, np.float32)                             # 每列水面上沿
    bottom = np.full(W, H, np.float32)
    if rng.random() < 0.75:                                   # 远侧池壁 + 栏杆
        y0, y1 = rng.uniform(0.05, 0.3) * H, rng.uniform(0.05, 0.3) * H
        top = np.linspace(y0, y1, W).astype(np.float32)
        wall = rng.uniform(135, 190) + (noise(rng, H, W, 60, 4) - 0.5) * 50
        yy = np.arange(H, dtype=np.float32)[:, None]
        m = (yy < top[None, :]).astype(np.float32)
        img = img * (1 - m[..., None]) + (wall[..., None] * np.array([1, 0.99, 0.95])) * m[..., None]
        wet = ((yy >= top[None, :] - rng.uniform(3, 9)) & (yy < top[None, :])).astype(np.float32)
        img *= (1 - 0.35 * wet)[..., None]
        if rng.random() < 0.6:
            rail = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(rail)
            col = jitter_color(rng, [(220, 190, 40), (60, 100, 170), (190, 190, 195)][rng.integers(3)], 10, 255)
            th, hgt = int(rng.integers(3, 8)), rng.uniform(25, 60)
            for k in (1.0, 0.55):
                d.line([(0, y0 - hgt * k), (W, y1 - hgt * k)], fill=col, width=th)
            for x in np.arange(rng.uniform(0, 80), W, rng.uniform(60, 140)):
                yb = y0 + (y1 - y0) * x / W
                d.line([(x, yb), (x, yb - hgt)], fill=col, width=th)
            composite(img, rail, 0, 0)
            distractors.append("railing")
        distractors.append("far_wall")
    if rng.random() < 0.3:                                    # 近侧走道
        yb = rng.uniform(0.82, 0.93) * H
        bottom[:] = yb
        walk = rng.uniform(120, 170) + (noise(rng, H, W, 30, 3) - 0.5) * 40
        m = (np.arange(H)[:, None] >= yb).astype(np.float32) * np.ones((1, W), np.float32)
        img = img * (1 - m[..., None]) + walk[..., None] * m[..., None]
        distractors.append("near_walkway")

    boxes, objects = [], []
    tries = 0
    while len(boxes) < n_targets and tries < n_targets * 25:
        tries += 1
        kind_o = DEBRIS[rng.integers(len(DEBRIS))]
        cx = rng.uniform(0, W)
        col = int(min(max(cx, 0), W - 1))
        if bottom[col] - top[col] < 40:
            continue
        cy = rng.uniform(top[col] + 10, bottom[col] - 5)
        persp = 0.45 + 0.75 * (cy - top[col]) / max(bottom[col] - top[col], 1)
        L = math.exp(rng.uniform(math.log(14), math.log(120))) * persp
        obj = render_debris(rng, kind_o, max(L, 10))
        ow, oh = obj.size
        x, y = int(cx - ow / 2), int(cy - oh / 2)
        a = np.asarray(obj)[..., 3] > 77
        ys, xs = np.nonzero(a)
        if len(xs) == 0:
            continue
        cand = [x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1]
        cand = [max(cand[0], 0), max(cand[1], int(top[col])), min(cand[2], W), min(cand[3], int(bottom[col]))]
        if cand[2] - cand[0] < 6 or cand[3] - cand[1] < 6 or any(iou(cand, b) > 0.05 for b in boxes):
            continue
        submerge = (rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 0.8)) if rng.random() < 0.4 else None
        vis = composite(img, obj, x, y, tint=base, tint_amt=rng.uniform(0, 0.25), submerge=submerge,
                        shadow=rng.uniform(0.1, 0.35))
        if vis is None:
            continue
        bb = box_of(vis, 0.3)
        if bb is None or bb[2] - bb[0] < 5 or bb[3] - bb[1] < 5:
            continue
        boxes.append(bb)
        objects.append({"cls": 0, "kind": kind_o, "box": bb})
    return img, objects, {"scene": "pool", "pool": kind, "distractors": distractors}


# ---------------------------------------------------------------- 管廊渗漏

def streak_poly(rng, x0, y0, length, w0):
    n = max(int(length / 8), 3)
    ys = np.linspace(y0, y0 + length, n)
    drift = np.cumsum(rng.normal(0, 1.2, n))
    widths = w0 * (1 - 0.55 * np.linspace(0, 1, n)) * (0.75 + 0.5 * rng.random(n))
    widths[-1] *= 1.6                                         # 末端积滴
    left = [(x0 + drift[i] - widths[i] / 2, ys[i]) for i in range(n)]
    right = [(x0 + drift[i] + widths[i] / 2, ys[i]) for i in range(n)][::-1]
    return left + right


def apply_wet(img, M, rng, *, puddle: bool):
    Mb = blur(M, 1.5)
    if puddle:                                                # 积水表面平滑
        sm = rng.uniform(0.4, 0.75)
        img[:] = img * (1 - sm * Mb[..., None]) + blur_rgb(img, 3) * (sm * Mb[..., None])
    dark = rng.uniform(0.45, 0.72)
    img *= (1 - (1 - dark) * Mb)[..., None]
    img *= (1 - rng.uniform(0.1, 0.25) * np.clip(Mb * (1 - Mb) * 4, 0, 1))[..., None]   # 湿痕边缘更深
    if puddle:
        b = box_of(M, 0.35)
        if b is not None:
            cx, cy = rng.uniform(b[0], b[2]), rng.uniform(b[1], b[3])
            g = gauss_spot(cx, cy, (b[2] - b[0]) * rng.uniform(0.08, 0.25), (b[3] - b[1]) * rng.uniform(0.05, 0.15))
            img += (g * Mb)[..., None] * rng.uniform(50, 130)    # 灯光倒影
        img += Mb[..., None] * np.array([0, 2, 7], np.float32)
    elif rng.random() < 0.7:                                  # 湿面泛光
        img += (np.clip(noise(rng, H, W, 6, 2) - 0.55, 0, 1) * 4 * M)[..., None] * rng.uniform(15, 40)


def gallery_scene(rng, n_targets: int):
    horizon = int(H * rng.uniform(0.52, 0.72))
    yy = np.arange(H, dtype=np.float32)[:, None]
    floor_m = (yy >= horizon).astype(np.float32) * np.ones((1, W), np.float32)
    wall = rng.uniform(140, 195) + (noise(rng, H, W, 90, 5) - 0.5) * rng.uniform(25, 45) + (noise(rng, H, W, 4, 2) - 0.5) * 14
    floor = rng.uniform(95, 150) + (noise(rng, H, W, 60, 5) - 0.5) * rng.uniform(25, 45) + (noise(rng, H, W, 3, 2) - 0.5) * 18
    tint = np.array([1, rng.uniform(0.97, 1.02), rng.uniform(0.92, 1.03)], np.float32)
    img = (wall[..., None] * (1 - floor_m[..., None]) + floor[..., None] * floor_m[..., None]) * tint
    lines = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(lines)
    vp = (W * rng.uniform(0.3, 0.7), horizon - rng.uniform(50, 150))
    for xb in np.linspace(-W * 0.5, W * 1.5, int(rng.integers(3, 7))):
        t = (horizon - vp[1]) / (H - vp[1])
        d.line([(vp[0] + (xb - vp[0]) * t, horizon), (xb, H)], fill=90, width=2)
    for x in np.arange(rng.uniform(0, 200), W, rng.uniform(150, 300)):
        d.line([(x, 0), (x, horizon)], fill=50, width=1)
    d.line([(0, horizon), (W, horizon)], fill=150, width=int(rng.integers(2, 5)))
    img *= (1 - 0.35 * np.asarray(lines, np.float32)[..., None] / 255)

    pipes, flanges, distractors = [], [], ["pipe_shadow"]
    for _ in range(rng.integers(1, 4)):
        D = rng.uniform(26, 80)
        yc = rng.uniform(H * 0.12, horizon - D * 0.7)
        if any(abs(yc - p[0]) < (D + p[1]) * 0.75 for p in pipes):
            continue
        pipes.append((yc, D))
    for yc, D in pipes:
        sh_h = D * rng.uniform(0.3, 1.0)
        sm = ((yy >= yc + D / 2) & (yy < min(yc + D / 2 + sh_h, horizon))).astype(np.float32) * np.ones((1, W))
        img *= (1 - rng.uniform(0.15, 0.35) * blur(sm, rng.uniform(0, 3)))[..., None]   # 管道阴影：困难负样本
        base = np.asarray(jitter_color(rng, PIPE_COLORS[rng.integers(len(PIPE_COLORS))], 10), np.float32)
        y0, y1 = int(yc - D / 2), int(yc + D / 2)
        v = np.linspace(-1, 1, y1 - y0)[:, None, None]
        shade = 0.45 + 0.65 * np.exp(-((v + 0.4) / 0.45) ** 2) - 0.25 * np.clip(v, 0, 1)
        img[y0:y1] = base * shade * np.ones((1, W, 1))
        for _ in range(rng.integers(1, 4)):
            fx, fw = rng.uniform(40, W - 40), D * 0.22
            fy0, fy1 = int(yc - D * 0.65), int(yc + D * 0.65)
            vv = np.linspace(-1, 1, fy1 - fy0)[:, None, None]
            fs = 0.4 + 0.6 * np.exp(-((vv + 0.4) / 0.5) ** 2)
            img[fy0:fy1, int(fx - fw / 2):int(fx + fw / 2)] = base * 0.85 * fs
            for by in (fy0 + 3, fy1 - 4):
                img[by - 2:by + 2, int(fx - fw / 2) - 2:int(fx + fw / 2) + 2] *= 0.6
            flanges.append((fx, yc, D))
        if rng.random() < 0.5:                                # 支架
            sx = rng.uniform(30, W - 30)
            sw = rng.uniform(8, 16)
            img[int(yc + D / 2):horizon + 8, int(sx - sw / 2):int(sx + sw / 2)] = rng.uniform(70, 100)
    for _ in range(rng.integers(1, 4)):                       # 顶灯照明
        img += gauss_spot(rng.uniform(0, W), rng.uniform(0, H * 0.4), rng.uniform(80, 260), rng.uniform(60, 160))[..., None] * rng.uniform(15, 50)

    # 困难负样本：先画，避免被误当成渗漏的一部分
    n_neg = int(rng.integers(1, 4)) + (2 if n_targets == 0 else 0)
    for _ in range(n_neg):
        k = ["rust", "dirt", "dry_mark", "paint", "cast_shadow", "floor_glare"][rng.integers(6)]
        if k == "rust" and flanges:
            fx, yc, D = flanges[rng.integers(len(flanges))]
            M = poly_mask([streak_poly(rng, fx + rng.uniform(-D * 0.1, D * 0.1), yc + D * 0.6,
                                       rng.uniform(30, max(horizon - yc, 40)), rng.uniform(4, 16))], blur_r=1.5)
            M *= np.clip(1 - (yy - yc) / rng.uniform(80, 250), 0.3, 1)
            img *= (1 - M[..., None] * (1 - np.array([0.92, 0.7, 0.5]) * rng.uniform(0.85, 1)))
        elif k == "dirt":
            cx, cy = rng.uniform(0, W), rng.uniform(horizon + 10, H)
            M = poly_mask([blob(rng, cx, cy, rng.uniform(30, 140), rng.uniform(10, 45))], blur_r=3)
            M *= noise(rng, H, W, 6, 3)
            img = img * (1 - 0.45 * M[..., None]) + np.array([95, 82, 62], np.float32) * 0.45 * M[..., None]
        elif k == "dry_mark":
            cx, cy = rng.uniform(0, W), rng.uniform(H * 0.1, H)
            M = poly_mask([blob(rng, cx, cy, rng.uniform(25, 100), rng.uniform(15, 60))], blur_r=2)
            ring = np.clip(M * (1 - M) * 4, 0, 1)
            img += (255 - img) * (0.25 * ring + 0.06 * M)[..., None]
        elif k == "paint":
            x0, y0 = rng.uniform(0, W - 60), rng.uniform(0, horizon - 30)
            img[int(y0):int(y0 + rng.uniform(25, 90)), int(x0):int(x0 + rng.uniform(40, 160))] *= rng.uniform(0.82, 1.15)
        elif k == "cast_shadow":
            x0, y0 = rng.uniform(0, W), rng.uniform(H * 0.2, H)
            ww, hh, sk = rng.uniform(30, 160), rng.uniform(20, 120), rng.uniform(-40, 40)
            M = poly_mask([[(x0, y0), (x0 + ww, y0), (x0 + ww + sk, y0 + hh), (x0 + sk, y0 + hh)]], blur_r=rng.uniform(0, 1.5))
            img *= (1 - rng.uniform(0.25, 0.5) * M)[..., None]
        else:
            g = gauss_spot(rng.uniform(0, W), rng.uniform(horizon, H), rng.uniform(20, 90), rng.uniform(6, 25))
            img += g[..., None] * rng.uniform(40, 110)
        distractors.append(k)

    objects = []
    for _ in range(n_targets):
        kind = ["streak", "puddle", "streak_puddle", "patch"][rng.integers(4)]
        if not flanges:
            kind = "puddle"
        fx, yc, D = flanges[rng.integers(len(flanges))] if flanges else (rng.uniform(80, W - 80), horizon - 40, 40)
        polys = []
        y_src = yc + D * 0.6
        if kind in ("streak", "streak_puddle"):
            length = (horizon - y_src + rng.uniform(0, 15)) if kind == "streak_puddle" else rng.uniform(30, max(horizon - y_src, 35))
            polys.append(streak_poly(rng, fx + rng.uniform(-D * 0.1, D * 0.1), y_src, length, rng.uniform(6, 30)))
            if rng.random() < 0.35:
                polys.append(streak_poly(rng, fx + rng.uniform(-D * 0.3, D * 0.3), y_src, length * rng.uniform(0.4, 0.9),
                                         rng.uniform(3, 10)))
        if kind in ("puddle", "streak_puddle"):
            cy = horizon + rng.uniform(8, max(min(H - horizon - 20, 120), 10))
            rx = rng.uniform(40, 160) * (0.6 + 0.6 * (cy - horizon) / max(H - horizon, 1))
            polys.append(blob(rng, fx + rng.uniform(-30, 30), cy, rx, rx * rng.uniform(0.2, 0.4), n=18, jitter=0.35))
        if kind == "patch":
            polys.append(blob(rng, fx, y_src + rng.uniform(0, 40), rng.uniform(25, 80), rng.uniform(30, 100), n=18, jitter=0.4))
        M = poly_mask(polys, blur_r=1.0)
        if kind == "patch":
            M *= np.clip(0.6 + noise(rng, H, W, 14, 3), 0, 1)
        bb = box_of(M, 0.35)
        if bb is None or bb[2] - bb[0] < 6 or bb[3] - bb[1] < 6 or any(iou(bb, o["box"]) > 0.2 for o in objects):
            continue
        apply_wet(img, M, rng, puddle=kind != "streak" and kind != "patch")
        objects.append({"cls": 1, "kind": kind, "box": bb})
    return img, objects, {"scene": "gallery", "pipes": len(pipes), "distractors": distractors}


# ---------------------------------------------------------------- 全局成像效果

def camera(rng, img, meta) -> Image.Image:
    img = (img - 128) * rng.uniform(0.8, 1.2) + 128 * rng.uniform(0.75, 1.2)
    img *= np.array([1 + rng.normal(0, 0.05), 1, 1 + rng.normal(0, 0.05)], np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    img *= (1 - rng.uniform(0, 0.3) * r2 / 2)[..., None]
    if rng.random() < 0.12:                                   # 夜间红外
        g = img.mean(-1, keepdims=True)
        img = g * np.array([0.95, 1.0, 0.92], np.float32)
        meta["ir"] = True
    img = blur_rgb(img, rng.uniform(0, 1.2)) if rng.random() < 0.6 else img
    img += rng.normal(0, rng.uniform(0, 7), img.shape)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    if rng.random() < 0.4:                                    # 摄像头时间水印
        d = ImageDraw.Draw(im)
        try:
            font = ImageFont.load_default(size=int(rng.integers(14, 22)))
        except TypeError:
            font = ImageFont.load_default()
        text = f"CAM{rng.integers(1, 40):02d}  2026-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d} " \
               f"{rng.integers(0, 24):02d}:{rng.integers(0, 60):02d}:{rng.integers(0, 60):02d}"
        y = 6 if rng.random() < 0.5 else H - 28
        if rng.random() < 0.5:
            d.rectangle([0, y - 4, W, y + 22], fill=(0, 0, 0))
        d.text((10, y), text, fill=(255, 255, 255), font=font)
        meta["osd"] = True
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(rng.integers(55, 95)))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def make(seed: int, split: str, idx: int):
    rng = np.random.default_rng([seed, SPLIT_SEED[split], idx])
    pool = rng.random() < POOL_SHARE
    negative = rng.random() < NEGATIVE_SHARE
    n = 0 if negative else int(rng.integers(1, 9) if pool else rng.integers(1, 3))
    img, objects, meta = water_scene(rng, n) if pool else gallery_scene(rng, n)
    return camera(rng, img, meta), objects, meta


def _job(args):
    seed, split, idx, out = args
    im, objects, meta = make(seed, split, idx)
    stem = f"{split}_{idx:05d}"
    im.save(Path(out) / "images" / split / f"{stem}.jpg", quality=95)
    (Path(out) / "labels" / split / f"{stem}.txt").write_text("".join(
        f"{o['cls']} {(o['box'][0] + o['box'][2]) / 2 / W:.6f} {(o['box'][1] + o['box'][3]) / 2 / H:.6f} "
        f"{(o['box'][2] - o['box'][0]) / W:.6f} {(o['box'][3] - o['box'][1]) / H:.6f}\n" for o in objects))
    return {"file": f"{stem}.jpg", **meta, "objects": objects}


def preview(seed: int, n: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        im, objects, meta = make(seed, "train", i)
        d = ImageDraw.Draw(im)
        for o in objects:
            d.rectangle(o["box"], outline=["red", "yellow"][o["cls"]], width=2)
        im.save(out / f"preview_{i:02d}_{meta['scene']}.jpg", quality=90)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data/vision/synth")
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--preview", type=int, default=0)
    p.add_argument("--workers", type=int, default=max((os.cpu_count() or 4) - 2, 1))
    a = p.parse_args(argv)
    if a.preview:
        preview(a.seed, a.preview, a.out / "preview")
        return
    manifest = {"classes": CLASSES, "seed": a.seed, "size": [W, H], "splits": {}}
    for split, n in SPLITS.items():
        (a.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(a.workers) as ex:
            items = list(ex.map(_job, [(a.seed, split, i, str(a.out)) for i in range(n)], chunksize=8))
        counts = {c: sum(1 for it in items for o in it["objects"] if o["cls"] == k) for k, c in enumerate(CLASSES)}
        manifest["splits"][split] = {"images": n, "boxes": counts,
                                     "negatives": sum(1 for it in items if not it["objects"]),
                                     "scenes": {s: sum(1 for it in items if it["scene"] == s) for s in ("pool", "gallery")},
                                     "items": items}
        print(split, n, counts, flush=True)
    (a.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())

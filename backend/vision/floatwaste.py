"""FloatWaste 公开漂浮物数据集 → 分层子集（YOLO 格式）。

数据集：OliviaWuYJ/sc-floatwaste（Hugging Face），CC BY 4.0，DOI 10.5281/zenodo.21934286。
全量 7906 张、38 GB（Hobart 多为 4K PNG），这里只按地区分层抽一部分：下载后在内存里缩到长边 1280、
存 JPEG，原图不落盘；25 个细类合并为单类 floating_debris；数据集自带的无标注负样本全部保留。
沿用数据集自己的 train/val 划分，不重新切分。

    uv run python -I vision/floatwaste.py --train 500 --val 200

下载内容按不可信数据处理：只用 Pillow 解码像素，不执行、不解包。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

BASE = "https://huggingface.co/datasets/OliviaWuYJ/sc-floatwaste/resolve/main/"
SOURCE = {"dataset": "OliviaWuYJ/sc-floatwaste", "url": "https://huggingface.co/datasets/OliviaWuYJ/sc-floatwaste",
          "doi": "10.5281/zenodo.21934286", "license": "CC BY 4.0",
          "note": "子集按地区分层抽样，长边缩至 1280 并转 JPEG，25 类合并为 floating_debris"}
# 各地区占比：Hobart 多为 4K PNG（单张约 6 MB），本机带宽约 300 KB/s，压低其占比以控制下载时间
REGION_SHARE = {"Hobart": 0.4, "NSW": 0.32, "Indonesia": 0.28}
MAX_SIDE = 1280
Image.MAX_IMAGE_PIXELS = 60_000_000


def pick(images: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_region: dict[str, list[dict]] = defaultdict(list)
    for im in images:
        by_region[im["region"]].append(im)
    chosen = []
    for region, share in REGION_SHARE.items():
        pool = sorted(by_region[region], key=lambda im: im["id"])
        rng.shuffle(pool)
        chosen += pool[:round(n * share)]
    return chosen


_gate = threading.Lock()
_next_at = [0.0]
MIN_INTERVAL = 0.8      # 全局请求间隔（秒）：匿名访问 Hugging Face 有频率限制


def fetch(path: str, retries: int = 8) -> bytes:
    for attempt in range(retries):
        with _gate:                                   # 所有线程共享的节流阀
            wait = _next_at[0] - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _next_at[0] = time.monotonic() + MIN_INTERVAL
        try:
            req = urllib.request.Request(BASE + path, headers={"User-Agent": "shuiwu-vision-demo"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if attempt == retries - 1 or e.code not in (429, 500, 502, 503, 504):
                raise
            delay = float(e.headers.get("Retry-After") or 0) or 30 * (attempt + 1)
            with _gate:                               # 被限流时全体暂停
                _next_at[0] = max(_next_at[0], time.monotonic() + delay)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    raise AssertionError


def convert(im: dict, anns: list[dict], raw: bytes, out_img: Path, out_lbl: Path) -> dict:
    with Image.open(io.BytesIO(raw)) as src:
        src.load()
        img = src.convert("RGB")
        # 标注里的宽高是基准；像素尺寸不符时按 EXIF 方向转正后再比，仍不符就丢弃，不猜旋转方向
        if img.size != (im["width"], im["height"]):
            img = ImageOps.exif_transpose(src).convert("RGB")
    if img.size != (im["width"], im["height"]):
        raise ValueError(f"尺寸不符：{img.size} vs {(im['width'], im['height'])}")
    w, h = img.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1:
        img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    img.save(out_img, "JPEG", quality=90)
    lines = []
    for a in anns:
        x, y, bw, bh = a["bbox"]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        lines.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
    return {"id": im["id"], "file": out_img.name, "source": im["file_name"], "region": im["region"],
            "orig_size": [w, h], "scale": round(scale, 5), "boxes": len(lines),
            "orig_categories": sorted({a["category_id"] for a in anns})}


def build(ann_dir: Path, out: Path, counts: dict[str, int], seed: int, workers: int) -> None:
    manifest = {"source": SOURCE, "classes": ["floating_debris"], "seed": seed, "splits": {}}
    for split, n in counts.items():
        if n <= 0:
            continue
        d = json.loads((ann_dir / f"instances_{split}.json").read_text())
        anns: dict[int, list[dict]] = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
        negatives = [im for im in d["images"] if not anns[im["id"]]]
        chosen = {im["id"]: im for im in pick(d["images"], n, seed) + negatives}
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        done, failed = [], []
        todo = []
        for im in chosen.values():
            stem = f"{im['region'].lower()}_{im['id']:05d}"
            img_p, lbl_p = out / "images" / split / f"{stem}.jpg", out / "labels" / split / f"{stem}.txt"
            todo.append((im, img_p, lbl_p))

        def job(t):
            im, img_p, lbl_p = t
            if img_p.exists() and lbl_p.exists():   # 断点续传：已转换的不重下
                with Image.open(img_p) as cached:
                    cw, ch = cached.size
                return {"id": im["id"], "file": img_p.name, "source": im["file_name"], "region": im["region"],
                        "orig_size": [im["width"], im["height"]],
                        "scale": round(min(1.0, MAX_SIDE / max(im["width"], im["height"])), 5),
                        "boxes": sum(1 for _ in lbl_p.read_text().splitlines()), "cached_size": [cw, ch],
                        "orig_categories": sorted({a["category_id"] for a in anns[im["id"]]})}
            return convert(im, anns[im["id"]], fetch(im["file_name"]), img_p, lbl_p)

        t0 = time.time()
        with cf.ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(job, t): t for t in todo}
            for k, f in enumerate(cf.as_completed(futs), 1):
                try:
                    done.append(f.result())
                except Exception as e:
                    failed.append({"source": futs[f][0]["file_name"], "error": repr(e)[:200]})
                if k % 20 == 0 or k == len(todo):
                    print(f"[{split}] {k}/{len(todo)} 成功 {len(done)} 失败 {len(failed)} "
                          f"{time.time() - t0:.0f}s", flush=True)
        keep = {t[1].name for t in todo}
        stale = [f for f in (out / "images" / split).glob("*.jpg") if f.name not in keep]
        for f in stale:                               # 调整抽样规模后，不在本次子集里的旧文件要删掉，否则会混进训练
            f.unlink()
            (out / "labels" / split / f"{f.stem}.txt").unlink(missing_ok=True)
        if stale:
            print(f"[{split}] 删除不在本次子集中的旧文件 {len(stale)} 个")
        for r in done:
            r.pop("cached_size", None)
        done.sort(key=lambda r: r["id"])
        regions = defaultdict(int)
        for r in done:
            regions[r["region"]] += 1
        manifest["splits"][split] = {"images": len(done), "boxes": sum(r["boxes"] for r in done),
                                     "negatives": sum(1 for r in done if r["boxes"] == 0),
                                     "regions": dict(regions), "failed": failed, "items": done}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: {x: v[x] for x in ("images", "boxes", "negatives", "regions")} | {"failed": len(v["failed"])}
                      for k, v in manifest["splits"].items()}, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    here = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, default=here / "data/floatwaste_raw/annotations")
    p.add_argument("--out", type=Path, default=here / "data/vision/floatwaste")
    p.add_argument("--train", type=int, default=500)
    p.add_argument("--val", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    build(a.annotations, a.out, {"train": a.train, "val": a.val}, a.seed, a.workers)


if __name__ == "__main__":
    sys.exit(main())

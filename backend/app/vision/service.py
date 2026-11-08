"""推理服务：限流 + 线程池执行，事件循环不被 CPU 推理阻塞。

onnxruntime 的 session.run 释放 GIL、可多线程并发调用同一个 session。并发数（信号量）× 每次推理的
算子线程数（intra_op_num_threads）应不超过可用核数。默认 2×2 兼顾单请求延迟与吞吐；M3 上实测 4×1 吞吐
约高 17%、单请求延迟约翻倍，1×4 单请求最快、吞吐最低（evals/vision/results/report.md）。
排队时间单独计时（timings_ms.queue），压测时能区分「算得慢」和「在排队」。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from app.vision.detector import Detector, Result

SAMPLE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}\.(?:jpg|jpeg|png)$")


@dataclass(frozen=True)
class Sample:
    name: str
    scene: str
    source: str
    camera: str
    note: str
    attribution: str | None
    description: str = ""

    def out(self) -> dict:
        return {"name": self.name, "scene": self.scene, "source": self.source, "camera": self.camera,
                "description": self.description, "note": self.note, "attribution": self.attribution,
                "url": f"/api/vision/samples/{self.name}"}


def load_samples(samples_dir: Path) -> dict[str, Sample]:
    index = samples_dir / "samples.json"
    if not index.exists():
        return {}
    out = {}
    for s in json.loads(index.read_text(encoding="utf-8")):
        if SAMPLE_NAME.match(s["name"]) and (samples_dir / s["name"]).is_file():
            out[s["name"]] = Sample(s["name"], s["scene"], s["source"], s["camera"], s.get("note", ""),
                                    s.get("attribution"), s.get("description", ""))
    return out


class VisionService:
    def __init__(self, detector: Detector, *, concurrency: int = 2, threads: int = 2,
                 samples_dir: Path | None = None):
        self.detector = detector
        self.concurrency, self.threads = concurrency, threads
        self._sem = asyncio.Semaphore(concurrency)
        self.samples_dir = samples_dir
        self.samples = load_samples(samples_dir) if samples_dir else {}
        self.model_id = detector.model_sha256[:12]

    @classmethod
    def load(cls, model_path: Path, *, concurrency: int, threads: int, samples_dir: Path | None) -> VisionService:
        return cls(Detector.load(model_path, threads), concurrency=concurrency, threads=threads,
                   samples_dir=samples_dir)

    async def detect(self, data: bytes, conf: float | None = None) -> Result:
        t0 = time.perf_counter()
        async with self._sem:
            t1 = time.perf_counter()
            res = await asyncio.to_thread(self.detector.detect_bytes, data, conf)
        res.timings_ms["queue"] = round((t1 - t0) * 1000, 2)
        res.timings_ms["total"] = round((time.perf_counter() - t0) * 1000, 2)
        return res

    def sample_path(self, name: str) -> Path | None:
        s = self.samples.get(name)
        return self.samples_dir / s.name if s and self.samples_dir else None

    def info(self) -> dict:
        return {**self.detector.info(), "model_id": self.model_id,
                "serving": {"concurrency": self.concurrency, "intra_op_threads": self.threads},
                "samples": len(self.samples)}

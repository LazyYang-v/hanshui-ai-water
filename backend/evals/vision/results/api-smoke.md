# 视觉巡检 HTTP 验收记录

生成于 2026-11-08T03:51:24；独立测试库 `shuiwu_test`，演示时钟从 2026-11-05T09:00:00+08:00 起由脚本设置；模型 `336ebeba28ef`（e2_mixed）。由 `evals/vision/api_smoke.py` 生成。

### 模型卡  `GET /api/vision/model` → 200

```json
{
 "name": "shuiwu-inspect-yolo11n-e2_mixed",
 "classes": [
  "floating_debris",
  "leak"
 ],
 "imgsz": 640,
 "conf": 0.25,
 "experiment": "e2_mixed",
 "model_id": "336ebeba28ef",
 "serving": {
  "concurrency": 2,
  "intra_op_threads": 2
 },
 "samples": 11,
 "real_val": {
  "gt": 787,
  "ap50": 0.6064841942320086,
  "ap50_95": 0.39164162370779965,
  "alarm_conf": 0.45,
  "precision": 0.7945492662473794,
  "recall": 0.4815756035578145,
  "image_hit_rate": 0.8080808080808081,
  "images_with_target": 198,
  "image_false_alarm_rate": 0.4,
  "images_without_target": 5
 }
}
```

### 样例清单  `GET /api/vision/samples` → 200

```json
[
 "synth_aeration.jpg  CAM-AER-01  expect:floating_debris",
 "synth_clarifier.jpg  CAM-CLR-01  expect:floating_debris",
 "synth_outfall.jpg  CAM-OUT-01  expect:floating_debris",
 "synth_leak_streak.jpg  CAM-GAL-01  expect:leak",
 "synth_leak_puddle.jpg  CAM-GAL-02  expect:leak",
 "synth_negative_rust.jpg  CAM-GAL-01  期望无检出",
 "synth_negative_glare.jpg  CAM-GAL-02  期望无检出",
 "real_hobart_00754.jpg  CAM-RIV-01  expect:floating_debris",
 "real_nsw_00119.jpg  CAM-RIV-01  expect:floating_debris",
 "real_indonesia_00073.jpg  CAM-RIV-01  expect:floating_debris",
 "real_indonesia_00057.jpg  CAM-RIV-01  "
]
```

### ① 2 个漂浮物 → 新告警（提示）  `POST /api/vision/samples/real_nsw_00119.jpg/inspect?plant=P01&camera=CAM-RIV-01` → 200

```json
{
 "source": "sample:real_nsw_00119.jpg",
 "camera": "CAM-RIV-01",
 "ts": "2026-11-05T09:00:00+08:00",
 "detections": [
  "floating_debris 0.78",
  "floating_debris 0.77",
  "floating_debris 0.39"
 ],
 "events": [
  {
   "event": "new",
   "alarm_id": 14,
   "rule": "vision_debris",
   "level": "提示"
  }
 ],
 "alarm_note": null,
 "timings_ms": {
  "decode": 9.57,
  "infer": 47.58,
  "total": 63.05
 }
}
```

### ② 6 个漂浮物 → 升级为预警  `POST /api/vision/samples/real_hobart_00754.jpg/inspect?plant=P01&camera=CAM-RIV-01` → 200

```json
{
 "source": "sample:real_hobart_00754.jpg",
 "camera": "CAM-RIV-01",
 "ts": "2026-11-05T09:05:00+08:00",
 "detections": [
  "floating_debris 0.88",
  "floating_debris 0.87",
  "floating_debris 0.86",
  "floating_debris 0.86",
  "floating_debris 0.85",
  "floating_debris 0.69"
 ],
 "events": [
  {
   "event": "escalated",
   "alarm_id": 14,
   "rule": "vision_debris",
   "level": "预警"
  }
 ],
 "alarm_note": null,
 "timings_ms": {
  "decode": 5.5,
  "infer": 48.77,
  "total": 61.11
 }
}
```

### ③ 只有低分框（0.25～0.45 之间）→ 保持，不恢复也不升级  `POST /api/vision/samples/real_indonesia_00057.jpg/inspect?plant=P01&camera=CAM-RIV-01` → 200

```json
{
 "source": "sample:real_indonesia_00057.jpg",
 "camera": "CAM-RIV-01",
 "ts": "2026-11-05T09:10:00+08:00",
 "detections": [
  "floating_debris 0.36",
  "floating_debris 0.35"
 ],
 "events": [],
 "alarm_note": null,
 "timings_ms": {
  "decode": 2.89,
  "infer": 45.52,
  "total": 53.53
 }
}
```

### ④ 无检出 → 恢复  `POST /api/vision/samples/synth_negative_glare.jpg/inspect?plant=P01&camera=CAM-RIV-01` → 200

```json
{
 "source": "sample:synth_negative_glare.jpg",
 "camera": "CAM-RIV-01",
 "ts": "2026-11-05T09:15:00+08:00",
 "detections": [],
 "events": [
  {
   "event": "recovered",
   "alarm_id": 14,
   "rule": "vision_debris"
  }
 ],
 "alarm_note": null,
 "timings_ms": {
  "decode": 1.54,
  "infer": 46.72,
  "total": 51.7
 }
}
```

### ⑤ 再次检出 → 开新事件（不并入已恢复的事件）  `POST /api/vision/samples/real_nsw_00119.jpg/inspect?plant=P01&camera=CAM-RIV-01` → 200

```json
{
 "source": "sample:real_nsw_00119.jpg",
 "camera": "CAM-RIV-01",
 "ts": "2026-11-05T09:20:00+08:00",
 "detections": [
  "floating_debris 0.78",
  "floating_debris 0.77",
  "floating_debris 0.39"
 ],
 "events": [
  {
   "event": "new",
   "alarm_id": 15,
   "rule": "vision_debris",
   "level": "提示"
  }
 ],
 "alarm_note": null,
 "timings_ms": {
  "decode": 3.34,
  "infer": 46.82,
  "total": 55.01
 }
}
```

### ⑥ 管廊渗漏 → 预警  `POST /api/vision/samples/synth_leak_streak.jpg/inspect?plant=P01` → 200

```json
{
 "source": "sample:synth_leak_streak.jpg",
 "camera": "CAM-GAL-01",
 "ts": "2026-11-05T09:21:00+08:00",
 "detections": [
  "leak 0.95",
  "leak 0.95"
 ],
 "events": [
  {
   "event": "new",
   "alarm_id": 16,
   "rule": "vision_leak",
   "level": "预警"
  }
 ],
 "alarm_note": null,
 "timings_ms": {
  "decode": 1.59,
  "infer": 46.03,
  "total": 51.13
 }
}
```

### ⑦ 难负样本（锈迹、管道阴影）→ 渗漏恢复  `POST /api/vision/samples/synth_negative_rust.jpg/inspect?plant=P01` → 200

```json
{
 "source": "sample:synth_negative_rust.jpg",
 "camera": "CAM-GAL-01",
 "ts": "2026-11-05T09:22:00+08:00",
 "detections": [],
 "events": [
  {
   "event": "recovered",
   "alarm_id": 16,
   "rule": "vision_leak"
  }
 ],
 "alarm_note": null,
 "timings_ms": {
  "decode": 1.74,
  "infer": 45.66,
  "total": 50.8
 }
}
```

### ⑧ 上传图片、不指定厂站 → 只记录不告警  `POST /api/vision/inspect` → 200

```json
{
 "source": "upload",
 "camera": null,
 "ts": "2026-11-05T09:23:00+08:00",
 "detections": [
  "floating_debris 0.91",
  "floating_debris 0.90",
  "floating_debris 0.90",
  "floating_debris 0.87",
  "floating_debris 0.83",
  "floating_debris 0.72"
 ],
 "events": [],
 "alarm_note": "未同时指定厂站和摄像头，只记录推理结果，不写告警",
 "timings_ms": {
  "decode": 1.21,
  "infer": 45.9,
  "total": 50.71
 }
}
```

### ⑨ 非图片 → 415  `POST /api/vision/inspect` → 415

```json
{
 "detail": "请求体须为图片（Content-Type: image/jpeg, image/png, image/webp, image/bmp）"
}
```

### 告警列表（视觉巡检）  `GET /api/alarms?kind=%E8%A7%86%E8%A7%89%E5%B7%A1%E6%A3%80` → 200

```json
[
 {
  "id": 16,
  "rule": "vision_leak",
  "level": "预警",
  "metric": "leak",
  "ts": "2026-11-05T09:21:00+08:00",
  "ended_at": "2026-11-05T09:22:00+08:00",
  "value": 0.9533,
  "message": "CAM-GAL-01 视觉巡检发现渗漏 2 处（最高置信度 0.95）"
 },
 {
  "id": 15,
  "rule": "vision_debris",
  "level": "提示",
  "metric": "floating_debris",
  "ts": "2026-11-05T09:20:00+08:00",
  "ended_at": null,
  "value": 0.7781,
  "message": "CAM-RIV-01 视觉巡检发现漂浮物 2 处（最高置信度 0.78）"
 },
 {
  "id": 14,
  "rule": "vision_debris",
  "level": "预警",
  "metric": "floating_debris",
  "ts": "2026-11-05T09:00:00+08:00",
  "ended_at": "2026-11-05T09:15:00+08:00",
  "value": 0.8807,
  "message": "CAM-RIV-01 视觉巡检发现漂浮物 6 处（最高置信度 0.88）"
 }
]
```

### 告警详情  `GET /api/alarms/15` → 200

```json
{
 "detail": {
  "hits": 1,
  "unit": "置信度",
  "count": 2,
  "model": "336ebeba28ef",
  "camera": "CAM-RIV-01",
  "images": [
   {
    "ts": "2026-11-05T09:20:00+08:00",
    "count": 2,
    "sha256": "77f5da0de2407603b25016b1b3a350ade0129d154acfbd8940ef5c0ea4a2257c",
    "source": "sample:real_nsw_00119.jpg",
    "max_score": 0.7781
   }
  ],
  "last_seen": "2026-11-05T09:20:00+08:00",
  "detected_at": "2026-11-05T09:20:00+08:00"
 }
}
```

### 工单草稿（模板步骤）  `POST /api/alarms/15/work-order-draft` → 200

```json
{
 "title": "南郑中心污水处理厂 视觉巡检（漂浮物）",
 "steps": [
  {
   "text": "调取对应摄像头实时画面，复核漂浮物的位置与范围（模型可能误检）",
   "source": "template",
   "citations": []
  },
  {
   "text": "现场确认后处置：漂浮物安排打捞并检查格栅、撇渣设施；渗漏先隔离、标记渗漏点并报修，处置前后拍照留档",
   "source": "template",
   "citations": []
  },
  {
   "text": "处置后跟踪漂浮物恢复情况，并在工单中记录处理结果",
   "source": "template",
   "citations": []
  }
 ],
 "source": {
  "kind": "template",
  "note": "没有可用的 AI 核查建议，以下为默认步骤，不是 AI 建议"
 }
}
```

### AI 处置建议 → 422（知识库无对应规程，不调用模型）  `POST /api/alarms/15/advice` → 422

```json
{
 "detail": "视觉巡检告警暂不生成 AI 处置建议（知识库没有对应规程），可直接生成工单草稿"
}
```

### 巡检记录  `GET /api/vision/inspections?limit=3` → 200

```json
[
 {
  "id": 17,
  "ts": "2026-11-05T09:23:00+08:00",
  "plant": null,
  "camera": null,
  "source": "upload",
  "alarm_ids": []
 },
 {
  "id": 16,
  "ts": "2026-11-05T09:22:00+08:00",
  "plant": "P01",
  "camera": "CAM-GAL-01",
  "source": "sample:synth_negative_rust.jpg",
  "alarm_ids": [
   16
  ]
 },
 {
  "id": 15,
  "ts": "2026-11-05T09:21:00+08:00",
  "plant": "P01",
  "camera": "CAM-GAL-01",
  "source": "sample:synth_leak_streak.jpg",
  "alarm_ids": [
   16
  ]
 }
]
```

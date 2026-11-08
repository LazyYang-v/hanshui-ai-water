# Text2SQL 评测报告：deepseek-flash

固定评测时钟：2026-11-05 12:00 CST；64 题；运行模式：`replay`。
生成时间：2026-11-07T05:28:32.925450+08:00。实验指纹：`55cfcc853c25d2a3`。

## 评测口径

这是本项目模拟数据上的开发集，包含 60 道可答题和 4 道拒答题；已用于发现 schema 裁剪等问题，并非独立保留的测试集。
结果匹配要求行数相同、金标列一一对应，允许列顺序不同和额外说明列。数值容差为相对 1% 或绝对 0.01；日期统一为北京时间。仅 E02 允许只答时刻，标记 ordered 的题保留行序。格式错误、调用失败不算正确拒答。此口径不能与官方 Spider/BIRD EX 直接比较。

每配置独立调用且关闭本地缓存；厂商上下文缓存可能命中。LLM 延迟为各轮调用延迟之和，回放沿用录制值；逐题 total_ms 是当次本地端到端耗时。token 是请求用量，非人民币成本。单次运行的准确率与延迟会有波动。

## 消融对比

| 配置 | 正确题数 | 结果匹配率 | 可答题匹配率 | 误拒 | 触发修复 | 修复后正确 | LLM P50 | LLM P95 | 平均 token |
|---|---|---|---|---|---|---|---|---|---|
| 完整（few-shot 检索 + schema 裁剪 + 自修复） | 61/64 | **95.3%** | 95.0% | 0 | 2% | 100% | 1.54s | 5.79s | 2083 |
| 去掉 few-shot | 61/64 | **95.3%** | 95.0% | 0 | 0% | 0% | 2.24s | 4.75s | 1826 |
| 去掉 schema 裁剪（全量 schema） | 63/64 | **98.4%** | 98.3% | 0 | 0% | 0% | 1.69s | 4.98s | 2450 |
| 去掉自修复 | 60/64 | **93.8%** | 93.3% | 0 | 0% | 0% | 1.85s | 5.14s | 2039 |
| 只有规则和全量 schema | 58/64 | **90.6%** | 90.0% | 0 | 0% | 0% | 2.37s | 7.86s | 2299 |

## 分类匹配率

| 配置 | 基础信息 | 实时值 | 统计汇总 | 时间序列 | 排序TopN | 达标超标 | 设备 | 事件追溯 | 复合计算 | 拒答 |
|---|---|---|---|---|---|---|---|---|---|---|
| full | 88% | 80% | 100% | 100% | 100% | 100% | 100% | 83% | 100% | 100% |
| no_fewshot | 100% | 100% | 100% | 75% | 83% | 100% | 100% | 83% | 100% | 100% |
| no_prune | 88% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| no_repair | 100% | 80% | 92% | 100% | 100% | 100% | 100% | 67% | 100% | 100% |
| bare | 88% | 80% | 100% | 75% | 100% | 100% | 86% | 83% | 83% | 100% |

## 完整配置的错题

### L05 额定功率最大的设备有哪些？

- 错误：结果与金标不一致
- 预测前几行：`[["南郑中心污水处理厂", "P01-BL-1", "鼓风机", "磁悬浮离心鼓风机 MBL-150", 110.0, "运行"], ["南郑中心污水处理厂", "P01-BL-2", "鼓风机", "磁悬浮离心鼓风机 MBL-150", 110.0, "运行"], ["南郑中心污水处理厂", "P01-BL-3", "鼓风机", "磁悬浮离心鼓风机 MBL-150", 110.0, "备用"]]`
- 金标前几行：`[["P01-BL-1", 110.0], ["P01-BL-2", 110.0], ["P01-BL-3", 110.0]]`

```sql
SELECT p.name AS 厂站, d.code AS 设备编码, d.kind AS 类型, d.model AS 型号, ROUND(CAST(d.rated_kw AS DECIMAL), 1) AS 额定功率_kW, d.status AS 状态 FROM device AS d JOIN plant AS p ON p.id = d.plant_id WHERE d.rated_kw IS NOT NULL ORDER BY d.rated_kw DESC, d.code LIMIT 10
```

### R05 现在哪个厂的出水总氮最高？

- 错误：结果与金标不一致
- 预测前几行：`[["南郑中心污水处理厂", "2026-11-05T11:59:00+08:00", 10.73], ["协税镇污水处理站", "2026-11-05T11:59:00+08:00", 10.8], ["梁山镇污水处理站", "2026-11-05T11:59:00+08:00", 10.82]]`
- 金标前几行：`[["牟家坝镇污水处理站", 11.3627]]`

```sql
SELECT DISTINCT ON (p.id) p.name AS 厂站, r.ts AS 时间, ROUND(CAST(r.tn_out AS DECIMAL), 2) AS 出水总氮_mgL FROM plant_reading AS r JOIN plant AS p ON p.id = r.plant_id WHERE r.ts > CAST('2026-11-05T12:00:00+08:00' AS TIMESTAMPTZ) - INTERVAL '30 MINUTES' AND r.tn_out IS NOT NULL ORDER BY p.id, r.ts DESC LIMIT 200
```

### E02 10月31日的降雨从几点开始、几点结束？

- 错误：结果与金标不一致
- 预测前几行：`[["2026-10-31T14:01:00+08:00", "2026-10-31T17:29:00+08:00", 149, 1680.0]]`
- 金标前几行：`[["2026-10-31T14:01:00+08:00", "2026-10-31T16:29:00+08:00"]]`

```sql
SELECT MIN(ts) AS 开始时间, MAX(ts) + INTERVAL '1 HOUR' AS 结束时间, COUNT(*) AS 降雨小时数, ROUND(CAST(SUM(rain_mm_h) AS DECIMAL), 1) AS 降雨量_mm FROM weather WHERE ts >= CAST('2026-10-31 00:00:00+08' AS TIMESTAMPTZ) AND ts < CAST('2026-11-01 00:00:00+08' AS TIMESTAMPTZ) AND rain_mm_h > 0 LIMIT 200
```

## 复现配置

```json
{
  "dataset_sha256": "4a10274619682e21fc973ff349d0eb883f674ed25156563232768ffcb407a460",
  "prompt_sha256": "10fea2034cd8b1c2aea9b45b1c2b43497ed68b08f0b5f68639988337fd814c57",
  "examples_sha256": "5a6daf17daa8ef07e5114ae096fa4f11c426d85ef0966ccf3b6fe3da60bbe39d",
  "schema_sha256": "4971ffbeeed8f02f1dae54a6564c3f43193c40f87cc5e57440460c2279a6be76",
  "max_tokens": 4000,
  "now": "2026-11-05T12:00:00+08:00",
  "seed": 7,
  "days": 7,
  "step_s": 60.0,
  "fingerprint": "55cfcc853c25d2a3",
  "model_requested": "deepseek-flash",
  "provider": "deepseek",
  "mode": "replay",
  "concurrency": 4,
  "generated_at": "2026-11-07T05:28:32.925450+08:00",
  "configs": {
    "full": {
      "use_fewshot": true,
      "fewshot_k": 3,
      "prune_schema": true,
      "max_repairs": 2,
      "max_tokens": 4000
    },
    "no_fewshot": {
      "use_fewshot": false,
      "fewshot_k": 3,
      "prune_schema": true,
      "max_repairs": 2,
      "max_tokens": 4000
    },
    "no_prune": {
      "use_fewshot": true,
      "fewshot_k": 3,
      "prune_schema": false,
      "max_repairs": 2,
      "max_tokens": 4000
    },
    "no_repair": {
      "use_fewshot": true,
      "fewshot_k": 3,
      "prune_schema": true,
      "max_repairs": 0,
      "max_tokens": 4000
    },
    "bare": {
      "use_fewshot": false,
      "fewshot_k": 3,
      "prune_schema": false,
      "max_repairs": 0,
      "max_tokens": 4000
    }
  },
  "models_returned": [
    "deepseek-flash"
  ]
}
```

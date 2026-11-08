# Text2SQL 评测报告：deepseek-flash

固定评测时钟：2026-11-05 12:00 CST；64 题；运行模式：`replay`。
生成时间：2026-11-07T22:31:04.269116+08:00。实验指纹：`ebcf2a528b536727`。

## 评测口径

这是本项目模拟数据上的开发集，包含 60 道可答题和 4 道拒答题；已用于发现 schema 裁剪等问题，并非独立保留的测试集。
结果匹配要求行数相同、金标列一一对应，允许列顺序不同和额外说明列。数值容差为相对 1% 或绝对 0.01；日期统一为北京时间。仅 E02 允许只答时刻，标记 ordered 的题保留行序。格式错误、调用失败不算正确拒答。此口径不能与官方 Spider/BIRD EX 直接比较。

每配置独立调用且关闭本地缓存；厂商上下文缓存可能命中。LLM 延迟为各轮调用延迟之和，回放沿用录制值；逐题 total_ms 是当次本地端到端耗时。token 是请求用量，非人民币成本。单次运行的准确率与延迟会有波动。

## 消融对比

| 配置 | 正确题数 | 结果匹配率 | 可答题匹配率 | 误拒 | 触发修复 | 修复后正确 | LLM P50 | LLM P95 | 平均 token |
|---|---|---|---|---|---|---|---|---|---|
| 完整（few-shot 检索 + schema 裁剪 + 自修复） | 63/64 | **98.4%** | 98.3% | 0 | 2% | 100% | 2.17s | 5.31s | 2313 |

## 分类匹配率

| 配置 | 基础信息 | 实时值 | 统计汇总 | 时间序列 | 排序TopN | 达标超标 | 设备 | 事件追溯 | 复合计算 | 拒答 |
|---|---|---|---|---|---|---|---|---|---|---|
| full | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 83% | 100% | 100% |

## 完整配置的错题

### E02 10月31日的降雨从几点开始、几点结束？

- 错误：结果与金标不一致
- 预测前几行：`[["2026-10-31T14:01:00+08:00", "2026-10-31T14:01:00+08:00", 1], ["2026-10-31T14:02:00+08:00", "2026-10-31T14:02:00+08:00", 1], ["2026-10-31T14:03:00+08:00", "2026-10-31T14:03:00+08:00", 1]]`
- 金标前几行：`[["2026-10-31T14:01:00+08:00", "2026-10-31T16:29:00+08:00"]]`

```sql
WITH rain AS (SELECT ts, ts - (ROW_NUMBER() OVER (ORDER BY ts) * INTERVAL '1 HOUR') AS grp FROM weather WHERE ts >= '2026-10-31 00:00:00+08' AND ts < '2026-11-01 00:00:00+08' AND rain_mm_h > 0) SELECT MIN(ts) AS 开始时间, MAX(ts) AS 结束时间, COUNT(*) AS 降雨读数小时数 FROM rain GROUP BY grp ORDER BY 开始时间 LIMIT 200
```

## 复现配置

```json
{
  "dataset_sha256": "4a10274619682e21fc973ff349d0eb883f674ed25156563232768ffcb407a460",
  "prompt_sha256": "44e0ac0ac7b644c4bcde37f110ac9fe4ecc73170844972bbc3a7cd57d5b95342",
  "examples_sha256": "f5d3e8cff3bc8b106c7d779f05d837e4b224f01eed14dba3214661876bbfc89d",
  "schema_sha256": "4971ffbeeed8f02f1dae54a6564c3f43193c40f87cc5e57440460c2279a6be76",
  "max_tokens": 4000,
  "now": "2026-11-05T12:00:00+08:00",
  "seed": 7,
  "days": 7,
  "step_s": 60.0,
  "fingerprint": "ebcf2a528b536727",
  "model_requested": "deepseek-flash",
  "provider": "deepseek",
  "mode": "replay",
  "concurrency": 4,
  "generated_at": "2026-11-07T22:31:04.269116+08:00",
  "configs": {
    "full": {
      "use_fewshot": true,
      "fewshot_k": 3,
      "prune_schema": true,
      "max_repairs": 2,
      "max_tokens": 4000
    }
  },
  "models_returned": [
    "deepseek-flash"
  ],
  "label": "raw-sample-v1-guard"
}
```

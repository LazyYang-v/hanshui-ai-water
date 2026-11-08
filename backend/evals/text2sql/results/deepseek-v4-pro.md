# Text2SQL 评测报告：deepseek-v4-pro

评测库 shuiwu_eval（演示时钟 2026-11-05 12:00），64 题，指标为执行准确率（EX）。生成时间 2026-11-07 05:16

## 消融对比

| 配置 | EX | 可答题 EX | 误拒 | 触发修复 | 修复后正确 | P50 延迟 | P95 延迟 | 平均 token | 平均输入 token |
|---|---|---|---|---|---|---|---|---|---|
| 完整（few-shot 检索 + schema 裁剪 + 自修复） | **98.4%** | 98.3% | 0 | 2% | 100% | 5.5s | 15.9s | 2626 | 1823 |
| 裸提示（只有规则和全量 schema） | **92.2%** | 93.3% | 1 | 0% | 0% | 8.0s | 20.4s | 2955 | 1939 |

## 分类准确率

| 配置 | 基础信息 | 实时值 | 统计汇总 | 时间序列 | 排序TopN | 达标超标 | 设备 | 事件追溯 | 复合计算 | 拒答 |
|---|---|---|---|---|---|---|---|---|---|---|
| full | 100% | 100% | 100% | 100% | 100% | 100% | 86% | 100% | 100% | 100% |
| bare | 100% | 100% | 100% | 75% | 100% | 100% | 86% | 83% | 83% | 75% |

## 完整配置的错题

### D06 昨天各厂污泥泵分别运行了多少分钟？（设备数据每分钟一条）

- 错误：结果与金标不一致
- 预测前几行：`[["南郑中心污水处理厂", 480], ["协税镇污水处理站", 480], ["梁山镇污水处理站", 480]]`
- 金标前几行：`[["P06-SP-1", 480], ["P04-SP-1", 480], ["P05-SP-1", 480]]`

```sql
SELECT p.name AS 厂站, COUNT(*) FILTER(WHERE dr.running) AS 运行分钟数 FROM device_reading AS dr JOIN device AS d ON d.id = dr.device_id JOIN plant AS p ON p.id = d.plant_id WHERE d.kind = '污泥泵' AND dr.ts >= DATE_TRUNC('DAY', CAST('2026-11-05T12:00:00+08:00' AS TIMESTAMPTZ)) - INTERVAL '1 DAY' AND dr.ts < DATE_TRUNC('DAY', CAST('2026-11-05T12:00:00+08:00' AS TIMESTAMPTZ)) GROUP BY p.id, p.name ORDER BY p.id LIMIT 200
```

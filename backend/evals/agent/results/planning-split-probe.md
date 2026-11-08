# 规划提示词“限制查数子问题项数”对比（2026-11-08，deepseek-flash，已撤回）

动机：Agent 实测里规划把十余项合成一条 `sql_query`，Text2SQL 因思考过长被截断、超时（见 `evals/text2sql/results/truncation-repair-probe.md`）。尝试在规划提示词中要求：诊断查数只列默认组合、不追加 MLSS/功率/药量/厂站编码，单条最多 5 项，默认不查设备台数。

方法：6 道未列明证据范围的诊断题，只跑首轮规划（与线上同参：厂商默认思考、4000 token），再按并发 3 对每条 `sql_query` 子问题实际跑一次 Text2SQL（已含截断修复）。改前/改后各一次，原始结果见 `planning-split-before.json`、`planning-split-after.json`。

| | 改前 | 改后 |
|---|---|---|
| 规划合格 | 6/6 | 3/6（2 次 4000 token 截断，1 次 plant 写成“高台镇”） |
| 规划输出 tokens | 450–2738 | 573–4000 |
| 查数子问题长度 | 79–213 字 | 86–113 字 |
| 查数首次截断 | 1/8 | 1/3 |
| 查数全部成功 | 8/8，最长 29.3 s | 3/3，最长 30.2 s |

结论：子问题变短，但计数类规则让规划推理变长，2/6 想满预算；规划失败比查数慢更伤，故撤回提示词改动。每组每题仅一次，只说明趋势。

衍生修改：改前组已有规划输出 2738 token，说明首轮规划同样可能截断；原代码截断后修复仍开思考。现改为首轮规划因 `finish_reason=length` 失败时，修复关闭思考并明确提示“推理过长”，其他修复不变（`tests/test_agent.py::test_first_plan_truncated_is_repaired_without_thinking`）。该修改未单独付费实测。

本次共约 12.1 万 tokens（改前 6.8 万、改后 5.3 万）。

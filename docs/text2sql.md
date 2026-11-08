# AI：Text2SQL

接口和评测统一使用 **`deepseek-flash`**。在项目根目录 `.env` 中填写自己的 `DEEPSEEK_API_KEY`，保留 `LLM_PROVIDERS=deepseek` 和 `DEEPSEEK_MODEL=deepseek-flash`，然后执行 `docker compose up -d --build api`。密钥只保存在本地 `.env`，不要提交到仓库。未配置模型时，查数接口返回 503，其余业务接口仍可使用。

```bash
curl -s localhost:8000/api/ai/gateway
curl -s localhost:8000/api/ai/text2sql \
  -H 'Content-Type: application/json' \
  -d '{"question":"昨天各厂分别处理了多少水？"}'
```

响应包含 `ok`、实际 `model`、生成的 `sql`、`columns` / `rows`、`attempts`、token 用量和耗时。模型明确拒答时 `refused=true`；SQL 执行失败、格式错误与模型调用失败分别记录错误原因，不算拒答。

查数流程：读取表字段注释 → 按问题裁剪 schema → 从 20 条示例中用字符 bigram BM25 检索 3 条 → Flash 输出 JSON → sqlglot 校验（表与函数白名单）并把 `now()`、`LOCALTIMESTAMP` 等改成演示时钟 → 只读账号执行 → 出错最多修复 2 次。Flash 思考过长、输出在 token 上限处被截断时，明确反馈原因并关闭思考修复；累计接近 50 秒停止修复并返回错误，避免超出 Agent 的 60 秒工具上限（[实测](../backend/evals/text2sql/results/truncation-repair-probe.md)）。查询限制为单条 SELECT、白名单表、最多 200 行、5 秒超时。网关支持多厂商顺序降级、重试与熔断、LRU 缓存、流式输出、调用统计以及 JSONL 录制/回放；当前只启用 DeepSeek Flash。

**网关并发压测**（`evals/gateway/results/report.md`）。用独立进程的可配置假厂商做压测，走真实 httpx 连接池，不花钱；结果分别记录改动前后的网关。

- 网关开销：并发 1～100 时与裸 httpx 一致；并发 50 流式首字的额外开销 p95 为 13.6 ms。
- 压测发现旧熔断器在两种情况下帮倒忙：
  - 厂商限流（同时最多 20 个、客户端并发 50）时，429 把唯一厂商熔断，成功率只有 **7.3%**；
  - 主厂商 30% 返回 503 时，连续 3 次失败就熔断，300 个请求里有 257 个被切到更慢的备用厂商。
- 改动：
  - 429 不计入熔断，按 Retry-After（可带小数）重试；
  - 熔断改为按窗口失败率判断，分母包含在途请求，否则高并发下先回来的快速失败会把 30% 误判成 100%；
  - 每个厂商可设并发上限 `LLM_MAX_CONCURRENCY`，超出在网关内排队，排队时间单独计时；
  - 退避结束后重新检查熔断。
- 改动后：
  - 限流场景成功率 81%，把并发上限设为厂商容量后为 **100%**（429 为 0，吞吐 39.2 rps，理论上限 40 rps）；
  - 30% 503 场景 95% 的请求留在主厂商，p50 从 826 ms 降到 523 ms；
  - 主厂商不响应时，最长耗时从 4.8 s 降到 2.8 s。
- 测量陷阱：本机 `HTTP_PROXY` 会让 httpx 访问 127.0.0.1 时也走代理，高并发下出现代理返回的空 502。压测脚本现在对本机地址绕过代理。
- 真实 DeepSeek Flash：流式、非流式各 100 次，并发 50，关闭思考，经过本机代理。
  - 结果：200/200 成功，无 429、无重试。流式首字 p50 / p95 为 **928 / 1539 ms**，非流式总耗时 p50 / p95 为 1290 / 1632 ms。
  - 第一轮没有关闭思考，Flash 把 120 token 的输出上限用在了推理上，流式只有 22/100 次输出了正文。这一轮作废，结果单独保存。为此网关 `stream()` 增加了 `thinking` 开关。

```bash
uv run python -m evals.gateway.load fake --label after
uv run python -m evals.gateway.load live --label flash --n 100 --concurrency 50   # 产生费用，只允许 deepseek-flash
```

评测在独立的 `shuiwu_eval` 数据库中运行：固定时钟为 **2026-11-05 12:00（北京时间）**，7 天模拟历史，64 道问题、10 类，包含 4 道拒答题。评测用例与 few-shot 的问题文本不重复。

```bash
cd backend
uv run python -m evals.text2sql.build_db           # 首次使用；会重建 shuiwu_eval，不改演示库
uv run python -m evals.text2sql.run --validate-only # 校验 60 条金标 SQL，不调用模型
uv run python -m evals.text2sql.run                # Flash：完整方案及 4 组消融，会产生 API 费用
uv run python -m evals.text2sql.run --mode replay  # 同配置离线回放，不调用外部模型
```

五组配置为 `full`、`no_fewshot`、`no_prune`、`no_repair`、`bare`，可用 `--configs full` 单独运行。API 和评测的输出预算均为 4000 token（含推理）。每组关闭本地缓存并独立录制，报告包含请求模型、实际模型、数据/提示词指纹和完整逐题结果；回放单独生成文件，不覆盖真实调用报告。

2026-11-07 的 Flash 实测结果如下（并发 4，所有配置实际返回的模型均为 `deepseek-flash`）：

| 配置 | 正确题数 | 结果匹配率 | LLM P50 | LLM P95 | 平均 token / 题 |
| --- | --- | --- | --- | --- | --- |
| 默认完整方案 | 61/64 | 95.3% | 1.54s | 5.79s | 2083 |
| 不加 few-shot | 61/64 | 95.3% | 2.24s | 4.75s | 1826 |
| 不裁剪 schema | 63/64 | 98.4% | 1.69s | 4.98s | 2450 |
| 不做自修复 | 60/64 | 93.8% | 1.85s | 5.14s | 2039 |
| 只有规则和全量 schema | 58/64 | 90.6% | 2.37s | 7.86s | 2299 |

同日针对诊断评测暴露的问题补充了原始采样查数约定（指定时刻须精确匹配、返回实际采样时间、原始值不取整，并新增一条精确时刻示例），完整配置单轮复测为 **63/64（98.4%）**，只剩 E02 错，LLM P50/P95 为 2.17s/5.31s，平均 token 2313。单轮有采样波动，不把 2 题之差全归因于这次改动；见 [复测报告](../backend/evals/text2sql/results/deepseek-flash-record-raw-sample-v1.md)。上表消融仍是改动前的结果。

320 条评测结果已逐题离线回放，SQL、结果、判分、token 与录制的 LLM 延迟均一致。schema 裁剪减少了上下文，但本次未取得最高匹配率；单轮消融仍受模型采样波动影响。

**裁剪和 few-shot 的保留理由不是准确率。** 这一轮裁剪没有多答对任何一题（关闭后 63/64，高于默认的 61/64），few-shot 的增减也是 2:2 打平，64 题单轮分不出差异。默认仍保留裁剪，因为它让平均每题少用约 15% token（2083 对 2450），且补充采样约定后默认配置复测同为 63/64；few-shot 用来传递查数约定（如指定时刻须精确匹配）。要证明两者值得保留，需要多轮重复评测，目前没有做。

默认方案的 3 道错题分别涉及并列最大值筛选、最新值排名、降雨结束时间，报告保留了生成的 SQL 和金标结果。

正式结果见 [Flash 评测报告](../backend/evals/text2sql/results/deepseek-flash-record.md)，逐题记录见 [JSONL](../backend/evals/text2sql/results/deepseek-flash-record.jsonl)。`results/` 下不带 `-record` / `-replay` 后缀的文件是早期试跑记录，使用旧判分规则，不作为当前验收结果。

**指标范围：**这是用于迭代的模拟数据开发集。匹配允许额外说明列、数值相对误差 1% 或绝对误差 0.01，不能与官方 Text2SQL 基准或真实业务准确率直接比较。SQL 能执行也可能答非所问，需要查看 SQL 与结果；当前自修复针对格式、校验和执行错误。水量、电量及投药总量由小时均值求和估算，未满小时与混合采样的精确积分留待后续完善。

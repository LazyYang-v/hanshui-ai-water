# AI：RAG 知识问答

知识库位于 `backend/knowledge/`，当前为 **10 篇文档、35 个片段**：8 篇项目自编的工艺/故障排查演示资料、1 篇 Demo 数据口径说明、1 篇用户提供案例 PDF 的摘要。每篇保留来源类型；案例摘要关联原文链接和页码，项目自编资料不会标成真实厂站规程或官方标准。

检索使用 **BGE-small-zh-v1.5（512 维、ONNX CPU）+ 字符 bigram BM25 + RRF 融合**，默认问答再由 `deepseek-flash` 对 6 个候选作相关性重排，选择最多 5 个片段生成回答。重排显式关闭思考，输出预算 1200 token，8 秒内未完成就保留检索排序并标明降级；独立知识回答也关闭思考，保留引用校验与一次修复。重排是 Flash 调用，不是本地 cross-encoder。

进程内缓存最多 128 条、有效期 10 分钟，键包含知识库、模型与策略版本；只缓存检索结果及校验成功的重排，失败不缓存。相同问题并发请求共享一次成功重排，取消继续向上传播。少量高置信短定义题可跳过重排；诊断、查数与能力外问题不走此捷径。响应增加 `rerank_status`、`cache_hit`、`cache_age_s` 与 `total_ms`。缓存命中不重复累计 token、模型耗时或调用尝试；缓存仅属于当前 API 进程，重启失效。

首次准备本地模型：

```bash
cd backend
uv sync
uv run python -m app.ai.rag.build   # 下载约 90MB 中文嵌入模型，构建语料向量缓存
cd ..
docker compose up -d --build api
```

只有 Docker 时也可先 `docker compose build api`，再执行 `docker compose run --rm --no-deps api python -m app.ai.rag.build`。Compose 挂载本地模型缓存和向量索引目录，避免每次构建都打包模型。服务启动只读取本地模型，不自动下载；模型未就绪时 `hybrid` 显式降级为 BM25，明确请求 `dense` 则返回 503。`GET /api/ai/knowledge` 可查看 `dense_available` 和具体状态。

```bash
curl -s localhost:8000/api/ai/knowledge
curl -s localhost:8000/api/ai/rag/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"暴雨后为什么氨氮可能升高？","mode":"hybrid","rerank":false}'
curl -s localhost:8000/api/ai/rag \
  -H 'Content-Type: application/json' \
  -d '{"question":"风机跳停、DO下降后，应该怎样排查氨氮升高？"}'
```

回答返回 `answer`、`citations`、`retrieval`、实际模型、token 和耗时。回答最多 4 条简短陈述，每条必须带引用；服务端检查片段 id 确实来自本次检索，并检查 `quote` 是该片段中的连续摘录。失败最多修复一次，反馈具体字段/引用错误，`validation_errors` 保留修复记录；仍无效就返回错误，不展示未通过校验的回答。每条引用带可访问的原文 API 链接、Markdown 行号，以及参考案例的 PDF 页码；引用摘录来自知识库 Markdown，案例文档本身为摘要。

没有足够资料时返回 `refused=true` 和具体原因。知识库不提供实时测点、员工联系方式、实际电费或天气预报。引用校验保证出处存在与摘录可核对，不等于已经证明所有结论都被证据支持；原因诊断仍需人工核查和实际数据。独立 RAG 接口仍只回答知识问题；需要结合查数的诊断请使用下文 Agent 对话接口。

检索评测使用 30 道可答题、4 道库外题，对比 BM25、纯向量、混合、混合 + Flash 重排。按章节计算 Recall@5，避免同一章节切为多块后重复计分。

```bash
cd backend
uv run python -m evals.rag.run --label rag-new-record                 # 四组检索配置；最多34次Flash重排
uv run python -m evals.rag.run --label rag-new-replay --mode replay    # 相同语料/策略下回放，另存结果
uv run python -m evals.rag.performance --label rag-new-comparison     # 同题比较legacy与optimized；会调用Flash
```

2026-11-07 早期版本实测（33 片段、8 候选重排；30 道可答题的检索指标，延迟统计含库外题；保留为历史记录）：

| 方案 | Recall@5 | MRR@5 | 检索 + 重排 P50 | 平均模型 token |
| --- | --- | --- | --- | --- |
| BM25 | 100.0% | 0.892 | 0.4ms | 0 |
| BGE 向量 | 100.0% | 0.808 | 133.4ms | 0 |
| 混合检索 | 100.0% | 0.917 | 131.6ms | 0 |
| 混合 + Flash 重排 | 100.0% | 0.983 | 7546.1ms | 3449 |

语料较小，BM25 已能在前 5 名找到证据；重排改善了靠前的排序，但增加了延迟和模型调用成本。四组共 136 条记录已离线回放核对，返回片段与检索指标一致；回放仍执行真实本地向量检索。不要将 100% 的开发集召回解释为生产效果。

正式结果见 [RAG 检索评测](../backend/evals/rag/results/record.md)，[逐题记录](../backend/evals/rag/results/record.jsonl)包含金标章节和返回片段。此结果是小规模人工开发集的检索指标，不是回答事实准确率。

当前版本在相同 35 片段、相同 34 题、并发 3 的成对对照中，冷请求真实墙钟 P50 为 **9.07 → 1.09 秒**，P95 为 **16.42 → 1.40 秒**，平均模型 token 为 **3683 → 1696**。30 道可答题 Recall@5、Hit@5 均保持 100%，MRR@5 为 **0.983 → 0.967**：故障恢复题的标注章节从第一位移到第二位，仍在前五；没有用这套小开发集证明生产效果。优化配置没有超时降级，1 题跳过重排。4 道库外题有 2 道返回空结果，其余返回能力边界说明，检索有片段不等于允许回答。68 条冷请求的片段、判分、token 与模型计时已逐条离线回放一致。详见 [同题对照报告](../backend/evals/rag/results/rag-speed-v1/report.md)和[原始记录](../backend/evals/rag/results/rag-speed-v1/rows.jsonl)。

新版本真实 API 验收通过 **4/4**：氨氮排查建议、区分参考案例与 Demo 成果、拒答员工联系方式、拒答知识库无法提供的实时读数，耗时 3.36 / 2.75 / 1.36 / 2.62 秒，原文链接与摘录核对通过。可执行 `uv run python -m evals.rag.smoke --output evals/rag/results/api-smoke-new.json` 复验（会调用 Flash，已有文件拒绝覆盖），记录见 [当前 API 验收](../backend/evals/rag/results/api-smoke-rag-speed.json)。另经实际 HTTP 验证，首查约 920ms、同题缓存约 2.5ms，缓存请求 token 与模型耗时均为零，见[缓存验收](../backend/evals/rag/results/api-cache-rag-speed.json)。这些只验证指定场景和引用结构，不是通用回答准确率评测。

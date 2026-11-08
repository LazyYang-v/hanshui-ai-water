# 水务知识检索评测

语料 10 篇、33 片段；30 道可答题 + 4 道库外题；模式 `replay`。

这是人工整理的小规模开发集，不能代表未知生产资料上的表现。按章节统计 Recall@5；每道题先算召回比例，再对可答题求平均。MRR@5 为首个相关片段的倒数排名；Hit@5 为至少命中一个相关章节的题目比例。四道库外题不计入召回率，其检索结果保留在逐题记录中；检索返回片段并不等同于回答有依据。

重排使用 DeepSeek Flash，为语义评分重排而非本地 cross-encoder。向量使用真实 BGE-small-zh-v1.5（ONNX CPU，512维），混合使用 RRF。本地缓存关闭，厂商可能命中上下文缓存。延迟包括本次检索和记录的 LLM 调用，不包括排队；回放时 LLM 延迟取录制值，本地检索延迟会变化。

| 配置 | Recall@5 | Hit@5 | MRR@5 | P50 | P95 | 平均 token |
|---|---|---|---|---|---|---|
| bm25 | 100.0% | 100.0% | 0.892 | 0.4ms | 1.9ms | 0 |
| dense | 100.0% | 100.0% | 0.808 | 70.4ms | 123.3ms | 0 |
| hybrid | 100.0% | 100.0% | 0.917 | 61.3ms | 227.3ms | 0 |
| hybrid_rerank | 100.0% | 100.0% | 0.983 | 7556.6ms | 15284.2ms | 3449 |

## 未完全召回的问题


## 复现配置

```json
{
  "documents": 10,
  "chunks": 33,
  "fingerprint": "1f0ede3e7086f49e",
  "dense_available": true,
  "embedding_model": "BAAI/bge-small-zh-v1.5",
  "dense_error": null,
  "mode": "replay",
  "reranker": "deepseek-flash",
  "top_k": 5,
  "candidates": 8,
  "dataset_sha256": "ad54c79ba12a9363f965dd50955d3ec6fa27d851016138984802c8cd28fa3e76"
}
```

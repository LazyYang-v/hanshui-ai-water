# API

所有时间都以**演示时钟**为准。不带时区的时间参数按北京时间解释。

**鉴权：** `GET` 接口和 WebSocket 公开。其余方法（写库、调用模型、注入场景、视觉推理）都要管理员会话，未登录返回 401；只有登录、注销和曝气仿真例外，曝气仿真只是确定性计算。命令行先登录拿 Cookie，之后的写请求加 `-b`：

```bash
curl -c /tmp/hs.jar -X POST localhost:8000/api/auth/login -H 'content-type: application/json' \
  -d "{\"username\":\"admin\",\"password\":\"$ADMIN_PASSWORD\"}"
```

下文写请求示例省略 `-b /tmp/hs.jar`。

| 接口 | 说明 |
| --- | --- |
| `GET /api/overview` | 集控总览：天气、当日 KPI（处理水量、吨水曝气电耗、药耗、达标率）、厂站状态、24 小时趋势、进行中的场景 |
| `GET /api/plants` | 厂站列表：最新读数，状态（normal / warn / alarm / offline），问题清单 |
| `GET /api/plants/{code}` | 厂站详情：当日统计、设备台账与最新监测值 |
| `GET /api/plants/{code}/readings` | 时间序列。参数：`metrics`（逗号分隔）、`start`、`end`、`bucket`（auto / 5s / 1m / 5m / 15m / 1h / 6h / 1d） |
| `GET /api/devices/{code}/readings` | 设备电流、振动、温度时间序列 |
| `GET /api/metrics` | 指标目录：中文名、单位、限值、预警边界 `warn_at` |
| `POST /api/auth/login`、`POST /api/auth/logout`、`GET /api/auth/me` | 管理员登录（写入 HttpOnly 会话 Cookie，8 小时）、注销（服务端立即删除会话）、当前登录状态 |
| `GET /api/clock` / `GET /api/health` | 演示时钟 / 健康检查 |
| `GET/POST/DELETE /api/demo/scenarios`、`PUT /api/demo/speed` | 演示控制：查看、注入、结束场景，设置倍速 |
| `WS /ws/realtime` | 连接后先推最新一帧，之后每步推送 |
| `POST /api/ai/text2sql` | 自然语言查数：`{"question": "昨天各厂处理了多少水？"}` → SQL、结果表、修复过程、token 与耗时 |
| `POST /api/ai/statistics` | 固定只读统计：明确厂站、指标与时段，返回累计/连续异常时长、原始与均值极值、排名、覆盖和服务端事实 |
| `GET /api/ai/gateway` | LLM 网关状态：逻辑请求总耗时、每次厂商尝试延迟、成功单次延迟、错误/token/缓存/熔断 |
| `GET /api/ai/knowledge` | 知识文档清单、片段数、向量模型是否就绪及降级原因 |
| `GET /api/ai/knowledge/{id}` / `GET /api/ai/knowledge/chunks/{id}` | 知识文档原文 / 被引用的片段、行号和页码 |
| `POST /api/ai/rag/search` | 检索调试：`question`、`mode`（bm25 / dense / hybrid）、`rerank`、`top_k` |
| `POST /api/ai/rag` | 知识问答：`{"question":"风机跳停后为什么氨氮可能升高？"}` → 回答、核验后的引用与原文链接 |
| `POST /api/ai/chat` | Agent 对话：`question`、可选 `session_id` → 查数/检索、带证据回答、完整工具记录 |
| `POST /api/ai/chat/stream` | 同样的请求体，SSE 返回规划进度、工具开始/结束、校验后的完整结果 |
| `GET /api/ai/chat/sessions/{id}` | 读取最近 6 轮已完成对话及 SQL、结果表、知识出处 |
| `GET /api/alarms` | 告警列表：按厂站、级别、类型、处理状态、数据是否已恢复、时间范围过滤，附各级别/状态数量 |
| `GET /api/alarms/{id}` | 告警详情：触发参数、峰值、前后 6 小时同厂告警、工单、最近一次处置建议 |
| `GET /api/alarms/monitor` / `POST /api/alarms/scan?hours=3` | 巡检状态与规则清单 / 立即扫描 |
| `POST /api/alarms/{id}/advice` | AI 处置建议：固定证据查询 + 知识检索 + 引用校验，`refresh=true` 重新生成 |
| `POST /api/alarms/{id}/work-order-draft` | 工单草稿：只存 Redis 30 分钟，不写库；从对话生成时可带 `chat: {session_id, run_id}`，步骤优先取该轮回答的核查建议 |
| `POST /api/alarms/{id}/close` | 关闭告警（处理状态），需填写说明 |
| `POST /api/work-orders` | 确认草稿并写库：`draft_id`、`assignee`，可改标题和步骤；从对话派单时可带 `chat: {session_id, run_id}`，响应的 `chat_recorded` 表示是否记回该轮对话 |
| `GET /api/work-orders`、`GET /api/work-orders/{id}`、`POST /api/work-orders/{id}/start`、`POST /api/work-orders/{id}/close` | 工单看板（附各状态数量）、详情、开始处理、关闭 |
| `POST /api/reports` | 生成日报/周报：`{"kind":"日报"}`，可带 `date` |
| `GET /api/reports`、`GET /api/reports/{id}`、`GET /api/reports/{id}/markdown`、`GET /api/reports/{id}/pdf` | 报告列表、详情（含统计事实与生成信息）、Markdown / PDF 下载 |
| `GET /api/vision/model` | 视觉模型卡：类别、阈值、训练数据、评测指标、许可、局限与运行配置 |
| `GET /api/vision/samples`、`GET /api/vision/samples/{name}` | 内置样例清单（含来源与署名）/ 原图 |
| `POST /api/vision/samples/{name}/inspect`、`POST /api/vision/inspect` | 样例推理 / 上传图片推理（原始请求体，≤10 MB）；带 `plant` 和 `camera` 时按阈值写入或合并告警 |
| `GET /api/vision/inspections` | 巡检记录：检测框、各阶段耗时、关联告警 |
| `GET /api/aeration/options` | 曝气仿真的策略、场景、设定值范围与当前整定参数 |
| `POST /api/aeration/simulate` | 运行一次对照仿真：`scenario`、`setpoint`（1.0–3.0，按 0.1 取整）、`seed`、`strategies`；返回各策略指标与 24 小时 5 分钟曲线，注明“仿真结果，仅用于说明方法” |

```bash
curl -b /tmp/hs.jar -X PUT localhost:8000/api/demo/speed -H 'content-type: application/json' -d '{"value":60}'
curl -b /tmp/hs.jar -X POST localhost:8000/api/demo/scenarios -H 'content-type: application/json' -d '{"kind":"storm"}'
curl "localhost:8000/api/plants/P03/readings?metrics=inflow_m3h,nh3n_out,mlss&bucket=15m"
```

设计要点：
- 列名只能来自指标目录白名单，经转义后才拼进 SQL。
- 统计按 5 分钟分桶做时间加权，避免回填数据（1 分钟一条）和实时数据（5 秒一条）采样密度不同带来偏差。
- 演示命令带请求 id，多个请求同时操作也不会拿错回复。

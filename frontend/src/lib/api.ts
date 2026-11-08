// 后端接口的最小封装：同源 /api（开发时 Vite 转发，部署时 nginx 转发）
import { withLogin } from './auth'

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
  }
}

async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const send = () => fetch(path, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  })
  // 读接口公开；写接口未登录时弹登录框，登录后重试
  const res = method === 'GET' ? await send() : await withLogin(send)
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try {
      const j = await res.json()
      if (typeof j.detail === 'string') msg = j.detail
      else if (Array.isArray(j.detail)) msg = j.detail.map((d: { msg: string }) => d.msg).join('；')
    } catch { /* 非 JSON 错误体 */ }
    throw new ApiError(res.status, msg)
  }
  return res.status === 204 ? (undefined as T) : res.json()
}

export const api = {
  get: <T>(path: string, signal?: AbortSignal) => request<T>('GET', path, undefined, signal),
  post: <T>(path: string, body?: unknown, signal?: AbortSignal) => request<T>('POST', path, body ?? {}, signal),
  put: <T>(path: string, body?: unknown) => request<T>('PUT', path, body ?? {}),
  del: <T>(path: string) => request<T>('DELETE', path),
}

// ---------------------------------------------------------------- 类型（只列前端用到的字段）

export type Status = 'normal' | 'warn' | 'alarm' | 'offline'

export interface Issue { metric: string | null; label: string; level: Status; value?: number; limit?: number }

export interface PlantBrief {
  code: string; name: string; kind: string; process: string; capacity_m3d: number
  lng: number; lat: number; status: Status; issues: Issue[]; latest_ts: string | null
  latest: Record<string, number>
}

export interface Overview {
  now: string
  weather: { ts: string; rain_mm_h: number; temp_c: number }
  today: { water_m3: number; aeration_kwh: number; aeration_kwh_per_m3: number; carbon_kg: number; pac_kg: number; compliance_rate: number }
  status_counts: Record<Status, number>
  plants: PlantBrief[]
  hourly: { ts: string; inflow_m3h: number; blower_kw: number; rain_mm_h: number }[]
  scenarios: ActiveScenario[]
}

export interface ActiveScenario { kind: string; plant_code: string | null; start: string; end: string; duration_min: number; description: string; params: Record<string, number> }

export interface MetricInfo { key: string; label: string; unit: string; group: string; limit: number | null; lower_is_bad: boolean; warn_at?: number | null }

export interface DeviceInfo {
  code: string; kind: string; model: string; rated_kw: number; installed_at: string; status: string
  level: 'normal' | 'warn' | 'alarm'; latest_ts: string | null; running: boolean | null
  current_a: number | null; vibration_mms: number | null; temp_c: number | null
}
export interface PlantDetail extends PlantBrief {
  today: { hours: number; water_m3: number; aeration_kwh: number; aeration_kwh_per_m3: number; carbon_kg: number; pac_kg: number; compliance_rate: number } | null
  devices: DeviceInfo[]
}

export interface Alarm {
  id: number; ts: string; ended_at: string | null; active: boolean; duration_min: number | null
  plant: { id: number; code: string; name: string } | null
  device: { id: number; code: string; kind?: string } | null
  level: '告警' | '预警' | '提示'; kind: string; rule: string; metric: string | null; metric_label: string | null
  unit: string | null; value: number | null; threshold: number | null; peak: number | null; peak_at: string | null
  message: string; status: '未处理' | '处理中' | '已关闭'
  detail: Record<string, any>; closed_at: string | null; close_note: string | null; advice_at: string | null
}

export interface AlarmList { total: number; items: Alarm[]; summary: { total: number; active: number; by_level: Record<string, number>; by_status: Record<string, number> } }

// 处置建议与 Agent 回答同构：结论逐条引用证据单元格或知识原文
export interface Advice {
  ok: boolean; refused: boolean; answer: string; reason: string | null; error: string | null
  claims: { kind: string; text: string; citation_numbers: number[] }[]
  citations: Citation[]; evidence: ToolCall[]; models: string[]; warnings: string[]
  prompt_tokens: number; completion_tokens: number; total_ms: number; generated_at: string
}
export interface AlarmDetail extends Alarm { advice: Advice | null; related: Alarm[]; work_orders: WorkOrder[] }

export interface WoStep { text: string; source: 'ai' | 'template' | 'user'; citations: Record<string, any>[] }
export interface WorkOrder {
  id: number; alarm_id: number | null; title: string; assignee: string; status: '待处理' | '处理中' | '已关闭'
  steps: WoStep[]; resolution: string | null
  plant: { id: number; code: string | null; name: string | null }
  alarm: { level: string; kind: string; message: string; status: string } | null
  created_at: string; started_at: string | null; closed_at: string | null; source: Record<string, any>
}
export interface WorkOrderList { total: number; counts: Record<string, number>; items: WorkOrder[] }
export interface Draft {
  draft_id: string; alarm_id: number; title: string; steps: WoStep[]
  source: { kind: 'chat_answer' | 'ai_advice' | 'template'; note?: string; models?: string[]; question?: string }; expires_in_s: number
}

export interface Series { entity: string; bucket: string; metrics: MetricInfo[]; ts: string[]; series: Record<string, (number | null)[]> }

// ---- 报告
export interface ReportBrief { id: number; kind: '日报' | '周报'; period_start: string; period_end: string; created_at: string; narrative: string | null; model: string | null; total_ms: number | null }
export interface Report {
  id: number; kind: string; period_start: string; period_end: string; created_at: string; content_md: string
  facts: Record<string, { id: string; label: string; text: string; value: unknown; llm?: boolean }>
  meta: { narrative?: string; model?: string; total_ms?: number; llm_ms?: number; prompt_tokens?: number; completion_tokens?: number
          attempts?: { errors: string[] }[]; fallback_reason?: string; [k: string]: unknown }
}

// ---- 视觉巡检
export interface Detection { cls: string; score: number; box: [number, number, number, number] }
export interface VisionSample { name: string; scene: string; source: string; camera: string; description: string; note: string; attribution: string | null; url: string }
export interface VisionResult {
  model_id: string; source: string; camera: string | null; plant: { code: string; name: string } | null
  width: number; height: number; detections: Detection[]; timings_ms: Record<string, number>
  inspection_id: number | null; events: { event: string; alarm_id: number; rule: string; level?: string }[]; alarm_note: string | null; ts?: string
}
export interface VisionModel {
  name: string; classes: string[]; class_labels: Record<string, string>; conf: number; model_id: string
  metrics: Record<string, Record<string, any>>; limitations: string[]; license: Record<string, string>
  training: Record<string, any>; runtime: { onnxruntime: string; providers: string[] }
}
export interface Inspection {
  id: number; ts: string; plant: string | null; camera: string | null; source: string; width: number; height: number
  detections: Detection[]; timings_ms: Record<string, number>; alarm_ids: number[] | null
}

// ---- 曝气仿真
export interface AerationMetrics {
  do_mae: number; do_p95_abs_err: number; do_low_pct: number; do_high_pct: number
  kwh: number; kwh_per_m3: number; nh3n_mean: number; nh3n_max: number; nh3n_over_limit_h: number
  air_moves_per_h: number; kwh_vs_fixed_pct?: number
}
export interface AerationResult {
  scenario: string; scenario_label: string; seed: number; setpoint: number; plant: string; notice: string
  series: { t_min: number[]; load: number[]; inflow: number[] }
  strategies: Record<string, { label: string; metrics: AerationMetrics; series: { do: number[]; air_frac: number[]; kw: number[]; nh3n_out: number[] } }>
}
export interface AerationOptions {
  strategies: { id: string; label: string }[]
  scenarios: { id: string; label: string }[]
  setpoint: { min: number; max: number; step: number; default: number }
  notice: string
}

// ---- Agent
export interface Citation {
  number: number; kind: 'data' | 'statistic' | 'knowledge'; call_id: string
  value?: unknown; column_name?: string; row?: number; text?: string; unit?: string
  title?: string; heading?: string; quote?: string; url?: string; source?: string
}
export interface ToolCall {
  id: string; tool: string; question: string; status: string; error: string | null; total_ms: number | null
  output: { sql?: string; columns?: string[]; rows?: unknown[][]; ok?: boolean } | null
}
export interface ChatResult {
  question: string; run_id: string; session_id: string; ok: boolean; refused: boolean; answer: string; reason: string | null; error: string | null
  claims: { kind: string; text: string; citation_numbers: number[] }[]
  citations: Citation[]; tool_calls: ToolCall[]; models: string[]
  prompt_tokens: number; completion_tokens: number; total_ms: number; warnings: string[]
  actions?: ChatAction[]
  followups?: ChatFollowup[]        // 本轮之后的人工操作（已派单），刷新后据此显示派单结果
}
export interface ChatFollowup { type: 'work_order'; work_order_id: number; alarm_id: number; at: string; text: string }

// 对话里的派单入口：服务端按本轮查询范围（厂站、时间段、指标）或用户写明的编号匹配告警，不经模型
export interface ChatAction {
  type: 'work_order'; alarm_id: number; plant_code: string; plant_name: string; device_code: string | null
  level: string; kind: string; metric: string | null; metric_label: string; message: string
  ts: string; ended_at: string | null; status: string; open_work_order_id: number | null; matched_by: 'explicit' | 'scope'
}

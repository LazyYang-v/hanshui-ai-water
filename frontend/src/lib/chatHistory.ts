// 对话历史：会话 id 存在本机，刷新后用它从服务端取回最近 6 轮（Redis 保留 24 小时）
import type { ChatResult, ToolCall } from './api'

export interface Step { id: string; text: string; state: 'run' | 'ok' | 'fail'; ms?: number | null }
export interface Turn {
  question: string
  steps: Step[]
  result: ChatResult | null
  error: string
  running: boolean
  startedAt: number
  elapsed?: number
}

const KEY = 'shuiwu.chat.session'
export const TOOL_LABEL: Record<string, string> = { sql_query: '查数（Text2SQL）', rag_search: '知识检索', plant_statistics: '固定统计' }
const STATUS_LABEL: Record<string, string> = { empty: '无结果', refused: '被拒绝', error: '出错', cancelled: '已停止' }

// 隐私模式等情况下存储可能不可用，此时只是不恢复历史
export function loadSessionId(): string | null {
  try { return localStorage.getItem(KEY) } catch { return null }
}
export function saveSessionId(id: string | null): void {
  try {
    if (id) localStorage.setItem(KEY, id)
    else localStorage.removeItem(KEY)
  } catch { /* 同上 */ }
}

export const stepOk = (status: string) => status === 'ok' || status === 'empty'
export const statusNote = (status: string, error?: string | null) =>
  status === 'ok' ? '' : `（${STATUS_LABEL[status] ?? status}${error ? '：' + error : ''}）`

function toolStep(c: ToolCall): Step {
  return {
    id: c.id,
    text: `${TOOL_LABEL[c.tool] ?? c.tool}：${c.question ?? ''}${statusNote(c.status, c.error)}`,
    state: stepOk(c.status) ? 'ok' : 'fail',
    ms: c.total_ms,
  }
}

export function restoredTurn(r: ChatResult): Turn {
  return { question: r.question, steps: r.tool_calls.map(toolStep), result: r, error: '', running: false, startedAt: 0 }
}

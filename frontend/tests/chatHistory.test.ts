// 刷新后恢复对话：node --test
import assert from 'node:assert/strict'
import { afterEach, test } from 'node:test'
import type { ChatResult, ToolCall } from '../src/lib/api.ts'
import { loadSessionId, restoredTurn, saveSessionId } from '../src/lib/chatHistory.ts'

const g = globalThis as { localStorage?: unknown }
afterEach(() => { delete g.localStorage })

const call = (over: Partial<ToolCall>): ToolCall =>
  ({ id: 't1', tool: 'sql_query', question: '查氨氮', status: 'ok', error: null, total_ms: 120, output: null, ...over })

test('恢复的一轮按工具调用重建步骤，失败原因保留', () => {
  const r = { question: '问题', tool_calls: [
    call({}),
    call({ id: 't2', tool: 'rag_search', status: 'empty' }),
    call({ id: 't3', status: 'error', error: '超时' }),
  ] } as ChatResult
  const t = restoredTurn(r)
  assert.equal(t.question, '问题')
  assert.equal(t.running, false)
  assert.equal(t.result, r)
  assert.equal(t.elapsed, undefined)              // 耗时回退到服务端记录的 total_ms
  assert.deepEqual(t.steps.map((s) => s.state), ['ok', 'ok', 'fail'])
  assert.equal(t.steps[0].text, '查数（Text2SQL）：查氨氮')
  assert.equal(t.steps[1].text, '知识检索：查氨氮（无结果）')
  assert.equal(t.steps[2].text, '查数（Text2SQL）：查氨氮（出错：超时）')
})

test('会话 id 可写入、读取、清除', () => {
  const m = new Map<string, string>()
  g.localStorage = { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => m.set(k, v), removeItem: (k: string) => m.delete(k) }
  assert.equal(loadSessionId(), null)
  saveSessionId('abc')
  assert.equal(loadSessionId(), 'abc')
  saveSessionId(null)
  assert.equal(loadSessionId(), null)
})

test('存储不可用时不抛错', () => {
  const boom = () => { throw new Error('denied') }
  g.localStorage = { getItem: boom, setItem: boom, removeItem: boom }
  assert.equal(loadSessionId(), null)
  assert.doesNotThrow(() => saveSessionId('abc'))
})

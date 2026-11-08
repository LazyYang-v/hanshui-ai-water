// POST + SSE：EventSource 只支持 GET，这里用 fetch 读流并按空行切事件
import { withLogin } from './auth'

export async function postSSE(path: string, body: unknown, onEvent: (event: string, data: any) => void,
                              signal?: AbortSignal): Promise<void> {
  const res = await withLogin(() => fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
  }), '使用 AI 助手需要管理员登录（会调用大模型）')
  if (!res.ok || !res.body) {
    let msg = `${res.status} ${res.statusText}`
    try { const j = await res.json(); if (typeof j.detail === 'string') msg = j.detail } catch { /* ignore */ }
    throw new Error(msg)
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += value
    let i: number
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const block = buf.slice(0, i)
      buf = buf.slice(i + 2)
      let event = 'message'
      const data: string[] = []
      for (const line of block.split('\n')) {
        if (line.startsWith(':')) continue            // 心跳注释
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
      }
      if (data.length) onEvent(event, JSON.parse(data.join('\n')))
    }
  }
}

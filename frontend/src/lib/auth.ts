// 管理员会话：会话号在 HttpOnly Cookie 里，前端只知道“是否已登录、用户名”。
// 写接口返回 401 时弹出登录框，登录成功后由调用方重试一次；取消则按原错误处理。
import { reactive } from 'vue'

export const auth = reactive({
  user: null as string | null,
  dialog: false,            // 登录框是否打开
  reason: '',               // 打开原因（如“生成报告需要管理员登录”）
})

let waiters: { resolve: () => void; reject: (e: Error) => void }[] = []

export async function refreshMe() {
  try {
    const r = await fetch('/api/auth/me')
    auth.user = r.ok ? ((await r.json()).username ?? null) : null
  } catch { auth.user = null }
}

/** 打开登录框，等待登录成功（resolve）或取消（reject）。并发的多个 401 共用一个登录框。 */
export function requireLogin(reason = '该操作需要管理员登录'): Promise<void> {
  auth.reason = reason
  auth.dialog = true
  return new Promise((resolve, reject) => waiters.push({ resolve, reject }))
}

export async function login(username: string, password: string): Promise<void> {
  const r = await fetch('/api/auth/login', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, password }),
  })
  if (!r.ok) {
    let msg = `${r.status}`
    try { const j = await r.json(); if (typeof j.detail === 'string') msg = j.detail } catch { /* 非 JSON */ }
    throw new Error(r.status === 422 ? '请输入用户名和密码' : msg)
  }
  auth.user = (await r.json()).username
  auth.dialog = false
  const w = waiters; waiters = []
  w.forEach((x) => x.resolve())
}

export function cancelLogin() {
  auth.dialog = false
  const w = waiters; waiters = []
  w.forEach((x) => x.reject(new Error('需要管理员登录')))
}

export async function logout() {
  await fetch('/api/auth/logout', { method: 'POST' }).catch(() => undefined)
  auth.user = null
}

/** 发起写请求；401 时先登录再重试一次。fetchFn 每次调用都要重新构造请求（请求体流不能复用）。 */
export async function withLogin(fetchFn: () => Promise<Response>, reason?: string): Promise<Response> {
  const res = await fetchFn()
  if (res.status !== 401) return res
  auth.user = null
  await requireLogin(reason)
  return fetchFn()
}

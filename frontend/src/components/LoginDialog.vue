<script setup lang="ts">
// 管理员登录框：写操作遇到 401 时自动弹出，也可从顶栏打开
import { nextTick, ref, watch } from 'vue'
import { auth, cancelLogin, login } from '../lib/auth'

const username = ref('admin')
const password = ref('')
const busy = ref(false)
const err = ref('')
const pwEl = ref<HTMLInputElement | null>(null)

watch(() => auth.dialog, async (open) => {
  if (!open) return
  err.value = ''; password.value = ''
  await nextTick(); pwEl.value?.focus()
})

async function submit() {
  busy.value = true; err.value = ''
  try { await login(username.value.trim(), password.value) } catch (e) { err.value = (e as Error).message } finally { busy.value = false }
}
</script>

<template>
  <div v-if="auth.dialog" class="mask" @click.self="cancelLogin" @keydown.esc="cancelLogin">
    <form class="panel dlg" @submit.prevent="submit">
      <h3>管理员登录</h3>
      <p class="dim">{{ auth.reason || '登录后可生成报告、调用 AI、派单与操作演示控制台' }}</p>
      <label>用户名<input v-model="username" class="input" autocomplete="username" /></label>
      <label>密码<input ref="pwEl" v-model="password" class="input" type="password" autocomplete="current-password" /></label>
      <p v-if="err" class="err">{{ err }}</p>
      <div class="ops">
        <button type="button" class="btn" @click="cancelLogin">取消</button>
        <button type="submit" class="btn primary" :disabled="busy || !password">{{ busy ? '登录中…' : '登录' }}</button>
      </div>
      <p class="dim hint">查看页面无需登录；会话 8 小时有效，可随时退出。</p>
    </form>
  </div>
</template>

<style scoped>
.mask { position: fixed; inset: 0; z-index: 100; background: rgba(2, 8, 20, 0.66); display: grid; place-items: center; padding: 16px; }
.dlg { width: min(360px, 100%); display: flex; flex-direction: column; gap: 10px; background: var(--bg-2); }
.dlg h3 { margin: 0; font-size: 16px; color: var(--strong); }
.dlg p { margin: 0; font-size: 12px; }
label { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--dim); }
.ops { display: flex; justify-content: flex-end; gap: 8px; margin-top: 4px; }
.hint { font-size: 11px; }
</style>

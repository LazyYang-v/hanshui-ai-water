<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import AiDrawer from './components/AiDrawer.vue'
import AlarmToasts from './components/AlarmToasts.vue'
import LoginDialog from './components/LoginDialog.vue'
import { api } from './lib/api'
import { auth, logout, refreshMe, requireLogin } from './lib/auth'
import { connectRealtime, realtime } from './lib/realtime'

const route = useRoute()
const tabs = [
  { to: '/', label: '集控总览' },
  { to: '/plants', label: '厂站详情' },
  { to: '/alarms', label: '告警工单' },
  { to: '/aeration', label: '智能曝气' },
  { to: '/vision', label: '视觉巡检' },
  { to: '/reports', label: '报告中心' },
  { to: '/demo', label: '演示控制' },
]
const drawer = ref(false)
const clock = ref<{ now: string; speed: number } | null>(null)
let timer: number | undefined

async function loadClock() {
  try { clock.value = await api.get('/api/clock') } catch { clock.value = null }
}
onMounted(() => { connectRealtime(); loadClock(); refreshMe(); timer = window.setInterval(loadClock, 10000) })
onUnmounted(() => clearInterval(timer))

const now = computed(() => {
  const iso = realtime.frame?.ts ?? clock.value?.now
  if (!iso) return '—'
  return new Date(iso).toLocaleString('zh-CN', { hour12: false, timeZone: 'Asia/Shanghai' })
})
</script>

<template>
  <header class="top">
    <div class="brand">
      <span class="logo">汉水智控</span>
      <span class="dim">AI + 水务智慧运营平台 · Demo（数据由模拟器生成）</span>
    </div>
    <nav>
      <RouterLink v-for="t in tabs" :key="t.to" :to="t.to" :class="{ on: t.to === '/' ? route.path === '/' : route.path.startsWith(t.to) }">{{ t.label }}</RouterLink>
    </nav>
    <div class="right">
      <span :class="realtime.connected ? 's-normal' : 's-alarm'" :title="realtime.connected ? '实时推送已连接' : '实时推送断开，正在重连'">●</span>
      <span class="num">{{ now }}</span>
      <span v-if="clock && clock.speed !== 1" class="tag s-warn">{{ clock.speed }}× 倍速</span>
      <span v-if="auth.user" class="user" title="管理员会话">{{ auth.user }}<button class="link" @click="logout">退出</button></span>
      <button v-else class="btn" @click="requireLogin('').catch(() => undefined)">登录</button>
      <button class="btn primary" @click="drawer = !drawer">AI 助手</button>
    </div>
  </header>
  <main>
    <RouterView />
  </main>
  <AlarmToasts />
  <AiDrawer v-model:open="drawer" />
  <LoginDialog />
</template>

<style scoped>
.top {
  height: 56px; display: flex; align-items: center; gap: 24px; padding: 0 20px;
  background: linear-gradient(180deg, rgba(20, 60, 130, 0.65), rgba(5, 13, 31, 0.2));
  border-bottom: 1px solid var(--panel-border);
  position: sticky; top: 0; z-index: 10; backdrop-filter: blur(6px);
}
.brand { display: flex; align-items: baseline; gap: 12px; min-width: 112px; flex: 0 1 auto; }
.brand .dim { font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.logo {
  font-size: 22px; font-weight: 700; letter-spacing: 2px; white-space: nowrap;
  background: linear-gradient(90deg, #fff, var(--cyan)); -webkit-background-clip: text; background-clip: text; color: transparent;
}
nav { display: flex; gap: 4px; min-width: 0; overflow-x: auto; scrollbar-width: none; }
nav a {
  color: var(--dim); text-decoration: none; padding: 6px 16px; border-radius: 4px; white-space: nowrap;
  border: 1px solid transparent;
}
nav a.on { color: var(--strong); border-color: rgba(47, 211, 255, 0.5); background: rgba(47, 211, 255, 0.12); }
.right { margin-left: auto; display: flex; align-items: center; gap: 12px; white-space: nowrap; }
.user { font-size: 12px; color: var(--dim); }
.link { background: none; border: 0; padding: 0 0 0 6px; color: var(--cyan); cursor: pointer; font: inherit; }
main { padding: 14px 16px 20px; }
@media (max-width: 1360px) {
  .brand .dim { display: none; }
  nav a { padding: 6px 10px; }
}
@media (max-width: 900px) {
  .top { flex-wrap: wrap; height: auto; padding: 8px 12px; gap: 8px; }
  .brand .dim { display: none; }
  main { padding: 10px; }
}
</style>

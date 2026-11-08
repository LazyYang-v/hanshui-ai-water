<script setup lang="ts">
// 新告警 / 升级时右下角弹出提示（10 秒后消失，最多 3 条），点击进入告警详情
import { ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { realtime } from '../lib/realtime'

interface Toast { key: number; id: number; level: string; message: string; escalated: boolean }
const router = useRouter()
const toasts = ref<Toast[]>([])
let key = 0

watch(() => realtime.alarmSeq, () => {
  const ev = realtime.alarms[0]
  const a = ev?.alarm as { id: number; level: string; message: string } | undefined
  if (!a || (ev.event !== 'new' && ev.event !== 'escalated')) return
  const t = { key: ++key, id: a.id, level: a.level, message: a.message, escalated: ev.event === 'escalated' }
  toasts.value = [t, ...toasts.value].slice(0, 3)
  setTimeout(() => { toasts.value = toasts.value.filter((x) => x.key !== t.key) }, 10000)
})

function open(t: Toast) {
  toasts.value = toasts.value.filter((x) => x.key !== t.key)
  router.push(`/alarms?id=${t.id}`)
}
</script>

<template>
  <TransitionGroup name="toast" tag="div" class="toasts">
    <button v-for="t in toasts" :key="t.key" class="toast" :class="t.level === '告警' ? 'lv-alarm' : 'lv-warn'" @click="open(t)">
      <b>{{ t.escalated ? '告警升级' : '新' + t.level }} #{{ t.id }}</b>
      <span>{{ t.message }}</span>
      <small class="dim">点击查看详情与处置</small>
    </button>
  </TransitionGroup>
</template>

<style scoped>
.toasts { position: fixed; right: 16px; bottom: 16px; z-index: 30; display: flex; flex-direction: column; gap: 8px; width: 340px; }
.toast {
  text-align: left; display: flex; flex-direction: column; gap: 3px; padding: 10px 12px; border-radius: 6px; cursor: pointer;
  background: rgba(8, 20, 45, 0.96); border: 1px solid; color: var(--text); box-shadow: 0 6px 24px rgba(0, 0, 0, 0.5);
}
.toast b { font-size: 14px; }
.lv-alarm { border-color: var(--red); animation: flash 1s ease-in-out 3; }
.lv-alarm b { color: var(--red); }
.lv-warn { border-color: var(--yellow); }
.lv-warn b { color: var(--yellow); }
@keyframes flash { 50% { box-shadow: 0 0 22px rgba(255, 90, 106, 0.75); } }
.toast-enter-active, .toast-leave-active { transition: all 0.25s ease; }
.toast-enter-from, .toast-leave-to { opacity: 0; transform: translateX(40px); }
</style>

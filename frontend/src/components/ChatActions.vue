<script setup lang="ts">
// AI 回答下方的派单入口。告警由服务端按本轮查询范围匹配（不是模型判断）；
// 草稿与确认派单走与告警页相同的接口，派单需人工确认并登录，结果记回本次对话。
import { reactive } from 'vue'
import { RouterLink } from 'vue-router'
import { api, type ChatAction, type Draft } from '../lib/api'
import { mdhm } from '../lib/format'
import WorkOrderDraftForm from './WorkOrderDraftForm.vue'

const props = defineProps<{ actions: ChatAction[]; chat: { session_id: string; run_id: string } }>()

interface State { busy: boolean; err: string; draft: Draft | null; dispatched: { id: number; recorded: boolean } | null }
const st = reactive<Record<number, State>>({})
for (const a of props.actions) st[a.alarm_id] = { busy: false, err: '', draft: null, dispatched: null }

const LEVEL_CLASS: Record<string, string> = { 告警: 's-alarm', 预警: 's-warn', 提示: 's-normal' }

async function makeDraft(a: ChatAction) {
  const s = st[a.alarm_id]
  s.busy = true; s.err = ''
  // 带上本轮对话：草稿步骤优先用这轮回答里的核查建议（服务端核对过这轮给过该告警的入口）
  try { s.draft = await api.post<Draft>(`/api/alarms/${a.alarm_id}/work-order-draft`, { chat: props.chat }) }
  catch (e) { s.err = (e as Error).message } finally { s.busy = false }
}
function onDone(a: ChatAction, wo: { id: number; chat_recorded?: boolean }) {
  const s = st[a.alarm_id]
  s.draft = null
  s.dispatched = { id: wo.id, recorded: !!wo.chat_recorded }
}
</script>

<template>
  <section class="acts">
    <div class="hd dim">
      {{ actions.some((a) => a.matched_by === 'scope') ? '相关的未关闭告警（服务端按本次查询的厂站和时间段匹配，不是 AI 判断）' : '你提到的告警' }}
    </div>
    <div v-for="a in actions" :key="a.alarm_id" class="act">
      <div class="line">
        <span class="tag" :class="LEVEL_CLASS[a.level]">{{ a.level }}</span>
        <RouterLink :to="`/alarms?id=${a.alarm_id}`" class="msg" :title="a.message">#{{ a.alarm_id }} {{ a.plant_code }} · {{ a.message }}</RouterLink>
        <span class="dim t">{{ mdhm(a.ts) }}{{ a.ended_at ? ' 已恢复' : ' 持续中' }}</span>
        <span v-if="st[a.alarm_id].dispatched" class="ok">
          已派单 #{{ st[a.alarm_id].dispatched!.id }}{{ st[a.alarm_id].dispatched!.recorded ? '，已记入本次对话' : '' }}
        </span>
        <RouterLink v-else-if="a.open_work_order_id" to="/alarms?tab=orders" class="dim">已有工单 #{{ a.open_work_order_id }}</RouterLink>
        <span v-else-if="a.status === '已关闭'" class="dim">已关闭</span>
        <button v-else-if="!st[a.alarm_id].draft" class="btn sm" :disabled="st[a.alarm_id].busy" @click="makeDraft(a)">
          {{ st[a.alarm_id].busy ? '生成中…' : '生成工单草稿' }}
        </button>
      </div>
      <div v-if="st[a.alarm_id].err" class="err">{{ st[a.alarm_id].err }}</div>
      <WorkOrderDraftForm v-if="st[a.alarm_id].draft" :draft="st[a.alarm_id].draft!" :chat="props.chat" compact
                          @done="(wo) => onDone(a, wo)" @cancel="st[a.alarm_id].draft = null" />
    </div>
  </section>
</template>

<style scoped>
.acts { margin-top: 8px; padding-top: 8px; border-top: 1px dashed rgba(64, 140, 255, 0.25); display: flex; flex-direction: column; gap: 6px; }
.hd { font-size: 11px; }
.act { display: flex; flex-direction: column; gap: 6px; }
.line { display: flex; align-items: center; gap: 6px; font-size: 12px; min-width: 0; }
.msg { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); text-decoration: none; }
.msg:hover { color: var(--cyan); }
.t { white-space: nowrap; }
.ok { color: var(--green); white-space: nowrap; }
.err { font-size: 12px; }
</style>

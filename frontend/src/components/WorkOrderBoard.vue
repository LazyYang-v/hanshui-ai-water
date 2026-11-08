<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api, type WorkOrder, type WorkOrderList } from '../lib/api'
import { mdhm } from '../lib/format'

const emit = defineEmits<{ alarm: [id: number] }>()
const COLS = ['待处理', '处理中', '已关闭'] as const
const list = ref<WorkOrderList | null>(null)
const err = ref('')
const busy = ref<number | null>(null)
const closingId = ref<number | null>(null)
const resolution = ref('')
const notice = ref('')
const open = ref<Set<number>>(new Set())

async function load() {
  try { list.value = await api.get<WorkOrderList>('/api/work-orders?limit=200') } catch (e) { err.value = (e as Error).message }
}
onMounted(load)
defineExpose({ load })

const byCol = (c: string) => list.value?.items.filter((w) => w.status === c) ?? []

async function act(w: WorkOrder, fn: () => Promise<{ notice?: string | null; alarm_closed?: boolean }>) {
  busy.value = w.id; err.value = ''; notice.value = ''
  try {
    const r = await fn()
    if (r.notice) notice.value = r.notice
    else if (r.alarm_closed) notice.value = `工单 #${w.id} 已关闭，关联告警 #${w.alarm_id} 数据已恢复，一并关闭`
    closingId.value = null; resolution.value = ''
    await load()
  } catch (e) { err.value = (e as Error).message } finally { busy.value = null }
}
const start = (w: WorkOrder) => act(w, () => api.post(`/api/work-orders/${w.id}/start`))
const close = (w: WorkOrder) => act(w, () => api.post(`/api/work-orders/${w.id}/close`, { resolution: resolution.value.trim() }))
function toggle(id: number) {
  const s = new Set(open.value)
  if (s.has(id)) s.delete(id); else s.add(id)
  open.value = s
}
const SRC_LABEL: Record<string, string> = { ai: 'AI', template: '模板', user: '人工' }
const LEVEL_CLASS: Record<string, string> = { 告警: 's-alarm', 预警: 's-warn', 提示: 's-normal' }
</script>

<template>
  <div class="board-wrap">
    <div v-if="notice" class="notice">{{ notice }}</div>
    <div v-if="err" class="err">{{ err }}</div>
    <div class="board">
      <section v-for="c in COLS" :key="c" class="panel col">
        <h3>{{ c }}<span class="sub num">{{ list?.counts[c] ?? 0 }}</span></h3>
        <div v-if="!byCol(c).length" class="dim empty">{{ c === '待处理' ? '在告警详情里生成草稿并确认后，工单出现在这里' : '无' }}</div>
        <article v-for="w in byCol(c)" :key="w.id" class="card">
          <div class="t"><b>#{{ w.id }}</b> {{ w.title }}</div>
          <div class="m dim">
            {{ w.plant.code }} · 负责人 {{ w.assignee }} · {{ mdhm(w.created_at) }}
            <template v-if="w.started_at"> · 开始 {{ mdhm(w.started_at) }}</template>
            <template v-if="w.closed_at"> · 关闭 {{ mdhm(w.closed_at) }}</template>
          </div>
          <div v-if="w.alarm" class="m">
            <span class="tag" :class="LEVEL_CLASS[w.alarm.level]">{{ w.alarm.level }}</span>
            <a href="#" @click.prevent="emit('alarm', w.alarm_id!)">告警 #{{ w.alarm_id }}</a>
            <span class="dim">{{ w.alarm.status }}</span>
            <span v-if="w.source.kind === 'ai_advice' || w.source.kind === 'chat_answer'" class="tag src-ai"
                  :title="w.source.kind === 'chat_answer' ? `取自 AI 对话：${w.source.question ?? ''}` : '取自告警的 AI 处置建议'">步骤含 AI 建议</span>
            <span v-if="w.source.via?.channel === 'chat'" class="tag">对话派单</span>
            <span v-if="w.source.edited?.steps || w.source.edited?.title" class="tag src-user">派单时人工修改</span>
          </div>
          <button class="link" @click="toggle(w.id)">{{ open.has(w.id) ? '收起步骤' : `查看 ${w.steps.length} 个步骤` }}</button>
          <ol v-if="open.has(w.id)" class="steps">
            <li v-for="(s, i) in w.steps" :key="i"><span class="dim">[{{ SRC_LABEL[s.source] }}]</span> {{ s.text }}</li>
          </ol>
          <div v-if="w.resolution" class="m">处理结果：{{ w.resolution }}</div>
          <div class="ops">
            <button v-if="w.status === '待处理'" class="btn sm primary" :disabled="busy === w.id" @click="start(w)">开始处理</button>
            <button v-if="w.status !== '已关闭' && closingId !== w.id" class="btn sm danger" :disabled="busy === w.id" @click="closingId = w.id; resolution = ''">关闭工单</button>
          </div>
          <div v-if="closingId === w.id" class="closing">
            <textarea v-model="resolution" class="input" rows="2" maxlength="500" placeholder="处理结果（必填），如：更换轴承，风机恢复运行" />
            <div class="ops">
              <button class="btn sm danger" :disabled="resolution.trim().length < 2 || busy === w.id" @click="close(w)">确认关闭</button>
              <button class="btn sm" @click="closingId = null">取消</button>
            </div>
            <div class="dim hint">关联告警数据已恢复时一并关闭告警；未恢复则告警保持处理中。</div>
          </div>
        </article>
      </section>
    </div>
  </div>
</template>

<style scoped>
.board-wrap { display: flex; flex-direction: column; gap: 10px; }
.board { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; align-items: start; }
.col { display: flex; flex-direction: column; gap: 8px; min-height: 300px; }
.empty { font-size: 12px; }
.card { border: 1px solid var(--panel-border); border-radius: 4px; padding: 8px 10px; background: rgba(5, 15, 40, 0.55); display: flex; flex-direction: column; gap: 5px; font-size: 13px; }
.card .t { color: var(--strong); }
.card .m { font-size: 12px; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.link { background: none; border: none; color: var(--cyan); padding: 0; cursor: pointer; font-size: 12px; text-align: left; }
.steps { margin: 0; padding-left: 18px; font-size: 12px; display: flex; flex-direction: column; gap: 3px; }
.ops { display: flex; gap: 6px; }
.closing { display: flex; flex-direction: column; gap: 6px; }
.closing textarea { resize: vertical; }
.hint { font-size: 11px; }
.src-ai { color: var(--cyan); }
.src-user { color: var(--yellow); }
@media (max-width: 1100px) { .board { grid-template-columns: 1fr; } }
</style>

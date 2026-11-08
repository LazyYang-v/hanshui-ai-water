<script setup lang="ts">
// 工单草稿编辑与人工确认：告警详情和 AI 对话共用。
// 步骤可改、可删、可加；未改动的步骤保留 AI 来源与引用。确认后才写库（需管理员登录）。
import { computed, ref, watch } from 'vue'
import { api, type Draft } from '../lib/api'

const props = defineProps<{ draft: Draft; chat?: { session_id: string; run_id: string } | null; compact?: boolean }>()
const emit = defineEmits<{ done: [wo: { id: number; chat_recorded?: boolean }]; cancel: [] }>()

const steps = ref<{ text: string; source: string; cites: number }[]>([])
const title = ref('')
const assignee = ref('')
const busy = ref(false)
const err = ref('')

watch(() => props.draft, (d) => {
  title.value = d.title
  steps.value = d.steps.map((s) => ({ text: s.text, source: s.source, cites: s.citations.length }))
  err.value = ''
}, { immediate: true })

const canConfirm = computed(() => assignee.value.trim() && title.value.trim().length >= 2 && steps.value.some((s) => s.text.trim()))

async function confirm() {
  busy.value = true; err.value = ''
  try {
    const wo = await api.post<{ id: number; chat_recorded?: boolean }>('/api/work-orders', {
      draft_id: props.draft.draft_id, assignee: assignee.value.trim(), title: title.value.trim(),
      steps: steps.value.map((s) => s.text.trim()).filter(Boolean),
      ...(props.chat ? { chat: props.chat } : {}),
    })
    emit('done', wo)
  } catch (e) { err.value = (e as Error).message } finally { busy.value = false }
}

const SRC_LABEL: Record<string, string> = { ai: 'AI 建议', template: '模板', user: '人工' }
</script>

<template>
  <div class="box" :class="{ compact }">
    <h4>工单草稿 <span class="dim">（未写库；确认后才派单，草稿 30 分钟内有效）</span></h4>
    <div v-if="draft.source.kind === 'template'" class="notice">{{ draft.source.note }}</div>
    <div v-else-if="draft.source.kind === 'chat_answer'" class="notice">
      {{ draft.source.note }}（提问：{{ draft.source.question }}）。保留引用；改写或新增的步骤标为人工。
    </div>
    <div v-else class="dim">步骤来自已保存的 AI 处置建议（{{ draft.source.models?.join(', ') }}），保留引用；改写或新增的步骤标为人工。</div>
    <label>标题<input v-model="title" class="input grow" maxlength="100" /></label>
    <ol class="steps">
      <li v-for="(s, i) in steps" :key="i">
        <span class="tag src" :class="'src-' + s.source" :title="s.source === 'ai' && s.cites ? `派单后保留 ${s.cites} 处引用` : ''">{{ SRC_LABEL[s.source] }}<template v-if="s.source === 'ai' && s.cites"> · {{ s.cites }} 引用</template></span>
        <textarea v-model="s.text" class="input grow" rows="2" maxlength="300" @input="s.source = 'user'" />
        <button class="btn sm danger" @click="steps.splice(i, 1)">删除</button>
      </li>
    </ol>
    <div class="row">
      <button class="btn sm" :disabled="steps.length >= 12" @click="steps.push({ text: '', source: 'user', cites: 0 })">添加步骤</button>
      <input v-model="assignee" class="input" maxlength="32" placeholder="负责人（必填）" />
      <button class="btn primary" :disabled="!canConfirm || busy" @click="confirm">{{ busy ? '派单中…' : '确认派单' }}</button>
      <button class="btn" :disabled="busy" @click="emit('cancel')">取消</button>
    </div>
    <div v-if="err" class="err">{{ err }}</div>
  </div>
</template>

<style scoped>
.box { border: 1px solid var(--panel-border); border-radius: 4px; padding: 10px; display: flex; flex-direction: column; gap: 8px; background: rgba(5, 15, 40, 0.5); }
.box label { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--dim); }
.row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.grow { flex: 1; min-width: 0; }
h4 { margin: 0; font-size: 14px; color: var(--strong); font-weight: 600; display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
h4 .dim { font-size: 12px; font-weight: 400; }
.steps { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 6px; }
.steps li { display: flex; gap: 6px; align-items: flex-start; }
.steps textarea { resize: vertical; font-size: 13px; }
.src { margin-top: 6px; }
.src-ai { color: var(--cyan); }
.src-template { color: var(--dim); }
.src-user { color: var(--yellow); }
.compact { padding: 8px; gap: 6px; }
.compact h4 { font-size: 13px; }
.compact .steps { padding-left: 16px; }
.compact .row .input { flex: 1; min-width: 110px; }
</style>

<script setup lang="ts">
import DOMPurify from 'dompurify'
import { marked } from 'marked'
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, type Report, type ReportBrief } from '../lib/api'
import { fmt, mdhm } from '../lib/format'

const route = useRoute()
const router = useRouter()
const selected = computed(() => (route.query.id ? Number(route.query.id) : null))

const items = ref<ReportBrief[]>([])
const kindFilter = ref('')
const report = ref<Report | null>(null)
const err = ref('')
const genKind = ref<'日报' | '周报'>('日报')
const genDate = ref('')
const genBusy = ref(false)
const genErr = ref('')

const day = (iso: string) => new Date(iso).toLocaleDateString('sv-SE', { timeZone: 'Asia/Shanghai' })   // YYYY-MM-DD
function period(r: { kind: string; period_start: string; period_end: string }) {
  const end = new Date(new Date(r.period_end).getTime() - 1)
  return r.kind === '日报' ? day(r.period_start) : `${day(r.period_start)} 至 ${day(end.toISOString())}`
}

async function loadList() {
  try {
    const qs = kindFilter.value ? `?kind=${encodeURIComponent(kindFilter.value)}&limit=50` : '?limit=50'
    items.value = (await api.get<{ items: ReportBrief[] }>(`/api/reports${qs}`)).items
    if (!selected.value && items.value.length) router.replace({ query: { id: String(items.value[0].id) } })
  } catch (e) { err.value = (e as Error).message }
}
watch(kindFilter, loadList)

async function loadReport(id: number | null) {
  report.value = null
  if (!id) return
  try { report.value = await api.get<Report>(`/api/reports/${id}`); err.value = '' } catch (e) { err.value = (e as Error).message }
}
watch(selected, loadReport, { immediate: true })

onMounted(async () => {
  loadList()
  try {
    const c = await api.get<{ now: string }>('/api/clock')
    // 默认生成当天的日报（截至演示时钟此刻），刚注入的场景能直接出现在报告里；要完整一天可改选昨天
    genDate.value = day(c.now)
  } catch { /* 忽略 */ }
})

async function generate() {
  genBusy.value = true; genErr.value = ''
  try {
    const r = await api.post<{ id: number }>('/api/reports', { kind: genKind.value, date: genDate.value || undefined })
    await loadList()
    router.replace({ query: { id: String(r.id) } })
  } catch (e) { genErr.value = (e as Error).message } finally { genBusy.value = false }
}

// 报告正文由服务端生成（含模型叙述），渲染前做 HTML 消毒
const html = computed(() => report.value ? DOMPurify.sanitize(marked.parse(report.value.content_md, { async: false }) as string) : '')
const facts = computed(() => Object.values(report.value?.facts ?? {}))
const attempts = computed(() => report.value?.meta.attempts ?? [])
</script>

<template>
  <div class="page">
    <section class="panel side">
      <h3>生成报告</h3>
      <div class="gen">
        <select v-model="genKind" class="input"><option>日报</option><option>周报</option></select>
        <input v-model="genDate" type="date" class="input" />
        <button class="btn primary" :disabled="genBusy" @click="generate">{{ genBusy ? '生成中（约 5–15 秒）…' : '生成' }}</button>
      </div>
      <div class="dim small">
        数字全部由 SQL 统计得出；DeepSeek Flash 只写叙述，用占位符引用统计值，服务端校验后填入。校验不通过会重试一次，仍失败则改用模板叙述。
        一次约 1 万 tokens。周报为截至所选日期的 7 天。
      </div>
      <div v-if="genErr" class="err">{{ genErr }}</div>

      <h3 class="mt">报告列表
        <span class="sub">
          <select v-model="kindFilter" class="input sm"><option value="">全部</option><option>日报</option><option>周报</option></select>
        </span>
      </h3>
      <ul>
        <li v-for="r in items" :key="r.id" :class="{ on: r.id === selected }" @click="router.replace({ query: { id: String(r.id) } })">
          <div><b>{{ r.kind }}</b> {{ period(r) }}</div>
          <div class="dim small">
            #{{ r.id }} · 生成于 {{ mdhm(r.created_at) }} ·
            <span :class="r.narrative === 'llm' ? 'llm' : 'tpl'">{{ r.narrative === 'llm' ? '模型叙述' : '模板叙述' }}</span>
          </div>
        </li>
        <li v-if="!items.length" class="dim none">还没有报告</li>
      </ul>
    </section>

    <section class="panel main">
      <div v-if="err" class="err">{{ err }}</div>
      <div v-if="!report" class="dim">{{ selected ? '加载中…' : '选择或生成一份报告' }}</div>
      <template v-else>
        <div class="bar">
          <span class="tag" :class="report.meta.narrative === 'llm' ? 'llm' : 'tpl'">{{ report.meta.narrative === 'llm' ? '模型叙述' : '模板叙述' }}</span>
          <span class="dim small">
            {{ report.meta.model ?? '未调用模型' }}
            <template v-if="report.meta.prompt_tokens"> · {{ report.meta.prompt_tokens }}+{{ report.meta.completion_tokens }} tokens</template>
            <template v-if="report.meta.total_ms"> · {{ fmt(report.meta.total_ms / 1000, 1) }} 秒</template>
            · {{ facts.length }} 项统计事实
          </span>
          <a class="btn sm" :href="`/api/reports/${report.id}/markdown`" download>下载 Markdown</a>
          <a class="btn sm" :href="`/api/reports/${report.id}/pdf`" download>下载 PDF</a>
        </div>
        <div class="mdwrap"><article class="md" v-html="html" /></div>
        <details class="block">
          <summary>生成与校验过程（{{ attempts.length }} 次尝试）</summary>
          <ol>
            <li v-for="(at, i) in attempts" :key="i">
              <span v-if="!at.errors.length" class="ok">第 {{ i + 1 }} 次：校验通过</span>
              <template v-else>
                <span class="warn">第 {{ i + 1 }} 次：{{ at.errors.length }} 处问题，已退回重写</span>
                <ul><li v-for="(e, k) in at.errors" :key="k" class="dim">{{ e }}</li></ul>
              </template>
            </li>
          </ol>
        </details>
        <details class="block">
          <summary>统计事实（叙述里的每个数字都来自这里）</summary>
          <table class="grid facts">
            <thead><tr><th>编号</th><th>含义</th><th>填入文本</th></tr></thead>
            <tbody><tr v-for="f in facts" :key="f.id"><td><code>{{ f.id }}</code></td><td>{{ f.label }}</td><td>{{ f.text }}</td></tr></tbody>
          </table>
        </details>
      </template>
    </section>
  </div>
</template>

<style scoped>
.page { padding: 14px 16px; display: grid; grid-template-columns: 340px 1fr; gap: 12px; align-items: start; }
.side { display: flex; flex-direction: column; gap: 8px; position: sticky; top: 70px; max-height: calc(100vh - 90px); }
.gen { display: flex; gap: 6px; }
.gen input { flex: 1; min-width: 0; }
.small { font-size: 12px; line-height: 1.6; }
.mt { margin-top: 8px !important; }
.input.sm { font-size: 12px; padding: 2px 6px; }
ul { list-style: none; margin: 0; padding: 0; overflow-y: auto; min-height: 0; }
li { padding: 7px 6px; border-bottom: 1px solid rgba(64, 140, 255, 0.12); cursor: pointer; font-size: 13px; }
li:hover { background: rgba(47, 211, 255, 0.05); }
li.on { background: rgba(47, 211, 255, 0.13); box-shadow: inset 2px 0 0 var(--cyan); }
.none { cursor: default; }
.llm { color: var(--cyan); }
.tpl { color: var(--yellow); }
.main { min-height: 400px; display: flex; flex-direction: column; gap: 10px; }
.bar { display: flex; align-items: center; gap: 10px; }
.bar a { margin-left: auto; text-decoration: none; }
.block summary { cursor: pointer; color: var(--cyan); font-size: 13px; }
.block ol { font-size: 13px; }
.ok { color: var(--green); }
.warn { color: var(--yellow); }
.facts td, .facts th { text-align: left; white-space: normal; }

/* Markdown 正文；宽表在容器内横向滚动 */
.mdwrap { overflow-x: auto; }
.md { line-height: 1.8; color: #dce9ff; max-width: 1100px; }
.md :deep(h1) { font-size: 22px; color: #fff; margin: 4px 0 12px; }
.md :deep(h2) { font-size: 17px; color: #fff; margin: 22px 0 8px; padding-left: 8px; border-left: 3px solid var(--cyan); }
.md :deep(h3) { font-size: 15px; color: #fff; margin: 16px 0 6px; }
.md :deep(p) { margin: 6px 0; }
.md :deep(blockquote) { margin: 8px 0; padding: 6px 12px; border-left: 3px solid var(--yellow); background: rgba(255, 197, 61, 0.06); color: var(--dim); font-size: 13px; }
.md :deep(table) { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0; }
.md :deep(td:first-child), .md :deep(th) { white-space: nowrap; }
.md :deep(th), .md :deep(td) { border: 1px solid rgba(64, 140, 255, 0.22); padding: 5px 8px; }
.md :deep(th) { background: rgba(47, 211, 255, 0.08); color: var(--dim); font-weight: 400; }
.md :deep(hr) { border: none; border-top: 1px solid rgba(64, 140, 255, 0.2); margin: 16px 0; }
.md :deep(code) { background: rgba(0, 0, 0, 0.3); padding: 0 4px; border-radius: 3px; }
@media (max-width: 1100px) { .page { grid-template-columns: 1fr; } .side { position: static; max-height: none; } }
</style>

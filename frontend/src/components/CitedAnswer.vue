<script setup lang="ts">
// 带引用的回答：正文里的 [n] 变成可点的角标，点开看对应的查询单元格 / 服务端统计 / 知识原文；
// 查数工具的结果能成图时自动画图，SQL 和结果表可展开核对。AI 助手和告警处置建议共用。
import { defineAsyncComponent, ref, watch } from 'vue'
import type { Citation, ToolCall } from '../lib/api'
// 图表组件连同 ECharts 按需加载，AI 助手抽屉在首屏就挂载，不能把图表库带进入口包
const AutoChart = defineAsyncComponent(() => import('./AutoChart.vue'))

const props = defineProps<{ answer: string; citations: Citation[]; tools?: ToolCall[] }>()
const picked = ref<Citation | null>(null)
watch(() => props.answer, () => { picked.value = null })

function segments(text: string) {
  return text.split(/(\[\d+\])/g).filter(Boolean).map((s) => {
    const m = s.match(/^\[(\d+)\]$/)
    return m ? { cite: +m[1] } : { text: s }
  })
}
function cite(n: number) {
  const c = props.citations.find((x) => x.number === n)
  if (c) picked.value = picked.value === c ? null : c
}
function citeText(c: Citation) {
  if (c.kind === 'data') return `${c.column_name} = ${String(c.value)}（${c.call_id} 第 ${(c.row ?? 0) + 1} 行，${c.source ?? '查询结果'}）`
  if (c.kind === 'statistic') return c.text ?? String(c.value)
  return `《${c.title}》${c.heading ? ' · ' + c.heading : ''}\n“${c.quote}”\n${c.source ?? ''}`
}
const KIND = { data: '查询结果单元格', statistic: '服务端统计', knowledge: '知识原文' }
const sqlTools = () => (props.tools ?? []).filter((x) => x.tool === 'sql_query' && x.output?.sql)
</script>

<template>
  <div class="answer">
    <template v-for="(seg, k) in segments(answer)" :key="k">
      <button v-if="'cite' in seg" class="cite" :class="{ on: picked?.number === seg.cite }" @click="cite(seg.cite!)">{{ seg.cite }}</button>
      <span v-else>{{ seg.text }}</span>
    </template>
  </div>
  <div v-if="picked" class="citebox">
    <div class="dim">引用 [{{ picked.number }}] · {{ KIND[picked.kind] }}</div>
    <pre>{{ citeText(picked) }}</pre>
    <a v-if="picked.url" :href="picked.url" target="_blank">查看原文片段</a>
  </div>
  <template v-for="tc in sqlTools()" :key="tc.id">
  <AutoChart :columns="tc.output!.columns" :rows="tc.output!.rows" :title="tc.id" />
  <details class="sql">
    <summary>{{ tc.id }} SQL：{{ tc.question }}</summary>
    <pre>{{ tc.output!.sql!.trim() }}</pre>
    <div v-if="tc.output!.columns" class="tbl">
      <table class="grid">
        <thead><tr><th>#</th><th v-for="c in tc.output!.columns" :key="c">{{ c }}</th></tr></thead>
        <tbody>
          <tr v-for="(r, ri) in (tc.output!.rows ?? []).slice(0, 30)" :key="ri">
            <td class="dim">{{ ri + 1 }}</td><td v-for="(v, ci) in r" :key="ci">{{ v }}</td>
          </tr>
        </tbody>
      </table>
      <div v-if="(tc.output!.rows?.length ?? 0) > 30" class="dim">共 {{ tc.output!.rows!.length }} 行，只显示前 30 行</div>
    </div>
  </details>
  </template>
</template>

<style scoped>
.answer { white-space: pre-wrap; line-height: 1.75; color: #e6f0ff; }
.cite { border: none; background: rgba(47, 211, 255, 0.2); color: var(--cyan); font-size: 11px; padding: 0 4px; margin: 0 1px; border-radius: 3px; cursor: pointer; vertical-align: super; line-height: 14px; }
.cite.on { background: var(--cyan); color: #00142e; }
.citebox { border-left: 2px solid var(--cyan); background: rgba(47, 211, 255, 0.06); padding: 6px 10px; font-size: 12px; }
.citebox pre { white-space: pre-wrap; margin: 4px 0; font-family: inherit; color: var(--text); }
.sql summary { cursor: pointer; font-size: 12px; color: var(--cyan); }
.sql pre { white-space: pre-wrap; word-break: break-all; font-size: 11px; background: rgba(0, 0, 0, 0.3); padding: 8px; border-radius: 4px; margin: 6px 0; }
.tbl { overflow-x: auto; font-size: 11px; }
.tbl td, .tbl th { text-align: left; }
</style>

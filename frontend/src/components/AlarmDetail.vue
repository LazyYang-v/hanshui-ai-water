<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { api, type AlarmDetail, type Draft, type MetricInfo, type Series } from '../lib/api'
import { C, axisStyle, tooltipStyle, useChart } from '../lib/chart'
import { fmt, mdhm } from '../lib/format'
import CitedAnswer from './CitedAnswer.vue'
import WorkOrderDraftForm from './WorkOrderDraftForm.vue'

const props = defineProps<{ id: number; metrics: { plant: MetricInfo[]; device: MetricInfo[] } | null }>()
const emit = defineEmits<{ select: [id: number]; changed: [] }>()

const a = ref<AlarmDetail | null>(null)
const err = ref('')
const busy = ref('')            // 正在进行的动作名
const actionErr = ref('')
const msg = ref('')
const chartEl = ref<HTMLElement | null>(null)
const chart = useChart(chartEl)
const hasChart = ref(false)

const LEVEL_CLASS: Record<string, string> = { 告警: 's-alarm', 预警: 's-warn', 提示: 's-normal' }
const STATUS_CLASS: Record<string, string> = { 未处理: 's-alarm', 处理中: 's-warn', 已关闭: 's-normal', 待处理: 's-alarm' }
const isVision = computed(() => a.value?.rule?.startsWith('vision_') ?? false)
const openOrder = computed(() => a.value?.work_orders.find((w) => w.status !== '已关闭'))

async function load(keep = false) {
  err.value = ''
  if (!keep) { a.value = null; draft.value = null; closing.value = false; actionErr.value = ''; msg.value = '' }
  try {
    a.value = await api.get<AlarmDetail>(`/api/alarms/${props.id}`)
    loadSeries()
  } catch (e) { err.value = (e as Error).message }
}

// ---------------------------------------------------------------- 指标曲线：告警前 3 小时到恢复后 1 小时
async function loadSeries() {
  const al = a.value
  if (!al || !props.metrics) return
  let path = '', metric = al.metric ?? ''
  if (al.device && props.metrics.device.some((m) => m.key === metric)) path = `/api/devices/${al.device.code}/readings`
  else if (al.device && metric === 'running') { path = `/api/devices/${al.device.code}/readings`; metric = 'current_a' }   // 停机看电流归零
  else if (al.plant && props.metrics.plant.some((m) => m.key === metric)) path = `/api/plants/${al.plant.code}/readings`
  hasChart.value = !!path
  if (!path) return
  const t0 = new Date(al.ts).getTime()
  const t1 = al.ended_at ? new Date(al.ended_at).getTime() + 3600e3 : null
  const qs = new URLSearchParams({ metrics: metric, start: new Date(t0 - 3 * 3600e3).toISOString() })
  if (t1) qs.set('end', new Date(Math.min(t1, t0 + 30 * 86400e3)).toISOString())
  try {
    const s = await api.get<Series>(`${path}?${qs}`)
    if (a.value?.id !== al.id) return
    const info = s.metrics[0]
    const data = s.ts.map((t, i) => [t, s.series[metric][i]])
    const threshold = metric === al.metric ? al.threshold : null
    chart.set({
      grid: { left: 48, right: 16, top: 30, bottom: 28 },
      tooltip: { ...tooltipStyle, valueFormatter: (v: number) => fmt(v, 2) + ' ' + info.unit },
      xAxis: { type: 'time', ...axisStyle },
      yAxis: { type: 'value', name: info.unit, scale: true, ...axisStyle },
      series: [{
        type: 'line', name: info.label, data, showSymbol: false, lineStyle: { width: 1.6, color: C.cyan }, itemStyle: { color: C.cyan },
        areaStyle: { color: 'rgba(47,211,255,0.08)' },
        markArea: { silent: true, itemStyle: { color: 'rgba(255,90,106,0.10)' },
                    data: [[{ xAxis: al.ts, name: '告警期间' }, { xAxis: al.ended_at ?? s.ts[s.ts.length - 1] }]],
                    label: { color: C.red, fontSize: 11 } },
        markLine: threshold == null ? undefined : { silent: true, symbol: 'none', lineStyle: { color: C.red, type: 'dashed' },
                    data: [{ yAxis: threshold }], label: { formatter: `阈值 ${threshold}`, color: C.red, position: 'insideEndTop' } },
      }],
      title: { text: `${info.label}（${s.bucket} 聚合）`, textStyle: { color: C.dim, fontSize: 12, fontWeight: 400 }, left: 0, top: 0 },
    })
  } catch { hasChart.value = false }
}

// ---------------------------------------------------------------- 动作
async function run(name: string, fn: () => Promise<void>) {
  busy.value = name; actionErr.value = ''; msg.value = ''
  try { await fn() } catch (e) { actionErr.value = (e as Error).message } finally { busy.value = '' }
}

const adviceInfo = ref('')
const genAdvice = (refresh: boolean) => run('advice', async () => {
  const r = await api.post<{ cached: boolean; saved: boolean; advice: AlarmDetail['advice'] }>(
    `/api/alarms/${props.id}/advice${refresh ? '?refresh=true' : ''}`)
  if (r.saved) { await load(true); emit('changed') }
  else if (a.value) { a.value = { ...a.value, advice: r.advice } }
  adviceInfo.value = r.saved ? '' : '本次结果未通过校验或被拒答，未保存'
})

// 工单草稿：编辑与确认见 WorkOrderDraftForm（与 AI 对话共用）
const draft = ref<Draft | null>(null)
const makeDraft = () => run('draft', async () => {
  draft.value = await api.post<Draft>(`/api/alarms/${props.id}/work-order-draft`)
})
async function onDispatched(wo: { id: number }) {
  draft.value = null
  await load(true)
  msg.value = `已派单 #${wo.id}，在「工单看板」跟进`
  emit('changed')
}

const closing = ref(false)
const note = ref('')
const closeAlarm = () => run('close', async () => {
  await api.post(`/api/alarms/${props.id}/close`, { note: note.value.trim() })
  closing.value = false; note.value = ''
  await load(true)
  emit('changed')
})

// 放在最后：load() 会用到上面声明的 draft / closing 等状态（immediate 回调在 setup 中同步执行）
watch(() => props.id, () => load(), { immediate: true })
watch(() => props.metrics, () => a.value && loadSeries())
</script>

<template>
  <div v-if="err" class="err">{{ err }}</div>
  <div v-else-if="!a" class="dim">加载中…</div>
  <div v-else class="detail">
    <div class="head">
      <span class="tag" :class="LEVEL_CLASS[a.level]">{{ a.level }}</span>
      <b class="title">#{{ a.id }} {{ a.kind }} · {{ a.message }}</b>
      <span class="tag" :class="STATUS_CLASS[a.status]">{{ a.status }}</span>
      <span v-if="a.active" class="tag s-alarm">数据仍异常</span>
      <span v-else class="tag s-normal">数据已恢复</span>
    </div>

    <dl class="info">
      <div><dt>厂站 / 设备</dt><dd>{{ a.plant?.code }} {{ a.plant?.name }}<template v-if="a.device"> · {{ a.device.code }}（{{ a.device.kind }}）</template></dd></div>
      <div><dt>开始</dt><dd class="num">{{ mdhm(a.ts) }}</dd></div>
      <div><dt>数据恢复</dt><dd class="num">{{ a.ended_at ? mdhm(a.ended_at) : '尚未恢复' }}</dd></div>
      <div><dt>持续</dt><dd class="num">{{ fmt(a.duration_min, 0) }} 分钟</dd></div>
      <div><dt>{{ a.metric_label || '指标' }}</dt><dd class="num">{{ fmt(a.value, 2) }} {{ a.unit }}<span class="dim"> / 阈值 {{ a.threshold ?? '—' }}</span></dd></div>
      <div><dt>最不利值</dt><dd class="num">{{ fmt(a.peak, 2) }} {{ a.unit }}<span v-if="a.peak_at" class="dim"> @ {{ mdhm(a.peak_at) }}</span></dd></div>
      <div><dt>规则</dt><dd><code>{{ a.rule }}</code></dd></div>
      <div v-if="a.closed_at"><dt>关闭</dt><dd>{{ mdhm(a.closed_at) }} · {{ a.close_note }}</dd></div>
    </dl>

    <div v-show="hasChart" ref="chartEl" class="chart" />
    <div v-if="isVision && a.detail?.camera" class="dim">摄像头 {{ a.detail.camera }}；检测框见「视觉巡检」页的巡检记录。</div>

    <div class="actions">
      <button class="btn primary" :disabled="!!busy || a.status === '已关闭' || !!openOrder" @click="makeDraft">
        {{ busy === 'draft' ? '生成中…' : '生成工单草稿' }}
      </button>
      <button class="btn danger" :disabled="!!busy || a.status === '已关闭' || !!openOrder" @click="closing = !closing">关闭告警</button>
      <span v-if="openOrder" class="dim">已有未关闭工单 #{{ openOrder.id }}（{{ openOrder.status }}），关闭工单后才能关闭告警</span>
      <span v-if="msg" class="ok">{{ msg }}</span>
      <span v-if="actionErr" class="err">{{ actionErr }}</span>
    </div>

    <div v-if="closing" class="box">
      <div class="dim">数据未恢复也可以关闭（如确认误报、计划内检修），须写明原因。</div>
      <div class="row">
        <input v-model="note" class="input grow" maxlength="300" placeholder="关闭说明，如：现场确认为仪表漂移，已校准" />
        <button class="btn danger" :disabled="note.trim().length < 2 || !!busy" @click="closeAlarm">确认关闭</button>
      </div>
    </div>

    <WorkOrderDraftForm v-if="draft" :draft="draft" @done="onDispatched" @cancel="draft = null" />

    <section class="block">
      <h4>AI 处置建议
        <span class="dim">固定证据查询 + 知识检索，结论逐条引用；原因只作假设，需人工核查</span>
      </h4>
      <div v-if="isVision" class="dim">视觉巡检告警暂不生成 AI 处置建议（知识库没有对应规程），可直接生成工单草稿（模板步骤）。</div>
      <template v-else>
        <template v-if="a.advice">
          <div v-if="a.advice.refused" class="notice">{{ a.advice.reason || a.advice.answer }}</div>
          <CitedAnswer v-else :answer="a.advice.answer" :citations="a.advice.citations" :tools="a.advice.evidence" />
          <div v-if="a.advice.warnings?.length" class="warn">注意：{{ a.advice.warnings.join('；') }}</div>
          <div class="dim meta">
            {{ a.advice.models.join(', ') }} · {{ a.advice.prompt_tokens + a.advice.completion_tokens }} tokens ·
            {{ (a.advice.total_ms / 1000).toFixed(1) }} 秒 · {{ a.advice.citations.length }} 处引用 · 生成于 {{ mdhm(a.advice.generated_at) }}
          </div>
        </template>
        <div v-else class="dim">尚未生成。</div>
        <div class="row">
          <button class="btn" :disabled="!!busy" @click="genAdvice(!!a.advice)">
            {{ busy === 'advice' ? '生成中（约 5–20 秒）…' : a.advice ? '重新生成' : '生成 AI 处置建议' }}
          </button>
          <span class="dim">调用 DeepSeek Flash，一次约 5 千 tokens</span>
          <span v-if="adviceInfo" class="warn">{{ adviceInfo }}</span>
        </div>
      </template>
    </section>

    <section v-if="a.work_orders.length" class="block">
      <h4>工单</h4>
      <div v-for="w in a.work_orders" :key="w.id" class="wo">
        <span class="tag" :class="STATUS_CLASS[w.status]">{{ w.status }}</span>
        #{{ w.id }} {{ w.title }} <span class="dim">· {{ w.assignee }} · {{ mdhm(w.created_at) }}</span>
        <div v-if="w.resolution" class="dim">处理结果：{{ w.resolution }}</div>
      </div>
    </section>

    <section class="block">
      <h4>前后 6 小时同厂告警 <span class="dim">{{ a.related.length }} 条</span></h4>
      <div v-if="!a.related.length" class="dim">无</div>
      <ul class="related">
        <li v-for="r in a.related" :key="r.id" @click="emit('select', r.id)">
          <span class="tag" :class="LEVEL_CLASS[r.level]">{{ r.level }}</span>
          <span class="num dim">{{ mdhm(r.ts) }}</span>
          <span class="m">#{{ r.id }} {{ r.message }}</span>
        </li>
      </ul>
    </section>
  </div>
</template>

<style scoped>
.detail { display: flex; flex-direction: column; gap: 12px; }
.head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.title { color: var(--strong); font-size: 15px; }
.info { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 6px 16px; margin: 0; }
.info div { display: flex; gap: 8px; font-size: 13px; }
.info dt { color: var(--dim); flex: none; width: 76px; }
.info dd { margin: 0; color: var(--strong); }
.chart { height: 220px; }
.actions, .row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.box { border: 1px solid var(--panel-border); border-radius: 4px; padding: 10px; display: flex; flex-direction: column; gap: 8px; background: rgba(5, 15, 40, 0.5); }
.grow { flex: 1; min-width: 0; }
h4 { margin: 0; font-size: 14px; color: var(--strong); font-weight: 600; display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
h4 .dim { font-size: 12px; font-weight: 400; }
.block { display: flex; flex-direction: column; gap: 8px; border-top: 1px solid rgba(64, 140, 255, 0.14); padding-top: 10px; }
.meta { font-size: 11px; }
.warn { color: var(--yellow); font-size: 12px; }
.ok { color: var(--green); }
.wo { font-size: 13px; }
.related { list-style: none; margin: 0; padding: 0; }
.related li { display: flex; gap: 8px; align-items: center; padding: 4px 0; font-size: 12px; cursor: pointer; border-bottom: 1px solid rgba(64, 140, 255, 0.1); }
.related li:hover { background: rgba(47, 211, 255, 0.06); }
.related .m { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>

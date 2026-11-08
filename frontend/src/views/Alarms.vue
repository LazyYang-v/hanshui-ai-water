<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AlarmDetail from '../components/AlarmDetail.vue'
import WorkOrderBoard from '../components/WorkOrderBoard.vue'
import { api, type AlarmList, type MetricInfo, type PlantBrief } from '../lib/api'
import { mdhm } from '../lib/format'
import { realtime } from '../lib/realtime'

const route = useRoute()
const router = useRouter()
const PAGE = 30

const tab = computed(() => (route.query.tab === 'orders' ? 'orders' : 'alarms'))
const selected = computed(() => (route.query.id ? Number(route.query.id) : null))
const setQuery = (q: Record<string, string | undefined>) => router.replace({ query: { ...route.query, ...q } })

const f = reactive({ status: '未关闭', level: '', kind: '', plant: typeof route.query.plant === 'string' ? route.query.plant : '', active: false })
const offset = ref(0)
const list = ref<AlarmList | null>(null)
const err = ref('')
const plants = ref<PlantBrief[]>([])
const kinds = ref<string[]>([])
const metrics = ref<{ plant: MetricInfo[]; device: MetricInfo[] } | null>(null)
const board = ref<InstanceType<typeof WorkOrderBoard> | null>(null)

async function load() {
  const qs = new URLSearchParams({ limit: String(PAGE), offset: String(offset.value) })
  for (const k of ['status', 'level', 'kind', 'plant'] as const) if (f[k]) qs.set(k, f[k])
  if (f.active) qs.set('active', 'true')
  try { list.value = await api.get<AlarmList>(`/api/alarms?${qs}`); err.value = '' } catch (e) { err.value = (e as Error).message }
}
watch(f, () => { offset.value = 0; load() })
watch(offset, load)
watch(() => realtime.alarmSeq, () => { load(); board.value?.load() })

onMounted(async () => {
  load()
  const [p, m, mon] = await Promise.all([
    api.get<PlantBrief[]>('/api/plants'),
    api.get<{ plant: MetricInfo[]; device: MetricInfo[] }>('/api/metrics'),
    api.get<{ rules: { kind: string }[] }>('/api/alarms/monitor'),
  ]).catch(() => [[], null, { rules: [] }] as const)
  plants.value = p as PlantBrief[]
  metrics.value = m
  kinds.value = [...new Set([...mon.rules.map((r) => r.kind), '视觉巡检'])]
})

function select(id: number) { setQuery({ id: String(id), tab: undefined }) }
function onChanged() { load(); board.value?.load() }

const LEVEL_CLASS: Record<string, string> = { 告警: 's-alarm', 预警: 's-warn', 提示: 's-normal' }
const STATUS_CLASS: Record<string, string> = { 未处理: 's-alarm', 处理中: 's-warn', 已关闭: 's-normal' }
</script>

<template>
  <div class="page">
    <div class="tabs">
      <button :class="{ on: tab === 'alarms' }" @click="setQuery({ tab: undefined })">告警事件</button>
      <button :class="{ on: tab === 'orders' }" @click="setQuery({ tab: 'orders' })">工单看板</button>
      <span class="dim hint">流程：告警 → AI 处置建议（可选）→ 工单草稿 → 人工确认派单 → 处理 → 关闭。AI 只给建议，不会自动派单或操作设备。</span>
    </div>

    <WorkOrderBoard v-show="tab === 'orders'" ref="board" @alarm="select" />

    <div v-show="tab === 'alarms'" class="split">
      <section class="panel list">
        <h3>告警事件<span class="sub">共 {{ list?.total ?? 0 }} 条 · 数据仍异常 {{ list?.summary.active ?? 0 }} 条</span></h3>
        <div class="filters">
          <select v-model="f.status" class="input">
            <option value="">全部状态</option><option>未关闭</option><option>未处理</option><option>处理中</option><option>已关闭</option>
          </select>
          <select v-model="f.level" class="input"><option value="">全部级别</option><option>告警</option><option>预警</option><option>提示</option></select>
          <select v-model="f.kind" class="input"><option value="">全部类型</option><option v-for="k in kinds" :key="k">{{ k }}</option></select>
          <select v-model="f.plant" class="input"><option value="">全部厂站</option><option v-for="p in plants" :key="p.code" :value="p.code">{{ p.code }} {{ p.name.replace(/污水处理(厂|站)$/, '') }}</option></select>
          <label class="chk"><input v-model="f.active" type="checkbox" /> 仅数据仍异常</label>
        </div>
        <div class="sum">
          <span v-for="(n, l) in list?.summary.by_level" :key="l" :class="LEVEL_CLASS[l]">{{ l }} <b class="num">{{ n }}</b></span>
          <span class="dim">|</span>
          <span v-for="(n, s) in list?.summary.by_status" :key="s" :class="STATUS_CLASS[s]">{{ s }} <b class="num">{{ n }}</b></span>
        </div>
        <div v-if="err" class="err">{{ err }}</div>
        <ul>
          <li v-for="a in list?.items" :key="a.id" :class="{ on: a.id === selected, active: a.active }" @click="select(a.id)">
            <div class="r1">
              <span class="tag" :class="LEVEL_CLASS[a.level]">{{ a.level }}</span>
              <span class="num dim">{{ mdhm(a.ts) }}</span>
              <span class="p">{{ a.plant?.code }}</span>
              <span class="k dim">{{ a.kind }}</span>
              <span class="tag st" :class="STATUS_CLASS[a.status]">{{ a.status }}</span>
            </div>
            <div class="r2">
              <span v-if="a.active" class="live">●</span>{{ a.message }}
              <span v-if="a.advice_at" class="ai" title="已有 AI 处置建议">AI</span>
            </div>
          </li>
          <li v-if="list && !list.items.length" class="dim none">没有符合条件的告警</li>
        </ul>
        <div class="pager">
          <button class="btn sm" :disabled="offset === 0" @click="offset = Math.max(0, offset - PAGE)">上一页</button>
          <span class="dim num">{{ list?.total ? `${offset + 1}–${Math.min(offset + PAGE, list.total)} / ${list.total}` : '' }}</span>
          <button class="btn sm" :disabled="!list || offset + PAGE >= list.total" @click="offset += PAGE">下一页</button>
        </div>
      </section>

      <section class="panel">
        <h3>告警详情</h3>
        <AlarmDetail v-if="selected" :id="selected" :metrics="metrics" @select="select" @changed="onChanged" />
        <div v-else class="dim">从左侧选择一条告警，查看指标曲线、关联告警、AI 处置建议并派单。</div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.page { padding: 14px 16px; display: flex; flex-direction: column; gap: 12px; }
.tabs { display: flex; align-items: center; gap: 6px; }
.tabs button { background: none; border: 1px solid transparent; color: var(--dim); padding: 5px 14px; border-radius: 4px; cursor: pointer; }
.tabs button.on { color: var(--strong); border-color: rgba(47, 211, 255, 0.5); background: rgba(47, 211, 255, 0.12); }
.hint { margin-left: 12px; font-size: 12px; }
.split { display: grid; grid-template-columns: 440px 1fr; gap: 12px; align-items: start; }
.list { display: flex; flex-direction: column; gap: 8px; position: sticky; top: 70px; max-height: calc(100vh - 140px); }
.filters { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.filters select { font-size: 12px; }
.chk { font-size: 12px; color: var(--dim); display: flex; align-items: center; gap: 4px; }
.sum { display: flex; gap: 10px; font-size: 12px; }
ul { list-style: none; margin: 0; padding: 0; overflow-y: auto; flex: 1; min-height: 0; }
li { padding: 7px 6px; border-bottom: 1px solid rgba(64, 140, 255, 0.12); cursor: pointer; display: flex; flex-direction: column; gap: 3px; }
li:hover { background: rgba(47, 211, 255, 0.05); }
li.on { background: rgba(47, 211, 255, 0.13); box-shadow: inset 2px 0 0 var(--cyan); }
li.active:not(.on) { background: linear-gradient(90deg, rgba(255, 90, 106, 0.1), transparent); }
.r1 { display: flex; gap: 6px; align-items: center; font-size: 12px; }
.r1 .p { color: var(--cyan); }
.r1 .st { margin-left: auto; }
.r2 { font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.live { color: var(--red); margin-right: 4px; font-size: 10px; }
.ai { margin-left: 6px; font-size: 10px; color: var(--cyan); border: 1px solid var(--cyan); border-radius: 3px; padding: 0 3px; }
.none { cursor: default; }
.pager { display: flex; align-items: center; justify-content: space-between; font-size: 12px; }
@media (max-width: 1100px) { .split { grid-template-columns: 1fr; } .list { position: static; max-height: 520px; } }
</style>

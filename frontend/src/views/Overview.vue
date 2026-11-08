<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { api, type AlarmList, type MetricInfo, type Overview } from '../lib/api'
import { C, axisStyle, tooltipStyle, useChart } from '../lib/chart'
import { fmt, hhmm, mdhm, statusLabel } from '../lib/format'
import { realtime } from '../lib/realtime'

const ov = ref<Overview | null>(null)
const alarms = ref<AlarmList | null>(null)
const metrics = ref<Record<string, MetricInfo>>({})
const selected = ref('P01')
const error = ref('')
let timer: number | undefined

async function load() {
  try {
    const [o, a] = await Promise.all([
      api.get<Overview>('/api/overview'),
      api.get<AlarmList>('/api/alarms?limit=12'),
    ])
    ov.value = o
    alarms.value = a
    error.value = ''
  } catch (e) {
    error.value = (e as Error).message
  }
}

onMounted(async () => {
  const m = await api.get<{ plant: MetricInfo[] }>('/api/metrics').catch(() => ({ plant: [] }))
  metrics.value = Object.fromEntries(m.plant.map((x) => [x.key, x]))
  await load()
  timer = window.setInterval(load, 30000)
})
onUnmounted(() => clearInterval(timer))
watch(() => realtime.alarmSeq, load)                // 告警变化时立即刷新

const STATUS_COLOR: Record<string, string> = { normal: C.green, warn: C.yellow, alarm: C.red, offline: '#5d6b85' }

// ---------------------------------------------------------------- 厂站分布（经纬度示意，不加底图瓦片）
const mapEl = ref<HTMLElement | null>(null)
const map = useChart(mapEl)
function drawMap() {
  const o = ov.value
  if (!o) return
  const pts = o.plants.map((p) => ({
    name: p.name, code: p.code, value: [p.lng, p.lat, p.capacity_m3d], status: p.status,
    itemStyle: { color: STATUS_COLOR[p.status], shadowBlur: 12, shadowColor: STATUS_COLOR[p.status] },
  }))
  const size = (v: number[]) => 12 + Math.sqrt(v[2]) / 10
  map.set({
    grid: { left: 10, right: 10, top: 10, bottom: 10 },
    tooltip: {
      ...tooltipStyle, trigger: 'item',
      formatter: (p: any) => {
        const pl = o.plants.find((x) => x.code === p.data.code)!
        const issues = pl.issues.map((i) => i.label).join('、') || '无'
        return `${pl.name}（${pl.code}）<br/>${pl.process} · ${pl.capacity_m3d.toLocaleString()} m³/d<br/>状态：${statusLabel[pl.status]}<br/>问题：${issues}`
      },
    },
    xAxis: { type: 'value', min: 106.86, max: 107.26, show: false },
    yAxis: { type: 'value', min: 32.83, max: 33.09, show: false },
    series: [
      { type: 'effectScatter', data: pts.filter((p) => p.status !== 'normal'), symbolSize: size,
        rippleEffect: { scale: 3.2, brushType: 'stroke' }, z: 3 },
      { type: 'scatter', data: pts, symbolSize: size, z: 4,
        label: { show: true, position: 'right', color: C.text, fontSize: 12, formatter: (p: any) => p.data.name.replace(/污水处理(厂|站)$/, '') } },
      { type: 'scatter', data: pts.filter((p) => p.code === selected.value), symbolSize: (v: number[]) => size(v) + 12, z: 2,
        itemStyle: { color: 'transparent', borderColor: C.cyan, borderWidth: 2 }, silent: true },
    ],
  })
}
watch([ov, selected], drawMap)
watch(map.chart, (c) => c?.on('click', (p: any) => { if (p.data?.code) selected.value = p.data.code }))

// ---------------------------------------------------------------- 24 小时趋势
const trendEl = ref<HTMLElement | null>(null)
const trend = useChart(trendEl)
watch(ov, (o) => {
  if (!o) return
  trend.set({
    grid: { left: 44, right: 44, top: 30, bottom: 24 },
    tooltip: tooltipStyle,
    legend: { top: 0, textStyle: { color: C.text, fontSize: 11 }, itemWidth: 12, itemHeight: 6 },
    xAxis: { type: 'category', data: o.hourly.map((h) => hhmm(h.ts)), ...axisStyle, splitLine: { show: false } },
    yAxis: [
      { type: 'value', name: 'm³/h · kW', nameTextStyle: { color: C.dim, fontSize: 10 }, ...axisStyle },
      { type: 'value', name: 'mm/h', nameTextStyle: { color: C.dim, fontSize: 10 }, ...axisStyle, splitLine: { show: false }, minInterval: 1 },
    ],
    series: [
      { name: '全区进水', type: 'line', smooth: true, showSymbol: false, data: o.hourly.map((h) => h.inflow_m3h),
        lineStyle: { color: C.cyan }, itemStyle: { color: C.cyan },
        areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: 'rgba(47,211,255,0.35)' }, { offset: 1, color: 'rgba(47,211,255,0)' }] } } },
      { name: '鼓风机功率', type: 'line', smooth: true, showSymbol: false, data: o.hourly.map((h) => h.blower_kw),
        lineStyle: { color: C.yellow }, itemStyle: { color: C.yellow } },
      { name: '降雨', type: 'bar', yAxisIndex: 1, data: o.hourly.map((h) => h.rain_mm_h), itemStyle: { color: C.blue }, barWidth: '50%' },
    ],
  })
})

// ---------------------------------------------------------------- 达标率环
const ringEl = ref<HTMLElement | null>(null)
const ring = useChart(ringEl)
watch(ov, (o) => {
  if (!o) return
  const v = Math.round(o.today.compliance_rate * 1000) / 10
  ring.set({
    series: [{
      type: 'gauge', startAngle: 90, endAngle: -270, radius: '92%', pointer: { show: false },
      progress: { show: true, roundCap: true, width: 10, itemStyle: { color: v >= 99.9 ? C.green : v >= 95 ? C.yellow : C.red } },
      axisLine: { lineStyle: { width: 10, color: [[1, 'rgba(120,170,255,0.15)']] } },
      splitLine: { show: false }, axisTick: { show: false }, axisLabel: { show: false },
      detail: { valueAnimation: true, formatter: '{value}%', color: '#fff', fontSize: 20, offsetCenter: [0, '-8%'] },
      title: { offsetCenter: [0, '30%'], color: C.dim, fontSize: 11 },
      data: [{ value: v, name: '出水达标率' }],
    }],
  })
})

// ---------------------------------------------------------------- 选中厂站的实时曲线（来自 WebSocket 帧）
const KEY_METRICS = ['inflow_m3h', 'do_aerobic_1', 'do_aerobic_2', 'nh3n_out', 'cod_out', 'tp_out', 'tn_out', 'ss_out', 'air_flow_m3h', 'blower_kw']
const history = ref<{ ts: string; v: Record<string, number> }[]>([])
watch(selected, () => { history.value = [] })
watch(() => realtime.frame, (f) => {
  if (!f) return
  const v = f.plants[selected.value]
  if (!v) return
  if (history.value.length && history.value[history.value.length - 1].ts === f.ts) return
  history.value.push({ ts: f.ts, v })
  if (history.value.length > 240) history.value.shift()
})
const live = computed(() => realtime.frame?.plants[selected.value] ?? ov.value?.plants.find((p) => p.code === selected.value)?.latest)
const plant = computed(() => ov.value?.plants.find((p) => p.code === selected.value))

const liveEl = ref<HTMLElement | null>(null)
const liveChart = useChart(liveEl)
watch(history, (h) => {
  liveChart.set({
    grid: { left: 36, right: 36, top: 26, bottom: 22 },
    tooltip: tooltipStyle,
    legend: { top: 0, textStyle: { color: C.text, fontSize: 11 }, itemWidth: 12, itemHeight: 6 },
    xAxis: { type: 'category', data: h.map((x) => new Date(x.ts).toLocaleTimeString('zh-CN', { hour12: false, timeZone: 'Asia/Shanghai' })), ...axisStyle, splitLine: { show: false } },
    yAxis: [{ type: 'value', name: 'DO', ...axisStyle, nameTextStyle: { color: C.dim, fontSize: 10 } },
            { type: 'value', name: '氨氮', ...axisStyle, splitLine: { show: false }, nameTextStyle: { color: C.dim, fontSize: 10 } }],
    series: [
      { name: '好氧区 DO1', type: 'line', showSymbol: false, data: h.map((x) => x.v.do_aerobic_1), lineStyle: { color: C.cyan }, itemStyle: { color: C.cyan } },
      { name: '出水氨氮', type: 'line', yAxisIndex: 1, showSymbol: false, data: h.map((x) => x.v.nh3n_out), lineStyle: { color: C.red }, itemStyle: { color: C.red },
        markLine: { silent: true, symbol: 'none', data: [{ yAxis: 5 }], lineStyle: { color: C.red, type: 'dashed', opacity: 0.6 }, label: { color: C.red, formatter: '限值 5' } } },
    ],
  }, false)
}, { deep: true })

function over(key: string, v: number | undefined) {
  const m = metrics.value[key]
  return m?.limit != null && v !== undefined && (m.lower_is_bad ? v < m.limit : v > m.limit)
}

const LEVEL_CLASS: Record<string, string> = { 告警: 's-alarm', 预警: 's-warn', 提示: 's-normal' }
</script>

<template>
  <div v-if="error && !ov" class="panel err">加载失败：{{ error }}</div>
  <div v-else class="ov">
    <!-- 左栏 -->
    <div class="col">
      <section class="panel">
        <h3>天气<span class="sub">汉中 · 演示时钟</span></h3>
        <div class="weather">
          <div><b class="num">{{ fmt(ov?.weather.temp_c, 1) }}</b><span class="dim">℃ 气温</span></div>
          <div><b class="num" :class="(ov?.weather.rain_mm_h ?? 0) > 0 ? 's-warn' : ''">{{ fmt(ov?.weather.rain_mm_h, 1) }}</b><span class="dim">mm/h 降雨</span></div>
        </div>
      </section>
      <section class="panel">
        <h3>今日运行<span class="sub">0 点至今</span></h3>
        <div class="kpis">
          <div><span class="dim">处理水量</span><b class="num">{{ fmt((ov?.today.water_m3 ?? 0) / 10000, 2) }}</b><i>万 m³</i></div>
          <div><span class="dim">吨水曝气电耗</span><b class="num">{{ fmt(ov?.today.aeration_kwh_per_m3, 3) }}</b><i>kWh/m³</i></div>
          <div><span class="dim">碳源投加</span><b class="num">{{ fmt(ov?.today.carbon_kg, 0) }}</b><i>kg</i></div>
          <div><span class="dim">PAC 投加</span><b class="num">{{ fmt(ov?.today.pac_kg, 0) }}</b><i>kg</i></div>
        </div>
        <div ref="ringEl" class="ring" />
      </section>
      <section class="panel">
        <h3>厂站状态</h3>
        <div class="counts">
          <div v-for="s in ['normal', 'warn', 'alarm', 'offline']" :key="s" :class="'s-' + s">
            <b class="num">{{ ov?.status_counts[s as keyof Overview['status_counts']] ?? 0 }}</b><span>{{ statusLabel[s] }}</span>
          </div>
        </div>
        <div v-if="ov?.scenarios.length" class="scen">
          <div class="dim">进行中的场景</div>
          <div v-for="s in ov.scenarios" :key="s.kind + s.start"><span class="tag s-warn">{{ s.kind }}</span> {{ s.plant_code ?? '全区' }} · 至 {{ hhmm(s.end) }}</div>
        </div>
      </section>
    </div>

    <!-- 中间 -->
    <div class="col mid">
      <section class="panel mapbox">
        <h3>厂站分布<span class="sub">按经纬度示意，点大小为设计规模，点击查看实时数据</span></h3>
        <div ref="mapEl" class="map" />
      </section>
      <section class="panel" v-if="plant">
        <h3>{{ plant.name }}（{{ plant.code }}）
          <span class="tag" :class="'s-' + plant.status">{{ statusLabel[plant.status] }}</span>
          <span class="sub">{{ plant.process }} · {{ plant.capacity_m3d.toLocaleString() }} m³/d · 实时 {{ history.length }} 帧 · <RouterLink :to="`/plants/${plant.code}`">工艺流程与设备 →</RouterLink></span>
        </h3>
        <div class="live">
          <div class="vals">
            <div v-for="k in KEY_METRICS" :key="k" :class="{ over: over(k, live?.[k]) }">
              <span class="dim">{{ metrics[k]?.label ?? k }}</span>
              <b class="num">{{ fmt(live?.[k], k.includes('m3h') || k === 'blower_kw' ? 0 : 2) }}</b>
              <i class="dim">{{ metrics[k]?.unit }}</i>
            </div>
          </div>
          <div ref="liveEl" class="livechart" />
        </div>
        <div v-if="plant.issues.length" class="issues">
          <span v-for="i in plant.issues" :key="i.label" class="tag" :class="'s-' + i.level">{{ i.label }}</span>
        </div>
      </section>
    </div>

    <!-- 右栏 -->
    <div class="col">
      <section class="panel">
        <h3>近 24 小时<span class="sub">全区小时均值</span></h3>
        <div ref="trendEl" class="trend" />
      </section>
      <section class="panel alarms">
        <h3>告警事件
          <span class="sub">共 {{ alarms?.total ?? 0 }} 条 · 数据仍异常 {{ alarms?.summary.active ?? 0 }} 条</span>
        </h3>
        <div class="levels">
          <span v-for="(n, l) in alarms?.summary.by_level" :key="l" :class="LEVEL_CLASS[l]">{{ l }} <b class="num">{{ n }}</b></span>
        </div>
        <ul>
          <li v-for="a in alarms?.items" :key="a.id" :class="{ active: a.active }" title="查看详情" @click="$router.push(`/alarms?id=${a.id}`)">
            <span class="tag" :class="LEVEL_CLASS[a.level]">{{ a.level }}</span>
            <span class="t num dim">{{ mdhm(a.ts) }}</span>
            <span class="p">{{ a.plant?.code }}</span>
            <span class="m">{{ a.message }}</span>
            <span v-if="a.active" class="tag s-alarm">进行中</span>
          </li>
        </ul>
      </section>
    </div>
  </div>
</template>

<style scoped>
.ov { display: grid; grid-template-columns: 300px minmax(0, 1fr) 380px; gap: 12px; }
.col { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.weather { display: flex; gap: 24px; }
.weather b { font-size: 28px; color: #fff; margin-right: 4px; }
.kpis { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.kpis > div { background: rgba(47, 211, 255, 0.06); border-radius: 4px; padding: 8px; display: flex; flex-direction: column; }
.kpis span { font-size: 12px; }
.kpis b { font-size: 22px; color: var(--cyan); }
.kpis i { font-style: normal; font-size: 11px; color: var(--dim); }
.ring { height: 150px; margin-top: 8px; }
.counts { display: grid; grid-template-columns: repeat(4, 1fr); text-align: center; }
.counts b { display: block; font-size: 26px; }
.counts span { font-size: 12px; }
.scen { margin-top: 10px; font-size: 12px; display: flex; flex-direction: column; gap: 4px; }
.mapbox .map { height: 380px; background:
  repeating-linear-gradient(0deg, rgba(64,140,255,0.06) 0 1px, transparent 1px 40px),
  repeating-linear-gradient(90deg, rgba(64,140,255,0.06) 0 1px, transparent 1px 40px); border-radius: 4px; }
.live { display: grid; grid-template-columns: 280px minmax(0, 1fr); gap: 12px; }
.vals { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px; font-size: 12px; }
.vals > div { display: flex; align-items: baseline; gap: 4px; }
.vals > div .dim:first-child { flex: 1; }
.vals b { color: #fff; font-size: 15px; }
.vals .over b { color: var(--red); }
.vals i { font-style: normal; font-size: 11px; width: 34px; }
.livechart { height: 190px; }
.issues { margin-top: 8px; display: flex; gap: 6px; flex-wrap: wrap; }
.trend { height: 220px; }
.alarms { flex: 1; display: flex; flex-direction: column; min-height: 300px; }
.levels { display: flex; gap: 16px; font-size: 13px; margin-bottom: 6px; }
.alarms ul { list-style: none; margin: 0; padding: 0; overflow-y: auto; max-height: 460px; }
.alarms li { cursor: pointer; display: flex; align-items: center; gap: 6px; padding: 6px 0; border-bottom: 1px solid rgba(64, 140, 255, 0.12); font-size: 12px; }
.alarms li.active { background: linear-gradient(90deg, rgba(255, 90, 106, 0.12), transparent); }
.alarms .t { width: 78px; flex: none; }
.alarms .p { color: var(--cyan); width: 28px; flex: none; }
.alarms .m { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
@media (max-width: 1400px) { .ov { grid-template-columns: 280px minmax(0, 1fr); } .ov > .col:last-child { grid-column: 1 / -1; } }
@media (max-width: 900px) { .ov { grid-template-columns: 1fr; } .live { grid-template-columns: 1fr; } }
</style>

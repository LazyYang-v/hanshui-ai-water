<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import MiniTrend from '../components/MiniTrend.vue'
import { api, type AlarmList, type DeviceInfo, type MetricInfo, type PlantBrief, type PlantDetail, type Series } from '../lib/api'
import { C } from '../lib/chart'
import { fmt, mdhm, statusLabel } from '../lib/format'
import { realtime } from '../lib/realtime'

const route = useRoute()
const router = useRouter()
const code = computed(() => String(route.params.code || 'P01'))

const plants = ref<PlantBrief[]>([])
const plant = ref<PlantDetail | null>(null)
const metrics = ref<Record<string, MetricInfo>>({})
const alarms = ref<AlarmList | null>(null)
const err = ref('')

// ---------------------------------------------------------------- 数据加载
async function loadPlant() {
  try {
    const [p, list, a] = await Promise.all([
      api.get<PlantDetail>(`/api/plants/${code.value}`),
      api.get<PlantBrief[]>('/api/plants'),
      api.get<AlarmList>(`/api/alarms?plant=${code.value}&status=${encodeURIComponent('未关闭')}&limit=8`),
    ])
    if (p.code !== code.value) return            // 切换厂站时丢弃过期响应
    plant.value = p; plants.value = list; alarms.value = a; err.value = ''
  } catch (e) { err.value = (e as Error).message }
}

const EFFLUENT = ['cod_out', 'nh3n_out', 'tn_out', 'tp_out', 'ss_out']
const range = ref<'24h' | '7d'>('24h')
const effluent = ref<Series | null>(null)
async function loadEffluent() {
  const c = code.value
  let qs = `metrics=${EFFLUENT.join(',')}`
  if (range.value === '7d') {
    const end = new Date(realtime.frame?.ts ?? (await api.get<{ now: string }>('/api/clock')).now)
    qs += `&start=${encodeURIComponent(new Date(end.getTime() - 7 * 86400e3).toISOString())}&end=${encodeURIComponent(end.toISOString())}`
  }
  try {
    const s = await api.get<Series>(`/api/plants/${c}/readings?${qs}`)
    if (c === code.value) effluent.value = s
  } catch (e) { err.value = (e as Error).message }
}

const device = ref<string | null>(null)
const devSeries = ref<Series | null>(null)
async function loadDevice() {
  const d = device.value
  if (!d) { devSeries.value = null; return }
  try {
    const s = await api.get<Series>(`/api/devices/${d}/readings?metrics=current_a,vibration_mms,temp_c`)
    if (d === device.value) devSeries.value = s
  } catch (e) { err.value = (e as Error).message }
}

let fast: number | undefined
let slow: number | undefined
onMounted(async () => {
  try {
    const m = await api.get<{ plant: MetricInfo[]; device: MetricInfo[] }>('/api/metrics')
    metrics.value = Object.fromEntries([...m.plant, ...m.device].map((x) => [x.key, x]))
  } catch { /* 无目录时只是不着色 */ }
  fast = window.setInterval(loadPlant, 15000)                      // 设备状态、今日统计
  slow = window.setInterval(() => { loadEffluent(); loadDevice() }, 60000)   // 曲线为 5 分钟桶
})
onUnmounted(() => { clearInterval(fast); clearInterval(slow) })
watch(() => realtime.alarmSeq, loadPlant)

// ---------------------------------------------------------------- 实时值与着色
const live = computed<Record<string, number>>(() => realtime.frame?.plants[code.value] ?? plant.value?.latest ?? {})
const liveFromWs = computed(() => !!realtime.frame?.plants[code.value])

function lv(key: string, v: number | null | undefined): string {
  const m = metrics.value[key]
  if (!m || m.limit == null || v == null) return ''
  if (m.lower_is_bad) return v < m.limit ? 'alarm' : m.warn_at != null && v < m.warn_at ? 'warn' : ''
  return v > m.limit ? 'alarm' : m.warn_at != null && v > m.warn_at ? 'warn' : ''
}
const DIGITS: Record<string, number> = { inflow_m3h: 0, air_flow_m3h: 0, mlss: 0, blower_kw: 0, cod_in: 0, cod_out: 1, carbon_dose_kgh: 1, pac_dose_kgh: 1 }
const SHORT: Record<string, string> = {
  inflow_m3h: '流量', cod_in: 'COD', nh3n_in: '氨氮', tn_in: '总氮', tp_in: '总磷',
  cod_out: 'COD', nh3n_out: '氨氮', tn_out: '总氮', tp_out: '总磷', ss_out: 'SS',
  do_aerobic_1: 'DO 1#', do_aerobic_2: 'DO 2#', air_flow_m3h: '风量', mlss: 'MLSS', blower_kw: '风机功率',
  carbon_dose_kgh: '碳源投加', pac_dose_kgh: 'PAC 投加',
}

// ---------------------------------------------------------------- 工艺流程图布局（SVG 坐标，viewBox 1400×340）
interface Unit { name: string; sub?: string; x: number; w: number; keys?: string[]; pumps?: boolean; ditch?: boolean }
const Y = 60
const H = 150
const isDitch = computed(() => plant.value?.process === '氧化沟')
const units = computed<Unit[]>(() => {
  const head: Unit[] = [
    { name: '进水', sub: '粗细格栅', x: 20, w: 170, keys: ['inflow_m3h', 'cod_in', 'nh3n_in', 'tn_in', 'tp_in'] },
    { name: '提升泵房', x: 225, w: 120, pumps: true },
    { name: '厌氧池', sub: '释磷', x: 380, w: 120 },
  ]
  const tail: Unit[] = [
    { name: '二沉池', sub: '泥水分离 · 化学除磷', x: 975, w: 130, keys: ['pac_dose_kgh'] },
    { name: '出水', sub: '一级 A 标准', x: 1140, w: 240, keys: EFFLUENT },
  ]
  const bio: Unit[] = isDitch.value
    ? [{ name: '氧化沟', sub: '同步硝化反硝化 · 曝气', x: 535, w: 405, ditch: true, keys: ['do_aerobic_1', 'do_aerobic_2', 'air_flow_m3h', 'mlss', 'blower_kw', 'carbon_dose_kgh'] }]
    : [{ name: '缺氧池', sub: '反硝化', x: 535, w: 130, keys: ['carbon_dose_kgh'] },
       { name: '好氧池', sub: '硝化 · 曝气', x: 700, w: 240, keys: ['do_aerobic_1', 'do_aerobic_2', 'air_flow_m3h', 'mlss', 'blower_kw'] }]
  return [...head, ...bio, ...tail]
})
// 宽单元（氧化沟）分两列排读数
const cols = (u: Unit) => (u.ditch ? 2 : 1)
const perCol = (u: Unit) => Math.ceil((u.keys?.length ?? 1) / cols(u))
const rowGap = (u: Unit) => Math.min(u.ditch ? 24 : 19, (H - 58) / Math.max(1, perCol(u)))
const colW = (u: Unit) => (u.w - (u.ditch ? 80 : 24) - (cols(u) - 1) * 40) / cols(u)
const rowX = (u: Unit, j: number) => u.x + (u.ditch ? 40 : 12) + Math.floor(j / perCol(u)) * (colW(u) + 40)
const rowY = (u: Unit, j: number) => Y + 60 + (j % perCol(u)) * rowGap(u)

const byKind = (k: string) => plant.value?.devices.filter((d) => d.kind === k) ?? []
const lift = computed(() => byKind('提升泵'))
const blowers = computed(() => byKind('鼓风机'))
const recirc = computed(() => byKind('回流泵')[0])
const sludge = computed(() => byKind('污泥泵')[0])
const dosing = computed(() => byKind('加药泵'))
const carbonPump = computed(() => dosing.value.find((d) => d.code.endsWith('-C')))
const pacPump = computed(() => dosing.value.find((d) => d.code.endsWith('-P')))
// 鼓风机放在生化池底部右侧，给内回流管留出左侧位置
const blowerX = (i: number) => 960 - 70 * (blowers.value.length - i)
const carbonX = 600

function devColor(d?: DeviceInfo) {
  if (!d) return C.dim
  if (d.level === 'alarm') return C.red
  if (d.level === 'warn') return C.yellow
  return d.running ? C.green : C.dim
}
const tail = (c: string) => c.split('-').slice(1).join('-')
function devTitle(d: DeviceInfo) {
  return `${d.code} ${d.kind} · ${d.status}${d.running ? '' : '（未运行）'}\n电流 ${fmt(d.current_a, 1)} A · 振动 ${fmt(d.vibration_mms, 2)} mm/s · 温度 ${fmt(d.temp_c, 1)} ℃\n${d.model}`
}
function pick(d?: DeviceInfo) { if (d) device.value = d.code }

// ---------------------------------------------------------------- 切换厂站
watch(code, () => {
  plant.value = null; effluent.value = null; device.value = null
  loadPlant(); loadEffluent()
}, { immediate: true })
watch(range, loadEffluent)
watch(device, loadDevice)

const deviceInfo = computed(() => plant.value?.devices.find((d) => d.code === device.value))
const series = (s: Series | null, k: string) => s?.series[k] ?? []
const LEVEL_CLASS: Record<string, string> = { 告警: 's-alarm', 预警: 's-warn', 提示: 's-normal' }
const DEV_LEVEL: Record<string, string> = { alarm: 's-alarm', warn: 's-warn', normal: 's-normal' }
</script>

<template>
  <div class="page">
    <div class="pick">
      <button v-for="p in plants" :key="p.code" :class="{ on: p.code === code }" @click="router.push(`/plants/${p.code}`)">
        <span :class="'s-' + p.status">●</span> {{ p.code }} {{ p.name.replace(/污水处理(厂|站)$/, '') }}
      </button>
    </div>
    <div v-if="err" class="err">{{ err }}</div>

    <template v-if="plant">
      <section class="panel head">
        <div class="title">
          <h2>{{ plant.name }}</h2>
          <span class="tag" :class="'s-' + plant.status">{{ statusLabel[plant.status] }}</span>
          <span class="dim">{{ plant.code }} · {{ plant.kind }} · {{ plant.process }} 工艺 · 设计规模 {{ plant.capacity_m3d.toLocaleString() }} m³/d</span>
          <span v-for="i in plant.issues" :key="i.label" class="tag" :class="'s-' + i.level">{{ i.label }}</span>
        </div>
        <div v-if="plant.today" class="kpis">
          <div><span class="dim">今日处理量</span><b class="num">{{ fmt(plant.today.water_m3, 0) }}</b><i>m³</i></div>
          <div><span class="dim">负荷率（按今日时长折算）</span><b class="num">{{ fmt(plant.today.water_m3 / Math.max(plant.today.hours, 0.1) * 24 / plant.capacity_m3d * 100, 0) }}</b><i>%</i></div>
          <div><span class="dim">曝气电耗</span><b class="num">{{ fmt(plant.today.aeration_kwh, 0) }}</b><i>kWh</i></div>
          <div><span class="dim">吨水曝气电耗</span><b class="num">{{ fmt(plant.today.aeration_kwh_per_m3, 3) }}</b><i>kWh/m³</i></div>
          <div><span class="dim">碳源 / PAC</span><b class="num">{{ fmt(plant.today.carbon_kg, 0) }} / {{ fmt(plant.today.pac_kg, 0) }}</b><i>kg</i></div>
          <div><span class="dim">出水达标率</span><b class="num" :class="plant.today.compliance_rate < 0.999 ? 's-warn' : 's-normal'">{{ fmt(plant.today.compliance_rate * 100, 1) }}</b><i>%</i></div>
        </div>
      </section>

      <section class="panel">
        <h3>工艺流程<span class="sub">{{ liveFromWs ? '实时推送' : '最近读数' }} · 数值着色：黄为预警、红为超限 · 点击设备查看曲线</span></h3>
        <svg class="flow" viewBox="0 0 1400 340" role="img" :aria-label="`${plant.name} ${plant.process} 工艺流程`">
          <defs>
            <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" :fill="C.cyan" />
            </marker>
            <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="#a98bff" />
            </marker>
          </defs>

          <!-- 主流程 -->
          <line v-for="(u, i) in units.slice(0, -1)" :key="'f' + i" class="pipe" :x1="u.x + u.w" :y1="Y + H / 2" :x2="units[i + 1].x - 2" :y2="Y + H / 2" marker-end="url(#arr)" />

          <!-- 内回流（A2O：好氧 → 缺氧） -->
          <g v-if="!isDitch">
            <path class="ret" d="M730,210 V275 H600 V212" marker-end="url(#arr2)" />
            <text class="lbl2" x="665" y="262" text-anchor="middle">混合液内回流</text>
          </g>
          <!-- 污泥回流（二沉 → 厌氧） -->
          <path class="ret" d="M1030,210 V310 H440 V212" marker-end="url(#arr2)" />
          <text class="lbl2" :x="isDitch ? 600 : 520" y="303" text-anchor="middle">污泥回流</text>
          <!-- 剩余污泥 -->
          <path class="ret dim2" d="M1080,210 V262 H1150" />
          <text class="lbl2" x="1156" y="266">剩余污泥 → 污泥处置</text>

          <!-- 工艺单元 -->
          <g v-for="u in units" :key="u.name">
            <rect :x="u.x" :y="Y" :width="u.w" :height="H" :rx="u.ditch ? 48 : 6" class="unit" />
            <rect v-if="u.ditch" :x="u.x + 18" :y="Y + 18" :width="u.w - 36" :height="H - 36" rx="34" class="ditch" />
            <text class="uname" :x="u.x + 12" :y="Y + 22">{{ u.name }}</text>
            <text v-if="u.sub" class="usub" :x="u.x + 12" :y="Y + 38">{{ u.sub }}</text>
            <text v-if="!u.keys && !u.pumps" class="usub" :x="u.x + 12" :y="Y + 64">无在线仪表</text>
            <g v-for="(k, j) in u.keys ?? []" :key="k">
              <text class="rk" :x="rowX(u, j)" :y="rowY(u, j)">{{ SHORT[k] ?? k }}</text>
              <text class="rv num" :class="lv(k, live[k])" :x="rowX(u, j) + colW(u)" :y="rowY(u, j)" text-anchor="end">
                {{ fmt(live[k], DIGITS[k] ?? 2) }}<tspan class="ru"> {{ metrics[k]?.unit }}</tspan>
              </text>
            </g>
            <!-- 提升泵画在泵房内 -->
            <template v-if="u.pumps">
              <g v-for="(d, j) in lift" :key="d.code" class="dev" @click="pick(d)">
                <title>{{ devTitle(d) }}</title>
                <circle :cx="u.x + 22" :cy="Y + 62 + j * 34" r="9" :fill="devColor(d)" :class="{ on: d.code === device }" />
                <text class="dk" :x="u.x + 38" :y="Y + 60 + j * 34">{{ tail(d.code) }}</text>
                <text class="dv num" :x="u.x + 38" :y="Y + 74 + j * 34">{{ fmt(d.current_a, 0) }} A</text>
              </g>
            </template>
          </g>

          <!-- 鼓风机（生化池下方，风管向上） -->
          <g v-for="(d, i) in blowers" :key="d.code" class="dev" @click="pick(d)">
            <title>{{ devTitle(d) }}</title>
            <line :x1="blowerX(i)" y1="238" :x2="blowerX(i)" y2="211" class="air" :class="{ off: !d.running }" />
            <circle :cx="blowerX(i)" cy="250" r="12" :fill="devColor(d)" :class="{ on: d.code === device }" />
            <text class="dk" :x="blowerX(i)" y="254" text-anchor="middle" fill="#051024">风</text>
            <text class="dk" :x="blowerX(i)" y="278" text-anchor="middle">{{ tail(d.code) }}</text>
            <text class="dv num" :x="blowerX(i)" y="292" text-anchor="middle">{{ d.running ? fmt(d.current_a, 0) + ' A' : d.status }}</text>
          </g>

          <!-- 加药泵（池顶） -->
          <g v-if="carbonPump" class="dev" @click="pick(carbonPump)">
            <title>{{ devTitle(carbonPump) }}</title>
            <line :x1="carbonX" y1="36" :x2="carbonX" :y2="Y - 1" class="dose" marker-end="url(#arr)" />
            <circle :cx="carbonX" cy="26" r="8" :fill="devColor(carbonPump)" :class="{ on: carbonPump.code === device }" />
            <text class="dk" :x="carbonX + 14" y="30">{{ tail(carbonPump.code) }} 碳源</text>
          </g>
          <g v-if="pacPump" class="dev" @click="pick(pacPump)">
            <title>{{ devTitle(pacPump) }}</title>
            <line x1="1040" y1="36" x2="1040" :y2="Y - 1" class="dose" marker-end="url(#arr)" />
            <circle cx="1040" cy="26" r="8" :fill="devColor(pacPump)" :class="{ on: pacPump.code === device }" />
            <text class="dk" x="1054" y="30">{{ tail(pacPump.code) }} PAC</text>
          </g>

          <!-- 回流泵与污泥泵 -->
          <g v-if="recirc" class="dev" @click="pick(recirc)">
            <title>{{ devTitle(recirc) }}</title>
            <circle :cx="isDitch ? 760 : 665" :cy="isDitch ? 310 : 275" r="8" :fill="devColor(recirc)" :class="{ on: recirc.code === device }" />
            <text class="dk" :x="isDitch ? 774 : 679" :y="isDitch ? 326 : 291">{{ tail(recirc.code) }}</text>
          </g>
          <g v-if="sludge" class="dev" @click="pick(sludge)">
            <title>{{ devTitle(sludge) }}</title>
            <circle cx="1110" cy="262" r="8" :fill="devColor(sludge)" :class="{ on: sludge.code === device }" />
            <text class="dk" x="1100" y="284" text-anchor="middle">{{ tail(sludge.code) }} {{ sludge.running ? '排泥中' : '间歇' }}</text>
          </g>
        </svg>
        <div class="legend dim">
          <span><i :style="{ background: C.green }" />运行</span><span><i :style="{ background: C.dim }" />备用 / 停止</span>
          <span><i :style="{ background: C.yellow }" />设备预警</span><span><i :style="{ background: C.red }" />设备告警</span>
          <span>流程为示意，单元划分与管线按 {{ plant.process }} 典型工艺绘制；数据来自模拟器。</span>
        </div>
      </section>

      <div class="lower">
        <section class="panel">
          <h3>出水五项
            <span class="sub">
              <button class="btn sm" :class="{ primary: range === '24h' }" @click="range = '24h'">近 24 小时</button>
              <button class="btn sm" :class="{ primary: range === '7d' }" @click="range = '7d'">近 7 天</button>
              <span v-if="effluent" class="dim"> · {{ effluent.bucket }} 均值</span>
            </span>
          </h3>
          <div class="trends">
            <MiniTrend v-for="(k, i) in EFFLUENT" :key="k" :label="metrics[k]?.label ?? k" :unit="metrics[k]?.unit ?? ''"
                       :ts="effluent?.ts ?? []" :values="series(effluent, k)" :limit="metrics[k]?.limit" :warn-at="metrics[k]?.warn_at"
                       :color="[C.cyan, C.green, C.blue, C.purple, C.yellow][i]" :digits="k === 'cod_out' ? 1 : 2" />
          </div>
        </section>

        <section class="panel">
          <h3>设备状态<span class="sub">{{ plant.devices.length }} 台 · 点击查看近 24 小时曲线</span></h3>
          <table class="grid devs">
            <thead><tr><th>设备</th><th>类型</th><th>状态</th><th>电流 A</th><th>振动 mm/s</th><th>温度 ℃</th></tr></thead>
            <tbody>
              <tr v-for="d in plant.devices" :key="d.code" :class="{ on: d.code === device }" @click="device = d.code">
                <td>{{ d.code }}</td>
                <td class="dim">{{ d.kind }}</td>
                <td><span class="tag" :class="DEV_LEVEL[d.level]">{{ d.status }}</span></td>
                <td class="num">{{ fmt(d.current_a, 1) }}</td>
                <td class="num" :class="'s-' + (lv('vibration_mms', d.vibration_mms) || 'x')">{{ fmt(d.vibration_mms, 2) }}</td>
                <td class="num" :class="'s-' + (lv('temp_c', d.temp_c) || 'x')">{{ fmt(d.temp_c, 1) }}</td>
              </tr>
            </tbody>
          </table>
          <div v-if="deviceInfo" class="devtrend">
            <div class="dim small">{{ deviceInfo.code }} · {{ deviceInfo.model }} · 额定 {{ deviceInfo.rated_kw }} kW · 投运 {{ deviceInfo.installed_at }}
              <button class="btn sm" @click="device = null">收起</button></div>
            <div class="trends three">
              <MiniTrend label="运行电流" unit="A" :ts="devSeries?.ts ?? []" :values="series(devSeries, 'current_a')" :digits="1" />
              <MiniTrend label="振动速度" unit="mm/s" :ts="devSeries?.ts ?? []" :values="series(devSeries, 'vibration_mms')"
                         :limit="metrics.vibration_mms?.limit" :warn-at="metrics.vibration_mms?.warn_at" :color="C.yellow" />
              <MiniTrend label="轴承温度" unit="℃" :ts="devSeries?.ts ?? []" :values="series(devSeries, 'temp_c')"
                         :limit="metrics.temp_c?.limit" :warn-at="metrics.temp_c?.warn_at" :color="C.red" :digits="1" />
            </div>
          </div>

          <h3 class="mt">未关闭告警<span class="sub"><RouterLink :to="`/alarms?plant=${plant.code}`">全部告警 →</RouterLink></span></h3>
          <ul class="alarms">
            <li v-for="a in alarms?.items" :key="a.id" @click="router.push(`/alarms?id=${a.id}`)">
              <span class="tag" :class="LEVEL_CLASS[a.level]">{{ a.level }}</span>
              <span class="num dim">{{ mdhm(a.ts) }}</span>
              <span v-if="a.active" class="s-alarm">●</span>
              <span class="msg">{{ a.message }}</span>
            </li>
            <li v-if="alarms && !alarms.items.length" class="dim none">没有未关闭的告警</li>
          </ul>
        </section>
      </div>
    </template>
    <div v-else-if="!err" class="dim">加载中…</div>
  </div>
</template>

<style scoped>
.page { padding: 14px 16px; display: flex; flex-direction: column; gap: 12px; }
.pick { display: flex; flex-wrap: wrap; gap: 6px; }
.pick button { background: rgba(5, 15, 40, 0.6); border: 1px solid var(--panel-border); color: var(--dim); padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 13px; }
.pick button.on { color: var(--strong); border-color: rgba(47, 211, 255, 0.6); background: rgba(47, 211, 255, 0.14); }
.head { display: flex; flex-direction: column; gap: 10px; }
.title { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
.title h2 { margin: 0; font-size: 20px; color: var(--strong); }
.title .dim { font-size: 13px; }
.kpis { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; }
.kpis div { display: flex; flex-direction: column; padding: 6px 10px; border-left: 2px solid rgba(47, 211, 255, 0.4); background: rgba(47, 211, 255, 0.04); }
.kpis span { font-size: 12px; }
.kpis b { font-size: 20px; color: var(--strong); }
.kpis b.s-warn { color: var(--yellow); }
.kpis b.s-normal { color: var(--green); }
.kpis i { font-style: normal; font-size: 11px; color: var(--dim); }

.flow { width: 100%; height: auto; display: block; max-height: 380px; }
.unit { fill: rgba(16, 52, 110, 0.55); stroke: rgba(47, 211, 255, 0.55); stroke-width: 1.2; }
.ditch { fill: none; stroke: rgba(47, 211, 255, 0.25); stroke-dasharray: 6 5; }
.uname { fill: #fff; font-size: 15px; font-weight: 600; }
.usub { fill: var(--dim); font-size: 11px; }
.rk { fill: var(--dim); font-size: 12px; }
.rv { fill: #e6f2ff; font-size: 13px; }
.rv.warn { fill: var(--yellow); }
.rv.alarm { fill: var(--red); font-weight: 700; }
.ru { fill: var(--dim); font-size: 10px; }
.pipe { stroke: var(--cyan); stroke-width: 3; stroke-dasharray: 10 6; animation: flow 1.2s linear infinite; }
.ret { fill: none; stroke: #a98bff; stroke-width: 2; stroke-dasharray: 8 5; animation: flow 1.6s linear infinite; }
.ret.dim2 { stroke: rgba(169, 139, 255, 0.5); }
.air { stroke: rgba(255, 255, 255, 0.6); stroke-width: 2; stroke-dasharray: 3 4; animation: rise 0.8s linear infinite; }
.air.off { animation: none; opacity: 0.2; }
.dose { stroke: var(--cyan); stroke-width: 1.5; stroke-dasharray: 3 3; }
.lbl2 { fill: #c4b3ff; font-size: 11px; }
.dev { cursor: pointer; }
.dev circle.on { stroke: #fff; stroke-width: 2.5; }
.dk { fill: var(--text); font-size: 11px; }
.dv { fill: var(--dim); font-size: 11px; }
@keyframes flow { to { stroke-dashoffset: -32; } }
@keyframes rise { to { stroke-dashoffset: 14; } }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; margin-top: 4px; }
.legend i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 4px; vertical-align: -1px; }

.lower { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 12px; align-items: start; }
.trends { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 10px 14px; }
.trends.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.devs tbody tr { cursor: pointer; }
.devs tbody tr:hover { background: rgba(47, 211, 255, 0.05); }
.devs tbody tr.on { background: rgba(47, 211, 255, 0.13); }
.devs .s-x { color: inherit; }
.devtrend { margin-top: 10px; display: flex; flex-direction: column; gap: 6px; }
.devtrend .btn { margin-left: 8px; }
.small { font-size: 12px; }
.mt { margin-top: 14px !important; }
.alarms { list-style: none; margin: 0; padding: 0; }
.alarms li { display: flex; gap: 6px; align-items: center; padding: 6px 4px; border-bottom: 1px solid rgba(64, 140, 255, 0.12); font-size: 13px; cursor: pointer; }
.alarms li:hover { background: rgba(47, 211, 255, 0.05); }
.alarms .num { white-space: nowrap; }
.alarms .msg { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
.alarms .none { cursor: default; }
@media (max-width: 1300px) { .kpis { grid-template-columns: repeat(3, minmax(0, 1fr)); } .lower { grid-template-columns: 1fr; } }
</style>

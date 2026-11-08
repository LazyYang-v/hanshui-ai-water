<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { api, type AerationOptions, type AerationResult } from '../lib/api'
import { C, axisStyle, tooltipStyle, useChart } from '../lib/chart'
import { fmt } from '../lib/format'

const COLORS: Record<string, string> = { fixed: C.yellow, pi: C.blue, ff_pi: C.green, ff_hw_pi: C.purple }

const opts = ref<AerationOptions | null>(null)
const scenario = ref('dry')
const setpoint = ref(2.0)
const seed = ref(101)
const enabled = ref<string[]>(['fixed', 'pi', 'ff_pi', 'ff_hw_pi'])
const result = ref<AerationResult | null>(null)
const loading = ref(false)
const error = ref('')
const elapsed = ref<number | null>(null)

let ctrl: AbortController | null = null
let debounce: number | undefined

async function run() {
  ctrl?.abort()
  ctrl = new AbortController()
  loading.value = true
  error.value = ''
  const t0 = performance.now()
  try {
    result.value = await api.post<AerationResult>('/api/aeration/simulate', {
      scenario: scenario.value, setpoint: setpoint.value, seed: seed.value, strategies: enabled.value,
    }, ctrl.signal)
    elapsed.value = Math.round(performance.now() - t0)
  } catch (e) {
    if ((e as Error).name !== 'AbortError') error.value = (e as Error).message
  } finally {
    loading.value = false
  }
}

function schedule() {
  clearTimeout(debounce)
  debounce = window.setTimeout(run, 250)          // 拖动滑块时合并请求
}
watch([scenario, setpoint, seed, enabled], () => { if (enabled.value.length) schedule() }, { deep: true })

const route = useRoute()
onMounted(async () => {
  opts.value = await api.get<AerationOptions>('/api/aeration/options')
  // 演示页的“暴雨”卡片链接到 ?scenario=storm，直接打开对应工况
  const q = String(route.query.scenario ?? '')
  if (opts.value.scenarios.some((s) => s.id === q)) scenario.value = q
  run()
})

function toggle(id: string) {
  const i = enabled.value.indexOf(id)
  if (i >= 0) { if (enabled.value.length > 1) enabled.value.splice(i, 1) }
  else enabled.value.push(id)
}

const ids = computed(() => (result.value ? Object.keys(result.value.strategies) : []))
const hours = computed(() => result.value?.series.t_min.map((m) => m / 60) ?? [])
const xLabel = (h: number) => `${String(Math.floor(h)).padStart(2, '0')}:${String(Math.round((h % 1) * 60)).padStart(2, '0')}`

// ---------------------------------------------------------------- 图表
const doEl = ref<HTMLElement | null>(null)
const airEl = ref<HTMLElement | null>(null)
const kwhEl = ref<HTMLElement | null>(null)
const nh3El = ref<HTMLElement | null>(null)
const doChart = useChart(doEl)
const airChart = useChart(airEl)
const kwhChart = useChart(kwhEl)
const nh3Chart = useChart(nh3El)

function lineBase(yName: string, extra: object = {}) {
  return {
    grid: { left: 48, right: 46, top: 52, bottom: 28 },
    tooltip: { ...tooltipStyle, valueFormatter: (v: number) => (typeof v === 'number' ? v.toFixed(2) : v) },
    legend: { top: 0, left: 0, textStyle: { color: C.text, fontSize: 12 }, itemWidth: 14, itemHeight: 8 },
    xAxis: { type: 'category', data: hours.value, boundaryGap: false, ...axisStyle,
             axisLabel: { ...axisStyle.axisLabel, formatter: (v: string) => xLabel(+v), interval: 35 }, splitLine: { show: false } },
    yAxis: { type: 'value', name: yName, nameTextStyle: { color: C.dim, fontSize: 11 }, ...axisStyle },
    ...extra,
  }
}

function strategyLines(key: 'do' | 'air_frac' | 'nh3n_out') {
  const r = result.value!
  return ids.value.map((id) => ({
    name: r.strategies[id].label, type: 'line', showSymbol: false, smooth: false,
    data: r.strategies[id].series[key], lineStyle: { width: id === 'fixed' ? 1.5 : 2, color: COLORS[id] },
    itemStyle: { color: COLORS[id] },
  }))
}

function draw() {
  const r = result.value
  if (!r) return
  const sp = r.setpoint
  const doSeries = strategyLines('do')
  ;(doSeries[0] as any).markArea = {
    silent: true, itemStyle: { color: 'rgba(54,211,154,0.08)' },
    data: [[{ yAxis: sp - 0.5 }, { yAxis: sp + 0.5 }]],
  }
  ;(doSeries[0] as any).markLine = {
    silent: true, symbol: 'none', label: { color: C.green, formatter: `设定值 ${sp}`, position: 'insideEndTop' },
    lineStyle: { color: C.green, type: 'dashed' }, data: [{ yAxis: sp }],
  }
  doChart.set(lineBase('mg/L', { series: doSeries, yAxis: { type: 'value', name: 'mg/L', min: 0, max: (v: any) => Math.max(5, Math.ceil(v.max)), nameTextStyle: { color: C.dim }, ...axisStyle } }))

  airChart.set(lineBase('× 设计风量', {
    yAxis: [
      { type: 'value', name: '× 设计风量', nameTextStyle: { color: C.dim, fontSize: 11 }, min: 0, ...axisStyle },
      { type: 'value', name: '× 设计负荷', nameTextStyle: { color: C.dim, fontSize: 11 }, min: 0, ...axisStyle, splitLine: { show: false } },
    ],
    series: [
      ...airLines(),
      { name: '进水需氧负荷（真实）', type: 'line', yAxisIndex: 1, showSymbol: false, data: r.series.load,
        lineStyle: { color: '#ffffff', type: 'dashed', width: 1, opacity: 0.55 }, itemStyle: { color: '#ffffff' }, z: 1 },
    ],
  }))

  function airLines() {
    const lines = strategyLines('air_frac')
    ;(lines[0] as any).markLine = {
      silent: true, symbol: 'none', label: { color: C.red, formatter: '风机上限 1.7', position: 'insideEndTop' },
      lineStyle: { color: C.red, type: 'dashed', opacity: 0.7 }, data: [{ yAxis: 1.7 }],
    }
    return lines
  }
  const nh3 = strategyLines('nh3n_out')
  ;(nh3[0] as any).markLine = {
    silent: true, symbol: 'none', label: { color: C.red, formatter: '一级 A 限值 5', position: 'insideEndTop' },
    lineStyle: { color: C.red, type: 'dashed' }, data: [{ yAxis: 5 }],
  }
  nh3Chart.set(lineBase('mg/L', { series: nh3 }))

  kwhChart.set({
    grid: { left: 60, right: 20, top: 40, bottom: 40 },
    tooltip: { ...tooltipStyle, trigger: 'item' },
    xAxis: { type: 'category', data: ids.value.map((id) => r.strategies[id].label), ...axisStyle,
             axisLabel: { ...axisStyle.axisLabel, interval: 0, formatter: (v: string) => v.replace('（', '\n（') } },
    yAxis: { type: 'value', name: 'kWh / 天', nameTextStyle: { color: C.dim }, ...axisStyle },
    series: [{
      type: 'bar', barWidth: '45%',
      data: ids.value.map((id) => ({ value: r.strategies[id].metrics.kwh, itemStyle: { color: COLORS[id], borderRadius: [3, 3, 0, 0] } })),
      label: {
        show: true, position: 'top', color: C.text, fontSize: 12,
        formatter: (p: any) => {
          const m = r.strategies[ids.value[p.dataIndex]].metrics
          return m.kwh_vs_fixed_pct === undefined || ids.value[p.dataIndex] === 'fixed'
            ? `${Math.round(m.kwh)}` : `${Math.round(m.kwh)}\n${m.kwh_vs_fixed_pct > 0 ? '+' : ''}${m.kwh_vs_fixed_pct.toFixed(1)}%`
        },
      },
    }],
  })
}
watch(result, draw)

const bestMae = computed(() => {
  if (!result.value) return null
  return ids.value.reduce((a, b) => (result.value!.strategies[a].metrics.do_mae <= result.value!.strategies[b].metrics.do_mae ? a : b))
})
</script>

<template>
  <div class="aer">
    <section class="panel controls">
      <div class="ctl">
        <label>场景</label>
        <select v-model="scenario" class="input">
          <option v-for="s in opts?.scenarios ?? []" :key="s.id" :value="s.id">{{ s.label }}</option>
        </select>
      </div>
      <div class="ctl slider">
        <label>DO 设定值 <b class="num">{{ setpoint.toFixed(1) }}</b> mg/L</label>
        <input v-model.number="setpoint" type="range" :min="opts?.setpoint.min ?? 1" :max="opts?.setpoint.max ?? 3" :step="opts?.setpoint.step ?? 0.1" />
      </div>
      <div class="ctl">
        <label>随机种子</label>
        <input v-model.number="seed" type="number" min="1" class="input seed" />
        <button class="btn sm" @click="seed = 100 + Math.floor(Math.random() * 900)">换一组扰动</button>
      </div>
      <div class="ctl strategies">
        <label>策略</label>
        <button v-for="s in opts?.strategies ?? []" :key="s.id" class="chip" :class="{ on: enabled.includes(s.id) }"
                :style="{ '--c': COLORS[s.id] }" @click="toggle(s.id)">{{ s.label }}</button>
      </div>
      <div class="state">
        <span v-if="loading" class="dim">计算中…</span>
        <span v-else-if="error" class="err">{{ error }}</span>
        <span v-else-if="elapsed !== null" class="dim">本次 {{ elapsed }} ms（4 天仿真，60 秒步长）</span>
      </div>
    </section>

    <div class="notice">
      {{ opts?.notice ?? '仿真结果，仅用于说明方法' }}。主厂 P01，同一种子下各策略的进水与测量噪声完全相同；控制器只能看到带滞后和噪声的测量值。
      节能比例取决于固定曝气的配风基准（这里按旱天高峰配风），不能直接当作真实厂效果。
    </div>

    <section v-if="result" class="cards">
      <div v-for="id in ids" :key="id" class="panel card" :style="{ '--c': COLORS[id] }">
        <div class="name">{{ result.strategies[id].label }}</div>
        <div class="kv">
          <div><span class="dim">DO 平均误差</span><b class="num" :class="{ best: id === bestMae }">{{ fmt(result.strategies[id].metrics.do_mae, 3) }}</b></div>
          <div><span class="dim">电耗 kWh/天</span><b class="num">{{ fmt(result.strategies[id].metrics.kwh, 0) }}</b></div>
          <div><span class="dim">较固定曝气</span>
            <b class="num" :class="(result.strategies[id].metrics.kwh_vs_fixed_pct ?? 0) < 0 ? 's-normal' : ''">
              {{ result.strategies[id].metrics.kwh_vs_fixed_pct === undefined ? '—' : `${result.strategies[id].metrics.kwh_vs_fixed_pct > 0 ? '+' : ''}${fmt(result.strategies[id].metrics.kwh_vs_fixed_pct, 1)}%` }}
            </b></div>
          <div><span class="dim">氨氮超 5 mg/L</span>
            <b class="num" :class="result.strategies[id].metrics.nh3n_over_limit_h > 0 ? 's-alarm' : ''">{{ fmt(result.strategies[id].metrics.nh3n_over_limit_h, 2) }} h</b></div>
          <div><span class="dim">DO 偏低时间</span><b class="num">{{ fmt(result.strategies[id].metrics.do_low_pct, 1) }}%</b></div>
          <div><span class="dim">过曝气时间</span><b class="num">{{ fmt(result.strategies[id].metrics.do_high_pct, 1) }}%</b></div>
        </div>
      </div>
    </section>

    <section class="charts">
      <div class="panel"><h3>好氧区 DO（真实值）<span class="sub">绿色带：设定值 ±0.5 mg/L</span></h3><div ref="doEl" class="chart" /></div>
      <div class="panel"><h3>风量与进水负荷<span class="sub">白色虚线：真实需氧负荷（右轴）</span></h3><div ref="airEl" class="chart" /></div>
      <div class="panel"><h3>出水氨氮</h3><div ref="nh3El" class="chart" /></div>
      <div class="panel"><h3>24 小时鼓风机电耗</h3><div ref="kwhEl" class="chart" /></div>
    </section>

    <section class="panel explain">
      <h3>怎么读这页</h3>
      <ul>
        <li><b>固定曝气</b>按旱天高峰配风，所以大部分时间过曝气。若把固定风量降到平均水平，电耗会和 PI 差不多，但负荷高峰时 DO 不足、氨氮超标。这里的“省电”是在不超标前提下少曝气，不是算法本身凭空省出来的。</li>
        <li><b>暴雨</b>场景最能说明反馈控制的价值：初期冲刷带来的浓度峰下，固定曝气氨氮超标约 3 小时，反馈控制压到约 0.2 小时，同时省电。</li>
        <li><b>前馈与负荷预测</b>在这个模型里几乎没有增益：模型的耗氧随进水即时变化，PI 跟得上；进水的不规则波动比日内规律大，Holt-Winters 预测不如沿用当前测量值。真实厂有管网和池容滞后时结论可能不同。</li>
        <li><b>超能力冲击</b>时所需风量超过运行风机上限，任何控制都会顶在上限，应启用备用风机或调蓄削峰。</li>
        <li>拖动<b>设定值</b>可以看到：在硝化允许的范围内，设定值每升 0.5 mg/L，电耗约增加 9%–10%。</li>
      </ul>
      <p class="dim">方法、整定规则（动作预算内误差最小）、10 个评测种子的汇总与局限见仓库 backend/evals/aeration/results/report.md。</p>
    </section>
  </div>
</template>

<style scoped>
.aer { display: flex; flex-direction: column; gap: 12px; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 28px; }
.ctl { display: flex; align-items: center; gap: 8px; }
.ctl label { color: var(--dim); white-space: nowrap; }
.ctl label b { color: var(--cyan); font-size: 18px; }
.slider input { width: 220px; accent-color: var(--cyan); }
.seed { width: 80px; }
.ctl select { width: 340px; max-width: 100%; }
.strategies { flex-wrap: wrap; }
.chip {
  border: 1px solid color-mix(in srgb, var(--c) 50%, transparent); background: transparent; color: var(--dim);
  padding: 3px 10px; border-radius: 12px; cursor: pointer; font-size: 12px;
}
.chip.on { color: #fff; background: color-mix(in srgb, var(--c) 28%, transparent); border-color: var(--c); }
.state { margin-left: auto; font-size: 12px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
.card { border-top: 2px solid var(--c); }
.card .name { color: var(--c); font-weight: 600; margin-bottom: 8px; }
.kv { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 14px; }
.kv > div { display: flex; justify-content: space-between; align-items: baseline; gap: 6px; font-size: 12px; }
.kv b { font-size: 16px; color: var(--strong); font-weight: 600; }
.kv b.best { color: var(--green); }
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.chart { height: 280px; }
.explain ul { margin: 0; padding-left: 20px; line-height: 1.8; }
.explain p { margin: 8px 0 0; font-size: 12px; }
@media (max-width: 1100px) { .charts { grid-template-columns: 1fr; } }
@media (max-width: 600px) { .slider input { width: 100%; } .slider { flex-wrap: wrap; width: 100%; } .state { margin-left: 0; } }
</style>

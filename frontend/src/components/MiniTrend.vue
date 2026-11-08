<script setup lang="ts">
// 单指标小曲线：标题带最新值，限值画红色虚线、预警边界画黄色虚线
import { computed, ref, watch } from 'vue'
import { axisStyle, C, tooltipStyle, useChart } from '../lib/chart'
import { fmt } from '../lib/format'

const props = defineProps<{
  label: string; unit: string; ts: string[]; values: (number | null)[]
  limit?: number | null; warnAt?: number | null; color?: string; digits?: number
}>()

const el = ref<HTMLElement | null>(null)
const chart = useChart(el)
const last = computed(() => [...props.values].reverse().find((v) => v !== null) ?? null)
const lineColor = computed(() => props.color ?? C.cyan)

function nice(v: number) {
  const p = 10 ** Math.floor(Math.log10(v))
  const m = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find((x) => x * p >= v - 1e-12) ?? 10
  return +(m * p).toPrecision(6)
}

watch(() => [props.ts, props.values, props.limit, props.warnAt], () => {
  const marks: any[] = []
  if (props.limit != null) marks.push({ yAxis: props.limit, lineStyle: { color: C.red, type: 'dashed', opacity: 0.7 }, label: { color: C.red, formatter: `限值 ${props.limit}`, fontSize: 10, position: 'insideEndTop' } })
  if (props.warnAt != null) marks.push({ yAxis: props.warnAt, lineStyle: { color: C.yellow, type: 'dotted', opacity: 0.6 }, label: { show: false } })
  const vals = props.values.filter((v): v is number => v !== null)
  // 纵轴包含限值，曲线远低于限值时也能看出余量；上界取整，避免出现 0.7000001 这类刻度
  const top = Math.max(...vals, props.limit ?? -Infinity) * 1.08
  chart.set({
    grid: { left: 40, right: 12, top: 10, bottom: 22 },
    tooltip: { ...tooltipStyle, valueFormatter: (v: number) => `${fmt(v, props.digits ?? 2)} ${props.unit}` },
    xAxis: { type: 'time', ...axisStyle, splitLine: { show: false }, axisLabel: { ...axisStyle.axisLabel, fontSize: 10, hideOverlap: true, formatter: { year: '{yyyy}', month: '{MM}-{dd}', day: '{MM}-{dd}', hour: '{HH}:{mm}', minute: '{HH}:{mm}' } } },
    yAxis: { type: 'value', ...axisStyle, min: 0, max: top > 0 && Number.isFinite(top) ? nice(top) : undefined,
             axisLabel: { ...axisStyle.axisLabel, fontSize: 10 } },
    series: [{
      name: props.label, type: 'line', showSymbol: false, connectNulls: false, data: props.ts.map((t, i) => [t, props.values[i]]),
      lineStyle: { color: lineColor.value, width: 1.5 }, itemStyle: { color: lineColor.value },
      areaStyle: { color: lineColor.value, opacity: 0.08 },
      markLine: marks.length ? { silent: true, symbol: 'none', data: marks } : undefined,
    }],
  })
}, { immediate: true })

const level = computed(() => {
  const v = last.value
  if (v === null || props.limit == null) return ''
  if (v > props.limit) return 's-alarm'
  return props.warnAt != null && v > props.warnAt ? 's-warn' : ''
})
</script>

<template>
  <div class="mini">
    <div class="hd">
      <span>{{ label }}</span>
      <b class="num" :class="level">{{ fmt(last, digits ?? 2) }}</b><i class="dim">{{ unit }}</i>
    </div>
    <div ref="el" class="cv" />
  </div>
</template>

<style scoped>
.mini { min-width: 0; }
.hd { display: flex; align-items: baseline; gap: 4px; font-size: 12px; color: var(--dim); }
.hd b { margin-left: auto; font-size: 16px; color: var(--strong); }
.hd b.s-alarm { color: var(--red); }
.hd b.s-warn { color: var(--yellow); }
.hd i { font-style: normal; font-size: 11px; }
.cv { height: 120px; }
</style>

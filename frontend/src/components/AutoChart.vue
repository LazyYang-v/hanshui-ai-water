<script setup lang="ts">
// 查询结果自动图表：规则见 lib/autochart.ts；判断不出合适图形时不渲染
import { computed, onMounted, ref, shallowRef, watch } from 'vue'
import type { MetricInfo } from '../lib/api'
import { inferChart } from '../lib/autochart'
import { axisStyle, C, tooltipStyle, useChart } from '../lib/chart'
import { metricCatalog } from '../lib/metrics'

const props = defineProps<{ columns?: string[]; rows?: unknown[][]; title?: string }>()
const metrics = shallowRef<MetricInfo[]>([])
onMounted(async () => { metrics.value = await metricCatalog() })
const spec = computed(() => inferChart(props.columns, props.rows, metrics.value))

const el = ref<HTMLElement | null>(null)
const chart = useChart(el)
const COLORS = [C.cyan, C.yellow, C.green, C.purple, C.red, C.blue, '#ff9f6e']

watch(spec, (s) => {
  if (!s) return
  const two = s.series.some((x) => x.axis === 1)
  const yAxis = (i: number) => ({ type: 'value', ...axisStyle, scale: s.type === 'line', splitLine: i ? { show: false } : axisStyle.splitLine,
                                  axisLabel: { ...axisStyle.axisLabel, fontSize: 10 } })
  chart.set({
    color: COLORS,
    grid: { left: 44, right: two ? 44 : 14, top: s.series.length > 1 ? 30 : 14, bottom: 24 },
    tooltip: { ...tooltipStyle, trigger: 'axis' },
    legend: s.series.length > 1 ? { top: 0, textStyle: { color: C.text, fontSize: 11 }, itemWidth: 12, itemHeight: 6, type: 'scroll' } : undefined,
    xAxis: { type: 'category', data: s.x, ...axisStyle, splitLine: { show: false }, axisLabel: { ...axisStyle.axisLabel, fontSize: 10, hideOverlap: true } },
    yAxis: two ? [yAxis(0), yAxis(1)] : yAxis(0),
    series: s.series.map((x) => ({
      name: x.name, type: s.type, yAxisIndex: x.axis, data: x.data, showSymbol: s.x.length <= 12, connectNulls: false,
      barMaxWidth: 26, lineStyle: { width: 1.6 },
      markLine: x.limit != null ? { silent: true, symbol: 'none', data: [{ yAxis: x.limit }],
                                    lineStyle: { color: C.red, type: 'dashed', opacity: 0.7 },
                                    label: { color: C.red, fontSize: 10, formatter: `限值 ${x.limit}`, position: 'insideEndTop' } } : undefined,
    })),
  })
}, { immediate: true })
</script>

<template>
  <figure v-if="spec" class="auto">
    <div ref="el" class="cv" />
    <figcaption class="dim">
      {{ title ? title + ' · ' : '' }}{{ spec.type === 'line' ? '折线' : '柱状' }}图由前端按查询结果自动绘制（横轴：{{ spec.xName }}）{{ spec.note ? '；' + spec.note : '' }}
    </figcaption>
  </figure>
</template>

<style scoped>
.auto { margin: 6px 0; padding: 6px 4px 4px; border: 1px solid rgba(64, 140, 255, 0.18); border-radius: 4px; background: rgba(5, 15, 40, 0.4); }
.cv { height: 190px; }
figcaption { font-size: 11px; padding: 2px 4px 0; }
</style>

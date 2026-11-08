// ECharts 组合式封装：挂载即初始化，容器尺寸变化自动 resize，卸载时释放
import { BarChart, EffectScatterChart, GaugeChart, LineChart, ScatterChart } from 'echarts/charts'
import { GridComponent, LegendComponent, MarkAreaComponent, MarkLineComponent, TitleComponent, TooltipComponent } from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { onBeforeUnmount, shallowRef, watch, type Ref } from 'vue'

echarts.use([LineChart, BarChart, ScatterChart, EffectScatterChart, GaugeChart, GridComponent, LegendComponent,
             TooltipComponent, TitleComponent, MarkLineComponent, MarkAreaComponent, CanvasRenderer])

export function useChart(el: Ref<HTMLElement | null>) {
  const chart = shallowRef<echarts.ECharts | null>(null)
  let ro: ResizeObserver | null = null
  let pending: echarts.EChartsCoreOption | null = null
  // 容器可能在 v-if 之后才出现，所以跟随 ref 初始化/释放，而不是只在 onMounted 里初始化
  watch(el, (node) => {
    ro?.disconnect()
    chart.value?.dispose()
    chart.value = null
    if (!node) return
    chart.value = echarts.init(node, undefined, { renderer: 'canvas' })
    ro = new ResizeObserver(() => chart.value?.resize())
    ro.observe(node)
    if (pending) chart.value.setOption(pending, true)
  }, { flush: 'post' })
  onBeforeUnmount(() => { ro?.disconnect(); chart.value?.dispose() })
  function set(option: echarts.EChartsCoreOption, notMerge = true) {
    pending = option
    chart.value?.setOption(option, notMerge)
  }
  return { chart, set }
}

// 大屏配色：与 style.css 的变量保持一致
export const C = {
  text: '#cfe3ff', dim: '#7f9cc2', grid: 'rgba(120,170,255,0.12)', axis: 'rgba(120,170,255,0.35)',
  cyan: '#2fd3ff', blue: '#3d7bff', green: '#36d39a', yellow: '#ffc53d', red: '#ff5a6a', purple: '#a98bff',
}

export const axisStyle = {
  axisLine: { lineStyle: { color: C.axis } },
  axisLabel: { color: C.dim, fontSize: 11 },
  splitLine: { lineStyle: { color: C.grid } },
}

export const tooltipStyle = {
  trigger: 'axis' as const,
  backgroundColor: 'rgba(8,20,45,0.92)', borderColor: 'rgba(47,211,255,0.4)',
  textStyle: { color: C.text, fontSize: 12 },
}

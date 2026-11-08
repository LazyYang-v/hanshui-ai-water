// 按查询结果自动选图：纯前端规则，不调用模型，结果可复现。
// 规则保守：判断不了就不画，表格始终保留。
import type { MetricInfo } from './api'

export interface ChartSeries { name: string; data: (number | null)[]; axis: 0 | 1; limit?: number | null }
export interface ChartSpec {
  type: 'line' | 'bar'
  x: string[]                 // 横轴标签（时间已格式化为 MM-DD HH:mm）
  xName: string
  series: ChartSeries[]
  note: string                // 说明画了什么、省略了什么
}

const TIME_NAME = /时间|时刻|日期|小时|time|date|^ts$|_ts$|day|hour|bucket|period/i
// 编号、计数类数值列不画：数值大小没有业务含义
const ID_NAME = /(^|_)id($|_)|ID|编号|编码|序号|行号|条数|样本数|采样数|count|^n$|_n$/i
const MAX_SERIES = 4
// 极值类列排在均值类之后，避免“DO 均值 / DO 最低”占满名额而挤掉其他指标
const EXTREME_NAME = /最低|最高|最大|最小|min|max/i
const MAX_BARS = 30
const MAX_PIVOT = 7

function toNum(v: unknown): number | null {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null
  if (typeof v === 'string' && /^-?\d+(\.\d+)?$/.test(v.trim())) return Number(v)
  return null
}
function isNumericCol(rows: unknown[][], c: number) {
  let n = 0, ok = 0
  for (const r of rows) {
    if (r[c] === null || r[c] === undefined || r[c] === '') continue
    n++
    if (toNum(r[c]) !== null) ok++
  }
  return n > 0 && ok / n >= 0.8
}

const pad = (n: number) => String(n).padStart(2, '0')
function parseTime(v: unknown): Date | null {
  if (typeof v !== 'string') return null
  const s = v.trim()
  if (/^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?(Z|[+-]\d{2}:?\d{2})?$/.test(s)) {
    const d = new Date(s.length === 10 ? `${s}T00:00:00+08:00` : s)
    return Number.isNaN(d.getTime()) ? null : d
  }
  // 处置建议里常见的“11-07 18:30”格式：年份无关紧要，只用于排序
  const m = s.match(/^(\d{2})-(\d{2}) (\d{2}):(\d{2})$/)
  if (m) return new Date(`2000-${m[1]}-${m[2]}T${m[3]}:${m[4]}:00+08:00`)
  return null
}
function fmtTime(d: Date, withDate: boolean) {
  const t = new Date(d.getTime() + 8 * 3600e3)        // 固定按北京时间显示
  const hm = `${pad(t.getUTCHours())}:${pad(t.getUTCMinutes())}`
  return withDate ? `${pad(t.getUTCMonth() + 1)}-${pad(t.getUTCDate())} ${hm}` : hm
}

/** 列名去掉单位后缀，用来匹配指标目录的中文名（如 “出水氨氮_mgl” → 出水氨氮）。 */
function matchMetric(name: string, metrics: MetricInfo[]): MetricInfo | undefined {
  const norm = (s: string) => s.toLowerCase().replace(/[\s（）()#_]/g, '')
  const n = norm(name)
  return metrics.find((m) => m.limit != null && (n.includes(norm(m.label)) || name.includes(m.key)))
}

function assignAxes(series: ChartSeries[]): { series: ChartSeries[]; dropped: string[] } {
  // 同一轴上峰值相差 5 倍以内（再宽小指标会被压成直线）；第二个量级放右轴；再不同的省略
  const peak = (s: ChartSeries) => Math.max(...s.data.map((v) => Math.abs(v ?? 0)), 1e-9)
  const base = peak(series[0])
  const close = (a: number, b: number) => a / b <= 5 && b / a <= 5
  let second: number | null = null
  const kept: ChartSeries[] = []
  const dropped: string[] = []
  for (const s of series) {
    const p = peak(s)
    if (close(p, base)) kept.push({ ...s, axis: 0 })
    else if (second === null || close(p, second)) { second ??= p; kept.push({ ...s, axis: 1 }) }
    else dropped.push(s.name)
  }
  return { series: kept, dropped }
}

export function inferChart(columns: string[] | undefined, rows: unknown[][] | undefined, metrics: MetricInfo[] = []): ChartSpec | null {
  if (!columns || !rows || rows.length < 2 || columns.length < 2) return null
  const numeric = columns.map((name, c) => !ID_NAME.test(name) && isNumericCol(rows, c))
  const timeCol = columns.findIndex((name, c) => TIME_NAME.test(name) && !numeric[c] &&
    rows.filter((r) => parseTime(r[c])).length >= rows.length * 0.9)
  const textCols = columns.map((_, c) => c).filter((c) => c !== timeCol && !numeric[c] && !ID_NAME.test(columns[c]))
  const varying = textCols.filter((c) => new Set(rows.map((r) => String(r[c]))).size > 1)
  const valueCols = columns.map((_, c) => c).filter((c) => numeric[c])
    .sort((a, b) => Number(EXTREME_NAME.test(columns[a])) - Number(EXTREME_NAME.test(columns[b])) || a - b)
  if (!valueCols.length) return null
  const notes: string[] = []

  // ---------------- 时间序列
  if (timeCol >= 0) {
    if (varying.length > 1) return null                   // 多个分类维度混在一起，画出来会误导
    const times = rows.map((r) => parseTime(r[timeCol])!.getTime())
    const order = times.map((_, i) => i).sort((a, b) => times[a] - times[b])
    const uniq = [...new Set(order.map((i) => times[i]))]
    // 跨日才显示日期
    const multiDay = new Set(uniq.map((t) => fmtTime(new Date(t), true).slice(0, 5))).size > 1
    const label = (t: number) => fmtTime(new Date(t), multiDay)

    let series: ChartSeries[]
    if (varying.length === 1) {
      // 长表（时间 + 分类 + 数值）：按分类拆成多条线，只画第一个数值列
      const cat = varying[0]
      const cats = [...new Set(rows.map((r) => String(r[cat])))]
      if (cats.length > MAX_PIVOT) return null
      const v = valueCols[0]
      series = cats.map((k) => {
        const byT = new Map(rows.filter((r) => String(r[cat]) === k).map((r) => [parseTime(r[timeCol])!.getTime(), toNum(r[v])]))
        return { name: k, data: uniq.map((t) => byT.get(t) ?? null), axis: 0 as const }
      })
      notes.push(`按「${columns[cat]}」分线，数值为「${columns[v]}」`)
      if (valueCols.length > 1) notes.push(`另有 ${valueCols.length - 1} 个数值列未画`)
      const m = matchMetric(columns[v], metrics)
      if (m) series.forEach((s) => { s.limit = m.limit })
      if (uniq.length < 2) return null
      return { type: 'line', x: uniq.map(label), xName: columns[timeCol], series, note: notes.join('；') }
    }
    if (uniq.length !== rows.length || rows.length < 3) return null   // 宽表时间重复说明不是一条序列
    series = valueCols.map((c) => ({ name: columns[c], data: order.map((i) => toNum(rows[i][c])), axis: 0 as const }))
    if (series.length > MAX_SERIES) { notes.push(`未画：${series.slice(MAX_SERIES).map((x) => x.name).join('、')}`); series = series.slice(0, MAX_SERIES) }
    const { series: kept, dropped } = assignAxes(series)
    if (dropped.length) notes.push(`量级不同未画：${dropped.join('、')}`)
    if (kept.some((s) => s.axis === 1)) notes.push('右轴：' + kept.filter((s) => s.axis === 1).map((s) => s.name).join('、'))
    // 只有一条线时画限值，多条线时限值线容易被误读为属于别的指标
    if (kept.length === 1) kept[0].limit = matchMetric(kept[0].name, metrics)?.limit
    return { type: 'line', x: order.map((i) => label(times[i])), xName: columns[timeCol], series: kept, note: notes.join('；') }
  }

  // ---------------- 分类对比（柱状）
  if (varying.length !== 1 || rows.length > MAX_BARS) return null
  const cat = varying[0]
  if (new Set(rows.map((r) => String(r[cat]))).size !== rows.length) return null
  let series: ChartSeries[] = valueCols.map((c) => ({ name: columns[c], data: rows.map((r) => toNum(r[c])), axis: 0 as const }))
  if (series.length > 3) { notes.push(`未画：${series.slice(3).map((x) => x.name).join('、')}`); series = series.slice(0, 3) }
  const { series: kept, dropped } = assignAxes(series)
  if (dropped.length) notes.push(`量级不同未画：${dropped.join('、')}`)
  if (kept.some((s) => s.axis === 1)) notes.push('右轴：' + kept.filter((s) => s.axis === 1).map((s) => s.name).join('、'))
  if (kept.length === 1) kept[0].limit = matchMetric(kept[0].name, metrics)?.limit
  return { type: 'bar', x: rows.map((r) => String(r[cat])), xName: columns[cat], series: kept, note: notes.join('；') }
}

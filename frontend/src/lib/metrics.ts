// 指标目录只取一次，供自动图表匹配限值
import { api, type MetricInfo } from './api'

let cache: Promise<MetricInfo[]> | null = null
export function metricCatalog(): Promise<MetricInfo[]> {
  cache ??= api.get<{ plant: MetricInfo[]; device: MetricInfo[]; derived: MetricInfo[] }>('/api/metrics')
    .then((m) => [...m.plant, ...m.device, ...(m.derived ?? [])])
    .catch(() => { cache = null; return [] })
  return cache
}

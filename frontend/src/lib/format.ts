export const fmt = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : v.toLocaleString('zh-CN', { maximumFractionDigits: d, minimumFractionDigits: d })

export const hhmm = (iso: string) => new Date(iso).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'Asia/Shanghai' })

export const mdhm = (iso: string) => new Date(iso).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'Asia/Shanghai' })

export const statusLabel: Record<string, string> = { normal: '正常', warn: '预警', alarm: '告警', offline: '离线' }

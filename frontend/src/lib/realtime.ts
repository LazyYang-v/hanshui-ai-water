// 全局实时连接：模拟器帧（ts/plants/devices）与告警消息（type=alarm），断线指数退避重连
import { reactive } from 'vue'

export interface Frame {
  ts: string
  weather: { rain_mm_h: number; temp_c: number }
  plants: Record<string, Record<string, number>>
  devices: { device_id: number; running: boolean; current_a: number; vibration_mms: number; temp_c: number }[]
  scenarios: unknown[]
}
export interface AlarmEvent { type: 'alarm'; event: string; [k: string]: unknown }

export const realtime = reactive({
  connected: false,
  frame: null as Frame | null,
  alarms: [] as AlarmEvent[],          // 最近 50 条告警变化
  alarmSeq: 0,                          // 每来一条告警加 1，供页面刷新列表
})

let ws: WebSocket | null = null
let backoff = 1000

export function connectRealtime() {
  if (ws) return
  const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/realtime`
  ws = new WebSocket(url)
  ws.onopen = () => { realtime.connected = true; backoff = 1000 }
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data)
    if (msg.type === 'alarm') {
      realtime.alarms.unshift(msg)
      realtime.alarms.splice(50)
      realtime.alarmSeq++
    } else if (msg.plants) {
      realtime.frame = msg
    }
  }
  ws.onclose = () => {
    realtime.connected = false
    ws = null
    setTimeout(connectRealtime, backoff)
    backoff = Math.min(backoff * 2, 15000)
  }
}

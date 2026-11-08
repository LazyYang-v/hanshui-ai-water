<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { api, type ActiveScenario } from '../lib/api'
import { hhmm } from '../lib/format'

const KINDS = [
  { kind: 'storm', label: '暴雨', desc: '全区降雨：雨污混接严重的厂站进水激增、浓度被稀释，持续大流量会冲走污泥', needsPlant: false, plant: '' },
  { kind: 'blower_fault', label: '鼓风机故障', desc: '1 号鼓风机振动和温度逐渐升高，约 30 分钟后跳停，DO 下降', needsPlant: true, plant: 'P01' },
  { kind: 'shock_load', label: '进水冲击负荷', desc: '进水浓度突增（如工业废水偷排）', needsPlant: true, plant: 'P03' },
  { kind: 'pipe_leak', label: '碳源管道渗漏', desc: '碳源实际进入生化池的量减少，控制系统加大投加，表现为药耗异常', needsPlant: true, plant: 'P05' },
  { kind: 'dosing_block', label: 'PAC 加药泵堵塞', desc: '除磷剂实际投加量骤降，出水总磷上升', needsPlant: true, plant: 'P02' },
]
const PLANTS = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07']
const SPEEDS = [1, 10, 60, 300]

const active = ref<ActiveScenario[]>([])
const speed = ref<number | null>(null)
const msg = ref('')
const err = ref('')
const busy = ref(false)
let timer: number | undefined

async function refresh() {
  try {
    active.value = await api.get('/api/demo/scenarios')
    speed.value = (await api.get<{ speed: number }>('/api/clock')).speed
  } catch (e) { err.value = (e as Error).message }
}
async function act(fn: () => Promise<{ message: string }>) {
  busy.value = true; err.value = ''; msg.value = ''
  try { msg.value = (await fn()).message } catch (e) { err.value = (e as Error).message }
  finally { busy.value = false; refresh() }
}
const inject = (k: typeof KINDS[number]) =>
  act(() => api.post('/api/demo/scenarios', { kind: k.kind, plant: k.needsPlant ? k.plant : null }))
const clearOne = (s: ActiveScenario) =>
  act(() => api.del(`/api/demo/scenarios?kind=${s.kind}${s.plant_code ? `&plant=${s.plant_code}` : ''}`))
const clearAll = () => act(() => api.del('/api/demo/scenarios'))
const setSpeed = (v: number) => act(() => api.put('/api/demo/speed', { value: v }))

onMounted(() => { refresh(); timer = window.setInterval(refresh, 5000) })
onUnmounted(() => clearInterval(timer))
</script>

<template>
  <div class="demo">
    <section class="panel">
      <h3>演示时钟<span class="sub">倍速只影响模拟器推进速度，所有接口以演示时钟为准</span></h3>
      <div class="row">
        <button v-for="v in SPEEDS" :key="v" class="btn" :class="{ primary: speed === v }" :disabled="busy" @click="setSpeed(v)">{{ v }}×</button>
      </div>
    </section>

    <section class="panel">
      <h3>注入场景<span class="sub">模拟器收到命令后立即生效；告警巡检约 10 秒扫描一次</span></h3>
      <div class="kinds">
        <div v-for="k in KINDS" :key="k.kind" class="kind">
          <div class="t">{{ k.label }} <code class="dim">{{ k.kind }}</code></div>
          <div class="dim d">{{ k.desc }}</div>
          <RouterLink v-if="k.kind === 'storm'" to="/aeration?scenario=storm" class="d">曝气优化页的暴雨工况 →</RouterLink>
          <div class="row">
            <select v-if="k.needsPlant" v-model="k.plant" class="input">
              <option v-for="p in PLANTS" :key="p">{{ p }}</option>
            </select>
            <span v-else class="dim">全区</span>
            <button class="btn primary" :disabled="busy" @click="inject(k)">注入</button>
          </div>
        </div>
      </div>
    </section>

    <section class="panel">
      <h3>进行中的场景<button class="btn danger sm" style="margin-left:auto" :disabled="busy || !active.length" @click="clearAll">全部结束</button></h3>
      <table v-if="active.length" class="grid">
        <thead><tr><th>场景</th><th>厂站</th><th>开始</th><th>结束</th><th>参数</th><th></th></tr></thead>
        <tbody>
          <tr v-for="s in active" :key="s.kind + s.start">
            <td>{{ KINDS.find((k) => k.kind === s.kind)?.label ?? s.kind }}</td>
            <td>{{ s.plant_code ?? '全区' }}</td>
            <td class="num">{{ hhmm(s.start) }}</td>
            <td class="num">{{ hhmm(s.end) }}</td>
            <td class="dim">{{ Object.entries(s.params ?? {}).map(([k, v]) => `${k}=${v}`).join(' ') || '默认' }}</td>
            <td><button class="btn sm" :disabled="busy" @click="clearOne(s)">结束</button></td>
          </tr>
        </tbody>
      </table>
      <div v-else class="dim">无</div>
    </section>
    <div v-if="msg" class="panel s-normal">{{ msg }}</div>
    <div v-if="err" class="panel err">{{ err }}</div>
  </div>
</template>

<style scoped>
.demo { display: flex; flex-direction: column; gap: 12px; max-width: 1100px; margin: 0 auto; }
.row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.kinds { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; }
.kind { border: 1px solid var(--panel-border); border-radius: 4px; padding: 10px; display: flex; flex-direction: column; gap: 6px; }
.kind .t { font-weight: 600; color: #fff; }
.kind .d { font-size: 12px; flex: 1; }
</style>

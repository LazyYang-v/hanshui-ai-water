<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { api, ApiError, type Inspection, type PlantBrief, type VisionModel, type VisionResult, type VisionSample } from '../lib/api'
import { withLogin } from '../lib/auth'
import { fmt, mdhm } from '../lib/format'

const model = ref<VisionModel | null>(null)
const samples = ref<VisionSample[]>([])
const plants = ref<PlantBrief[]>([])
const history = ref<Inspection[]>([])
const loadErr = ref('')

const plant = ref('P01')
const camera = ref('')
const writeAlarm = ref(true)
const showLow = ref(true)

const current = ref<{ src: string; title: string; sample?: VisionSample } | null>(null)
const result = ref<VisionResult | null>(null)
const busy = ref(false)
const err = ref('')
const fileEl = ref<HTMLInputElement | null>(null)
let objectUrl: string | null = null

const CLS_COLOR: Record<string, string> = { floating_debris: '#ffc53d', leak: '#2fd3ff' }
const label = (c: string) => model.value?.class_labels[c] ?? c
// 告警阈值取模型卡里合成测试集的 alarm_conf（漂浮物 0.45、渗漏 0.5），低于阈值的框只展示不告警
const alarmConf = (c: string) => model.value?.metrics.synth_test?.[c]?.alarm_conf ?? 0.5
const shown = computed(() => (result.value?.detections ?? []).filter((d) => showLow.value || d.score >= alarmConf(d.cls)))

async function loadHistory() {
  try { history.value = (await api.get<{ items: Inspection[] }>('/api/vision/inspections?limit=15')).items } catch { /* 忽略 */ }
}
onMounted(async () => {
  try {
    const [m, s, p] = await Promise.all([
      api.get<VisionModel>('/api/vision/model'),
      api.get<{ items: VisionSample[] }>('/api/vision/samples'),
      api.get<PlantBrief[]>('/api/plants'),
    ])
    model.value = m; samples.value = s.items; plants.value = p
  } catch (e) { loadErr.value = (e as Error).message }
  loadHistory()
})

const askedAlarm = ref(false)      // 本次请求是否要求写告警（勾选框之后可能被改动）
function params(cam: string | null) {
  askedAlarm.value = writeAlarm.value && !!plant.value
  const qs = new URLSearchParams({ alarm: String(askedAlarm.value) })
  if (plant.value) qs.set('plant', plant.value)
  if (cam) qs.set('camera', cam)
  return qs
}

async function runSample(s: VisionSample) {
  current.value = { src: s.url, title: s.description, sample: s }
  camera.value = s.camera
  await inspect(() => api.post<VisionResult>(`/api/vision/samples/${s.name}/inspect?${params(s.camera)}`))
}

async function onFile(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (!file) return
  if (file.size > 10 * 1024 * 1024) { err.value = '图片超过 10 MB'; return }
  if (objectUrl) URL.revokeObjectURL(objectUrl)
  objectUrl = URL.createObjectURL(file)
  current.value = { src: objectUrl, title: `上传：${file.name}` }
  await inspect(async () => {
    // 后端收原始请求体（不走 multipart），Content-Type 写图片类型
    const res = await withLogin(() => fetch(`/api/vision/inspect?${params(camera.value.trim() || null)}`, {
      method: 'POST', headers: { 'Content-Type': file.type || 'application/octet-stream' }, body: file,
    }))
    if (!res.ok) {
      let msg = `${res.status}`
      try { const j = await res.json(); msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail) } catch { /* 忽略 */ }
      throw new ApiError(res.status, msg)
    }
    return res.json()
  })
  if (fileEl.value) fileEl.value.value = ''
}

async function inspect(fn: () => Promise<VisionResult>) {
  busy.value = true; err.value = ''; result.value = null
  try { result.value = await fn(); loadHistory() } catch (e) { err.value = (e as Error).message } finally { busy.value = false }
}

const EVENT_LABEL: Record<string, string> = { new: '新告警', updated: '合并到已有告警', escalated: '升级', recovered: '恢复' }
const pct = (v: number | undefined) => (v === undefined ? '—' : (v * 100).toFixed(1) + '%')
const srcLabel = (s: string) => s.startsWith('sample:') ? '样例 ' + s.slice(7) : s === 'upload' ? '上传' : s
</script>

<template>
  <div class="page">
    <div class="notice">
      巡检模型为 YOLO11n，用程序合成图 + 公开数据集 FloatWaste（真实河道图片）训练，ONNX 在 CPU 上推理。
      漂浮物在真实图片上的召回为 {{ pct(model?.metrics.real_val?.floating_debris?.recall) }}，渗漏只有合成数据，没有真实图片验证。检出只作为提示，需要人工复核画面。
    </div>
    <div v-if="loadErr" class="err">{{ loadErr }}</div>

    <div class="cols">
      <!-- 左：样例与上传 -->
      <section class="panel side">
        <h3>图片来源</h3>
        <div class="opts">
          <label>厂站
            <select v-model="plant" class="input">
              <option value="">不关联（只推理）</option>
              <option v-for="p in plants" :key="p.code" :value="p.code">{{ p.code }} {{ p.name.replace(/污水处理(厂|站)$/, '') }}</option>
            </select>
          </label>
          <label>摄像头<input v-model="camera" class="input" maxlength="32" placeholder="如 CAM-AER-01" /></label>
          <label class="chk"><input v-model="writeAlarm" type="checkbox" :disabled="!plant" /> 命中阈值时写入告警（需厂站 + 摄像头）</label>
        </div>
        <div class="upload">
          <input ref="fileEl" type="file" accept="image/jpeg,image/png,image/webp,image/bmp" hidden @change="onFile" />
          <button class="btn primary" :disabled="busy" @click="fileEl?.click()">上传图片推理</button>
          <span class="dim">≤ 10 MB；上传时用上面填的摄像头编号</span>
        </div>
        <h4>内置样例 <span class="dim">点击即推理</span></h4>
        <div class="samples">
          <button v-for="s in samples" :key="s.name" class="sample" :class="{ on: current?.sample?.name === s.name }" :disabled="busy" @click="runSample(s)">
            <img :src="s.url" :alt="s.name" loading="lazy" />
            <span class="cap">
              <span class="tag" :class="s.source === 'synthetic' ? 'syn' : 'real'">{{ s.source === 'synthetic' ? '合成' : '真实' }}</span>
              {{ s.description.split(/[：，]/)[0] }}
            </span>
          </button>
        </div>
      </section>

      <!-- 中：图像与检测框 -->
      <section class="panel viewer">
        <h3>检测结果
          <span class="sub">
            <label class="chk"><input v-model="showLow" type="checkbox" /> 显示低于告警阈值的框（虚线）</label>
          </span>
        </h3>
        <div v-if="!current" class="dim placeholder">从左侧选一张样例，或上传图片。</div>
        <template v-else>
          <div class="stage">
            <img :src="current.src" :alt="current.title" />
            <svg v-if="result" :viewBox="`0 0 ${result.width} ${result.height}`" preserveAspectRatio="none">
              <g v-for="(d, i) in shown" :key="i">
                <rect :x="d.box[0]" :y="d.box[1]" :width="d.box[2] - d.box[0]" :height="d.box[3] - d.box[1]"
                      :stroke="CLS_COLOR[d.cls]" fill="none" :stroke-width="Math.max(2, result.width / 400)"
                      :stroke-dasharray="d.score < alarmConf(d.cls) ? `${result.width / 120} ${result.width / 160}` : undefined" />
                <text :x="d.box[0] + 2" :y="Math.max(d.box[1] - 4, result.width / 50)" :fill="CLS_COLOR[d.cls]"
                      :font-size="Math.max(12, result.width / 55)" font-weight="600" paint-order="stroke" stroke="#000" stroke-width="3">
                  {{ label(d.cls) }} {{ d.score.toFixed(2) }}
                </text>
              </g>
            </svg>
            <div v-if="busy" class="busy">推理中…</div>
          </div>
          <div class="cap2">{{ current.title }}</div>
          <div v-if="current.sample?.note" class="dim">样例说明：{{ current.sample.note.replace('expect:', '期望检出 ') }}</div>
          <div v-if="current.sample?.attribution" class="attr">图片来源：{{ current.sample.attribution }}</div>
          <div v-if="err" class="err">{{ err }}</div>
        </template>
      </section>

      <!-- 右：结论、告警、模型卡 -->
      <section class="panel side">
        <h3>本次推理</h3>
        <div v-if="!result" class="dim">—</div>
        <template v-else>
          <table class="grid">
            <thead><tr><th>类别</th><th>置信度</th><th>告警阈值</th></tr></thead>
            <tbody>
              <tr v-for="(d, i) in result.detections" :key="i">
                <td><span class="dot" :style="{ background: CLS_COLOR[d.cls] }" />{{ label(d.cls) }}</td>
                <td class="num">{{ d.score.toFixed(3) }}</td>
                <td :class="d.score >= alarmConf(d.cls) ? 's-alarm' : 'dim'">{{ d.score >= alarmConf(d.cls) ? '达到' : '未达' }} {{ alarmConf(d.cls) }}</td>
              </tr>
              <tr v-if="!result.detections.length"><td colspan="3" class="dim">未检出（展示下限 {{ model?.conf }}）</td></tr>
            </tbody>
          </table>
          <div class="dim small">
            {{ result.width }}×{{ result.height }} · 推理 {{ fmt(result.timings_ms.infer, 0) }} ms · 总计 {{ fmt(result.timings_ms.total, 0) }} ms（解码 {{ fmt(result.timings_ms.decode, 0) }} / 预处理 {{ fmt(result.timings_ms.preprocess, 0) }}）· 模型 {{ result.model_id }}
          </div>
          <div v-for="e in result.events" :key="e.alarm_id + e.event" class="event">
            {{ EVENT_LABEL[e.event] ?? e.event }}：<RouterLink :to="`/alarms?id=${e.alarm_id}`">告警 #{{ e.alarm_id }}</RouterLink>
            <span v-if="e.level" class="dim">（{{ e.level }}）</span>
          </div>
          <div v-if="result.alarm_note" class="dim small">{{ result.alarm_note }}</div>
          <div v-else-if="!askedAlarm" class="dim small">本次只推理，未写告警。</div>
          <div v-else-if="!result.events.length" class="dim small">没有新增或变化的告警（同一摄像头的未结束告警会合并，连续多次检出才升级为预警）。</div>
        </template>

        <h3 class="mt">模型卡</h3>
        <template v-if="model">
          <div class="dim small">{{ model.name }} · {{ model.classes.map(label).join(' / ') }} · 输入 {{ model.training?.imgsz }} · onnxruntime {{ model.runtime.onnxruntime }}（CPU）</div>
          <table class="grid">
            <thead><tr><th>评测集</th><th>类别</th><th>AP50</th><th>精确率</th><th>召回</th></tr></thead>
            <tbody>
              <template v-for="(set, name) in model.metrics" :key="name">
                <tr v-for="c in model.classes.filter((c) => set[c]?.ap50 !== undefined)" :key="name + c">
                  <td>{{ name === 'synth_test' ? '合成测试' : name === 'real_val' ? '真实验证' : name }}</td>
                  <td>{{ label(c) }}</td>
                  <td class="num">{{ fmt(set[c].ap50, 3) }}</td>
                  <td class="num">{{ pct(set[c].precision) }}</td>
                  <td class="num">{{ pct(set[c].recall) }}</td>
                </tr>
              </template>
            </tbody>
          </table>
          <div class="dim small">合成测试集与训练同一生成器，指标偏乐观；以真实验证集为准。</div>
          <ul class="lim"><li v-for="l in model.limitations" :key="l">{{ l }}</li></ul>
          <details class="small"><summary>许可</summary><p v-for="(v, k) in model.license" :key="k" class="dim">{{ v }}</p></details>
        </template>
      </section>
    </div>

    <section class="panel">
      <h3>最近巡检记录<span class="sub">每次推理都会记录；写告警时记录关联的告警编号</span></h3>
      <table class="grid hist">
        <thead><tr><th>时间（演示时钟）</th><th>厂站</th><th>摄像头</th><th>来源</th><th>检出</th><th>推理 ms</th><th>告警</th></tr></thead>
        <tbody>
          <tr v-for="h in history" :key="h.id">
            <td class="num">{{ mdhm(h.ts) }}</td>
            <td>{{ h.plant ?? '—' }}</td>
            <td>{{ h.camera ?? '—' }}</td>
            <td>{{ srcLabel(h.source) }}</td>
            <td>{{ h.detections.length ? Object.entries(h.detections.reduce((m: Record<string, number>, d) => (m[d.cls] = (m[d.cls] ?? 0) + 1, m), {})).map(([c, n]) => `${label(c)} ${n}`).join('，') : '无' }}</td>
            <td class="num">{{ fmt(h.timings_ms?.infer, 0) }}</td>
            <td><RouterLink v-for="id in h.alarm_ids ?? []" :key="id" :to="`/alarms?id=${id}`" class="aid">#{{ id }}</RouterLink><span v-if="!h.alarm_ids?.length" class="dim">—</span></td>
          </tr>
          <tr v-if="!history.length"><td colspan="7" class="dim">暂无</td></tr>
        </tbody>
      </table>
    </section>
  </div>
</template>

<style scoped>
.page { padding: 14px 16px; display: flex; flex-direction: column; gap: 12px; }
.cols { display: grid; grid-template-columns: 320px 1fr 380px; gap: 12px; align-items: start; }
.side { display: flex; flex-direction: column; gap: 10px; }
.opts { display: flex; flex-direction: column; gap: 6px; }
.opts label { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--dim); }
.opts label .input { flex: 1; min-width: 0; }
.chk { display: flex; align-items: center; gap: 4px; font-size: 12px; color: var(--dim); }
.upload { display: flex; align-items: center; gap: 8px; font-size: 12px; }
h4 { margin: 4px 0 0; font-size: 14px; color: var(--strong); }
h4 .dim { font-size: 12px; font-weight: 400; }
.samples { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; max-height: 560px; overflow-y: auto; }
.sample { padding: 0; border: 1px solid var(--panel-border); border-radius: 4px; background: rgba(5, 15, 40, 0.6); cursor: pointer; overflow: hidden; text-align: left; }
.sample.on { border-color: var(--cyan); box-shadow: 0 0 0 1px var(--cyan); }
.sample img { width: 100%; aspect-ratio: 16 / 10; object-fit: cover; display: block; }
.sample .cap { display: block; padding: 4px 6px; font-size: 11px; color: var(--text); line-height: 1.4; }
.syn { color: var(--dim); }
.real { color: var(--green); }
.viewer { display: flex; flex-direction: column; gap: 8px; }
.placeholder { padding: 120px 0; text-align: center; }
.stage { position: relative; line-height: 0; }
.stage img { width: 100%; display: block; border-radius: 4px; }
.stage svg { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; }
.busy { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; background: rgba(0, 0, 0, 0.45); color: #fff; line-height: 1; }
.cap2 { font-size: 13px; }
.attr { font-size: 11px; color: var(--dim); }
.small { font-size: 12px; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.event { font-size: 13px; color: var(--yellow); }
.mt { margin-top: 8px !important; }
.lim { margin: 0; padding-left: 18px; font-size: 12px; color: var(--yellow); display: flex; flex-direction: column; gap: 3px; }
details summary { cursor: pointer; color: var(--cyan); }
.hist td, .hist th { text-align: left; }
.aid { margin-right: 6px; }
@media (max-width: 1300px) { .cols { grid-template-columns: 1fr; } }
</style>

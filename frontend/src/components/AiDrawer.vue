<script setup lang="ts">
import { nextTick, ref } from 'vue'
import type { ChatResult } from '../lib/api'
import ChatActions from './ChatActions.vue'
import CitedAnswer from './CitedAnswer.vue'
import { postSSE } from '../lib/sse'

const open = defineModel<boolean>('open', { default: false })

interface Step { id: string; text: string; state: 'run' | 'ok' | 'fail'; ms?: number | null }
interface Turn {
  question: string
  steps: Step[]
  result: ChatResult | null
  error: string
  running: boolean
  startedAt: number
  elapsed?: number
}

const TOOL_LABEL: Record<string, string> = { sql_query: '查数（Text2SQL）', rag_search: '知识检索', plant_statistics: '固定统计' }
const STATUS_LABEL: Record<string, string> = { empty: '无结果', refused: '被拒绝', error: '出错', cancelled: '已停止' }
const EXAMPLES = [
  '各厂站当前出水氨氮是多少？哪座最高？',
  '黄官镇昨天是否有氨氮超标？请结合 DO、进水量和鼓风机状态分析可能原因',
  '鼓风机跳停后为什么氨氮可能升高？应该先查什么？',
  '南郑中心厂昨天的吨水曝气电耗是多少？',
]

const turns = ref<Turn[]>([])
const input = ref('')
const sessionId = ref<string | null>(null)
const listEl = ref<HTMLElement | null>(null)
let ctrl: AbortController | null = null

const busy = () => turns.value.some((t) => t.running)

async function scroll() {
  await nextTick()
  listEl.value?.scrollTo({ top: listEl.value.scrollHeight, behavior: 'smooth' })
}

async function ask(q?: string) {
  const question = (q ?? input.value).trim()
  if (question.length < 2 || busy()) return
  input.value = ''
  turns.value.push({ question, steps: [], result: null, error: '', running: true, startedAt: performance.now() })
  const t = turns.value[turns.value.length - 1]     // 取响应式代理，后续修改才会触发渲染
  scroll()
  ctrl = new AbortController()
  try {
    await postSSE('/api/ai/chat/stream', { question, session_id: sessionId.value ?? undefined }, (event, data) => {
      if (event === 'run_start') sessionId.value = data.session_id
      else if (event === 'planning') t.steps.push({ id: `plan${data.round}`, text: data.round > 1 ? `第 ${data.round} 轮规划：判断证据是否足够` : '理解问题、规划查询', state: 'ok' })
      else if (event === 'tool_start') t.steps.push({ id: data.id, text: `${TOOL_LABEL[data.tool] ?? data.tool}：${data.question ?? ''}`, state: 'run' })
      else if (event === 'tool_end') {
        const s = t.steps.find((x) => x.id === data.id)
        if (s) {
          s.state = data.status === 'ok' || data.status === 'empty' ? 'ok' : 'fail'
          s.ms = data.total_ms
          if (data.status !== 'ok') s.text += `（${STATUS_LABEL[data.status] ?? data.status}${data.error ? '：' + data.error : ''}）`
        }
      } else if (event === 'answer_start') t.steps.push({ id: 'answer', text: data.message ?? '综合证据并校验引用', state: 'run' })
      else if (event === 'done') {
        t.result = data
        sessionId.value = data.session_id
        const a = t.steps.find((x) => x.id === 'answer')
        if (a) a.state = 'ok'
      } else if (event === 'error') t.error = data.error ?? '对话执行失败'
      scroll()
    }, ctrl.signal)
  } catch (e) {
    t.error = (e as Error).name === 'AbortError' ? '已取消' : (e as Error).message
  } finally {
    t.running = false
    t.elapsed = Math.round(performance.now() - t.startedAt)
    t.steps.forEach((s) => { if (s.state === 'run') s.state = t.error ? 'fail' : 'ok' })
    scroll()
  }
}

function reset() {
  ctrl?.abort()
  turns.value = []
  sessionId.value = null
}

</script>

<template>
  <transition name="slide">
    <aside v-if="open" class="drawer">
      <header>
        <b>AI 助手</b>
        <span class="dim">查数 · 知识检索 · 固定统计，回答附可核对的引用</span>
        <button class="btn sm" @click="reset">新对话</button>
        <button class="btn sm" @click="open = false">收起</button>
      </header>

      <div ref="listEl" class="list">
        <div v-if="!turns.length" class="empty">
          <p class="dim">数据来自模拟器。回答里的每个数字都对应查询结果的单元格、服务端统计或知识原文；原因分析标为假设，需要人工核查。</p>
          <button v-for="e in EXAMPLES" :key="e" class="ex" @click="ask(e)">{{ e }}</button>
        </div>

        <div v-for="(t, i) in turns" :key="i" class="turn">
          <div class="q">{{ t.question }}</div>
          <div class="a">
            <ol class="steps">
              <li v-for="s in t.steps" :key="s.id" :class="s.state">
                <span class="dot" />{{ s.text }}<span v-if="s.ms" class="dim num"> · {{ Math.round(s.ms) }} ms</span>
              </li>
            </ol>
            <div v-if="t.error" class="err">{{ t.error }}</div>
            <template v-if="t.result">
              <div v-if="t.result.refused" class="refused">{{ t.result.answer || t.result.reason }}</div>
              <CitedAnswer v-else :answer="t.result.answer" :citations="t.result.citations" :tools="t.result.tool_calls" />
              <div v-if="t.result.warnings?.length" class="warn">注意：{{ t.result.warnings.join('；') }}</div>
              <ChatActions v-if="t.result.actions?.length" :actions="t.result.actions"
                           :chat="{ session_id: t.result.session_id, run_id: t.result.run_id }" />
              <div class="meta dim">
                {{ t.result.models.join(', ') }} · {{ t.result.prompt_tokens + t.result.completion_tokens }} tokens ·
                {{ ((t.elapsed ?? t.result.total_ms) / 1000).toFixed(1) }} 秒 · {{ t.result.citations.length }} 处引用
              </div>
            </template>
            <div v-else-if="t.running" class="dim">处理中… 复杂诊断约需 20–40 秒</div>
          </div>
        </div>
      </div>

      <footer>
        <textarea v-model="input" class="input" rows="2" maxlength="500" placeholder="例如：3 号厂氨氮为什么升高？（Enter 发送，Shift+Enter 换行）"
                  @keydown.enter.exact.prevent="ask()" />
        <button v-if="busy()" class="btn danger" @click="ctrl?.abort()">停止</button>
        <button v-else class="btn primary" :disabled="input.trim().length < 2" @click="ask()">发送</button>
      </footer>
    </aside>
  </transition>
</template>

<style scoped>
.drawer {
  position: fixed; top: 56px; right: 0; bottom: 0; width: min(560px, 100vw); z-index: 20;
  background: rgba(6, 18, 44, 0.97); border-left: 1px solid var(--panel-border);
  box-shadow: -12px 0 32px rgba(0, 0, 0, 0.45); display: flex; flex-direction: column;
}
header { display: flex; align-items: center; gap: 10px; padding: 10px 14px; border-bottom: 1px solid var(--panel-border); }
header b { color: #fff; font-size: 16px; }
header .dim { font-size: 12px; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.list { flex: 1; overflow-y: auto; padding: 12px 14px; display: flex; flex-direction: column; gap: 16px; }
.empty { display: flex; flex-direction: column; gap: 8px; }
.empty p { font-size: 13px; line-height: 1.6; margin: 0 0 6px; }
.ex { text-align: left; background: rgba(47, 211, 255, 0.06); border: 1px solid var(--panel-border); border-radius: 4px; padding: 8px 10px; cursor: pointer; color: var(--text); }
.ex:hover { border-color: var(--cyan); }
.q { align-self: flex-end; margin-left: 15%; background: linear-gradient(90deg, #1b5fd6, #1590d8); color: #fff; padding: 8px 12px; border-radius: 8px 8px 2px 8px; white-space: pre-wrap; }
.turn { display: flex; flex-direction: column; gap: 8px; }
.a { background: var(--panel); border: 1px solid var(--panel-border); border-radius: 2px 8px 8px 8px; padding: 10px 12px; display: flex; flex-direction: column; gap: 8px; }
.steps { list-style: none; margin: 0; padding: 0; font-size: 12px; color: var(--dim); display: flex; flex-direction: column; gap: 3px; }
.steps li { display: flex; align-items: center; gap: 6px; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); flex: none; }
.steps li.run .dot { background: var(--cyan); animation: pulse 1s infinite; }
.steps li.fail .dot { background: var(--red); }
@keyframes pulse { 50% { opacity: 0.3; } }
.refused { color: var(--yellow); white-space: pre-wrap; }
.warn { font-size: 12px; color: var(--yellow); }
.meta { font-size: 11px; }
footer { display: flex; gap: 8px; padding: 10px 14px; border-top: 1px solid var(--panel-border); align-items: flex-end; }
footer textarea { flex: 1; resize: none; }
.slide-enter-active, .slide-leave-active { transition: transform 0.2s ease; }
.slide-enter-from, .slide-leave-to { transform: translateX(100%); }
</style>

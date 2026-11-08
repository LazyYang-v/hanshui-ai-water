// 自动图表规则的单元测试：node --test（Node 22.6+ 直接运行 TypeScript）
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { inferChart } from '../src/lib/autochart.ts'

const M = [
  { key: 'nh3n_out', label: '出水氨氮', unit: 'mg/L', group: '出水', limit: 5, lower_is_bad: false },
  { key: 'do_aerobic_1', label: '好氧区 DO（1#）', unit: 'mg/L', group: '工艺', limit: 1, lower_is_bad: true },
]

test('宽表逐时数据：折线，编号/计数列不画，量级不同的放右轴', () => {
  // 氨氮峰值 3.4、DO 2.0 同轴；流量 143 是氨氮的 40 倍，放右轴
  // 取自 Agent 验收记录的列结构
  const cols = ['时间', '出水氨氮_mgl', '好氧区do1_mgl', '进水流量_m3', '鼓风机设备_id编码', '采样条数']
  const rows = [0, 1, 2, 3].map((h) => [`2026-11-06T0${h}:00:00+08:00`, 0.4 + h, 2.0 - h * 0.3, 140 + h, '121号(P06-BL-1)', 120])
  const s = inferChart(cols, rows, M)!
  assert.equal(s.type, 'line')
  assert.deepEqual(s.x, ['00:00', '01:00', '02:00', '03:00'])
  assert.deepEqual(s.series.map((x) => [x.name, x.axis]), [['出水氨氮_mgl', 0], ['好氧区do1_mgl', 0], ['进水流量_m3', 1]])
  assert.ok(s.series.every((x) => x.limit == null), '多条线时不画限值')
  assert.match(s.note, /右轴：进水流量_m3/)
})

test('乱序时间按时间排序；跨日显示日期；单线匹配限值', () => {
  const s = inferChart(['时间', '出水氨氮'], [['2026-11-07T01:00:00+08:00', 3], ['2026-11-06T23:00:00+08:00', 6], ['2026-11-07T00:00:00+08:00', 4]], M)!
  assert.deepEqual(s.x, ['11-06 23:00', '11-07 00:00', '11-07 01:00'])
  assert.deepEqual(s.series[0].data, [6, 4, 3])
  assert.equal(s.series[0].limit, 5)
})

test('处置建议的“MM-DD HH:mm”时间格式也能识别', () => {
  const s = inferChart(['时间', '好氧区 DO（1#）均值(mg/L)'], [['11-07 18:15', 1.9], ['11-07 18:30', 0.43], ['11-07 18:45', 0.05]], M)!
  assert.equal(s.type, 'line')
  assert.equal(s.series[0].limit, 1)
})

test('长表（时间 + 厂站 + 数值）按厂站分线', () => {
  const rows = ['P01', 'P02'].flatMap((p, k) => [0, 1, 2].map((h) => [`2026-11-07 0${h}:00:00+08:00`, p, h + k]))
  const s = inferChart(['小时', '厂站', '出水氨氮均值'], rows, M)!
  assert.deepEqual(s.series.map((x) => x.name), ['P01', 'P02'])
  assert.deepEqual(s.series[1].data, [1, 2, 3])
  assert.ok(s.series.every((x) => x.limit === 5))
})

test('分类对比画柱状图', () => {
  const s = inferChart(['厂站', '处理水量_m3', '达标率_pct'], [['P01', 50208, 100], ['P04', 2994, 96.18], ['P07', 1998, 100]], M)!
  assert.equal(s.type, 'bar')
  assert.deepEqual(s.x, ['P01', 'P04', 'P07'])
  assert.equal(s.series.find((x) => x.name === '达标率_pct')!.axis, 1)
})

test('不该画的情况返回 null', () => {
  assert.equal(inferChart(['时间', '出水氨氮'], [['2026-11-07T00:00:00+08:00', 1]], M), null, '只有一行')
  assert.equal(inferChart(['厂站', '名称'], [['P01', '南郑'], ['P02', '协税']], M), null, '没有数值列')
  // 告警列表：设备、指标两个分类维度混在一起
  assert.equal(inferChart(['告警ID', '开始时间', '设备', '指标', '数值'], [
    [24, '2026-11-07T18:35:00+08:00', 'P04-BL-1', '停机', 0],
    [22, '2026-11-07T18:15:00+08:00', 'P04-BL-1', '轴承温度', 78.6],
    [23, '2026-11-07T18:15:00+08:00', 'P04-BL-2', '振动', 5],
  ], M), null)
  // 宽表时间重复：不是一条序列
  assert.equal(inferChart(['时间', '数值'], [['2026-11-07T00:00:00+08:00', 1], ['2026-11-07T00:00:00+08:00', 2], ['2026-11-07T01:00:00+08:00', 3]], M), null)
  // 只有编号类数值
  assert.equal(inferChart(['厂站', '告警ID'], [['P01', 1], ['P02', 2]], M), null)
})

test('列多时先画均值类，极值类列让位（告警 #25 处置建议的真实列）', () => {
  const cols = ['时间', '好氧区 DO（1#）均值(mg/L)', '好氧区 DO（1#）最低(mg/L)', '曝气风量均值(m³/h)', '鼓风机功率均值(kW)', '出水氨氮均值(mg/L)', '降雨强度最大(mm/h)']
  const rows = [['11-07 17:30', 1.908, 1.739, 744.3, 12.4, 0.227, 0], ['11-07 18:45', 0.053, 0, 0, 0, 6.32, 0], ['11-07 19:00', 1.167, 0.2, 892.4, 14.1, 7.847, 0]]
  const s = inferChart(cols, rows, M)!
  // 功率峰值约为 DO 的 7 倍，放左轴会把 DO 压扁，右轴已被风量占用，所以省略并注明
  assert.deepEqual(s.series.map((x) => [x.name, x.axis]), [['好氧区 DO（1#）均值(mg/L)', 0], ['曝气风量均值(m³/h)', 1], ['出水氨氮均值(mg/L)', 0]])
  assert.match(s.note, /未画：好氧区 DO（1#）最低\(mg\/L\)、降雨强度最大/)
  assert.match(s.note, /量级不同未画：鼓风机功率均值/)
})

test('AI 助手回放数据：氨氮约 10、流量约 200 时流量放右轴，DO 不被压扁', () => {
  const rows = [0, 1, 2].map((h) => [`2026-11-06T1${h}:00:00+08:00`, h === 1 ? 10.1 : 0.4, 2.3, 2.4, 160 + 20 * h])
  const s = inferChart(['时间', '出水氨氮_mgl', '好氧区do1_mgl', '好氧区do2_mgl', '进水流量_m3'], rows, M)!
  assert.deepEqual(s.series.map((x) => x.axis), [0, 0, 0, 1])
})

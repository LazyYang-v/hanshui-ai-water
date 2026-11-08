# 告警因果回放：causal-load-60s-final

模拟器评测步长 60 秒；默认负荷参数 {'ratio': 1.5, 'concentration_ratio': 1.25, 'mad_sigma': 3.0, 'on': 3, 'off': 3}。
每种子7天历史+6天评测，8个异常场景与1场阴性小雨；按5分钟推进演示时钟，实际调用生产scan/持久化，含6小时停机。
基线对每个桶取其整点以前7天完整小时；不使用场景标签剔除训练数据。预载SQL聚合通过6个时刻与生产取数逐字段对照。
检出延迟=实际首次写入扫描时刻−场景开始；缺失可能延缓确认。kg/h按同采样乘积聚合，浓度和负荷同时升高才预警。

| 种子 | 场景 | 检出 | 首个规则 | 实际延迟(分钟) | 负荷预警较出水提前(分钟) |
|---|---|---|---|---:|---:|
| 1 | shock_load P04 | True | inflow_load:cod_load_kgh | 15.0 | 15.0 |
| 1 | dosing_block P02 | True | effluent_trend:tp_out | 25.0 | None |
| 1 | blower_fault P06 | True | device_temp:temp_c | 25.0 | None |
| 1 | pipe_leak P05 | True | dosing_anomaly:carbon_dose_kgh | 200.0 | None |
| 1 | blower_fault P01 | True | device_temp:temp_c | 25.0 | None |
| 1 | dosing_block P03 | True | effluent_trend:tp_out | 25.0 | None |
| 1 | shock_load P07 | True | inflow_load:cod_load_kgh | 15.0 | 5.0 |
| 1 | storm 全区 | True | inflow_surge:inflow_m3h | 45.0 | 30.0 |
| 1 | 小雨（阴性） 全区 | False | — | None | None |
| 2 | shock_load P04 | True | inflow_load:cod_load_kgh | 15.0 | 45.0 |
| 2 | dosing_block P02 | True | effluent_trend:tp_out | 25.0 | None |
| 2 | blower_fault P06 | True | device_temp:temp_c | 25.0 | None |
| 2 | pipe_leak P05 | True | dosing_anomaly:carbon_dose_kgh | 210.0 | None |
| 2 | blower_fault P01 | True | device_temp:temp_c | 25.0 | None |
| 2 | dosing_block P03 | True | effluent_trend:tp_out | 25.0 | None |
| 2 | shock_load P07 | True | inflow_load:cod_load_kgh | 15.0 | 5.0 |
| 2 | storm 全区 | True | inflow_surge:inflow_m3h | 45.0 | 30.0 |
| 2 | 小雨（阴性） 全区 | False | — | None | None |
| 3 | shock_load P04 | True | inflow_load:cod_load_kgh | 15.0 | None |
| 3 | dosing_block P02 | True | effluent_trend:tp_out | 25.0 | None |
| 3 | blower_fault P06 | True | device_temp:temp_c | 25.0 | None |
| 3 | pipe_leak P05 | True | dosing_anomaly:carbon_dose_kgh | 200.0 | None |
| 3 | blower_fault P01 | True | device_temp:temp_c | 25.0 | None |
| 3 | dosing_block P03 | True | effluent_trend:tp_out | 25.0 | None |
| 3 | shock_load P07 | True | inflow_load:cod_load_kgh | 15.0 | 10.0 |
| 3 | storm 全区 | True | inflow_surge:inflow_m3h | 45.0 | None |
| 3 | 小雨（阴性） 全区 | False | — | None | None |

| 种子 | 检出 | 误报事件 | 全量/增量 | 缺失/多出/字段差异 | 恢复改写 | 扫描次数 | 扫描中位/最大ms |
|---|---|---:|---|---|---:|---:|---|
| 1 | 8/8 | 0 | 43/43 | 0/0/0 | 0 | 1656 | 44.1/1477.9 |
| 2 | 8/8 | 0 | 39/39 | 0/0/0 | 0 | 1656 | 28.4/2465.6 |
| 3 | 8/8 | 0 | 38/38 | 0/0/0 | 0 | 1656 | 20.7/1474.2 |

## 负荷参数对照

只在60秒种子1/2比较参数，种子3作为保留验证；候选规则仍走同一因果基线与状态机。参数对照的延迟是条件确认时间，不含扫描停机。

| 种子 | 负荷倍数 | 持续桶 | 冲击检出 | 延迟(分钟) | 无场景/阴性误报 | 雨期负荷事件 |
|---|---:|---:|---|---|---:|---:|
| 1 | 1.4 | 2 | 2/2 | [10.0, 10.0] | 0 | 3 |
| 1 | 1.4 | 3 | 2/2 | [15.0, 15.0] | 0 | 2 |
| 1 | 1.4 | 4 | 2/2 | [20.0, 20.0] | 0 | 2 |
| 1 | 1.5 | 2 | 2/2 | [10.0, 10.0] | 0 | 3 |
| 1 | 1.5 | 3 | 2/2 | [15.0, 15.0] | 0 | 2 |
| 1 | 1.5 | 4 | 2/2 | [20.0, 20.0] | 0 | 2 |
| 1 | 1.6 | 2 | 2/2 | [10.0, 10.0] | 0 | 3 |
| 1 | 1.6 | 3 | 2/2 | [15.0, 15.0] | 0 | 2 |
| 1 | 1.6 | 4 | 2/2 | [20.0, 20.0] | 0 | 2 |
| 2 | 1.4 | 2 | 2/2 | [10.0, 10.0] | 0 | 1 |
| 2 | 1.4 | 3 | 2/2 | [15.0, 15.0] | 0 | 1 |
| 2 | 1.4 | 4 | 2/2 | [20.0, 20.0] | 0 | 0 |
| 2 | 1.5 | 2 | 2/2 | [10.0, 10.0] | 0 | 1 |
| 2 | 1.5 | 3 | 2/2 | [15.0, 15.0] | 0 | 1 |
| 2 | 1.5 | 4 | 2/2 | [20.0, 20.0] | 0 | 0 |
| 2 | 1.6 | 2 | 2/2 | [10.0, 10.0] | 0 | 1 |
| 2 | 1.6 | 3 | 2/2 | [15.0, 15.0] | 0 | 1 |
| 2 | 1.6 | 4 | 2/2 | [20.0, 20.0] | 0 | 0 |

## 真实写入时序回归

- 种子1：{'missing_is_not_recovery': True, 'outage_catchup': True, 'late_report_actual_detection': True, 'future_samples_hidden': True, 'over_7_days_stays_open': True, 'over_7_days_recovers_from_normal_values': True}。
- 种子2：{'missing_is_not_recovery': True, 'outage_catchup': True, 'late_report_actual_detection': True, 'future_samples_hidden': True, 'over_7_days_stays_open': True, 'over_7_days_recovers_from_normal_values': True}。
- 种子3：{'missing_is_not_recovery': True, 'outage_catchup': True, 'late_report_actual_detection': True, 'future_samples_hidden': True, 'over_7_days_stays_open': True, 'over_7_days_recovers_from_normal_values': True}。

## 限制

- 全部来自模拟器。标签只用于评测归因；场景前后的6小时伴随窗口可能使其他异常不计为误报，不等同真实业务精确率。
- 连续桶是连续有效判断桶，缺失不计数也不中断。迟到数据可能改写历史条件，detected_at保留实际首次发现时刻；普通补扫最近3小时，超窗迟到需手动补扫，已有开放事件会延长窗口。
- 开放事件超过7天仍从开头重放，去掉coverage兜底；需保留原始历史，长事件增加计算开销。初次启动补扫仍为7天。
- 全量/增量一致性只针对本次按时采样；迟到写入另做SQL回归。alarm.peak为REAL，一致性容差0.001，其余字段精确比较。
- 60秒与5秒AR(1)噪声的相关时间不同，分开报告；三个相关污染物分别产生事件，不按事件数冒充独立场景数。

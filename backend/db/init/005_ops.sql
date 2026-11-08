-- M4：告警事件化、工单流转与报告元数据。
-- 本脚本可重复执行：新库由 docker-entrypoint 按序执行；已有库由 API 启动时（app/db_upgrade.py）再执行一次。

-- ---------------------------------------------------------------- 告警：一条记录对应一次持续事件
ALTER TABLE alarm ADD COLUMN IF NOT EXISTS rule TEXT;
ALTER TABLE alarm ADD COLUMN IF NOT EXISTS ended_at TIMESTAMPTZ;
ALTER TABLE alarm ADD COLUMN IF NOT EXISTS peak REAL;
ALTER TABLE alarm ADD COLUMN IF NOT EXISTS peak_at TIMESTAMPTZ;
ALTER TABLE alarm ADD COLUMN IF NOT EXISTS detail JSONB NOT NULL DEFAULT '{}';
ALTER TABLE alarm ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;
ALTER TABLE alarm ADD COLUMN IF NOT EXISTS advice JSONB;
ALTER TABLE alarm ADD COLUMN IF NOT EXISTS closed_at TIMESTAMPTZ;
ALTER TABLE alarm ADD COLUMN IF NOT EXISTS close_note TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS alarm_episode_key ON alarm (plant_id, (coalesce(device_id, 0)), rule, metric, ts);
CREATE INDEX IF NOT EXISTS alarm_open ON alarm (plant_id) WHERE ended_at IS NULL;
COMMENT ON COLUMN alarm.ts IS '告警开始时间（演示时钟）';
COMMENT ON COLUMN alarm.kind IS '类型：水质超标 / 水质预警 / 工艺异常 / 进水冲击 / 药耗异常 / 设备异常';
COMMENT ON COLUMN alarm.rule IS '触发规则编码，见 app/analytics/rules.py';
COMMENT ON COLUMN alarm.value IS '触发时的指标值';
COMMENT ON COLUMN alarm.ended_at IS '数据恢复时间；NULL 表示异常仍在持续。与处理状态 status 相互独立';
COMMENT ON COLUMN alarm.peak IS '持续期间最不利值（DO 为最低值，其余为最高值）';
COMMENT ON COLUMN alarm.peak_at IS '最不利值出现的时间';
COMMENT ON COLUMN alarm.detail IS '规则参数与中间量，如趋势斜率、预计超标时间、常态基线';
COMMENT ON COLUMN alarm.advice IS '最近一次 AI 处置建议的完整记录（陈述、引用、证据表、模型与耗时）';
COMMENT ON COLUMN alarm.closed_at IS '处理关闭时间（演示时钟）';
COMMENT ON COLUMN alarm.close_note IS '关闭说明';

-- ---------------------------------------------------------------- 工单：草稿在 Redis，用户确认后才写入本表
ALTER TABLE work_order ADD COLUMN IF NOT EXISTS plant_id SMALLINT REFERENCES plant(id);
ALTER TABLE work_order ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ;
ALTER TABLE work_order ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;
ALTER TABLE work_order ADD COLUMN IF NOT EXISTS source JSONB NOT NULL DEFAULT '{}';
CREATE INDEX IF NOT EXISTS work_order_alarm ON work_order (alarm_id);
-- 每个告警同一时间只能有一张未关闭的工单；并发确认时由唯一约束兜底
CREATE UNIQUE INDEX IF NOT EXISTS work_order_one_open ON work_order (alarm_id) WHERE status <> '已关闭';
COMMENT ON COLUMN work_order.created_at IS '确认写入时间（演示时钟）';
COMMENT ON COLUMN work_order.started_at IS '开始处理时间（演示时钟）';
COMMENT ON COLUMN work_order.source IS '草稿来源：AI 建议的模型、生成时间、引用，以及用户是否修改';

-- ---------------------------------------------------------------- 报告
ALTER TABLE report ADD COLUMN IF NOT EXISTS facts JSONB NOT NULL DEFAULT '{}';
ALTER TABLE report ADD COLUMN IF NOT EXISTS meta JSONB NOT NULL DEFAULT '{}';
COMMENT ON COLUMN report.facts IS '报告使用的全部统计事实（SQL 汇总结果）';
COMMENT ON COLUMN report.meta IS '生成信息：模型、token、耗时、叙述来源与校验记录';

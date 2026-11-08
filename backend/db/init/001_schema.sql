-- 汉水智控 数据库表结构
-- 表与字段的 COMMENT 会被注入 Text2SQL 提示词，请保持「中文含义 + 单位」的写法。

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ---------------------------------------------------------------- 厂站
CREATE TABLE plant (
    id              SMALLINT PRIMARY KEY,
    code            TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('中心厂', '集镇厂')),
    process         TEXT NOT NULL CHECK (process IN ('A2O', '氧化沟')),
    capacity_m3d    INTEGER NOT NULL,
    lng             DOUBLE PRECISION NOT NULL,
    lat             DOUBLE PRECISION NOT NULL,
    commissioned_at DATE NOT NULL
);
COMMENT ON TABLE  plant IS '污水处理厂站基本信息。1 座中心厂 + 6 座集镇厂';
COMMENT ON COLUMN plant.code IS '厂站编码，如 P01';
COMMENT ON COLUMN plant.name IS '厂站名称，如 南郑中心污水厂、大河坎镇污水厂';
COMMENT ON COLUMN plant.kind IS '厂站类型：中心厂 / 集镇厂';
COMMENT ON COLUMN plant.process IS '处理工艺：A2O / 氧化沟';
COMMENT ON COLUMN plant.capacity_m3d IS '设计处理规模，立方米/日（吨/日）';
COMMENT ON COLUMN plant.lng IS '经度';
COMMENT ON COLUMN plant.lat IS '纬度';
COMMENT ON COLUMN plant.commissioned_at IS '投运日期';

-- ---------------------------------------------------------------- 厂站运行数据
CREATE TABLE plant_reading (
    ts              TIMESTAMPTZ NOT NULL,
    plant_id        SMALLINT NOT NULL REFERENCES plant(id),
    inflow_m3h      REAL,
    cod_in          REAL,
    nh3n_in         REAL,
    tn_in           REAL,
    tp_in           REAL,
    cod_out         REAL,
    nh3n_out        REAL,
    tn_out          REAL,
    tp_out          REAL,
    ss_out          REAL,
    do_aerobic_1    REAL,
    do_aerobic_2    REAL,
    air_flow_m3h    REAL,
    blower_kw       REAL,
    carbon_dose_kgh REAL,
    pac_dose_kgh    REAL,
    mlss            REAL,
    PRIMARY KEY (plant_id, ts)
);
SELECT create_hypertable('plant_reading', by_range('ts', INTERVAL '1 day'));
COMMENT ON TABLE  plant_reading IS '厂站运行实时数据（原始粒度 5 秒~1 分钟）。按小时统计请优先查 plant_reading_1h';
COMMENT ON COLUMN plant_reading.ts IS '采集时间（带时区）';
COMMENT ON COLUMN plant_reading.plant_id IS '厂站 ID，关联 plant.id';
COMMENT ON COLUMN plant_reading.inflow_m3h IS '进水瞬时流量，立方米/小时';
COMMENT ON COLUMN plant_reading.cod_in IS '进水化学需氧量 COD，mg/L';
COMMENT ON COLUMN plant_reading.nh3n_in IS '进水氨氮 NH3-N，mg/L';
COMMENT ON COLUMN plant_reading.tn_in IS '进水总氮 TN，mg/L';
COMMENT ON COLUMN plant_reading.tp_in IS '进水总磷 TP，mg/L';
COMMENT ON COLUMN plant_reading.cod_out IS '出水 COD，mg/L，一级A标准 ≤50';
COMMENT ON COLUMN plant_reading.nh3n_out IS '出水氨氮，mg/L，一级A标准 ≤5（水温≤12℃时 ≤8）';
COMMENT ON COLUMN plant_reading.tn_out IS '出水总氮，mg/L，一级A标准 ≤15';
COMMENT ON COLUMN plant_reading.tp_out IS '出水总磷，mg/L，一级A标准 ≤0.5';
COMMENT ON COLUMN plant_reading.ss_out IS '出水悬浮物 SS，mg/L，一级A标准 ≤10';
COMMENT ON COLUMN plant_reading.do_aerobic_1 IS '好氧区 1 号溶解氧 DO，mg/L，设定值约 2.0';
COMMENT ON COLUMN plant_reading.do_aerobic_2 IS '好氧区 2 号溶解氧 DO，mg/L';
COMMENT ON COLUMN plant_reading.air_flow_m3h IS '曝气总风量，立方米/小时';
COMMENT ON COLUMN plant_reading.blower_kw IS '鼓风机总功率，千瓦';
COMMENT ON COLUMN plant_reading.carbon_dose_kgh IS '碳源（乙酸钠）投加量，千克/小时';
COMMENT ON COLUMN plant_reading.pac_dose_kgh IS '除磷剂（PAC）投加量，千克/小时';
COMMENT ON COLUMN plant_reading.mlss IS '好氧池污泥浓度 MLSS，mg/L';

-- 小时聚合（连续聚合视图），Text2SQL 首选
CREATE MATERIALIZED VIEW plant_reading_1h
-- materialized_only = false：未物化的部分（最近时段、演示加速后领先墙上时间的部分）实时从原始表计算
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket(INTERVAL '1 hour', ts) AS hour,
    plant_id,
    avg(inflow_m3h)      AS inflow_m3h,
    avg(cod_in)          AS cod_in,
    avg(nh3n_in)         AS nh3n_in,
    avg(tn_in)           AS tn_in,
    avg(tp_in)           AS tp_in,
    avg(cod_out)         AS cod_out,
    avg(nh3n_out)        AS nh3n_out,
    max(nh3n_out)        AS nh3n_out_max,
    avg(tn_out)          AS tn_out,
    avg(tp_out)          AS tp_out,
    max(tp_out)          AS tp_out_max,
    avg(ss_out)          AS ss_out,
    avg(do_aerobic_1)    AS do_aerobic_1,
    avg(air_flow_m3h)    AS air_flow_m3h,
    avg(blower_kw)       AS blower_kw,
    avg(carbon_dose_kgh) AS carbon_dose_kgh,
    avg(pac_dose_kgh)    AS pac_dose_kgh
FROM plant_reading
GROUP BY hour, plant_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('plant_reading_1h',
    start_offset => INTERVAL '30 days',
    end_offset   => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');

COMMENT ON VIEW plant_reading_1h IS '厂站运行数据小时均值（连续聚合）。hour 为整点；avg(inflow_m3h) 即该小时处理水量（立方米）；blower_kw 均值即该小时鼓风机耗电量（千瓦时）';
COMMENT ON COLUMN plant_reading_1h.hour IS '整点时间';
COMMENT ON COLUMN plant_reading_1h.inflow_m3h IS '小时平均进水流量，数值上等于该小时处理水量（立方米）';
COMMENT ON COLUMN plant_reading_1h.nh3n_out_max IS '该小时出水氨氮最大值，mg/L';
COMMENT ON COLUMN plant_reading_1h.tp_out_max IS '该小时出水总磷最大值，mg/L';
COMMENT ON COLUMN plant_reading_1h.blower_kw IS '小时平均鼓风机功率，数值上等于该小时鼓风机耗电量（千瓦时）';

-- ---------------------------------------------------------------- 设备
CREATE TABLE device (
    id           SMALLINT PRIMARY KEY,
    plant_id     SMALLINT NOT NULL REFERENCES plant(id),
    code         TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL CHECK (kind IN ('鼓风机', '提升泵', '回流泵', '污泥泵', '加药泵')),
    model        TEXT NOT NULL,
    rated_kw     REAL NOT NULL,
    installed_at DATE NOT NULL,
    status       TEXT NOT NULL DEFAULT '运行' CHECK (status IN ('运行', '备用', '检修', '故障'))
);
COMMENT ON TABLE  device IS '设备台账';
COMMENT ON COLUMN device.code IS '设备编码，如 P01-BL-1（P01 厂 1 号鼓风机）';
COMMENT ON COLUMN device.kind IS '设备类型：鼓风机 / 提升泵 / 回流泵 / 污泥泵 / 加药泵';
COMMENT ON COLUMN device.model IS '设备型号';
COMMENT ON COLUMN device.rated_kw IS '额定功率，千瓦';
COMMENT ON COLUMN device.installed_at IS '安装日期';
COMMENT ON COLUMN device.status IS '当前状态：运行 / 备用 / 检修 / 故障';

CREATE TABLE device_reading (
    ts            TIMESTAMPTZ NOT NULL,
    device_id     SMALLINT NOT NULL REFERENCES device(id),
    running       BOOLEAN NOT NULL,
    current_a     REAL,
    vibration_mms REAL,
    temp_c        REAL,
    PRIMARY KEY (device_id, ts)
);
SELECT create_hypertable('device_reading', by_range('ts', INTERVAL '1 day'));
COMMENT ON TABLE  device_reading IS '设备运行监测数据';
COMMENT ON COLUMN device_reading.running IS '是否运行';
COMMENT ON COLUMN device_reading.current_a IS '运行电流，安培';
COMMENT ON COLUMN device_reading.vibration_mms IS '振动速度有效值，mm/s。>4.5 需关注，>7.1 报警';
COMMENT ON COLUMN device_reading.temp_c IS '轴承温度，摄氏度。>80 报警';

-- ---------------------------------------------------------------- 气象
CREATE TABLE weather (
    ts         TIMESTAMPTZ NOT NULL PRIMARY KEY,
    rain_mm_h  REAL NOT NULL,
    temp_c     REAL NOT NULL
);
SELECT create_hypertable('weather', by_range('ts', INTERVAL '7 days'));
COMMENT ON TABLE  weather IS '区域气象数据（南郑区）';
COMMENT ON COLUMN weather.rain_mm_h IS '降雨强度，毫米/小时';
COMMENT ON COLUMN weather.temp_c IS '气温，摄氏度';

-- ---------------------------------------------------------------- 告警 / 工单 / 报告
CREATE TABLE alarm (
    id            BIGSERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    plant_id      SMALLINT NOT NULL REFERENCES plant(id),
    device_id     SMALLINT REFERENCES device(id),
    level         TEXT NOT NULL CHECK (level IN ('提示', '预警', '告警')),
    kind          TEXT NOT NULL,
    metric        TEXT,
    value         REAL,
    threshold     REAL,
    message       TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT '未处理' CHECK (status IN ('未处理', '处理中', '已关闭')),
    ai_suggestion TEXT
);
CREATE INDEX ON alarm (ts DESC);
COMMENT ON TABLE  alarm IS '告警记录';
COMMENT ON COLUMN alarm.level IS '级别：提示 / 预警（趋势预测将超标） / 告警（已超标或设备故障）';
COMMENT ON COLUMN alarm.kind IS '类型：水质超标 / 水质预警 / 设备异常 / 药耗异常 / 进水冲击';
COMMENT ON COLUMN alarm.metric IS '相关指标字段名，如 nh3n_out';
COMMENT ON COLUMN alarm.status IS '处理状态：未处理 / 处理中 / 已关闭';
COMMENT ON COLUMN alarm.ai_suggestion IS 'AI 生成的处置建议';

CREATE TABLE work_order (
    id          BIGSERIAL PRIMARY KEY,
    alarm_id    BIGINT REFERENCES alarm(id),
    title       TEXT NOT NULL,
    assignee    TEXT,
    status      TEXT NOT NULL DEFAULT '待处理' CHECK (status IN ('待处理', '处理中', '已关闭')),
    steps       JSONB NOT NULL DEFAULT '[]',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at   TIMESTAMPTZ,
    resolution  TEXT
);
COMMENT ON TABLE  work_order IS '处置工单';
COMMENT ON COLUMN work_order.steps IS '处置步骤列表（JSON 数组）';
COMMENT ON COLUMN work_order.resolution IS '处理结果说明';

CREATE TABLE report (
    id           BIGSERIAL PRIMARY KEY,
    kind         TEXT NOT NULL CHECK (kind IN ('日报', '周报')),
    period_start TIMESTAMPTZ NOT NULL,
    period_end   TIMESTAMPTZ NOT NULL,
    content_md   TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE report IS '自动生成的运行报告';

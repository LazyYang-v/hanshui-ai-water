-- 视觉巡检（可重复执行）：每次推理一条记录，命中阈值的检测写入 alarm（rule = vision_debris / vision_leak）
CREATE TABLE IF NOT EXISTS vision_inspection (
    id           BIGSERIAL PRIMARY KEY,
    ts           TIMESTAMPTZ NOT NULL,
    plant_id     SMALLINT REFERENCES plant(id),
    camera       TEXT,
    source       TEXT NOT NULL,
    image_sha256 TEXT NOT NULL,
    width        INT NOT NULL,
    height       INT NOT NULL,
    model        TEXT NOT NULL,
    detections   JSONB NOT NULL,
    timings_ms   JSONB NOT NULL DEFAULT '{}',
    alarm_ids    BIGINT[] NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS vision_inspection_camera ON vision_inspection (plant_id, camera, ts DESC);
COMMENT ON TABLE vision_inspection IS '视觉巡检记录：一次推理一行（演示时钟）';
COMMENT ON COLUMN vision_inspection.source IS '图片来源：upload 上传 / sample:<文件名> 内置样例';
COMMENT ON COLUMN vision_inspection.model IS '模型文件 sha256 前 12 位';
COMMENT ON COLUMN vision_inspection.detections IS '检测结果：[{cls, score, box:[x1,y1,x2,y2]}]，原图像素坐标';
COMMENT ON COLUMN vision_inspection.alarm_ids IS '本次推理新建、更新或恢复的告警';
COMMENT ON COLUMN alarm.kind IS '类型：水质超标 / 水质预警 / 工艺异常 / 进水冲击 / 进水负荷 / 药耗异常 / 设备异常 / 视觉巡检';
CREATE INDEX IF NOT EXISTS alarm_vision_open ON alarm (plant_id, rule, (detail->>'camera'))
    WHERE ended_at IS NULL AND rule LIKE 'vision_%';

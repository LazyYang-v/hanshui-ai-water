-- 厂站与设备台账（虚构的南郑区水务集团，坐标为南郑区附近的近似位置，数据均为模拟）

INSERT INTO plant (id, code, name, kind, process, capacity_m3d, lng, lat, commissioned_at) VALUES
    (1, 'P01', '南郑中心污水处理厂', '中心厂', 'A2O',    50000, 107.035, 33.040, '2012-06-01'),
    (2, 'P02', '协税镇污水处理站',   '集镇厂', '氧化沟',  2000, 107.100, 32.985, '2018-09-15'),
    (3, 'P03', '梁山镇污水处理站',   '集镇厂', 'A2O',     5000, 106.935, 33.050, '2017-04-20'),
    (4, 'P04', '高台镇污水处理站',   '集镇厂', 'A2O',     3000, 106.985, 33.000, '2019-11-01'),
    (5, 'P05', '新集镇污水处理站',   '集镇厂', '氧化沟',  3000, 107.180, 32.960, '2019-03-10'),
    (6, 'P06', '黄官镇污水处理站',   '集镇厂', 'A2O',     4000, 107.215, 33.020, '2020-07-01'),
    (7, 'P07', '牟家坝镇污水处理站', '集镇厂', '氧化沟',  2000, 106.900, 32.860, '2021-05-18');

-- 设备编码规则：{厂站编码}-{类型缩写}-{序号}；device.id = plant.id * 20 + 序号偏移
-- 中心厂 3 台鼓风机（2 用 1 备），集镇厂 2 台（1 用 1 备）
WITH t(idx, abbr, kind, model, kw_per_m3d, kw_min, standby_from_center, standby_from_town) AS (
    VALUES
        (1, 'BL-1', '鼓风机', '磁悬浮离心鼓风机 MBL-150', 0.0022, 7.5, false, false),
        (2, 'BL-2', '鼓风机', '磁悬浮离心鼓风机 MBL-150', 0.0022, 7.5, false, true),
        (3, 'BL-3', '鼓风机', '磁悬浮离心鼓风机 MBL-150', 0.0022, 7.5, true,  true),
        (4, 'LP-1', '提升泵', '潜水排污泵 WQ-300',        0.0006, 4.0, false, false),
        (5, 'LP-2', '提升泵', '潜水排污泵 WQ-300',        0.0006, 4.0, false, false),
        (6, 'RP-1', '回流泵', '潜水推进回流泵 QJB-W',     0.0004, 3.0, false, false),
        (7, 'SP-1', '污泥泵', '螺杆污泥泵 G60-1',         0.0002, 2.2, false, false),
        (8, 'DP-C', '加药泵', '计量泵 GM-0500（碳源）',   0.0000, 0.75, false, false),
        (9, 'DP-P', '加药泵', '计量泵 GM-0500（PAC）',    0.0000, 0.75, false, false)
)
INSERT INTO device (id, plant_id, code, kind, model, rated_kw, installed_at, status)
SELECT
    p.id * 20 + t.idx,
    p.id,
    p.code || '-' || t.abbr,
    t.kind,
    t.model,
    greatest(t.kw_min, round((p.capacity_m3d * t.kw_per_m3d)::numeric, 1)),
    p.commissioned_at,
    CASE
        WHEN p.kind = '中心厂' AND t.standby_from_center THEN '备用'
        WHEN p.kind = '集镇厂' AND t.standby_from_town   THEN '备用'
        ELSE '运行'
    END
FROM plant p
CROSS JOIN t
-- 集镇厂只有 2 台鼓风机
WHERE NOT (p.kind = '集镇厂' AND t.abbr = 'BL-3');

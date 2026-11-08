-- Text2SQL 专用只读账号：只能 SELECT，语句超时 5 秒
-- 角色是集群级对象，脚本可重复执行（测试库也会用到）。密码由 004_readonly_password.sh 设置
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'text2sql_ro') THEN
        CREATE ROLE text2sql_ro LOGIN PASSWORD 'change_me';
    END IF;
END $$;
ALTER ROLE text2sql_ro SET statement_timeout = '5s';
ALTER ROLE text2sql_ro SET default_transaction_read_only = on;

DO $$ BEGIN EXECUTE format('GRANT CONNECT ON DATABASE %I TO text2sql_ro', current_database()); END $$;
GRANT USAGE ON SCHEMA public TO text2sql_ro;
-- 只授权 Text2SQL 可见的表（与 app/ai/text2sql/schema.py 的 VISIBLE_TABLES 一致），新建的表默认不可读
GRANT SELECT ON plant, plant_reading, plant_reading_1h, device, device_reading, weather TO text2sql_ro;

-- 模拟器检查点：与读数同事务写入，重启后恢复，曲线和进行中的场景不中断。
-- 只有一行；text2sql_ro 未授权，模型查询不到。
-- 已有数据卷不会重新执行 init 脚本，模拟器启动时也会执行本文件，故须可重复执行。
CREATE TABLE IF NOT EXISTS sim_checkpoint (
    id    smallint PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    ts    timestamptz NOT NULL,
    state jsonb NOT NULL
);

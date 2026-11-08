import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 本地开发读项目根目录的 .env（不覆盖已有环境变量；容器里由 compose 注入）
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "postgresql://shuiwu:shuiwu@localhost:15432/shuiwu")
    # Text2SQL 专用只读账号（库级只读 + 5 秒语句超时）
    readonly_database_url: str = os.getenv(
        "READONLY_DATABASE_URL",
        f"postgresql://text2sql_ro:{os.getenv('TEXT2SQL_RO_PASSWORD', 'readonly')}@localhost:15432/shuiwu")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:16379/0")
    cors_origins: tuple[str, ...] = tuple(os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","))
    sim_reply_timeout_s: float = 5.0
    rag_corpus_dir: Path = Path(os.getenv("RAG_CORPUS_DIR", str(Path(__file__).resolve().parents[1] / "knowledge")))
    rag_model_cache: Path = Path(os.getenv("RAG_MODEL_CACHE", str(Path(__file__).resolve().parents[1] / ".cache/rag-models")))
    rag_vector_cache: Path = Path(os.getenv("RAG_VECTOR_CACHE", str(Path(__file__).resolve().parents[1] / "data/rag-index")))
    rag_dense_enabled: bool = os.getenv("RAG_DENSE_ENABLED", "true").lower() == "true"
    # 告警巡检：API 进程内每隔 interval 秒扫描最近 3 小时；多实例部署时由 Redis 锁保证同一时刻只有一个在扫
    alarm_monitor_enabled: bool = os.getenv("ALARM_MONITOR_ENABLED", "true").lower() == "true"
    alarm_scan_interval_s: float = float(os.getenv("ALARM_SCAN_INTERVAL_S", "10"))
    # 视觉巡检：ONNX 模型旁边放同名 .json 模型卡；并发数 × 算子线程数不宜超过可用核数
    vision_enabled: bool = os.getenv("VISION_ENABLED", "true").lower() == "true"
    vision_model_path: Path = Path(os.getenv("VISION_MODEL_PATH", str(Path(__file__).resolve().parents[1] / "models/vision/inspect.onnx")))
    vision_samples_dir: Path = Path(os.getenv("VISION_SAMPLES_DIR", str(Path(__file__).resolve().parents[1] / "models/vision/samples")))
    vision_concurrency: int = int(os.getenv("VISION_CONCURRENCY", "2"))
    vision_threads: int = int(os.getenv("VISION_THREADS", "2"))
    # 唯一的管理员账号：读接口公开，写入与付费接口需要登录。未配置密码时无法登录（写接口一律 401）
    admin_username: str = os.getenv("ADMIN_USERNAME", "admin")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "")
    session_ttl_s: int = int(os.getenv("SESSION_TTL_S", str(8 * 3600)))
    # 经 HTTPS 部署时设为 true；本地 http 下浏览器不会回传 Secure Cookie
    cookie_secure: bool = os.getenv("COOKIE_SECURE", "false").lower() == "true"


settings = Settings()

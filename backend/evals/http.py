"""评测脚本调用 HTTP 接口前先登录管理员：写接口和模型调用接口需要会话 Cookie。

口令与被测服务相同，取自环境变量或本地 .env 的 ADMIN_PASSWORD（`app.config` 已加载 .env）。
"""

import httpx

from app.config import settings


def _body(password: str | None) -> dict:
    pw = password or settings.admin_password
    if not pw:
        raise SystemExit("评测需要 ADMIN_PASSWORD（与被测服务一致），请写在本地 .env 或环境变量里")
    return {"username": settings.admin_username, "password": pw}


def login(client: httpx.Client, password: str | None = None) -> None:
    client.post("/api/auth/login", json=_body(password)).raise_for_status()


async def alogin(client: httpx.AsyncClient, password: str | None = None) -> None:
    (await client.post("/api/auth/login", json=_body(password))).raise_for_status()

"""管理员会话：Redis 存会话，HttpOnly Cookie 携带会话号。

目标用户只有管理员一人，所以不建用户表、不分角色：
- 读接口（GET）和 WebSocket 公开，大屏和详情页无需登录；
- 其余方法（写库、调用模型、注入场景、推理）一律要求已登录，新加的接口默认受保护；
- 会话号只在 Cookie 里，Redis 键是它的 SHA-256，Redis 被读到也拿不到可用的会话号。

选 Cookie 而不是 Authorization 头的原因：浏览器建 WebSocket 时不能自定义请求头，
同源部署下 Cookie 会被 fetch、SSE、WebSocket 和下载链接自动带上。
CSRF：Cookie 设 SameSite=Lax，跨站的 POST/PUT/DELETE 不会带上它。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets

from redis.asyncio import Redis

COOKIE = "hs_session"
KEY_SESSION = "auth:session:{}"
KEY_FAIL = "auth:fail:{}"
MAX_FAILS = 5                       # 同一来源 15 分钟内最多失败 5 次
FAIL_WINDOW_S = 15 * 60
PUBLIC_METHODS = {"GET", "HEAD", "OPTIONS"}
# 未登录也能调用的非 GET 接口。曝气仿真只是确定性计算（不写库、不调模型、结果有缓存），
# 页面打开即自动运行，用 POST 只是因为参数是结构体
OPEN_PATHS = {"/api/auth/login", "/api/auth/logout", "/api/aeration/simulate"}


def _key(token: str) -> str:
    return KEY_SESSION.format(hashlib.sha256(token.encode()).hexdigest())


def check_password(given_user: str, given_pw: str, user: str, pw: str) -> bool:
    if not pw:                       # 未配置密码：任何口令都不通过
        return False
    # 两项都比较完再合并，耗时不随哪一项先错而变化
    ok_user = hmac.compare_digest(given_user.encode(), user.encode())
    ok_pw = hmac.compare_digest(given_pw.encode(), pw.encode())
    return ok_user and ok_pw


async def create_session(redis: Redis, username: str, ttl_s: int) -> str:
    token = secrets.token_urlsafe(32)
    await redis.set(_key(token), username, ex=ttl_s)
    return token


async def session_user(redis: Redis, token: str | None) -> str | None:
    if not token or len(token) > 128:
        return None
    return await redis.get(_key(token))


async def drop_session(redis: Redis, token: str | None) -> None:
    if token:
        await redis.delete(_key(token))


async def too_many_failures(redis: Redis, source: str) -> bool:
    n = await redis.get(KEY_FAIL.format(source))
    return n is not None and int(n) >= MAX_FAILS


async def record_failure(redis: Redis, source: str) -> None:
    key = KEY_FAIL.format(source)
    if await redis.incr(key) == 1:          # 窗口从第一次失败算起，不随后续失败延长
        await redis.expire(key, FAIL_WINDOW_S)


def _cookie(headers: list[tuple[bytes, bytes]]) -> str | None:
    for name, value in headers:
        if name == b"cookie":
            for part in value.decode("latin-1").split(";"):
                k, _, v = part.strip().partition("=")
                if k == COOKIE:
                    return v
    return None


class AdminOnlyWrites:
    """纯 ASGI 中间件（不缓冲响应体，SSE 流式输出不受影响）：非只读方法要求有效会话。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope["method"] in PUBLIC_METHODS
                or not scope["path"].startswith("/api/") or scope["path"] in OPEN_PATHS):
            return await self.app(scope, receive, send)
        redis: Redis = scope["app"].state.redis
        user = await session_user(redis, _cookie(scope["headers"]))
        if user is None:
            body = json.dumps({"detail": "需要管理员登录"}, ensure_ascii=False).encode()
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode())]})
            return await send({"type": "http.response.body", "body": body})
        scope.setdefault("state", {})["user"] = user
        return await self.app(scope, receive, send)

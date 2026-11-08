"""管理员登录、注销与登录状态。"""

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app import auth
from app.deps import Rds

router = APIRouter(prefix="/api/auth", tags=["登录"])


class LoginBody(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class Me(BaseModel):
    authenticated: bool
    username: str | None = None


def _source(request: Request) -> str:
    """限流按来源 IP：经 nginx 时取它追加在 X-Forwarded-For 末尾的地址（客户端无法伪造这一项）。"""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.rsplit(",", 1)[-1].strip()
    return request.client.host if request.client else "unknown"


@router.post("/login", response_model=Me, summary="管理员登录（写入 HttpOnly 会话 Cookie）")
async def login(body: LoginBody, request: Request, response: Response, redis: Rds) -> Me:
    s = request.app.state.settings
    source = _source(request)
    if await auth.too_many_failures(redis, source):
        raise HTTPException(429, "登录失败次数过多，请 15 分钟后再试")
    if not auth.check_password(body.username, body.password, s.admin_username, s.admin_password):
        await auth.record_failure(redis, source)
        raise HTTPException(401, "用户名或密码错误" if s.admin_password else "服务端未配置 ADMIN_PASSWORD，无法登录")
    token = await auth.create_session(redis, s.admin_username, s.session_ttl_s)
    response.set_cookie(auth.COOKIE, token, max_age=s.session_ttl_s, httponly=True, samesite="lax",
                        secure=s.cookie_secure, path="/")
    return Me(authenticated=True, username=s.admin_username)


@router.post("/logout", status_code=204, summary="注销（服务端立即删除会话）")
async def logout(request: Request, redis: Rds) -> Response:
    await auth.drop_session(redis, request.cookies.get(auth.COOKIE))
    response = Response(status_code=204)
    response.delete_cookie(auth.COOKIE, path="/", httponly=True, samesite="lax",
                           secure=request.app.state.settings.cookie_secure)
    return response


@router.get("/me", response_model=Me, summary="当前登录状态")
async def me(request: Request, redis: Rds) -> Me:
    user = await auth.session_user(redis, request.cookies.get(auth.COOKIE))
    return Me(authenticated=user is not None, username=user)

"""管理员会话：读公开、写需登录、注销立即失效、登录失败限流。"""

from app import auth
from tests.conftest import ADMIN_PASSWORD


def test_reads_are_public_and_writes_need_login(client):
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").json() == {"authenticated": False, "username": None}
    assert client.get("/api/plants").status_code == 200
    assert client.get("/api/reports").status_code == 200
    # 曝气仿真页打开即运行，属于公开计算接口（参数错误返回 422 而不是 401）
    assert client.post("/api/aeration/simulate", json={"strategies": []}).status_code != 401
    for method, path in [("POST", "/api/reports"), ("POST", "/api/ai/chat"), ("POST", "/api/work-orders"),
                         ("PUT", "/api/demo/speed"), ("DELETE", "/api/demo/scenarios"),
                         ("POST", "/api/alarms/1/advice"), ("POST", "/api/vision/inspect")]:
        r = client.request(method, path, json={})
        assert r.status_code == 401, (method, path)
        assert r.json()["detail"] == "需要管理员登录"


def test_login_session_cookie_and_logout(client, sync_redis):
    client.post("/api/auth/logout")
    r = client.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD})
    assert r.status_code == 200 and r.json() == {"authenticated": True, "username": "admin"}
    cookie = r.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie and "Max-Age=28800" in cookie
    token = client.cookies.get(auth.COOKIE)
    # Redis 只存会话号的哈希
    assert sync_redis.get(f"auth:session:{token}") is None
    assert sync_redis.get(auth._key(token)) == "admin"
    assert client.get("/api/auth/me").json()["authenticated"] is True
    # 登录后写接口进入正常校验（这里是参数错误，而不是 401）
    assert client.post("/api/reports", json={}).status_code == 422

    assert client.post("/api/auth/logout").status_code == 204
    assert sync_redis.get(auth._key(token)) is None
    client.cookies.set(auth.COOKIE, token)            # 旧会话号即使还留在浏览器也已失效
    assert client.post("/api/reports", json={}).status_code == 401


def test_wrong_password_and_rate_limit(client):
    client.post("/api/auth/logout")
    for _ in range(auth.MAX_FAILS):
        r = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
        assert r.status_code == 401 and r.json()["detail"] == "用户名或密码错误"
    assert client.post("/api/auth/login", json={"username": "root", "password": ADMIN_PASSWORD}).status_code == 429
    # 正确密码在锁定期内同样拒绝，不泄露口令是否正确
    assert client.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD}).status_code == 429
    # 换一个来源（经 nginx 时取 X-Forwarded-For 末项）不受影响
    r = client.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD},
                    headers={"X-Forwarded-For": "1.2.3.4, 10.0.0.9"})
    assert r.status_code == 200


def test_no_password_configured_blocks_login():
    assert auth.check_password("admin", "", "admin", "") is False
    assert auth.check_password("admin", "anything", "admin", "") is False
    assert auth.check_password("admin", "pw", "admin", "pw") is True
    assert auth.check_password("Admin", "pw", "admin", "pw") is False


def test_websocket_stays_public(client, sync_redis):
    client.post("/api/auth/logout")
    sync_redis.set("realtime:latest", '{"ts": "latest"}')
    with client.websocket_connect("/ws/realtime") as ws:
        assert ws.receive_json() == {"ts": "latest"}

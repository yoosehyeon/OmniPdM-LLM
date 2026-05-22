"""services.auth.sessions + auth blueprint + before_request guard 통합 unit tests.

Tier 1.5 P1-b 인증 흐름:
- /login GET → 200 + CSRF token in form
- /login POST without CSRF → 400
- /login POST wrong creds → 401
- /login POST right creds → 302 to / (or next_url)
- Authenticated request → no redirect
- /logout → 302 to /login + session cleared
- Open redirect 방어 — next=http://evil.com → / 로 redirect

CI smoke 외부 의존 0 — UserService 를 mock 으로 주입, SQLite/DB 없음.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 인증 ON + DB OFF 환경에서 import 직전에 강제.
os.environ["OMNIPDM_AUTH_REQUIRED"] = "true"
os.environ["OMNIPDM_DB_ENABLED"] = "false"
# SECRET_KEY 고정 — test 안정성. 미설정 시 random fallback 도 동작은 하지만 결정적 X.
os.environ["OMNIPDM_SECRET_KEY"] = "test-secret-key-for-pytest-only-do-not-use"

from services.auth.users import User  # noqa: E402


# ---------------------------------------------------------------------------
# Mock UserService — DB 없이도 인증 라우트 동작 검증.
# ---------------------------------------------------------------------------
class _MockUserService:
    def __init__(self, valid_users: dict):
        # valid_users = {"alice": "wonderland"}
        self._users = valid_users

    def get_user(self, user_id: str) -> Optional[User]:
        if user_id not in self._users:
            return None
        return _make_user(user_id)

    def verify_credentials(self, user_id: str, plaintext: str) -> Optional[User]:
        if self._users.get(user_id) == plaintext:
            return _make_user(user_id)
        return None

    def close(self) -> None: ...


def _make_user(user_id: str) -> User:
    now = datetime.now(timezone.utc)
    return User(
        user_id=user_id,
        email=f"{user_id}@example.com",
        display_name=user_id.title(),
        role="admin" if user_id == "admin" else "operator",
        active=True,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def client():
    """Mock UserService 를 주입한 Flask test client. 매 테스트마다 새 app 인스턴스."""
    import importlib
    import app as app_module
    importlib.reload(app_module)  # before_request 가드 재등록

    # blueprint 의 module-level _user_service 와 role_required 의 _rbac_user_service
    # 둘 다 mock 으로 교체. init_auth_blueprint 가 NullUserService 를 RBAC 에 주입한 상태를 덮어씀.
    from services.auth import routes
    from services.auth import sessions as auth_sessions
    mock_users = _MockUserService({"alice": "wonderland", "admin": "adminpass"})
    routes._user_service = mock_users
    routes._audit_conn_provider = lambda: None  # audit 비활성
    auth_sessions.set_user_service_for_rbac(mock_users)

    flask_app = app_module.server
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def _extract_csrf(html_text: str) -> str:
    m = re.search(r'name="csrf_token" value="([^"]+)"', html_text)
    assert m, "CSRF token not found in login form"
    return m.group(1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestLoginPage:
    def test_get_login_returns_form_with_csrf(self, client):
        r = client.get("/login")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert 'name="csrf_token"' in body
        assert 'name="user_id"' in body
        assert 'name="password"' in body

    def test_csrf_token_persists_within_session(self, client):
        r1 = client.get("/login")
        token1 = _extract_csrf(r1.get_data(as_text=True))
        r2 = client.get("/login")
        token2 = _extract_csrf(r2.get_data(as_text=True))
        assert token1 == token2  # session 동안은 같은 token


class TestLoginPost:
    def test_missing_csrf_rejected(self, client):
        r = client.post(
            "/login",
            data={"user_id": "alice", "password": "wonderland"},
        )
        assert r.status_code == 400

    def test_bad_csrf_rejected(self, client):
        client.get("/login")  # session 시작
        r = client.post(
            "/login",
            data={
                "user_id": "alice",
                "password": "wonderland",
                "csrf_token": "WRONG_TOKEN",
            },
        )
        assert r.status_code == 400

    def test_wrong_password_returns_401(self, client):
        r1 = client.get("/login")
        csrf = _extract_csrf(r1.get_data(as_text=True))
        r = client.post(
            "/login",
            data={"user_id": "alice", "password": "WRONG", "csrf_token": csrf},
        )
        assert r.status_code == 401

    def test_correct_credentials_redirect_to_root(self, client):
        r1 = client.get("/login")
        csrf = _extract_csrf(r1.get_data(as_text=True))
        r = client.post(
            "/login",
            data={"user_id": "alice", "password": "wonderland", "csrf_token": csrf},
        )
        assert r.status_code == 302
        assert r.headers["Location"] == "/"

    def test_next_url_safe_path_preserved(self, client):
        r1 = client.get("/login?next=/status")
        csrf = _extract_csrf(r1.get_data(as_text=True))
        r = client.post(
            "/login",
            data={
                "user_id": "alice",
                "password": "wonderland",
                "csrf_token": csrf,
                "next": "/status",
            },
        )
        assert r.status_code == 302
        assert r.headers["Location"] == "/status"

    def test_open_redirect_blocked(self, client):
        # next= 가 외부 URL 이면 무시되어야 함.
        r1 = client.get("/login")
        csrf = _extract_csrf(r1.get_data(as_text=True))
        r = client.post(
            "/login",
            data={
                "user_id": "alice",
                "password": "wonderland",
                "csrf_token": csrf,
                "next": "http://evil.com/x",
            },
        )
        assert r.status_code == 302
        # Open redirect 차단 → / 로
        assert r.headers["Location"] == "/"


class TestAuthGuard:
    def test_unauth_request_redirects_to_login(self, client):
        r = client.get("/")
        assert r.status_code == 302
        assert r.headers["Location"] == "/login?next=/"

    def test_unauth_with_query_preserved(self, client):
        r = client.get("/status?foo=bar")
        assert r.status_code == 302
        assert r.headers["Location"].startswith("/login?next=/status")
        assert "foo=bar" in r.headers["Location"]

    def test_authenticated_request_passes(self, client):
        # 로그인 → 동일 client 가 session cookie 유지.
        r1 = client.get("/login")
        csrf = _extract_csrf(r1.get_data(as_text=True))
        client.post(
            "/login",
            data={"user_id": "alice", "password": "wonderland", "csrf_token": csrf},
        )
        r = client.get("/")
        # PRD v7.0: / 는 React SPA(frontend/dist/index.html) 정적 서빙.
        # 테스트 환경에 빌드 산출물이 없으면 503 (안내 페이지), 있으면 200.
        # 핵심은 redirect(302) 가 아니라는 것 — 인증 가드 통과 확인.
        assert r.status_code in (200, 503)

    def test_login_path_exempt(self, client):
        # /login 은 인증 없이도 200 (가드에서 제외).
        r = client.get("/login")
        assert r.status_code == 200

    def test_api_health_exempt(self, client):
        # /api/health 는 가드에서 제외 — Docker healthcheck 용.
        r = client.get("/api/health")
        assert r.status_code == 200


class TestLogout:
    def test_logout_clears_session_and_redirects(self, client):
        r1 = client.get("/login")
        csrf = _extract_csrf(r1.get_data(as_text=True))
        client.post(
            "/login",
            data={"user_id": "alice", "password": "wonderland", "csrf_token": csrf},
        )
        # 로그인 상태 확인 (인증 통과 — 빌드 산출물 유무에 따라 200/503).
        r_in = client.get("/")
        assert r_in.status_code in (200, 503)

        # logout
        r_logout = client.get("/logout")
        assert r_logout.status_code == 302
        assert r_logout.headers["Location"] == "/login"

        # logout 후 다시 protected 접근 → redirect.
        r_out = client.get("/")
        assert r_out.status_code == 302


# ---------------------------------------------------------------------------
# RBAC — role_required (P1-c)
# ---------------------------------------------------------------------------
def _login_as(client, user_id: str, password: str) -> None:
    """헬퍼: 주어진 사용자로 로그인. CSRF token 처리 포함."""
    r1 = client.get("/login")
    csrf = _extract_csrf(r1.get_data(as_text=True))
    r = client.post(
        "/login",
        data={"user_id": user_id, "password": password, "csrf_token": csrf},
    )
    assert r.status_code == 302, f"login failed for {user_id}: {r.status_code}"


class TestRoleRequired:
    def test_admin_can_access_admin_route(self, client):
        _login_as(client, "admin", "adminpass")
        r = client.get("/admin/health-check")
        assert r.status_code == 200
        assert r.json["role_check"] == "admin"

    def test_operator_cannot_access_admin_route(self, client):
        # alice 는 operator role — admin 전용 라우트에서 403.
        _login_as(client, "alice", "wonderland")
        r = client.get("/admin/health-check")
        assert r.status_code == 403
        # 403 페이지가 Bootstrap HTML 로 렌더링되는지 간단 확인.
        body = r.get_data(as_text=True)
        assert "403 Forbidden" in body
        assert "permission" in body.lower()

    def test_unauthenticated_admin_route_redirects(self, client):
        # 로그인 안 한 상태로 admin 라우트 접근 → before_request 가 /login 으로.
        r = client.get("/admin/health-check")
        assert r.status_code == 302
        assert "/login" in r.headers["Location"]


class TestRoleRequiredDecorator:
    """role_required 데코레이터 자체의 단위 동작."""

    def test_decorator_requires_at_least_one_role(self):
        from services.auth.sessions import role_required
        with pytest.raises(ValueError):
            role_required()

    def test_session_cleared_when_user_disappears(self, client):
        # 로그인 후 mock UserService 에서 사용자 삭제 → 다음 요청에서 401 → /login redirect.
        _login_as(client, "alice", "wonderland")

        # mock UserService 에서 alice 제거
        from services.auth import routes
        routes._user_service._users.pop("alice", None)

        # admin 라우트 접근 → role_required 가 get_user(alice) None 받으면 401 → /login.
        r = client.get("/admin/health-check")
        assert r.status_code == 302
        assert "/login" in r.headers["Location"]

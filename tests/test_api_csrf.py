"""GET /api/csrf smoke test.

- 인증 미요구 환경에서 토큰 발급 200 + JSON 응답
- Cache-Control: no-store 헤더로 토큰 캐싱 방지 (OWASP)
- 같은 client(=같은 세션) 두 번 호출 시 동일 토큰 (per-session, 재사용 패턴)
"""
from __future__ import annotations

import pytest


@pytest.fixture
def flask_client(monkeypatch):
    # Application factory 패턴 — importlib.reload 회피 (Flask 공식 Testing 권장).
    # auth_required 는 create_app() 내부에서 평가되므로 호출 직전 setenv 만으로 충분.
    monkeypatch.setenv("OMNIPDM_AUTH_REQUIRED", "false")
    from app import create_app
    return create_app().test_client()


def test_csrf_returns_token(flask_client):
    resp = flask_client.get("/api/csrf")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "token" in data
    assert isinstance(data["token"], str)
    # secrets.token_urlsafe(32) → ~43 chars. 최소 32 보장 (OWASP 권장 엔트로피).
    assert len(data["token"]) >= 32


def test_csrf_no_store_header(flask_client):
    resp = flask_client.get("/api/csrf")
    cache = resp.headers.get("Cache-Control", "")
    assert "no-store" in cache, f"Cache-Control missing no-store: {cache!r}"


def test_csrf_same_session_same_token(flask_client):
    r1 = flask_client.get("/api/csrf")
    r2 = flask_client.get("/api/csrf")
    assert r1.get_json()["token"] == r2.get_json()["token"]

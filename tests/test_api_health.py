"""GET /api/health smoke test.

인증 미요구 — _AUTH_EXEMPT_PREFIXES 에 /api/health 가 포함되어야 통과.
외부 의존(DB/MQTT) 없이 동작해야 함. 회귀 방지가 목적.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def flask_client(monkeypatch):
    # 인증 가드 켜진 상태에서도 /api/health 는 통과해야 함을 검증.
    monkeypatch.setenv("OMNIPDM_AUTH_REQUIRED", "true")
    monkeypatch.setenv("OMNIPDM_DB_ENABLED", "false")
    # app.py 는 모듈 import 시 create_app() 을 호출 → 환경변수 선반영 위해 재import.
    import importlib
    import app as app_module
    importlib.reload(app_module)
    return app_module.server.test_client()


def test_health_ok(flask_client):
    resp = flask_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["service"] == "omnipdm"
    assert "uptime_seconds" in data


def test_health_does_not_require_auth(flask_client):
    # 세션 쿠키 없이 호출했을 때 302 redirect 가 발생하면 안 됨.
    resp = flask_client.get("/api/health", follow_redirects=False)
    assert resp.status_code == 200, (
        f"/api/health redirected to {resp.headers.get('Location')} — "
        "확인: app.py _AUTH_EXEMPT_PREFIXES 에 /api/health 포함 여부"
    )

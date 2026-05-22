"""E2E 통합 테스트 — Flask + /api/* + auth + CSRF + SPA 정적 서빙 전체 경로 검증.

기존 단위 테스트들이 각각의 blueprint 만 검증하는 것과 달리, 본 테스트는
React SPA 가 실제로 백엔드와 주고받는 흐름을 한 번에 통과시켜 회귀를 차단한다.

검증 흐름:
  1. /api/health        (인증 면제)
  2. /api/datasets      (인증 면제, 5종 카탈로그)
  3. /login (GET → POST) 로 세션 쿠키 획득
  4. /api/csrf          (로그인 후 토큰 발급)
  5. /api/analyze POST (X-CSRF-Token 동봉) → 200 또는 500 (모델 체크포인트 의존)
  6. /api/analyze POST (CSRF 누락) → 403
  7. /                  (SPA index.html 정적 서빙 — dist 유무에 따라 200/503)

외부 의존 0 — DB/MQTT/모델 체크포인트 부재 환경에서도 흐름은 통과해야 함.
fixture 는 app.create_app() factory 호출 — importlib.reload 회피 (Flask 권장).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from services.auth.users import User


class _MockUserService:
    """tests/test_auth_sessions.py 의 패턴 재사용 — alice/wonderland 계정만 검증."""

    def __init__(self, valid_users: dict):
        self._users = valid_users

    def verify_credentials(self, user_id: str, password: str):
        if self._users.get(user_id) == password:
            return _make_user(user_id)
        return None

    def get_user(self, user_id: str):
        if user_id in self._users:
            return _make_user(user_id)
        return None


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
    from app import create_app
    from services.auth import routes
    from services.auth import sessions as auth_sessions

    # routes module-level singleton 을 mock 으로 덮어씀 — create_app() 후에 적용해야
    # init_auth_blueprint 가 주입한 NullUserService 가 덮인다.
    mock_users = _MockUserService({"alice": "wonderland"})

    flask_app = create_app()
    flask_app.config["TESTING"] = True

    routes._user_service = mock_users
    routes._audit_conn_provider = lambda: None
    auth_sessions.set_user_service_for_rbac(mock_users)

    return flask_app.test_client()


def _extract_login_csrf(html: str) -> str:
    m = re.search(r'name="csrf_token"\s+value="([^"]+)"', html)
    assert m, "login form CSRF token not found"
    return m.group(1)


def _login(client) -> None:
    r = client.get("/login")
    csrf = _extract_login_csrf(r.get_data(as_text=True))
    r = client.post(
        "/login",
        data={"user_id": "alice", "password": "wonderland", "csrf_token": csrf},
    )
    assert r.status_code in (302, 303), f"login failed: {r.status_code} {r.get_data(as_text=True)[:200]}"


def test_e2e_full_pipeline(client):
    # 1) liveness — 인증 불요
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"

    # 2) datasets — 인증 불요, 카탈로그 5종
    r = client.get("/api/datasets")
    assert r.status_code == 200
    catalog = r.get_json()
    assert isinstance(catalog, list) and len(catalog) == 5
    ai4i = next(d for d in catalog if d["id"] == "ai4i")
    ai4i_sensor_names = {s["name"] for s in ai4i["sensors"]}

    # 3) 로그인 → 세션 쿠키 획득
    _login(client)

    # 4) CSRF 토큰 발급
    r = client.get("/api/csrf")
    assert r.status_code == 200
    token = r.get_json()["token"]
    assert isinstance(token, str) and len(token) >= 32

    # 5) /api/analyze 정상 path — CSRF + validation 통과해 service layer 도달
    payload = {
        "dataset": "ai4i",
        "model": "bilstm",
        "riskFusionMethod": "weighted",
        "llmEnabled": False,
        "sensorValues": {name: 50.0 for name in ai4i_sensor_names},
    }
    r = client.post(
        "/api/analyze",
        json=payload,
        headers={"X-CSRF-Token": token},
    )
    # 체크포인트 부재 환경에서 500 가능 — 핵심은 400/403/501 이 아닌 것.
    assert r.status_code in (200, 500), (
        f"unexpected status {r.status_code}: {r.get_data(as_text=True)[:200]}"
    )

    # 6) CSRF 누락 → 403
    r = client.post("/api/analyze", json=payload)
    assert r.status_code == 403
    assert r.get_json()["error"] == "csrf_invalid"


def test_e2e_spa_serving(client):
    # 로그인 후 / 가 SPA index.html 을 서빙 (또는 dist 부재 시 503).
    _login(client)
    r = client.get("/")
    assert r.status_code in (200, 503)

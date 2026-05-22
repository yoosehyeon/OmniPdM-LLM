"""GET /api/csrf — CSRF 토큰 발급.

React SPA 가 상태 변경 요청(POST/PUT/DELETE) 전에 호출.
응답의 `token` 을 `X-CSRF-Token` 헤더로 동봉하면 백엔드 미들웨어가 검증한다.

- 세션 쿠키가 있어야 정상 동작 (auth_required=true 인 경우 가드를 통과해야 도달).
- 토큰은 services.auth.sessions.get_or_create_csrf_token() 가 재사용 (per-session, Flask-WTF/Flask-Security 패턴).
- 인증 미요구 환경(테스트)에서도 빈 세션에 새 토큰을 발급해 200 반환.
- Cache-Control: no-store — 토큰이 프록시/브라우저 캐시에 보관되어 다음 사용자에게 노출되는 사고 방지 (OWASP CSRF Cheat Sheet).
"""
from __future__ import annotations

from flask import Blueprint, Response, jsonify

from services.auth.sessions import get_or_create_csrf_token

bp_csrf = Blueprint("api_csrf", __name__, url_prefix="/api")


@bp_csrf.route("/csrf", methods=["GET"])
def get_csrf_token() -> Response:
    token = get_or_create_csrf_token()
    resp = jsonify({"token": token})
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp

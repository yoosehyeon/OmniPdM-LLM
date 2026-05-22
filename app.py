"""
OmniPdM — Flask + React SPA entry point (PRD v7.0).

이전 (v6.x) 의 Dash 다페이지 UI 는 제거되었고, 모든 화면은 frontend/dist/ 의 React SPA 가
담당한다. Flask 는 다음 세 역할만 한다:
  1. /api/* REST 엔드포인트 (services/api/* blueprint)
  2. /login, /logout 등 auth 라우트 (services/auth/routes.py)
  3. SPA 정적 서빙 (frontend/dist/ → / 및 client-side routing 의 catch-all)

사용:
    python app.py

환경변수:
    OMNIPDM_HOST           (default: 0.0.0.0)
    OMNIPDM_PORT           (default: 8050)
    OMNIPDM_DEBUG          (default: false)
    OMNIPDM_SECRET_KEY     (production 필수 — 미설정 시 random fallback + 경고, config.py 참조)
    OMNIPDM_AUTH_REQUIRED  (default: true — false 로 두면 인증 가드 비활성, 개발/테스트용)
"""
from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, redirect, request, send_from_directory

from models_core import config
from services.api import bp_analyze, bp_csrf, bp_datasets, bp_health
from services.auth import create_default_user_service
from services.auth import sessions as auth_sessions
from services.auth.routes import init_auth_blueprint


# 인증 가드 면제 경로.
# /api/health 는 liveness probe 라 무인증.
# /api/csrf 도 토큰 발급 자체에는 인증 불요 (세션은 있어야 함 — 후속 endpoint 에서 처리).
# /assets 는 React build 산출물의 정적 자원.
_AUTH_EXEMPT_PREFIXES = (
    "/login",
    "/logout",
    "/favicon.ico",
    "/assets",
    "/api/health",
    "/api/datasets",
)

# React build 산출물 경로. Dockerfile Stage 1 (vite build) 의 결과를 Stage 2 가 복사.
_FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"


def _is_exempt(path: str) -> bool:
    return any(path.startswith(p) for p in _AUTH_EXEMPT_PREFIXES)


def _make_audit_conn_provider():
    """audit_log 용 psycopg connection 을 캐시해 매 인증 이벤트마다 새 connect 비용 회피.

    PoC 단일 worker process 가정 — Flask multi-worker 환경에서는 각 worker 가 자기
    closure 를 가지므로 자연스럽게 분리. 다중 thread 동시 호출은 closed check 가 1회
    있을 수 있으나 audit 실패가 인증 흐름을 막지 않으므로 영향 미미.
    """
    state: dict = {"conn": None}

    def _provider():
        if not config.DB_ENABLED:
            return None
        conn = state["conn"]
        if conn is not None and not getattr(conn, "closed", True):
            return conn
        try:
            import psycopg
            new_conn = psycopg.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                dbname=config.DB_NAME,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                autocommit=True,
                application_name="omnipdm-auth",
            )
            state["conn"] = new_conn
            return new_conn
        except Exception:
            return None

    return _provider


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.secret_key = config.SECRET_KEY

    # OWASP Session Management — cookie 보안 기본값.
    # SECURE 는 HTTPS 전용이라 PoC dev (http://localhost) 에서는 OFF.
    # 운영 진입 시 OMNIPDM_SESSION_COOKIE_SECURE=true 로 강제.
    cookie_secure = os.getenv("OMNIPDM_SESSION_COOKIE_SECURE", "false").lower() in ("1", "true", "yes", "on")
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=cookie_secure,
    )

    # ----------------------------------------------------------------------
    # Auth blueprint + before_request guard (Tier 1.5 P1-b)
    # ----------------------------------------------------------------------
    auth_required = os.getenv("OMNIPDM_AUTH_REQUIRED", "true").lower() in ("1", "true", "yes", "on")
    user_service = create_default_user_service()
    audit_conn_provider = _make_audit_conn_provider()

    bp = init_auth_blueprint(user_service, audit_conn_provider)
    app.register_blueprint(bp)

    # REST API blueprints (React frontend 의 entry).
    app.register_blueprint(bp_health)
    app.register_blueprint(bp_csrf)
    app.register_blueprint(bp_datasets)
    app.register_blueprint(bp_analyze)

    if auth_required:
        @app.before_request
        def _require_login():
            if _is_exempt(request.path):
                return None
            if auth_sessions.is_authenticated():
                return None
            # 원래 가려던 경로를 ?next= 로 보존 — 로그인 후 자동 복귀.
            next_url = request.full_path if request.query_string else request.path
            return redirect(f"/login?next={next_url}")

    # ----------------------------------------------------------------------
    # React SPA 정적 서빙
    # ----------------------------------------------------------------------
    # build 산출물이 없으면 (dev: Vite 가 :5173 에서 직접 서빙) 안내 메시지만 노출.
    # /assets/* 는 Vite 가 hash 화한 JS/CSS — auth 면제 (이미 _AUTH_EXEMPT_PREFIXES).

    @app.route("/assets/<path:filename>")
    def assets(filename: str):
        return send_from_directory(_FRONTEND_DIST / "assets", filename)

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def spa(path: str):
        # API/auth 경로는 위 blueprint 가 먼저 매칭되므로 여기는 SPA 라우트만 도달.
        index = _FRONTEND_DIST / "index.html"
        if not index.exists():
            return (
                "<h1>OmniPdM</h1>"
                "<p>React build 산출물이 없습니다. "
                "<code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code> "
                "또는 dev 모드에서는 <code>npm run dev</code> (:5173) 를 사용하세요.</p>",
                503,
            )
        # 정적 자원이면 그 파일을, 아니면 index.html (client-side router).
        candidate = _FRONTEND_DIST / path
        if path and candidate.is_file():
            return send_from_directory(_FRONTEND_DIST, path)
        return send_from_directory(_FRONTEND_DIST, "index.html")

    return app


app = create_app()
server = app  # WSGI entry for production (gunicorn 등). 하위호환 alias.


if __name__ == "__main__":
    host = os.getenv("OMNIPDM_HOST", "0.0.0.0")
    port = int(os.getenv("OMNIPDM_PORT", "8050"))
    debug = os.getenv("OMNIPDM_DEBUG", "false").lower() == "true"

    app.run(host=host, port=port, debug=debug)

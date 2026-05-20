"""
OmniPdM - Dash multi-page application entry point.

사용:
    python app.py

환경변수:
    DASH_HOST              (default: 0.0.0.0)
    DASH_PORT              (default: 8050)
    DASH_DEBUG             (default: false)
    OMNIPDM_SECRET_KEY     (production 필수 — 미설정 시 random fallback + 경고, config.py 참조)
    OMNIPDM_AUTH_REQUIRED  (default: true — false 로 두면 인증 가드 비활성, 개발/테스트용)
"""
from __future__ import annotations

import os

import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc
from flask import redirect, request

from models_core import config
from services.auth import create_default_user_service
from services.auth import sessions as auth_sessions
from services.auth.routes import init_auth_blueprint


# /login, /logout, _dash 내부 정적 경로 등은 인증 가드에서 제외.
# Dash 내부 경로 (_dash-*) 가 redirect 되면 페이지 자체가 부팅 안 됨.
_AUTH_EXEMPT_PREFIXES = (
    "/login",
    "/logout",
    "/_dash",
    "/static",
    "/assets",
    "/favicon.ico",
)


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


def create_app() -> Dash:
    app = Dash(
        __name__,
        use_pages=True,
        pages_folder="pages",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="OmniPdM",
    )

    server = app.server  # Flask app
    server.secret_key = config.SECRET_KEY

    # ----------------------------------------------------------------------
    # Auth blueprint + before_request guard (Tier 1.5 P1-b)
    # ----------------------------------------------------------------------
    auth_required = os.getenv("OMNIPDM_AUTH_REQUIRED", "true").lower() in ("1", "true", "yes", "on")
    user_service = create_default_user_service()
    audit_conn_provider = _make_audit_conn_provider()

    bp = init_auth_blueprint(user_service, audit_conn_provider)
    server.register_blueprint(bp)

    if auth_required:
        @server.before_request
        def _require_login():
            if _is_exempt(request.path):
                return None
            if auth_sessions.is_authenticated():
                return None
            # 원래 가려던 경로를 ?next= 로 보존 — 로그인 후 자동 복귀.
            next_url = request.full_path if request.query_string else request.path
            return redirect(f"/login?next={next_url}")

    # ----------------------------------------------------------------------
    # Layout
    # ----------------------------------------------------------------------
    nav_items = [
        dbc.NavItem(dcc.Link(page["name"], href=page["path"], className="nav-link"))
        for page in dash.page_registry.values()
    ]
    # Logout 은 Flask route 라 Dash router 가 가로채면 안 됨 → html.A 로 full-page navigation.
    if auth_required:
        nav_items.append(
            dbc.NavItem(html.A("Logout", href="/logout", className="nav-link"))
        )

    navbar = dbc.NavbarSimple(
        brand="OmniPdM",
        brand_href="/",
        color="dark",
        dark=True,
        fluid=True,
        children=nav_items,
    )

    app.layout = dbc.Container(
        [
            navbar,
            html.Div(dash.page_container, className="mt-4"),
        ],
        fluid=True,
    )

    return app


app = create_app()
server = app.server  # WSGI entry for production (gunicorn 등)


if __name__ == "__main__":
    host = os.getenv("DASH_HOST", "0.0.0.0")
    port = int(os.getenv("DASH_PORT", "8050"))
    debug = os.getenv("DASH_DEBUG", "false").lower() == "true"

    app.run(host=host, port=port, debug=debug)

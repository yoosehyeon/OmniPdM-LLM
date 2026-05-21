"""Flask login / logout routes — Tier 1.5 P1-b.

Dash 페이지로 등록하지 않는 이유:
- dash.register_page 로 등록하면 page_registry 가 6 → 7 로 늘어 test_smoke 가 깨진다.
- 인증 페이지는 layout / callback 시스템이 필요 없는 단순 form POST — 순수 Flask 가 적합.

설계:
- Blueprint 로 분리해 app.py 가 한 줄 register_blueprint 호출만 하면 됨.
- HTML 은 Bootstrap CDN 사용 (Dash 와 동일 dbc.themes.BOOTSTRAP) — 추가 정적 파일 없음.
- POST 흐름: CSRF 검증 → UserService.verify_credentials → login_user → next 또는 / 로 redirect.
- 실패 시 form 재표시 + 일반화된 에러 메시지 (사용자 enumeration 방어).

audit_log: login_user / logout_user / record_login_failure 가 내부적으로 처리.
"""

from __future__ import annotations

import html
from typing import Any, Optional
from urllib.parse import urlparse

from flask import Blueprint, redirect, request

from services.auth import sessions
from services.auth.sessions import role_required, set_user_service_for_rbac
from services.auth.users import UserService


# 의존성을 module-level 변수로 보관 — app.py 의 init_auth_blueprint() 가 주입.
# 이렇게 한 이유: Flask blueprint 라우트는 instance method 가 아니라서 self 가 없고,
# UserService / audit_conn 을 매 요청 시 어떻게든 주입해야 한다. Flask app context 에
# 매달기보다 단순 module 변수가 PoC 에 가장 가볍다.
_user_service: Optional[UserService] = None
_audit_conn_provider: Any = None  # 호출 가능한 callable -> psycopg.Connection or None


bp = Blueprint("omnipdm_auth", __name__)


def init_auth_blueprint(
    user_service: UserService,
    audit_conn_provider: Any = None,
) -> Blueprint:
    """app.py 에서 Blueprint 등록 직전 호출. UserService 와 audit_conn provider 주입.

    audit_conn_provider: 매 호출 시 psycopg.Connection 또는 None 을 반환하는 callable.
                        None 이면 audit_log 호출 자체 skip.

    role_required 데코레이터가 사용할 UserService 도 함께 등록 — 한 곳에서 일관 주입.
    """
    global _user_service, _audit_conn_provider
    _user_service = user_service
    _audit_conn_provider = audit_conn_provider
    set_user_service_for_rbac(user_service)
    return bp


def _is_safe_next(next_url: Optional[str]) -> bool:
    """`?next=` open redirect 방어 — 외부 host 로 보내지 못하게.

    상대 경로 ('/status') 만 허용. 절대 URL ('http://evil.com/x') 거부.
    """
    if not next_url:
        return False
    parsed = urlparse(next_url)
    return not parsed.scheme and not parsed.netloc and next_url.startswith("/")


def _render_login_page(error: Optional[str] = None, next_url: str = "") -> str:
    """간단한 Bootstrap 로그인 폼. CSRF token hidden input 포함.

    XSS 방어: next_url, error 모두 attribute / text context 에 들어가기 전 html.escape.
    `_is_safe_next` 가 scheme/netloc 만 막아 `/"><script>...` 같은 path-embedded payload 는
    통과 가능 — 출력 시점에 escape 가 최종 방어선.
    """
    csrf = sessions.get_or_create_csrf_token()
    safe_error = html.escape(error, quote=True) if error else ""
    safe_next = html.escape(next_url, quote=True) if next_url else ""
    error_html = (
        f'<div class="alert alert-danger" role="alert">{safe_error}</div>' if safe_error else ""
    )
    next_input = f'<input type="hidden" name="next" value="{safe_next}">' if safe_next else ""

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OmniPdM - Login</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
  <div class="container" style="max-width: 420px; margin-top: 80px;">
    <div class="card shadow-sm">
      <div class="card-body">
        <h4 class="card-title mb-3">OmniPdM</h4>
        <p class="text-muted small mb-3">Sign in to access the dashboard</p>
        {error_html}
        <form method="post" action="/login">
          <input type="hidden" name="csrf_token" value="{csrf}">
          {next_input}
          <div class="mb-3">
            <label class="form-label" for="user_id">User ID</label>
            <input class="form-control" type="text" name="user_id" id="user_id"
                   autocomplete="username" required autofocus>
          </div>
          <div class="mb-3">
            <label class="form-label" for="password">Password</label>
            <input class="form-control" type="password" name="password" id="password"
                   autocomplete="current-password" required>
          </div>
          <button class="btn btn-primary w-100" type="submit">Sign in</button>
        </form>
      </div>
    </div>
  </div>
</body>
</html>
"""


@bp.route("/login", methods=["GET"])
def login_get():
    if sessions.is_authenticated():
        return redirect("/")
    next_url = request.args.get("next", "")
    if not _is_safe_next(next_url):
        next_url = ""
    return _render_login_page(next_url=next_url)


@bp.route("/login", methods=["POST"])
def login_post():
    user_id = (request.form.get("user_id") or "").strip()
    password = request.form.get("password") or ""
    submitted_token = request.form.get("csrf_token")
    next_url = request.form.get("next", "")
    if not _is_safe_next(next_url):
        next_url = ""

    audit_conn = _audit_conn_provider() if _audit_conn_provider else None

    if not sessions.verify_csrf_token(submitted_token):
        sessions.record_login_failure(user_id, reason="csrf", audit_conn=audit_conn)
        return _render_login_page(error="Session expired. Please try again.", next_url=next_url), 400

    if not user_id or not password:
        sessions.record_login_failure(user_id, reason="empty_fields", audit_conn=audit_conn)
        return _render_login_page(error="Invalid credentials.", next_url=next_url), 401

    if _user_service is None:
        # blueprint init 안 됨 — 운영 사고. 인증 자체 거부.
        sessions.record_login_failure(user_id, reason="service_unavailable", audit_conn=audit_conn)
        return _render_login_page(error="Authentication temporarily unavailable.", next_url=next_url), 503

    user = _user_service.verify_credentials(user_id, password)
    if user is None:
        sessions.record_login_failure(user_id, reason="bad_credentials", audit_conn=audit_conn)
        return _render_login_page(error="Invalid credentials.", next_url=next_url), 401

    sessions.login_user(user, audit_conn=audit_conn)
    return redirect(next_url or "/")


@bp.route("/logout", methods=["GET", "POST"])
def logout():
    audit_conn = _audit_conn_provider() if _audit_conn_provider else None
    sessions.logout_user(audit_conn=audit_conn)
    return redirect("/login")


# ---------------------------------------------------------------------------
# Admin-only stub — role_required 데코레이터 데모. P1-d 의 admin UI 진입 전 안전망.
# ---------------------------------------------------------------------------
@bp.route("/admin/health-check")
@role_required("admin")
def admin_health_check():
    """admin 만 접근 가능한 단순 health-check. RBAC 동작 확인용 stub.

    P1-d 의 maintenance_orders / users 관리 UI 가 들어오기 전 placeholder.
    """
    return {"status": "ok", "role_check": "admin"}, 200


# ---------------------------------------------------------------------------
# Error handlers — 401/403 간단 페이지. abort(401/403) 에서 자동 호출.
# blueprint app_errorhandler 는 app 전체에 적용 — Flask 공식 권장 방식.
# ---------------------------------------------------------------------------
def _render_forbidden_page(status_label: str, message: str) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>OmniPdM - {status_label}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
  <div class="container" style="max-width: 480px; margin-top: 80px;">
    <div class="card shadow-sm">
      <div class="card-body text-center">
        <h4 class="card-title mb-3">{status_label}</h4>
        <p class="text-muted">{message}</p>
        <a class="btn btn-primary mt-2" href="/">Back to dashboard</a>
        <a class="btn btn-outline-secondary mt-2 ms-2" href="/logout">Sign out</a>
      </div>
    </div>
  </div>
</body>
</html>
"""


@bp.app_errorhandler(401)
def _handle_401(_err):
    # before_request 가 보통 가로채지만, role_required 에서 abort(401) 시 도달.
    return redirect("/login?next=/"), 302


@bp.app_errorhandler(403)
def _handle_403(_err):
    return _render_forbidden_page(
        "403 Forbidden",
        "You do not have permission to access this page.",
    ), 403

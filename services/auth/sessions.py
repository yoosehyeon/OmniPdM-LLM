"""Flask session 기반 인증 헬퍼 — Tier 1.5 P1-b.

설계:
- Flask-Login 도입 비용 회피 (PoC 단일 사용자). Flask 표준 `session` dict 직접 사용.
- session 에는 `user_id` 만 저장 — User 객체 전체 pickle 회피 + DB role 변경 즉시 반영.
- current_user() 는 session 에서 user_id 읽고 UserService.get_user() 로 매번 조회 →
  비활성/삭제된 user 가 즉시 로그아웃. 캐시 안 함 (PoC 트래픽 매우 낮음).
- audit_log 자동 기록: LOGIN_SUCCESS / LOGIN_FAILURE / LOGOUT.

CSRF (login 폼):
- session 에 csrf_token 저장 + 폼 hidden input 으로 회수 → 검증.
- 다른 form 이 없으므로 Flask-WTF 도입 불요 (T1.5 P1-b 결정).

future migration: 사용자 5명+ 시점에 Flask-Login + Flask-WTF 검토 (PRD §13.1.b 후속 메모).
"""

from __future__ import annotations

import secrets
from typing import Any, Optional

from flask import session

from services.audit_actions import AuditAction, TARGET_TYPE_SESSION
from services.auth.users import User, UserService

_SESSION_USER_KEY = "omnipdm_user_id"
_SESSION_CSRF_KEY = "omnipdm_csrf_token"


# ---------------------------------------------------------------------------
# CSRF
# ---------------------------------------------------------------------------
def get_or_create_csrf_token() -> str:
    """session 에 csrf_token 이 없으면 새로 생성. 있으면 그대로 반환.

    login 폼이 hidden input 으로 이 값을 회수 → POST 시 검증.
    """
    token = session.get(_SESSION_CSRF_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[_SESSION_CSRF_KEY] = token
    return token


def verify_csrf_token(submitted: Optional[str]) -> bool:
    """폼에서 받은 token 이 session 의 그것과 정확히 일치하는지.

    secrets.compare_digest 로 timing-safe 비교.
    """
    stored = session.get(_SESSION_CSRF_KEY, "")
    if not stored or not submitted:
        return False
    return secrets.compare_digest(stored, submitted)


def rotate_csrf_token() -> str:
    """기존 token 폐기 + 새 token 생성. 로그인 성공 시 호출 (session fixation 방어)."""
    session.pop(_SESSION_CSRF_KEY, None)
    return get_or_create_csrf_token()


# ---------------------------------------------------------------------------
# Login / Logout
# ---------------------------------------------------------------------------
def login_user(user: User, *, audit_conn: Optional[Any] = None) -> None:
    """user 를 session 에 저장 + LOGIN_SUCCESS audit_log.

    audit_conn 은 psycopg.Connection 또는 None — None 이면 audit 생략 (test / Null backend).
    session fixation 방어: 새 csrf_token 생성.
    """
    session[_SESSION_USER_KEY] = user.user_id
    rotate_csrf_token()
    _safe_audit(
        audit_conn,
        actor_id=user.user_id,
        action=AuditAction.LOGIN_SUCCESS,
        target_id=user.user_id,
        meta={"role": user.role},
    )


def logout_user(*, audit_conn: Optional[Any] = None) -> None:
    """session 완전 비움 + LOGOUT audit_log.

    csrf_token 도 함께 폐기. 다음 로그인 시도가 새 token 으로 시작.
    """
    user_id = session.get(_SESSION_USER_KEY)
    session.pop(_SESSION_USER_KEY, None)
    session.pop(_SESSION_CSRF_KEY, None)
    if user_id:
        _safe_audit(
            audit_conn,
            actor_id=user_id,
            action=AuditAction.LOGOUT,
            target_id=user_id,
        )


def record_login_failure(
    attempted_user_id: str, *, reason: str, audit_conn: Optional[Any] = None
) -> None:
    """비밀번호 불일치 / CSRF / 미존재 등 LOGIN_FAILURE audit_log.

    attempted_user_id 는 PII 일 수 있으나 NIST AC-2 가 요구하는 "계정 활동" 감사에
    필수 — meta 에 평문 비밀번호는 절대 넣지 않는다 (audit_actions docstring 정책).
    """
    _safe_audit(
        audit_conn,
        actor_id=None,
        action=AuditAction.LOGIN_FAILURE,
        target_id=attempted_user_id,
        meta={"reason": reason},
    )


# ---------------------------------------------------------------------------
# Current user / guard
# ---------------------------------------------------------------------------
def current_user_id() -> Optional[str]:
    """session 에서 user_id 만 가볍게 추출 (DB 조회 없음). before_request 가드에 사용."""
    return session.get(_SESSION_USER_KEY)


def current_user(user_service: UserService) -> Optional[User]:
    """session user_id → UserService.get_user() 로 매번 조회.

    user 가 inactive / 삭제되면 즉시 None 반환 (session 도 정리됨이 자연스럽지만
    여기서는 read-only). 캐시 안 함 — PoC 트래픽 낮음 + 보안 우선.
    """
    user_id = current_user_id()
    if not user_id:
        return None
    user = user_service.get_user(user_id)
    if user is None:
        # session 에 있지만 DB 에서 사라진 경우 — session 도 정리.
        session.pop(_SESSION_USER_KEY, None)
    return user


def is_authenticated() -> bool:
    """before_request 가드용 — DB 조회 없이 session 만 확인."""
    return _SESSION_USER_KEY in session


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _safe_audit(
    conn: Optional[Any],
    *,
    actor_id: Optional[str],
    action: AuditAction,
    target_id: Optional[str],
    meta: Optional[dict] = None,
) -> None:
    """audit_log 호출을 try/except 로 감싸 인증 흐름을 죽이지 않는다.

    audit 실패는 stdout 로그만 남기고 진행 — 인증 자체가 실패해서 사용자가 로그인
    못하는 것이 더 큰 비용. NullDbWriter 환경에서는 conn=None 으로 호출되어 no-op.
    """
    if conn is None:
        return
    try:
        from services import audit_actions  # 지연 import — psycopg optional 호환
        audit_actions.log(
            conn,
            actor_id=actor_id,
            action=action,
            target_type=TARGET_TYPE_SESSION,
            target_id=target_id,
            meta=meta,
        )
    except Exception as e:
        print(
            f"[sessions] audit_log {action.value} failed: {type(e).__name__}: {e}",
            flush=True,
        )

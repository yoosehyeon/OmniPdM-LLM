"""Audit action 상수 + log() 헬퍼 — PRD v6.3 §13.1.b T1.5 결정 #7 (audit_log 도입).

설계 의도:
- DB 측 audit_log.action 컬럼은 CHECK 가 없다 (스키마 진화 자유도 우선 — T1.5 결정 #7
  거부 항목 정책). 그 대신 여기 Python enum 으로 action 종류를 한 곳에 모아 관리한다.
  새 action 추가 시 이 파일만 수정 → grep 으로 사용처 추적 가능.
- target_type 은 의도적으로 자유 텍스트 — 같은 정책 ("schema 진화 자유도 우선"). 권장
  값만 TARGET_TYPE_* 상수로 명시.
- log() 는 psycopg connection 을 받아 1행 INSERT — 호출자가 트랜잭션 정책 책임.

보안 (NIST AC-6 + OWASP Logging Cheat Sheet):
- meta JSONB 에 **절대** PII, password, API key, session token 등 민감 데이터를 넣지 말 것.
  audit_log 는 SELECT 권한이 광범위해질 가능성이 큰 테이블이다.
- meta JSONB 가 1MB 넘으면 WAL 폭주 + 쿼리 성능 저하 — old/new value 가 클 경우
  hash 나 요약만 저장하고 본문은 별도 저장소로.

호환성:
- psycopg 는 optional dependency. NullDeviceService / NullDbWriter 경로에서는
  log() 가 호출되지 않으므로 import 시점 비용 없음.
- module-level try/except ImportError 로 가드 — psycopg 미설치 환경에서도 이 모듈을
  import 하는 것 자체는 안전. AuditAction enum 만 사용하려는 코드도 무탈.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

try:
    from psycopg.types.json import Jsonb as _Jsonb  # psycopg v3 권장 JSONB adapter
except ImportError:  # psycopg 미설치 환경 — log() 자체가 호출되지 않는다는 전제.
    _Jsonb = None  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    # 타입 체커에는 실제 타입을 보여주지만 런타임 import 는 안 한다 — psycopg optional 유지.
    from psycopg import Connection


class AuditAction(str, Enum):
    """audit_log.action 에 들어가는 모든 값. 새 종류 추가는 여기서만.

    주의: enum value 는 DB 에 그대로 저장되므로 변경하면 기존 로그와 불일치한다.
    rename 이 불가피하면 maintenance window + UPDATE 마이그레이션 동반.
    """

    # User lifecycle
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_SOFT_DELETED = "user.soft_deleted"
    USER_ROLE_CHANGED = "user.role_changed"

    # Session
    LOGIN_SUCCESS = "session.login_success"
    LOGIN_FAILURE = "session.login_failure"
    LOGOUT = "session.logout"

    # Device lifecycle
    DEVICE_CREATED = "device.created"
    DEVICE_UPDATED = "device.updated"
    DEVICE_SOFT_DELETED = "device.soft_deleted"
    DEVICE_STATUS_CHANGED = "device.status_changed"

    # Maintenance orders
    ORDER_CREATED = "order.created"
    ORDER_ASSIGNED = "order.assigned"
    ORDER_STATUS_CHANGED = "order.status_changed"
    ORDER_CLOSED = "order.closed"
    ORDER_CANCELLED = "order.cancelled"


# audit_log.target_type 에 들어가는 권장 값. CHECK 없는 자유 텍스트라 강제는 아니지만
# 호출자가 일관 사용하면 조회 / grep 이 편하다.
TARGET_TYPE_USER = "user"
TARGET_TYPE_DEVICE = "device"
TARGET_TYPE_ORDER = "order"
TARGET_TYPE_SESSION = "session"


def log(
    conn: "Connection",
    *,
    actor_id: Optional[str],
    action: AuditAction,
    target_type: str,
    target_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """audit_log 에 1행 INSERT.

    Parameters
    ----------
    conn
        psycopg.Connection — 호출자가 트랜잭션 정책 책임 (autocommit / 명시 commit).
    actor_id
        행위자 user_id. 시스템 자동 이벤트는 None — meta 에 ``system_event=True`` 자동 추가.
    action
        AuditAction 상수.
    target_type
        TARGET_TYPE_* 권장. 자유 텍스트지만 일관 사용 권장.
    target_id
        대상 식별자. user_id / device_id / order_id (str 변환). None 은 그대로 NULL 저장.
    meta
        임의 JSONB. **PII/password/token 금지** (모듈 docstring 참조). 1MB 미만.

    예외: psycopg 가 던지는 예외는 그대로 전파 — 호출자가 운영 안정성 결정.
    DeviceService 처럼 워커 메인 루프에서 호출하는 코드는 try/except 로 감싸 audit 실패가
    워커를 죽이지 않도록 한다.
    """
    if _Jsonb is None:
        # psycopg 미설치 — 정상 경로에서는 도달 불가 (호출자가 NullDbWriter 분기에서 가드).
        raise RuntimeError(
            "audit_actions.log() requires psycopg. Caller must guard with "
            "isinstance(db, NullDbWriter) or equivalent before calling."
        )

    full_meta: Dict[str, Any] = dict(meta or {})
    if actor_id is None and "system_event" not in full_meta:
        full_meta["system_event"] = True

    target_id_str = str(target_id) if target_id is not None else None

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO audit_log (actor_id, action, target_type, target_id, meta)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                actor_id,
                action.value,
                target_type,
                target_id_str,
                _Jsonb(full_meta),
            ),
        )

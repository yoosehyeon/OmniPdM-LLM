"""UserService — Tier 1.5 P1-b. 인증용 user 조회 + verify.

설계 (DeviceService 패턴 그대로):
- Protocol + NullUserService (no-op) + TimescaleUserService (psycopg) + create_default factory.
- 모든 조회는 active=TRUE 인 사용자만 — soft delete 된 user 는 로그인 시도조차 안 됨.
- verify_credentials 는 passwords.verify_password 위임 → placeholder hash 자동 reject.
- 로그인 시도 결과 (성공/실패) audit_log 기록은 sessions.py 책임 (UserService 는 storage 만).

Timing attack 방어:
- user 미존재 / 비활성 / 비밀번호 불일치 / placeholder 모두 동일한 None 반환 + 동일한
  verify_password 한 번 호출 → 분기에 따른 timing 차이 최소화 (OWASP / NIST 권장).

Thread safety:
- psycopg connection + RLock (db_writer / device_service 와 동일 패턴).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from services.auth.passwords import PLACEHOLDER_HASH, verify_password

if TYPE_CHECKING:
    from psycopg import Connection  # noqa: F401 — 타입 힌트 전용


@dataclass(frozen=True)
class User:
    """users 행의 표현. password_hash 는 외부에 절대 노출하지 않는다 —
    UserService 내부 verify 전용. UI / session 으로는 user_id, role, display_name 만."""

    user_id: str
    email: str
    display_name: str
    role: str  # 'admin' / 'operator' / 'viewer'
    active: bool
    created_at: datetime
    updated_at: datetime


@runtime_checkable
class UserService(Protocol):
    """user storage + verify 추상화."""

    def get_user(self, user_id: str) -> Optional[User]: ...

    def verify_credentials(self, user_id: str, plaintext: str) -> Optional[User]:
        """평문 비밀번호 검증. 성공 시 User, 실패 시 None.

        실패 사유 (미존재 / 비활성 / 비밀번호 불일치 / placeholder) 를 호출자에게 노출하지
        않음 — timing / enumeration 공격 방지.
        """
        ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Null backend — DB 없이 인증 라우트 import 가 깨지지 않도록.
# ---------------------------------------------------------------------------
class NullUserService:
    """DB 미기동 환경용 no-op. 모든 인증 시도가 None 을 반환 → 로그인 불가."""

    def get_user(self, user_id: str) -> Optional[User]: return None
    def verify_credentials(self, user_id: str, plaintext: str) -> Optional[User]: return None
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Timescale backend
# ---------------------------------------------------------------------------
class TimescaleUserService:
    """TimescaleDB (Postgres) 백엔드. psycopg v3.

    단일 connection + RLock (PoC 단일 워커 가정). 다중 워커 시 ConnectionPool 도입.
    """

    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        connect_timeout: int = 5,
        application_name: str = "omnipdm-user-service",
    ) -> None:
        import psycopg

        self._conn = psycopg.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            connect_timeout=connect_timeout,
            application_name=application_name,
            autocommit=True,
        )
        self._lock = RLock()

    def get_user(self, user_id: str) -> Optional[User]:
        """active=TRUE 인 user 만. 비활성 / 미존재는 동일하게 None."""
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, email, display_name, role, active, created_at, updated_at
                      FROM users
                     WHERE user_id = %s AND active = TRUE
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return User(
            user_id=row[0],
            email=row[1],
            display_name=row[2],
            role=row[3],
            active=row[4],
            created_at=row[5],
            updated_at=row[6],
        )

    def verify_credentials(self, user_id: str, plaintext: str) -> Optional[User]:
        """user_id 로 user + password_hash 조회 → verify_password → User 반환.

        실패 사유 (미존재 / 비활성 / 비밀번호 불일치 / placeholder) 를 분리하지 않음 —
        timing 차이를 최소화하기 위해 모든 분기에서 verify_password 정확히 1회 호출.
        """
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, email, display_name, role, active, created_at, updated_at,
                           password_hash
                      FROM users
                     WHERE user_id = %s
                    """,
                    (user_id,),
                )
                row = cur.fetchone()

        if row is None:
            # user 없음 — placeholder hash 로 dummy verify (timing 평탄화).
            # placeholder 는 verify_password 가 무조건 False 라 결과는 안 바뀜.
            verify_password(PLACEHOLDER_HASH, plaintext)
            return None

        active = row[4]
        stored_hash = row[7]
        # active=False 일 때도 verify 호출 — 분기에 따른 시간 차이 최소화.
        ok = verify_password(stored_hash, plaintext)
        if not active or not ok:
            return None

        return User(
            user_id=row[0],
            email=row[1],
            display_name=row[2],
            role=row[3],
            active=active,
            created_at=row[5],
            updated_at=row[6],
        )

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


def create_default() -> UserService:
    """config 보고 적절한 백엔드 선택. DeviceService 와 동일 정책."""
    from models_core import config  # 지연 import — 테스트에서 mock 가능

    if not config.DB_ENABLED:
        return NullUserService()
    try:
        return TimescaleUserService(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
        )
    except Exception as e:
        print(
            f"[user_service] DB unavailable ({type(e).__name__}: {e}) - "
            f"falling back to NullUserService (login disabled).",
            flush=True,
        )
        return NullUserService()

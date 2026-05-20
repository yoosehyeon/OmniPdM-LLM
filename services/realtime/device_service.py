"""DeviceService — CMMS Tier 1.5 P0 (PRD v6.3 §13.1.b T1.5 결정 #1, #6, #7).

설계 의도 (db_writer.py 패턴 그대로):
- 워커 / app 은 DB 종류를 모른다. DeviceService protocol 만 받아서 호출한다.
- 기본 구현 NullDeviceService — psycopg 없이도 동작, smoke 테스트 가능.
- 실제 운영은 TimescaleDeviceService — psycopg + audit_log 통합.
- create_default() factory 가 config 보고 dispatch (NullDbWriter 정책과 동일).

T1.5 결정 반영:
- #1 device_id TEXT PK + ON DELETE RESTRICT → upsert 만 지원, 삭제는 soft delete (mark_deleted).
- #6 scope_required(roles) 데코레이터 placeholder — 시그니처만, 본문은 pass-through.
  사용자 2명+ 시 본문 활성화.
- #7 audit_log 도입 — set_status / mark_deleted 가 자동으로 audit_actions.log() 호출.

Thread safety:
- psycopg connection 은 thread-safe 하지 않으므로 RLock 으로 보호 (db_writer 와 동일).
- multi-statement (set_status: UPDATE + INSERT + audit) 는 conn.transaction() 으로 묶는다.

실패 정책 (의도된 비대칭):
- upsert_device 의 device INSERT 성공과 audit_log INSERT 는 **별도 트랜잭션**이다. audit 실패가
  device 생성을 rollback 시키면 telemetry FK 가 위반되어 워커 핵심 기능 (실시간 적재) 이
  깨진다. audit_log 는 부가 정보, device 는 필수 — 의도된 비대칭. audit 실패는 stdout 로그만.
- set_status / mark_deleted 는 사용자 명시 행동이라 audit 까지 한 트랜잭션으로 묶는다. audit 실패
  시 상태 변경 rollback 이 안전.
- 그 외 DB 실패는 호출자가 try/except — 본 클래스는 그대로 전파한다. 워커 측 wrapper
  (_db_safe) 에서 워커 루프를 살린다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Protocol, runtime_checkable

from services import audit_actions
from services.audit_actions import AuditAction

if TYPE_CHECKING:
    from psycopg import Connection  # noqa: F401 — 타입 힌트 전용


@dataclass(frozen=True)
class Device:
    """devices 행의 표현. soft delete 된 device 도 active=False 로 그대로 들어온다 —
    list_active_devices() 만 필터링한다."""

    device_id: str
    dataset_key: str
    name: str
    plant_id: str
    equipment_group_id: Optional[str]
    current_status: str
    active: bool
    created_at: datetime
    updated_at: datetime


@runtime_checkable
class DeviceService(Protocol):
    """Device 마스터 + 상태 머신 추상화. 새 백엔드는 이 protocol 만 따르면 됨."""

    def upsert_device(
        self,
        device_id: str,
        dataset_key: str,
        *,
        name: Optional[str] = None,
        plant_id: Optional[str] = None,
    ) -> None: ...

    def get_device(self, device_id: str) -> Optional[Device]: ...

    def list_active_devices(self) -> List[Device]: ...

    def set_status(
        self,
        device_id: str,
        new_status: str,
        *,
        reason: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> None: ...

    def mark_deleted(self, device_id: str, *, actor_id: Optional[str] = None) -> None: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# scope_required — T1.5 결정 #6 placeholder
# ---------------------------------------------------------------------------
def scope_required(*roles: str) -> Callable:
    """RBAC scope placeholder. 사용자 2명+ 시 본문 활성화 (T1.5 결정 #6).

    현재는 데코레이터 시그니처만 제공하며 본문은 pass-through — 정의만 있으면
    미래 라우트 코드가 미리 부착해 둘 수 있어 활성화 비용이 0 이 된다.
    """

    def _decorator(fn: Callable) -> Callable:
        # 현재는 no-op. 활성화 시 Flask `g.user.role in roles` 검증 + 403.
        return fn

    return _decorator


# ---------------------------------------------------------------------------
# Null backend — broker/DB 없이 워커 동작 보장 (NullDbWriter 와 짝)
# ---------------------------------------------------------------------------
class NullDeviceService:
    """no-op 백엔드. 워커가 device_service 호출 코드 경로를 막지 않으면서 zero-cost."""

    def upsert_device(self, device_id, dataset_key, *, name=None, plant_id=None): ...
    def get_device(self, device_id) -> Optional[Device]: return None
    def list_active_devices(self) -> List[Device]: return []
    def set_status(self, device_id, new_status, *, reason=None, actor_id=None): ...
    def mark_deleted(self, device_id, *, actor_id=None): ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Timescale backend — psycopg + audit_log 통합
# ---------------------------------------------------------------------------
class TimescaleDeviceService:
    """TimescaleDB (Postgres) 백엔드.

    설계:
    - 단일 psycopg connection + RLock (db_writer 와 동일 패턴). 다중 워커 / 고 TPS 진입 시
      ConnectionPool 도입.
    - upsert_device: 워커 hot path. 인메모리 set 으로 중복 INSERT 방지 → 같은 device 의
      2번째+ telemetry 부터는 DB 왕복 0회.
    - set_status / mark_deleted: multi-statement 트랜잭션 — conn.transaction() 으로 묶어
      audit_log 까지 atomicity 보장. autocommit=True 라도 transaction 컨텍스트는 동작.
    """

    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        connect_timeout: int = 5,
        application_name: str = "omnipdm-device-service",
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
        # 인메모리 캐시 — upsert 중복 차단. 워커 재시작 시 비워지지만 ON CONFLICT 가 보완.
        self._upsert_cache: set[str] = set()

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def upsert_device(
        self,
        device_id: str,
        dataset_key: str,
        *,
        name: Optional[str] = None,
        plant_id: Optional[str] = None,
    ) -> None:
        """첫 호출 시 INSERT (ON CONFLICT DO NOTHING). 이후 같은 device_id 는 캐시 hit.

        호출 측에서는 telemetry publish 직전마다 자유롭게 부르면 된다 — 캐시가 비용을 0 에 수렴.

        Atomicity 비대칭: device INSERT 와 audit_log INSERT 를 별도 트랜잭션으로 유지한다.
        audit 실패가 device 생성을 rollback 시키면 telemetry FK 가 위반되어 워커 hot path 가
        깨진다 (의도된 비대칭). 클래스 docstring 의 "실패 정책" 참조.
        """
        if device_id in self._upsert_cache:
            return

        with self._lock:
            # double-check: 다른 thread 가 먼저 채웠을 수 있다.
            if device_id in self._upsert_cache:
                return
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO devices (device_id, dataset_key, name, plant_id, active)
                    VALUES (%s, %s, %s, %s, TRUE)
                    ON CONFLICT (device_id) DO NOTHING
                    """,
                    (
                        device_id,
                        dataset_key,
                        name if name is not None else device_id,
                        plant_id if plant_id is not None else "plant-01",
                    ),
                )
                inserted = cur.rowcount > 0

            # device INSERT 성공/충돌 무관하게 캐시 채운다 — 어차피 device 는 존재.
            self._upsert_cache.add(device_id)

            # audit_log — 실제로 새로 추가된 경우에만. 실패해도 device 생성은 유지.
            if inserted:
                try:
                    audit_actions.log(
                        self._conn,
                        actor_id=None,  # 워커 자동 등록 — system event
                        action=AuditAction.DEVICE_CREATED,
                        target_type=audit_actions.TARGET_TYPE_DEVICE,
                        target_id=device_id,
                        meta={"dataset_key": dataset_key, "source": "mqtt_worker"},
                    )
                except Exception as e:
                    # audit 실패가 워커 hot path 를 죽이지 않는다 — 로그만 남기고 계속.
                    print(
                        f"[DeviceService] audit_log for DEVICE_CREATED failed: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )

    def get_device(self, device_id: str) -> Optional[Device]:
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT device_id, dataset_key, name, plant_id, equipment_group_id,
                           current_status, active, created_at, updated_at
                      FROM devices
                     WHERE device_id = %s
                    """,
                    (device_id,),
                )
                row = cur.fetchone()
        return _row_to_device(row) if row else None

    def list_active_devices(self) -> List[Device]:
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT device_id, dataset_key, name, plant_id, equipment_group_id,
                           current_status, active, created_at, updated_at
                      FROM devices
                     WHERE active = TRUE
                     ORDER BY device_id
                    """
                )
                rows = cur.fetchall()
        return [_row_to_device(r) for r in rows]

    def set_status(
        self,
        device_id: str,
        new_status: str,
        *,
        reason: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> None:
        """devices.current_status UPDATE + device_status_history INSERT + audit_log 를
        하나의 트랜잭션으로 묶는다.

        존재하지 않거나 동일 상태인 경우 no-op (history 도 생성 안 함).
        """
        with self._lock, self._conn.transaction():
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT current_status FROM devices WHERE device_id = %s FOR UPDATE",
                    (device_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return  # 미등록 device — 호출 측이 사전에 upsert 했어야 함.
                from_status = row[0]
                if from_status == new_status:
                    return  # 동일 상태 — 이력 / audit 생략.

                cur.execute(
                    "UPDATE devices SET current_status = %s WHERE device_id = %s",
                    (new_status, device_id),
                )
                cur.execute(
                    """
                    INSERT INTO device_status_history
                           (device_id, from_status, to_status, actor_id, reason)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (device_id, from_status, new_status, actor_id, reason),
                )

            audit_actions.log(
                self._conn,
                actor_id=actor_id,
                action=AuditAction.DEVICE_STATUS_CHANGED,
                target_type=audit_actions.TARGET_TYPE_DEVICE,
                target_id=device_id,
                meta={"from": from_status, "to": new_status, "reason": reason},
            )

    def mark_deleted(self, device_id: str, *, actor_id: Optional[str] = None) -> None:
        """Soft delete — devices.active=FALSE. FK ON DELETE RESTRICT 라 hard delete 불가.

        UPDATE + audit_log + 캐시 discard 모두 같은 트랜잭션 안에서 처리한다 — audit 실패
        시 active=TRUE 가 유지되고 캐시도 그대로라 일관성 보장.
        """
        with self._lock, self._conn.transaction():
            with self._conn.cursor() as cur:
                cur.execute(
                    "UPDATE devices SET active = FALSE WHERE device_id = %s AND active = TRUE",
                    (device_id,),
                )
                changed = cur.rowcount > 0
            if changed:
                audit_actions.log(
                    self._conn,
                    actor_id=actor_id,
                    action=AuditAction.DEVICE_SOFT_DELETED,
                    target_type=audit_actions.TARGET_TYPE_DEVICE,
                    target_id=device_id,
                )
                # transaction 안에서 discard — rollback 시 캐시도 함께 원복되도록.
                # 다음 publish 가 다시 INSERT 시도하면 ON CONFLICT 로 무탈.
                self._upsert_cache.discard(device_id)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _row_to_device(row: Iterable[Any]) -> Device:
    (
        device_id, dataset_key, name, plant_id, equipment_group_id,
        current_status, active, created_at, updated_at,
    ) = row
    return Device(
        device_id=device_id,
        dataset_key=dataset_key,
        name=name,
        plant_id=plant_id,
        equipment_group_id=equipment_group_id,
        current_status=current_status,
        active=active,
        created_at=created_at,
        updated_at=updated_at,
    )


def create_default() -> DeviceService:
    """config 보고 적절한 백엔드 선택. DbWriter 와 동일 정책 — DB 불가 시 NullDeviceService."""
    from models_core import config  # 지연 import — 테스트에서 환경변수 mock 가능

    if not config.DB_ENABLED:
        return NullDeviceService()
    try:
        return TimescaleDeviceService(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
        )
    except Exception as e:
        # ASCII hyphen 사용 — Windows cp949 환경에서 em-dash 가 UnicodeEncodeError 유발.
        print(
            f"[device_service] DB unavailable ({type(e).__name__}: {e}) - "
            f"falling back to NullDeviceService",
            flush=True,
        )
        return NullDeviceService()

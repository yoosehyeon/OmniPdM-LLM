"""MaintenanceOrderService — CMMS Tier 1.5 P1-d.

설계 (DeviceService / UserService 패턴 그대로):
- Protocol + NullMaintenanceOrderService (no-op) + TimescaleMaintenanceOrderService + factory.
- list/get/create/update_status/close — full CRUD 의 P1 부분집합.
- create/update_status/close 시 audit_log 자동 INSERT (AuditAction.ORDER_*).
- closed_at 은 DB 트리거가 자동 설정 (001_cmms_schema.sql 의 maintenance_orders_set_closed_at).

Thread safety:
- psycopg connection + RLock (db_writer / device_service 와 동일).

T1.5 결정 반영:
- #1 device_id FK 강결합: 미등록 device 에 order 생성 불가 (ForeignKeyViolation 전파).
- #5 3-role RBAC: 이 서비스는 storage 만 책임. role 검사는 호출자 (Dash 페이지 + role_required) 책임.
- #7 audit_log: 모든 변경에 자동 INSERT.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Protocol, runtime_checkable

from services import audit_actions
from services.audit_actions import AuditAction

if TYPE_CHECKING:
    from psycopg import Connection  # noqa: F401


# maintenance_orders.status CHECK 와 동기 — 001_cmms_schema.sql 참조.
ORDER_STATUSES = ("open", "in_progress", "closed", "cancelled")
ORDER_PRIORITIES = ("low", "normal", "high", "critical")


@dataclass(frozen=True)
class MaintenanceOrder:
    """maintenance_orders 행의 표현."""

    order_id: int
    device_id: str
    assigned_to: Optional[str]
    title: str
    description: Optional[str]
    priority: str
    status: str
    triggered_by_alert_at: Optional[datetime]
    active: bool
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime]


@runtime_checkable
class MaintenanceOrderService(Protocol):
    """정비 주문 storage + 상태 전이 추상화."""

    def list_orders(
        self,
        *,
        status: Optional[str] = None,
        device_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[MaintenanceOrder]: ...

    def get_order(self, order_id: int) -> Optional[MaintenanceOrder]: ...

    def create_order(
        self,
        *,
        device_id: str,
        title: str,
        description: Optional[str] = None,
        priority: str = "normal",
        assigned_to: Optional[str] = None,
        triggered_by_alert_at: Optional[datetime] = None,
        actor_id: Optional[str] = None,
    ) -> Optional[MaintenanceOrder]: ...

    def update_status(
        self,
        order_id: int,
        new_status: str,
        *,
        actor_id: Optional[str] = None,
    ) -> bool: ...

    def close_order(
        self,
        order_id: int,
        *,
        actor_id: Optional[str] = None,
    ) -> bool: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Null backend
# ---------------------------------------------------------------------------
class NullMaintenanceOrderService:
    """DB 미기동 환경용 no-op. 모든 조회 빈 결과, 모든 변경 False."""

    def list_orders(self, *, status=None, device_id=None, limit=100): return []
    def get_order(self, order_id): return None
    def create_order(self, **kw): return None
    def update_status(self, order_id, new_status, *, actor_id=None): return False
    def close_order(self, order_id, *, actor_id=None): return False
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Timescale backend
# ---------------------------------------------------------------------------
class TimescaleMaintenanceOrderService:
    """TimescaleDB (Postgres) 백엔드. psycopg v3.

    update_status / close_order 는 transaction 안에서 UPDATE + audit_log 동시 처리.
    closed_at 은 DB 트리거 자동 처리 — 호출자가 신경 안 써도 됨.
    """

    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        connect_timeout: int = 5,
        application_name: str = "omnipdm-order-service",
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

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def list_orders(
        self,
        *,
        status: Optional[str] = None,
        device_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[MaintenanceOrder]:
        """active=TRUE 인 주문만. 최신 created_at 순."""
        clauses = ["active = TRUE"]
        params: List[Any] = []
        if status is not None:
            if status not in ORDER_STATUSES:
                raise ValueError(f"invalid status: {status}")
            clauses.append("status = %s")
            params.append(status)
        if device_id is not None:
            clauses.append("device_id = %s")
            params.append(device_id)
        params.append(int(limit))

        sql = (
            "SELECT order_id, device_id, assigned_to, title, description, "
            "       priority, status, triggered_by_alert_at, active, "
            "       created_at, updated_at, closed_at "
            "  FROM maintenance_orders "
            " WHERE " + " AND ".join(clauses) +
            " ORDER BY created_at DESC LIMIT %s"
        )
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
        return [_row_to_order(r) for r in rows]

    def get_order(self, order_id: int) -> Optional[MaintenanceOrder]:
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT order_id, device_id, assigned_to, title, description,
                           priority, status, triggered_by_alert_at, active,
                           created_at, updated_at, closed_at
                      FROM maintenance_orders
                     WHERE order_id = %s
                    """,
                    (order_id,),
                )
                row = cur.fetchone()
        return _row_to_order(row) if row else None

    def create_order(
        self,
        *,
        device_id: str,
        title: str,
        description: Optional[str] = None,
        priority: str = "normal",
        assigned_to: Optional[str] = None,
        triggered_by_alert_at: Optional[datetime] = None,
        actor_id: Optional[str] = None,
    ) -> Optional[MaintenanceOrder]:
        """주문 1건 INSERT + audit_log. device_id FK 위반 시 psycopg.errors.ForeignKeyViolation.

        priority 가 유효 범위가 아니면 CHECK 위반으로 DB 에서 거부 — 클라이언트 입력 검증.
        """
        if priority not in ORDER_PRIORITIES:
            raise ValueError(f"invalid priority: {priority}")

        with self._lock, self._conn.transaction():
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO maintenance_orders (
                        device_id, assigned_to, title, description, priority, status,
                        triggered_by_alert_at
                    ) VALUES (%s, %s, %s, %s, %s, 'open', %s)
                    RETURNING order_id, device_id, assigned_to, title, description,
                              priority, status, triggered_by_alert_at, active,
                              created_at, updated_at, closed_at
                    """,
                    (
                        device_id, assigned_to, title, description, priority,
                        triggered_by_alert_at,
                    ),
                )
                row = cur.fetchone()
            if row is None:
                return None
            order = _row_to_order(row)

            audit_actions.log(
                self._conn,
                actor_id=actor_id,
                action=AuditAction.ORDER_CREATED,
                target_type=audit_actions.TARGET_TYPE_ORDER,
                target_id=str(order.order_id),
                meta={
                    "device_id": device_id,
                    "priority": priority,
                    "title": title,
                    "triggered_by_alert_at": (
                        triggered_by_alert_at.isoformat() if triggered_by_alert_at else None
                    ),
                },
            )
            return order

    def update_status(
        self,
        order_id: int,
        new_status: str,
        *,
        actor_id: Optional[str] = None,
    ) -> bool:
        """상태 전이 + audit_log. 동일 상태로 갈 때는 no-op (False 반환).

        'closed' 로 전이하면 DB 트리거가 closed_at 자동 설정.
        반환값: 실제 변경 여부.
        """
        if new_status not in ORDER_STATUSES:
            raise ValueError(f"invalid status: {new_status}")

        with self._lock, self._conn.transaction():
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT status FROM maintenance_orders WHERE order_id = %s FOR UPDATE",
                    (order_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return False
                from_status = row[0]
                if from_status == new_status:
                    return False

                cur.execute(
                    "UPDATE maintenance_orders SET status = %s WHERE order_id = %s",
                    (new_status, order_id),
                )

            # audit: closed 는 별도 action, 그 외는 STATUS_CHANGED.
            if new_status == "closed":
                action = AuditAction.ORDER_CLOSED
            elif new_status == "cancelled":
                action = AuditAction.ORDER_CANCELLED
            else:
                action = AuditAction.ORDER_STATUS_CHANGED

            audit_actions.log(
                self._conn,
                actor_id=actor_id,
                action=action,
                target_type=audit_actions.TARGET_TYPE_ORDER,
                target_id=str(order_id),
                meta={"from": from_status, "to": new_status},
            )
            return True

    def close_order(self, order_id: int, *, actor_id: Optional[str] = None) -> bool:
        """update_status(..., 'closed') 의 편의 alias."""
        return self.update_status(order_id, "closed", actor_id=actor_id)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _row_to_order(row: Iterable[Any]) -> MaintenanceOrder:
    (
        order_id, device_id, assigned_to, title, description,
        priority, status, triggered_by_alert_at, active,
        created_at, updated_at, closed_at,
    ) = row
    return MaintenanceOrder(
        order_id=order_id,
        device_id=device_id,
        assigned_to=assigned_to,
        title=title,
        description=description,
        priority=priority,
        status=status,
        triggered_by_alert_at=triggered_by_alert_at,
        active=active,
        created_at=created_at,
        updated_at=updated_at,
        closed_at=closed_at,
    )


def create_default() -> MaintenanceOrderService:
    """config 보고 적절한 백엔드 선택. DeviceService 와 동일 정책."""
    from models_core import config

    if not config.DB_ENABLED:
        return NullMaintenanceOrderService()
    try:
        return TimescaleMaintenanceOrderService(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
        )
    except Exception as e:
        print(
            f"[maintenance_order_service] DB unavailable ({type(e).__name__}: {e}) - "
            f"falling back to Null.",
            flush=True,
        )
        return NullMaintenanceOrderService()

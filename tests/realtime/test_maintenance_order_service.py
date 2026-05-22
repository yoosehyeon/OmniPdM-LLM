"""MaintenanceOrderService unit + integration tests — Tier 1.5 P1-d.

Unit test (외부 의존 0):
- NullMaintenanceOrderService 의 protocol 호환 + 모든 메서드 no-op 동작.
- ORDER_STATUSES / ORDER_PRIORITIES 상수와 schema 동기.
- TimescaleMaintenanceOrderService 의 입력 검증 — invalid priority / invalid status.

Integration test (OMNIPDM_TEST_DB_URL 설정 시):
- 실 TimescaleDB + 001_cmms_schema.sql 적용 상태.
- conftest.py 의 db_conn fixture 가 BEGIN → 테스트 → ROLLBACK 으로 격리.
- DB 측 제약 (FK, CHECK, trigger) 검증 — service 가 별도 connection 으로
  autocommit=True 라 db_conn 의 ROLLBACK 으로 격리 안 됨. service-level
  end-to-end (create → list → close round-trip + audit_log 검증) 통합 검증은
  service constructor 의 connection injection 리팩토링 또는 dedicated DSN +
  명시 cleanup 이 필요 — P1-d follow-up 으로 분리 (PRD §13.1.b 참조).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from services.realtime.maintenance_order_service import (
    ORDER_PRIORITIES,
    ORDER_STATUSES,
    NullMaintenanceOrderService,
    TimescaleMaintenanceOrderService,
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
class TestNullMaintenanceOrderService:
    def test_all_methods_callable(self):
        svc = NullMaintenanceOrderService()
        assert svc.list_orders() == []
        assert svc.list_orders(status="open", device_id="x", limit=10) == []
        assert svc.get_order(1) is None
        assert svc.create_order(device_id="x", title="t") is None
        assert svc.update_status(1, "closed") is False
        assert svc.close_order(1) is False
        svc.close()


class TestConstantsConsistency:
    def test_statuses_match_schema(self):
        # 001_cmms_schema.sql 의 CHECK 와 동기.
        assert set(ORDER_STATUSES) == {"open", "in_progress", "closed", "cancelled"}

    def test_priorities_match_schema(self):
        assert set(ORDER_PRIORITIES) == {"low", "normal", "high", "critical"}


class TestTimescaleServiceValidation:
    """psycopg.connect 를 mock 으로 대체해 service 의 클라이언트 측 검증 분기 확인.

    device_service 의 mock pattern 그대로 — DB 왕복 없이 service 내부 ValueError
    분기만 검증.
    """

    def _make_service_with_mock_conn(self) -> TimescaleMaintenanceOrderService:
        with patch("psycopg.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            svc = TimescaleMaintenanceOrderService(
                host="x", port=5432, dbname="x", user="x", password="x",
            )
        return svc

    def test_create_order_rejects_invalid_priority(self):
        svc = self._make_service_with_mock_conn()
        with pytest.raises(ValueError, match="invalid priority"):
            svc.create_order(device_id="d1", title="t", priority="ultra")

    def test_update_status_rejects_invalid_status(self):
        svc = self._make_service_with_mock_conn()
        with pytest.raises(ValueError, match="invalid status"):
            svc.update_status(1, "foo")

    def test_list_orders_rejects_invalid_status(self):
        svc = self._make_service_with_mock_conn()
        with pytest.raises(ValueError, match="invalid status"):
            svc.list_orders(status="not_a_status")

    def test_close_order_alias_invokes_update_status(self):
        # close_order 는 update_status('closed', ...) 의 alias — 동일 SQL 호출 경로.
        svc = self._make_service_with_mock_conn()
        # cursor().__enter__().fetchone() 가 None 을 반환하면 update_status 가 False.
        cursor_mock = svc._conn.cursor.return_value.__enter__.return_value
        cursor_mock.fetchone.return_value = None
        assert svc.close_order(99) is False


# ---------------------------------------------------------------------------
# Integration tests — schema 제약 (FK / CHECK / trigger) 검증.
# Service-level end-to-end 는 별도 DSN + cleanup 필요 — P1-d follow-up.
# ---------------------------------------------------------------------------
class TestMaintenanceOrderSchemaIntegration:
    def test_create_via_raw_sql_succeeds(self, db_conn):
        # device 가 존재해야 FK 위반 안 남. 미리 INSERT.
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO devices (device_id, dataset_key, name, plant_id, active)
                VALUES ('mo-test-dev-01', 'ai4i_cnn', 'mo-test-dev-01', 'plant-01', TRUE)
                ON CONFLICT (device_id) DO NOTHING
                """
            )
            cur.execute(
                """
                INSERT INTO maintenance_orders (device_id, title, priority, status)
                VALUES ('mo-test-dev-01', 'integration-test', 'high', 'open')
                RETURNING order_id, status, priority
                """
            )
            row = cur.fetchone()
            assert row is not None
            assert row[1] == "open"
            assert row[2] == "high"

    def test_fk_violation_on_missing_device(self, db_conn):
        import psycopg
        with db_conn.cursor() as cur:
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                cur.execute(
                    """
                    INSERT INTO maintenance_orders (device_id, title)
                    VALUES ('nonexistent-mo-dev', 'test')
                    """
                )

    def test_check_constraint_blocks_bad_status(self, db_conn):
        import psycopg
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO devices (device_id, dataset_key, name, plant_id, active)
                VALUES ('mo-test-dev-02', 'ai4i_cnn', 'mo-test-dev-02', 'plant-01', TRUE)
                ON CONFLICT (device_id) DO NOTHING
                """
            )
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    """
                    INSERT INTO maintenance_orders (device_id, title, status)
                    VALUES ('mo-test-dev-02', 'test', 'invalid_status')
                    """
                )

    def test_closed_at_trigger_fires(self, db_conn):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO devices (device_id, dataset_key, name, plant_id, active)
                VALUES ('mo-test-dev-03', 'ai4i_cnn', 'mo-test-dev-03', 'plant-01', TRUE)
                ON CONFLICT (device_id) DO NOTHING
                """
            )
            cur.execute(
                """
                INSERT INTO maintenance_orders (device_id, title, status)
                VALUES ('mo-test-dev-03', 'closed-trigger-test', 'open')
                RETURNING order_id
                """
            )
            order_id = cur.fetchone()[0]
            cur.execute(
                "UPDATE maintenance_orders SET status = 'closed' WHERE order_id = %s",
                (order_id,),
            )
            cur.execute(
                "SELECT status, closed_at FROM maintenance_orders WHERE order_id = %s",
                (order_id,),
            )
            row = cur.fetchone()
            assert row[0] == "closed"
            assert row[1] is not None  # 트리거가 NOW() 로 채움

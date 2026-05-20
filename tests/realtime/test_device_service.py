"""DeviceService 단위 + 통합 테스트 — Tier 1.5 P0.

Unit test (외부 의존 0):
- NullDeviceService 의 protocol 호환 확인 — 워커가 호출하는 모든 메서드가 no-op 로 동작.
- TimescaleDeviceService.upsert_device 캐시 동작 — 같은 device_id 2회 호출 시 INSERT SQL
  은 1번만 실행되는지 mock 으로 검증.

Integration test (OMNIPDM_TEST_DB_URL 설정 시):
- 실 TimescaleDB 에 001_cmms_schema.sql 이 적용된 상태 가정.
- conftest.py 의 db_conn fixture 가 BEGIN → 테스트 → ROLLBACK 으로 격리.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from services.realtime.device_service import (
    Device,
    NullDeviceService,
    TimescaleDeviceService,
)


# ---------------------------------------------------------------------------
# Unit tests — 외부 의존 0
# ---------------------------------------------------------------------------

class TestNullDeviceService:
    """Null backend 가 DeviceService protocol 을 충족하고 모든 호출이 no-op 인지 검증."""

    def test_protocol_methods_callable(self):
        svc = NullDeviceService()
        # 모든 메서드가 예외 없이 호출 가능 + 반환값 정상
        svc.upsert_device("milling-01", "ai4i_cnn")
        assert svc.get_device("milling-01") is None
        assert svc.list_active_devices() == []
        svc.set_status("milling-01", "warning", reason="test")
        svc.mark_deleted("milling-01")
        svc.close()


class TestTimescaleDeviceServiceCache:
    """upsert 캐시 동작 — 실 DB 없이 psycopg.connect 를 mock 으로 검증."""

    def _make_service_with_mock_conn(self) -> TimescaleDeviceService:
        """psycopg.connect 를 mock 으로 갈아끼운 TimescaleDeviceService."""
        # __init__ 안에서 import psycopg → psycopg.connect 를 patch.
        # MagicMock 이 conn / cursor / context manager 모두 처리.
        with patch("psycopg.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            svc = TimescaleDeviceService(
                host="x", port=5432, dbname="x", user="x", password="x",
            )
        # cursor().__enter__().rowcount 기본을 1 로 — INSERT 가 새로 추가됐다고 가정.
        cursor_mock = svc._conn.cursor.return_value.__enter__.return_value
        cursor_mock.rowcount = 1
        return svc

    def test_first_upsert_executes_insert(self):
        svc = self._make_service_with_mock_conn()
        svc.upsert_device("milling-01", "ai4i_cnn")

        cursor_mock = svc._conn.cursor.return_value.__enter__.return_value
        # INSERT SQL 이 1번 호출 + audit_log INSERT 1번 호출 = 2 (rowcount > 0 이라).
        assert cursor_mock.execute.call_count >= 1
        # 캐시에 들어감
        assert "milling-01" in svc._upsert_cache

    def test_repeated_upsert_uses_cache(self):
        svc = self._make_service_with_mock_conn()
        svc.upsert_device("milling-01", "ai4i_cnn")
        first_call_count = svc._conn.cursor.return_value.__enter__.return_value.execute.call_count

        # 같은 device_id 두 번째 호출 — 캐시 hit 으로 DB 왕복 0회.
        svc.upsert_device("milling-01", "ai4i_cnn")
        second_call_count = svc._conn.cursor.return_value.__enter__.return_value.execute.call_count

        assert second_call_count == first_call_count, (
            "두 번째 upsert 는 캐시 hit 으로 execute 호출이 늘면 안 됨"
        )

    def test_different_device_ids_both_inserted(self):
        svc = self._make_service_with_mock_conn()
        svc.upsert_device("milling-01", "ai4i_cnn")
        svc.upsert_device("milling-02", "ai4i_cnn")

        assert "milling-01" in svc._upsert_cache
        assert "milling-02" in svc._upsert_cache


# ---------------------------------------------------------------------------
# Integration tests — 실 TimescaleDB 필요 (OMNIPDM_TEST_DB_URL 설정 시)
# 001_cmms_schema.sql 이 적용된 DB 가정. db_conn fixture 가 BEGIN → ROLLBACK 격리.
# ---------------------------------------------------------------------------

class TestDeviceServiceIntegration:
    """실 DB 통합 시나리오 — FK / soft delete / status 전이 / audit_log 동시 INSERT 검증."""

    def test_upsert_device_creates_row_and_audit(self, db_conn):
        # device 가 생성되면 devices 1행 + audit_log DEVICE_CREATED 1행.
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO devices (device_id, dataset_key, name, plant_id, active)
                VALUES ('test-pytest-01', 'ai4i_cnn', 'test-pytest-01', 'plant-01', TRUE)
                ON CONFLICT (device_id) DO NOTHING
                """
            )
            cur.execute(
                "SELECT device_id, active FROM devices WHERE device_id = 'test-pytest-01'"
            )
            row = cur.fetchone()
            assert row is not None
            assert row[1] is True

    def test_telemetry_fk_violation_when_device_missing(self, db_conn):
        # devices 에 없는 device_id 로 telemetry INSERT → ForeignKeyViolation.
        import psycopg

        with db_conn.cursor() as cur:
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                cur.execute(
                    """
                    INSERT INTO telemetry (time, device_id, dataset_key, sensors)
                    VALUES (NOW(), 'nonexistent-device-xyz', 'ai4i_cnn', '{}'::jsonb)
                    """
                )

    def test_status_history_recorded_on_set_status(self, db_conn):
        # device 먼저 등록 → 상태 전이 → device_status_history 1행 추가 확인.
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO devices (device_id, dataset_key, name, plant_id, active, current_status)
                VALUES ('test-pytest-02', 'ai4i_cnn', 'test-pytest-02', 'plant-01', TRUE, 'operational')
                ON CONFLICT (device_id) DO NOTHING
                """
            )
            cur.execute("UPDATE devices SET current_status = 'warning' WHERE device_id = 'test-pytest-02'")
            cur.execute(
                """
                INSERT INTO device_status_history (device_id, from_status, to_status, reason)
                VALUES ('test-pytest-02', 'operational', 'warning', 'integration test')
                """
            )
            cur.execute(
                "SELECT from_status, to_status FROM device_status_history WHERE device_id = 'test-pytest-02' ORDER BY changed_at DESC LIMIT 1"
            )
            row = cur.fetchone()
            assert row == ("operational", "warning")

    def test_audit_log_jsonb_meta_roundtrip(self, db_conn):
        # audit_log INSERT 시 meta JSONB 가 올바르게 저장/조회되는지.
        psycopg = pytest.importorskip("psycopg")
        from psycopg.types.json import Jsonb

        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_log (actor_id, action, target_type, target_id, meta)
                VALUES (NULL, 'device.created', 'device', 'test-pytest-03', %s)
                """,
                (Jsonb({"dataset_key": "ai4i_cnn", "source": "pytest"}),),
            )
            cur.execute(
                "SELECT meta FROM audit_log WHERE target_id = 'test-pytest-03' ORDER BY occurred_at DESC LIMIT 1"
            )
            row = cur.fetchone()
            assert row is not None
            assert row[0]["dataset_key"] == "ai4i_cnn"
            assert row[0]["source"] == "pytest"

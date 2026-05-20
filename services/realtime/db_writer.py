"""Pluggable DbWriter — Tier 1 T1-03.

설계 의도 (Notifier 와 동일 pluggable 패턴):
- 워커는 DB 종류를 모른다. DbWriter protocol 만 받아서 호출한다.
- 기본 구현은 NullDbWriter — 외부 의존성 0, broker/DB 없이도 smoke 가능.
- 실제 운영 시 TimescaleDbWriter 를 주입 (psycopg + TimescaleDB hypertable).
- 향후 다른 DB (Influx / S3 / Parquet 등) 도 같은 protocol 로 추가 가능.

스키마:
- telemetry      : raw 센서 메시지 (1 publish = 1 row)
- predictions    : 워커 추론 결과 + 위험 등급
- alerts         : Notifier 발송 시도 결과 (SENT/FILTERED/RATE_LIMITED/FAILED)

infra/timescaledb/init.sql 참조.

Thread safety:
- psycopg connection 은 thread-safe 하지 않다 (cursor 단위 동시 사용 불가).
- TimescaleDbWriter 는 단일 connection + RLock 으로 paho-mqtt callback thread 와
  publish thread 가 안전하게 공유할 수 있게 한다.
- 다중 워커 / 고 TPS 진입 시 psycopg.pool.ConnectionPool 도입.

실패 정책:
- DB insert 실패는 워커 메인 루프를 죽이지 않는다 (운영 안정성).
- 실패 카운터 (fail_count) 만 누적 + print 로 로그. 호출자는 fire-and-forget.
"""

from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from services.schemas import PredictionResult, RiskResult


@runtime_checkable
class DbWriter(Protocol):
    """이력 저장 채널 추상화. 새 백엔드는 이 protocol 을 따르기만 하면 된다."""

    def write_telemetry(
        self,
        device_id: str,
        dataset_key: str,
        sensors: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> None: ...

    def write_prediction(
        self,
        device_id: str,
        dataset_key: str,
        pred: PredictionResult,
        risk: RiskResult,
        timestamp: Optional[datetime] = None,
    ) -> None: ...

    def write_alert(
        self,
        device_id: str,
        dataset_key: str,
        risk: RiskResult,
        channel: str,
        notify_result: str,
        payload_text: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None: ...

    def close(self) -> None: ...


def _now() -> datetime:
    return datetime.now(timezone.utc)


class NullDbWriter:
    """기본 백엔드 — 아무것도 안 함. broker/DB 없이 smoke 가능.

    워커가 DbWriter 호출하는 코드 경로를 막지 않으면서 zero-cost no-op.
    """

    def write_telemetry(self, device_id, dataset_key, sensors, timestamp=None): ...
    def write_prediction(self, device_id, dataset_key, pred, risk, timestamp=None): ...
    def write_alert(self, device_id, dataset_key, risk, channel, notify_result, payload_text=None, timestamp=None): ...
    def close(self) -> None: ...


class TimescaleDbWriter:
    """TimescaleDB (Postgres + hypertable) 백엔드.

    psycopg v3 사용 (binary). 단일 connection + RLock 으로 thread-safe 보장.
    PoC 단계 단일 워커 가정 — 다중 워커 / 고 TPS 시 connection pool (psycopg_pool) 도입.

    autocommit=True: 각 insert 가 즉시 commit → 워커 비정상 종료 시 손실 최소화.
    """

    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        connect_timeout: int = 5,
        application_name: str = "omnipdm-worker",
    ) -> None:
        # psycopg 는 lazy import — TimescaleDbWriter 를 인스턴스화 안 하면 import 자체 안 됨.
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
        self.fail_count = 0
        self.ok_count = 0

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def write_telemetry(
        self,
        device_id: str,
        dataset_key: str,
        sensors: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> None:
        ts = timestamp or _now()
        self._execute(
            """
            INSERT INTO telemetry (time, device_id, dataset_key, sensors)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (time, device_id, dataset_key) DO NOTHING
            """,
            (ts, device_id, dataset_key, self._json(sensors)),
            op="write_telemetry",
        )

    def write_prediction(
        self,
        device_id: str,
        dataset_key: str,
        pred: PredictionResult,
        risk: RiskResult,
        timestamp: Optional[datetime] = None,
    ) -> None:
        ts = timestamp or _now()
        # top_contributors 는 [(name, value), ...] tuple 리스트 — JSON 직렬화 가능 형태로 변환.
        top = [[str(n), float(v)] for n, v in (pred.top_contributors or [])]
        self._execute(
            """
            INSERT INTO predictions (
                time, device_id, dataset_key, model_name, model_mode,
                failure_probability, anomaly_score, rul_norm,
                risk_score, risk_level, risk_method, top_contributors
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (time, device_id, dataset_key) DO NOTHING
            """,
            (
                ts, device_id, dataset_key, pred.model_name, pred.model_mode,
                float(pred.failure_probability), float(pred.anomaly_score), float(pred.rul_norm),
                float(risk.risk_score), risk.risk_level, risk.method,
                self._json(top),
            ),
            op="write_prediction",
        )

    def write_alert(
        self,
        device_id: str,
        dataset_key: str,
        risk: RiskResult,
        channel: str,
        notify_result: str,
        payload_text: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        ts = timestamp or _now()
        payload_jsonb = self._json({"text": payload_text}) if payload_text else self._json({})
        self._execute(
            """
            INSERT INTO alerts (
                time, device_id, dataset_key, risk_level, risk_score,
                channel, notify_result, payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (time, device_id, dataset_key) DO NOTHING
            """,
            (
                ts, device_id, dataset_key, risk.risk_level, float(risk.risk_score),
                channel, notify_result, payload_jsonb,
            ),
            op="write_alert",
        )

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _execute(self, sql: str, params: tuple, op: str) -> None:
        try:
            with self._lock:
                with self._conn.cursor() as cur:
                    cur.execute(sql, params)
            self.ok_count += 1
        except Exception as e:
            # 워커 루프는 살린다 — insert 1건 실패가 모니터링 시스템을 죽이면 안 됨.
            self.fail_count += 1
            print(f"[TimescaleDbWriter] {op} failed: {type(e).__name__}: {e}", flush=True)

    @staticmethod
    def _json(obj: Any) -> str:
        import json as _json
        return _json.dumps(obj, ensure_ascii=False, default=str)

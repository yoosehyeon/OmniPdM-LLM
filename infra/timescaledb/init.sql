-- OmniPdM TimescaleDB schema — PRD v6.3 §10 T1-03.
-- 컨테이너 최초 기동 시 1회 실행 (POSTGRES_DB=omnipdm 안에서).
-- 이후 마이그레이션이 필요하면 alembic 도입 검토.

-- TimescaleDB extension 활성화 (이미지에 기본 설치됨)
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =====================================================================
-- 1) telemetry — 모든 raw 센서 메시지 (MQTT publish 1건 = 1 row)
-- PK 에 dataset_key 포함: 동일 device 가 향후 다중 센서 그룹(예: vibration vs temperature)
-- 을 발행할 수 있게 확장성 확보. 첫 컬럼은 time → TimescaleDB chunk 최적.
-- =====================================================================
CREATE TABLE IF NOT EXISTS telemetry (
    time           TIMESTAMPTZ      NOT NULL,
    device_id      TEXT             NOT NULL,
    dataset_key    TEXT             NOT NULL,
    sensors        JSONB            NOT NULL,            -- 원본 센서 dict 그대로
    PRIMARY KEY (time, device_id, dataset_key)
);

SELECT create_hypertable('telemetry', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS telemetry_device_time_idx
    ON telemetry (device_id, time DESC);

-- =====================================================================
-- 2) predictions — 워커가 추론한 결과 + 위험 등급
-- =====================================================================
CREATE TABLE IF NOT EXISTS predictions (
    time                 TIMESTAMPTZ   NOT NULL,
    device_id            TEXT          NOT NULL,
    dataset_key          TEXT          NOT NULL,
    model_name           TEXT          NOT NULL,
    model_mode           TEXT          NOT NULL
        CHECK (model_mode IN ('lite', 'full')),
    failure_probability  DOUBLE PRECISION,
    anomaly_score        DOUBLE PRECISION,
    rul_norm             DOUBLE PRECISION,
    risk_score           DOUBLE PRECISION NOT NULL,
    risk_level           TEXT          NOT NULL
        CHECK (risk_level IN ('Critical', 'Warning', 'Advisory', 'Normal')),
    risk_method          TEXT          NOT NULL,
    top_contributors     JSONB,                           -- [(name, value), ...]
    PRIMARY KEY (time, device_id, dataset_key)
);

SELECT create_hypertable('predictions', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS predictions_device_time_idx
    ON predictions (device_id, time DESC);
CREATE INDEX IF NOT EXISTS predictions_risk_level_time_idx
    ON predictions (risk_level, time DESC);

-- =====================================================================
-- 3) alerts — Notifier 가 실제 발송 시도한 알람 이력
-- notify_result 는 NotifyResult enum 4종 (services/realtime/notifier.py 참조)
-- =====================================================================
CREATE TABLE IF NOT EXISTS alerts (
    time         TIMESTAMPTZ   NOT NULL,
    device_id    TEXT          NOT NULL,
    dataset_key  TEXT          NOT NULL,
    risk_level   TEXT          NOT NULL
        CHECK (risk_level IN ('Critical', 'Warning', 'Advisory', 'Normal')),
    risk_score   DOUBLE PRECISION NOT NULL,
    channel      TEXT          NOT NULL,                 -- 'stdout' / 'telegram' / 'email' / ...
    notify_result TEXT         NOT NULL
        CHECK (notify_result IN ('SENT', 'FILTERED', 'RATE_LIMITED', 'FAILED')),
    payload      JSONB,                                  -- 발송 본문 (포맷된 텍스트 등)
    PRIMARY KEY (time, device_id, dataset_key)
);

SELECT create_hypertable('alerts', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS alerts_device_time_idx
    ON alerts (device_id, time DESC);

-- =====================================================================
-- 4) 보존 정책 (PoC 기본 — 운영 시 조정)
-- =====================================================================
-- telemetry: 90일 보존 (raw 가 무겁다)
-- predictions: 1년 보존
-- alerts: 무기한 (감사 추적)
SELECT add_retention_policy('telemetry',   INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('predictions', INTERVAL '365 days', if_not_exists => TRUE);

-- =====================================================================
-- 5) Compression — telemetry 디스크 절감 (보존 정책과 함께 동작)
-- 7일 이상 된 chunk 를 압축. 운영 측정 결과 raw insert 후 1주일 미접근 데이터에
-- 효과적 (보통 70~90% 디스크 절감).
-- segmentby = device_id: 같은 기기 데이터를 한 group 으로 정렬 → 압축률 + 쿼리 성능
-- =====================================================================
ALTER TABLE telemetry SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id'
);
SELECT add_compression_policy('telemetry', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE predictions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id'
);
SELECT add_compression_policy('predictions', INTERVAL '30 days', if_not_exists => TRUE);

-- alerts 는 무기한 보존 + 상대적으로 적은 volume → 압축 보류 (필요 시 추가).

-- =====================================================================
-- 권한 (PoC — omnipdm 사용자가 owner. Grafana 등 read-only role 은 외부 노출 시점에 도입)
-- =====================================================================
-- CREATE ROLE grafana_reader LOGIN PASSWORD '...';
-- GRANT CONNECT ON DATABASE omnipdm TO grafana_reader;
-- GRANT USAGE ON SCHEMA public TO grafana_reader;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_reader;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public
--   GRANT SELECT ON TABLES TO grafana_reader;

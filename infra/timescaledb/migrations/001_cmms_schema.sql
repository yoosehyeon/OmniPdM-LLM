-- OmniPdM CMMS Tier 1.5 P0 — PRD v6.3 §13.1.b T1.5 사전 의사결정 v1.0.
--
-- 적용 시점: docker compose 첫 기동 시 /docker-entrypoint-initdb.d/01_cmms_schema.sql 로
-- 자동 실행 (00_init.sql 다음 순서). 이미 init.sql 이 telemetry/predictions/alerts 3
-- hypertable 을 생성해 둔 상태 위에서 동작한다.
--
-- 핵심 결정 (PRD §13.1.b T1.5 결정 7건 + 외부 리뷰 보완 흡수):
--   - device_id 는 TEXT PK (기존 'milling-01' 식별자 호환). hypertable → regular table
--     FK 는 ON DELETE RESTRICT 강결합 (소속 변경은 soft delete 로).
--   - users / devices / maintenance_orders 모두 active BOOLEAN soft delete.
--   - users.role: admin/operator/viewer 3-role (CHECK constraint — 안정적).
--   - maintenance_orders.status: open/in_progress/closed/cancelled (CHECK — 안정적).
--   - devices.current_status: operational/warning/critical/maintenance/offline. CHECK 없음
--     — 운영 중 상태 추가 가능성 유지 (audit_log.action 과 동일 정책).
--   - audit_log.action: CHECK 없음. services/audit_actions.py 의 Python enum 으로 관리.
--   - scoping (plant_id / equipment_group_id) 는 컬럼만 추가. 쿼리 필터링은 P1 이후.
--   - updated_at / closed_at: DB 트리거로 자동 갱신 (외부 리뷰 채택 — 애플리케이션 누락
--     버그 클래스 자체 제거).
--
-- 멱등성: CREATE 는 IF NOT EXISTS, INSERT 는 ON CONFLICT DO NOTHING. ALTER TABLE FK 와
-- 트리거는 pg_constraint / pg_trigger 존재 검사로 가드. docker-entrypoint-initdb.d 는
-- 최초 기동 1회만 실행되지만, 수동 `psql -f` 재실행도 안전.

-- =====================================================================
-- 0) 공통 트리거 함수 — updated_at 자동 NOW()
-- 외부 리뷰 채택. CREATE OR REPLACE 라 멱등.
-- =====================================================================
CREATE OR REPLACE FUNCTION omnipdm_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =====================================================================
-- 1) users — 인증/RBAC 주체. TimescaleDB 동일 인스턴스 (T1.5 결정 #2)
-- password_hash: P1 에서 werkzeug.security.generate_password_hash 적용.
-- =====================================================================
CREATE TABLE IF NOT EXISTS users (
    user_id        TEXT             PRIMARY KEY,
    email          TEXT             NOT NULL UNIQUE,
    password_hash  TEXT             NOT NULL,
    display_name   TEXT             NOT NULL,
    role           TEXT             NOT NULL
        CHECK (role IN ('admin', 'operator', 'viewer')),
    active         BOOLEAN          NOT NULL DEFAULT TRUE,
    created_at     TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

-- =====================================================================
-- 2) devices — 설비 마스터. TEXT PK (기존 telemetry.device_id 호환, T1.5 결정 #1)
-- current_status 는 CHECK 없음 — 운영 중 'commissioning' 등 추가 가능성.
-- plant_id / equipment_group_id 는 scoping placeholder (결정 #6) — 쿼리 필터는 P1+.
-- =====================================================================
CREATE TABLE IF NOT EXISTS devices (
    device_id           TEXT             PRIMARY KEY,
    dataset_key         TEXT             NOT NULL,
    name                TEXT             NOT NULL,
    plant_id            TEXT             NOT NULL DEFAULT 'plant-01',
    equipment_group_id  TEXT             NULL,
    current_status      TEXT             NOT NULL DEFAULT 'operational',
    active              BOOLEAN          NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

-- devices(current_status) — 외부 리뷰 채택. Fleet Overview 의 'Critical 인 device 수'
-- 같은 집계 쿼리에서 사용.
CREATE INDEX IF NOT EXISTS devices_current_status_idx
    ON devices (current_status);

-- =====================================================================
-- 3) maintenance_orders — 정비 워크플로. assigned_to → users(user_id) FK.
-- status / priority 는 안정적이라 CHECK 적용.
-- =====================================================================
CREATE TABLE IF NOT EXISTS maintenance_orders (
    order_id      BIGSERIAL        PRIMARY KEY,
    device_id     TEXT             NOT NULL REFERENCES devices(device_id) ON DELETE RESTRICT,
    assigned_to   TEXT             NULL REFERENCES users(user_id) ON DELETE SET NULL,
    title         TEXT             NOT NULL,
    description   TEXT             NULL,
    priority      TEXT             NOT NULL DEFAULT 'normal'
        CHECK (priority IN ('low', 'normal', 'high', 'critical')),
    status        TEXT             NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'in_progress', 'closed', 'cancelled')),
    triggered_by_alert_at TIMESTAMPTZ NULL,  -- 알람 자동 생성 시 원본 alert.time
    active        BOOLEAN          NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    closed_at     TIMESTAMPTZ      NULL
);

-- 외부 리뷰 채택 — 정비 큐 / 내 작업 목록 두 가지 핵심 조회 패턴.
CREATE INDEX IF NOT EXISTS maintenance_orders_device_status_idx
    ON maintenance_orders (device_id, status);
CREATE INDEX IF NOT EXISTS maintenance_orders_assigned_status_idx
    ON maintenance_orders (assigned_to, status);

-- closed_at 자동 채움 — status 가 'closed' 로 전이될 때만. 외부 리뷰 채택.
CREATE OR REPLACE FUNCTION omnipdm_set_closed_at()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'closed' AND (OLD.status IS DISTINCT FROM 'closed') AND NEW.closed_at IS NULL THEN
        NEW.closed_at := NOW();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =====================================================================
-- 4) device_status_history — 상태 머신 이력. 모든 상태 전이를 1 row 로 보존.
-- devices.current_status UPDATE 와 같은 트랜잭션 안에서 INSERT — device_service.set_status.
-- =====================================================================
CREATE TABLE IF NOT EXISTS device_status_history (
    history_id  BIGSERIAL        PRIMARY KEY,
    device_id   TEXT             NOT NULL REFERENCES devices(device_id) ON DELETE RESTRICT,
    from_status TEXT             NULL,        -- 최초 상태 진입 시 NULL
    to_status   TEXT             NOT NULL,
    actor_id    TEXT             NULL REFERENCES users(user_id) ON DELETE SET NULL,
    reason      TEXT             NULL,
    changed_at  TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS device_status_history_device_changed_idx
    ON device_status_history (device_id, changed_at DESC);

-- =====================================================================
-- 5) audit_log — 감사 로그 (NIST AC-6). action 은 CHECK 없음 — Python enum 으로 관리.
-- meta JSONB 로 임의 컨텍스트 (예: maintenance_order_id, old/new value 등).
-- =====================================================================
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id     BIGSERIAL        PRIMARY KEY,
    actor_id     TEXT             NULL REFERENCES users(user_id) ON DELETE SET NULL,
    action       TEXT             NOT NULL,
    target_type  TEXT             NOT NULL,         -- 'device' / 'order' / 'user' / 'session' ...
    target_id    TEXT             NULL,
    meta         JSONB            NOT NULL DEFAULT '{}'::jsonb,
    occurred_at  TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

-- 외부 리뷰 채택 — 최근 활동 조회 핵심 패턴.
CREATE INDEX IF NOT EXISTS audit_log_occurred_at_idx
    ON audit_log (occurred_at DESC);

-- =====================================================================
-- 6) 트리거 부착 — pg_trigger 존재 확인으로 재실행 안전.
-- =====================================================================
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'users_set_updated_at') THEN
        CREATE TRIGGER users_set_updated_at
            BEFORE UPDATE ON users
            FOR EACH ROW EXECUTE FUNCTION omnipdm_set_updated_at();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'devices_set_updated_at') THEN
        CREATE TRIGGER devices_set_updated_at
            BEFORE UPDATE ON devices
            FOR EACH ROW EXECUTE FUNCTION omnipdm_set_updated_at();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'maintenance_orders_set_updated_at') THEN
        CREATE TRIGGER maintenance_orders_set_updated_at
            BEFORE UPDATE ON maintenance_orders
            FOR EACH ROW EXECUTE FUNCTION omnipdm_set_updated_at();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'maintenance_orders_set_closed_at') THEN
        CREATE TRIGGER maintenance_orders_set_closed_at
            BEFORE UPDATE ON maintenance_orders
            FOR EACH ROW EXECUTE FUNCTION omnipdm_set_closed_at();
    END IF;
END $$;

-- =====================================================================
-- 7) backfill — 기존 telemetry 의 distinct device_id 를 devices 로 끌어올린다.
-- 이 단계 직후 hypertable 3종에 FK 를 거는 것이 안전 (orphan device_id 0건 보장).
-- 외부 리뷰 조정: 리뷰의 'legacy' 하드코딩 대신 실제 telemetry.dataset_key 그대로 사용.
-- =====================================================================
DO $$
DECLARE
    n_backfilled INTEGER;
BEGIN
    INSERT INTO devices (device_id, dataset_key, name, plant_id, active)
    SELECT DISTINCT ON (device_id)
        device_id,
        dataset_key,
        device_id,           -- 표시명은 일단 device_id 그대로. P1 에서 수정 UI 추가.
        'plant-01',
        TRUE
    FROM telemetry
    ORDER BY device_id, time DESC
    ON CONFLICT (device_id) DO NOTHING;

    GET DIAGNOSTICS n_backfilled = ROW_COUNT;
    RAISE NOTICE 'omnipdm: backfilled % devices from telemetry', n_backfilled;
END $$;

-- =====================================================================
-- 8) FK 추가 — telemetry / predictions / alerts → devices (강결합, ON DELETE RESTRICT)
-- pg_constraint 존재 확인으로 재실행 안전. backfill 이후이므로 무결성 위반 없음.
-- =====================================================================
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'telemetry_device_id_fk'
    ) THEN
        ALTER TABLE telemetry
            ADD CONSTRAINT telemetry_device_id_fk
            FOREIGN KEY (device_id) REFERENCES devices(device_id)
            ON DELETE RESTRICT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'predictions_device_id_fk'
    ) THEN
        ALTER TABLE predictions
            ADD CONSTRAINT predictions_device_id_fk
            FOREIGN KEY (device_id) REFERENCES devices(device_id)
            ON DELETE RESTRICT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'alerts_device_id_fk'
    ) THEN
        ALTER TABLE alerts
            ADD CONSTRAINT alerts_device_id_fk
            FOREIGN KEY (device_id) REFERENCES devices(device_id)
            ON DELETE RESTRICT;
    END IF;
END $$;

-- =====================================================================
-- 9) seed admin (PoC) — 첫 사용자. P1 에서 비밀번호 재설정 강제.
-- password_hash: placeholder. P1 에서 werkzeug.security.generate_password_hash 결과로 교체.
-- 의도적으로 hash 형태가 아닌 명백한 placeholder — 운영 배포 직전 누락을 발견하기 쉽게.
-- =====================================================================
INSERT INTO users (user_id, email, password_hash, display_name, role)
VALUES (
    'admin',
    'admin@omnipdm.local',
    'PLACEHOLDER_NOT_A_REAL_HASH_REPLACE_BEFORE_PRODUCTION',
    'OmniPdM Admin',
    'admin'
)
ON CONFLICT (user_id) DO NOTHING;

"""POST /api/analyze smoke test.

검증 범위:
- CSRF 헤더 없거나 잘못된 경우 403
- 잘못된 body (risk method, sensorValues 누락) → 400
- sequence dataset → 501 (graceful Not Implemented)
- unknown dataset → 400
- 잘못된 sensor key → 400 (dataset_catalog SSOT 검증)
- 정상 호출 → 200 + dict (실제 모델 load 가 무거워 ai4i scalar 만 시도)

실모델 inference 가 환경 (체크포인트 부재 등) 에 따라 500 도 나올 수 있으므로
test_analyze_success 는 200 또는 500 모두 허용 — 핵심은 CSRF/validation 경로가 통과해
service layer 까지 도달했다는 것.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def flask_client(monkeypatch):
    monkeypatch.setenv("OMNIPDM_AUTH_REQUIRED", "false")  # 가드 우회, CSRF 만 검증.
    from app import create_app
    return create_app().test_client()


def _csrf(client) -> str:
    return client.get("/api/csrf").get_json()["token"]


def test_analyze_csrf_missing_403(flask_client):
    resp = flask_client.post("/api/analyze", json={
        "dataset": "ai4i",
        "riskFusionMethod": "weighted",
        "sensorValues": {"Torque [Nm]": 40},
    })
    assert resp.status_code == 403
    assert resp.get_json()["error"] == "csrf_invalid"


def test_analyze_csrf_wrong_403(flask_client):
    resp = flask_client.post(
        "/api/analyze",
        json={"dataset": "ai4i", "sensorValues": {"x": 1}},
        headers={"X-CSRF-Token": "deadbeef" * 8},
    )
    assert resp.status_code == 403


def test_analyze_invalid_risk_method_400(flask_client):
    token = _csrf(flask_client)
    resp = flask_client.post(
        "/api/analyze",
        json={
            "dataset": "ai4i",
            "riskFusionMethod": "INVALID",
            "sensorValues": {"Torque [Nm]": 40},
        },
        headers={"X-CSRF-Token": token},
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "invalid_risk_fusion_method"


def test_analyze_missing_sensors_400(flask_client):
    token = _csrf(flask_client)
    resp = flask_client.post(
        "/api/analyze",
        json={"dataset": "ai4i", "sensorValues": {}},
        headers={"X-CSRF-Token": token},
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "sensorValues_required"


def test_analyze_sequence_pipeline_reachable(flask_client):
    """sequence dataset (cmapss-fd001) + 유효 model 조합이 LSTM 경로로 진입.

    체크포인트 부재 시 500, 정상 환경에서는 200. 핵심: 501 이 아니어야 함
    (P1 작업에서 sequence path 가 추가됨).
    """
    token = _csrf(flask_client)
    resp = flask_client.post(
        "/api/analyze",
        json={
            "dataset": "cmapss-fd001",
            "model": "bilstm",
            "riskFusionMethod": "weighted",
            "sequenceLength": 30,
            "sensorValues": {"T30 (HPC Temp)": 512},
        },
        headers={"X-CSRF-Token": token},
    )
    assert resp.status_code in (200, 500), (
        f"unexpected status {resp.status_code}: {resp.get_data(as_text=True)[:200]}"
    )


def test_analyze_sequence_invalid_model_combo_400(flask_client):
    token = _csrf(flask_client)
    resp = flask_client.post(
        "/api/analyze",
        json={
            "dataset": "cmapss-fd001",
            "model": "no-such-model",
            "riskFusionMethod": "weighted",
            "sensorValues": {"T30 (HPC Temp)": 512},
        },
        headers={"X-CSRF-Token": token},
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "unsupported_dataset_model_combo"


def test_analyze_sequence_invalid_length_400(flask_client):
    token = _csrf(flask_client)
    resp = flask_client.post(
        "/api/analyze",
        json={
            "dataset": "cmapss-fd001",
            "model": "bilstm",
            "riskFusionMethod": "weighted",
            "sequenceLength": 1,
            "sensorValues": {"T30 (HPC Temp)": 512},
        },
        headers={"X-CSRF-Token": token},
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "invalid_sequenceLength"


def test_analyze_unknown_dataset_400(flask_client):
    token = _csrf(flask_client)
    resp = flask_client.post(
        "/api/analyze",
        json={
            "dataset": "made-up",
            "riskFusionMethod": "weighted",
            "sensorValues": {"x": 1},
        },
        headers={"X-CSRF-Token": token},
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "unknown_dataset"


def test_analyze_invalid_sensor_keys_400(flask_client):
    token = _csrf(flask_client)
    resp = flask_client.post(
        "/api/analyze",
        json={
            "dataset": "ai4i",
            "riskFusionMethod": "weighted",
            "sensorValues": {"Made Up Sensor": 1.0},
        },
        headers={"X-CSRF-Token": token},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "invalid_sensor_keys"
    assert "Made Up Sensor" in data["invalid"]


def test_analyze_pipeline_reachable(flask_client):
    """CSRF + validation 모두 통과해 service layer 까지 도달.

    체크포인트 부재 시 500, 정상 환경에서는 200 — 두 경로 모두 허용.
    400/403/501 은 절대 나오면 안 됨 (validation 통과 확인).
    """
    token = _csrf(flask_client)
    resp = flask_client.post(
        "/api/analyze",
        json={
            "dataset": "ai4i",
            "riskFusionMethod": "weighted",
            "sensorValues": {
                "Torque [Nm]": 40.0,
                "Rotational Speed [rpm]": 1510.0,
                "Tool Wear [min]": 50.0,
                "Process Temp [K]": 308.0,
            },
        },
        headers={"X-CSRF-Token": token},
    )
    assert resp.status_code in (200, 500), f"unexpected status {resp.status_code}: {resp.get_data(as_text=True)[:200]}"

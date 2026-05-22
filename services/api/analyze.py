"""POST /api/analyze — 단일 분석 파이프라인 진입.

요청:
    {
      "dataset": "ai4i" | "cmapss-fd001" | "cwru" | ...,    (frontend dataset id)
      "model": "bilstm" | "dlinear" | "itransformer",        (학습 알고리즘, scalar 경로 무시)
      "riskFusionMethod": "weighted" | "noisy_or" | "max",
      "sensorValues": {"<sensor name>": <float>, ...},
      "sequenceLength": int (선택, LSTM 경로 — 미지원, 501 반환),
      "llmEnabled": bool (현재 env 로 제어, 무시)
    }

응답:
    services.analyze_service.AnalyzeService.run() 의 결과 dict 그대로 직렬화.
    feature_plot/sensor_plot 키는 항상 null (PRD v7.0 — React 가 raw dict 로 직접 렌더).

보안:
    상태 변경 endpoint 이므로 X-CSRF-Token 헤더 검증 (per-session token).
    /api/csrf 로 발급받은 토큰을 동봉해야 함.

성능:
    AnalyzeService 인스턴스는 (dataset_key, risk_method) 별로 app.extensions 에 캐시 —
    모델 load 비용 1회만 부담. gunicorn worker 별로 독립 캐시 보유.

설계 메모:
    Pydantic 미사용 — 5개 필드를 manual 검증하는 비용이 새 의존성 추가보다 작음.
    필드가 늘어나면 (>15) 재검토.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

from flask import Blueprint, Response, current_app, jsonify, request

from services.analyze_service import AnalyzeService
from services.api._sequence_synth import synthesize_sequence
from services.auth.sessions import verify_csrf_token
from services.dataset_catalog import list_datasets

logger = logging.getLogger(__name__)

bp_analyze = Blueprint("api_analyze", __name__, url_prefix="/api")

# UI dataset id + model algorithm → 백엔드 dataset_key.
# scalar (CNN/GBDT): 슬라이더 1 시점만으로 추론.
# sequence (LSTM/DLinear/iTransformer): sequenceLength × N 시퀀스 필요. PoC v0 는
# _sequence_synth.synthesize_sequence 로 슬라이더 + jitter 합성. v1 에서 실 이력 입력.
_SCALAR_KEYS: Dict[str, str] = {
    "ai4i": "ai4i_cnn",
    "cwru": "cwru_cnn",
}
# (dataset_id, model) → 백엔드 dataset_key. models_core/data_pipeline.py PIPELINE 과 키 동기화.
_SEQUENCE_KEYS: Dict[Tuple[str, str], str] = {
    ("cmapss-fd001", "bilstm"):       "cmapss_lstm_fd001",
    ("cmapss-fd001", "dlinear"):      "cmapss_dlinear_fd001",
    ("cmapss-fd001", "itransformer"): "cmapss_itransformer_fd001",
    ("cmapss-fd002", "bilstm"):       "cmapss_lstm_fd002",
    ("cmapss-fd002", "dlinear"):      "cmapss_dlinear_fd002",
    ("cmapss-fd002", "itransformer"): "cmapss_itransformer_fd002",
    ("n-cmapss", "bilstm"):           "ncmapss_lstm",
    ("n-cmapss", "dlinear"):          "ncmapss_dlinear",
    ("n-cmapss", "itransformer"):     "ncmapss_itransformer",
}
_SEQUENCE_DATASETS: Set[str] = {ds for (ds, _) in _SEQUENCE_KEYS.keys()}
_VALID_FUSIONS: Set[str] = {"weighted", "noisy_or", "max"}
_DEFAULT_SEQUENCE_LENGTH = 60
_MAX_SEQUENCE_LENGTH = 200

_CACHE_EXT_KEY = "analyze_services"


def _get_service(dataset_key: str, risk_method: str) -> AnalyzeService:
    """app.extensions 에 (dataset_key, risk_method) 별 AnalyzeService 캐시.

    Module-level dict 가 아니라 app context 에 두는 이유:
    - 테스트마다 create_app() 으로 새 캐시 시작 → fixture 간 모델 상태 누수 방지
    - 멀티-앱 (Flask app multiple instances) 에서도 충돌 없음
    """
    cache: Dict[Tuple[str, str], AnalyzeService] = current_app.extensions.setdefault(
        _CACHE_EXT_KEY, {}
    )
    key = (dataset_key, risk_method)
    svc = cache.get(key)
    if svc is None:
        svc = AnalyzeService(mode="lite", risk_method=risk_method, dataset_key=dataset_key)
        cache[key] = svc
        logger.info("AnalyzeService initialized: dataset_key=%s risk=%s", dataset_key, risk_method)
    return svc


def _allowed_sensors_for(dataset_id: str) -> Set[str]:
    """dataset_catalog 의 sensors[].name 을 추출 — sensor key 검증 SSOT."""
    for ds in list_datasets():
        if ds["id"] == dataset_id:
            return {s["name"] for s in ds["sensors"]}
    return set()


def _sensor_order_for(dataset_id: str) -> List[str]:
    """LSTM feature 순서 — catalog 정의 순서 그대로 (학습 시 컬럼 순서 가정)."""
    for ds in list_datasets():
        if ds["id"] == dataset_id:
            return [s["name"] for s in ds["sensors"]]
    return []


def _default_value_for(dataset_id: str, sensor_name: str) -> float:
    for ds in list_datasets():
        if ds["id"] == dataset_id:
            for s in ds["sensors"]:
                if s["name"] == sensor_name:
                    return float(s["defaultValue"])
    return 0.0


@bp_analyze.route("/analyze", methods=["POST"])
def analyze() -> Response | Tuple[Response, int]:
    # CSRF 검증 — React 는 /api/csrf 에서 받은 토큰을 X-CSRF-Token 헤더로 전달.
    if not verify_csrf_token(request.headers.get("X-CSRF-Token", "")):
        return jsonify({"error": "csrf_invalid"}), 403

    body = request.get_json(silent=True) or {}

    dataset = body.get("dataset", "")
    risk_method = body.get("riskFusionMethod", "weighted")
    sensor_values = body.get("sensorValues") or {}

    if risk_method not in _VALID_FUSIONS:
        return jsonify({"error": "invalid_risk_fusion_method", "value": risk_method}), 400

    if not isinstance(sensor_values, dict) or not sensor_values:
        return jsonify({"error": "sensorValues_required"}), 400

    # sensor key 검증 — dataset_catalog 의 sensors[].name 와 일치해야 함.
    # 모델 load 같은 무거운 작업 전에 차단해 비용/에러 메시지 명확성 모두 확보.
    allowed = _allowed_sensors_for(dataset)
    if not allowed:
        return jsonify({"error": "unknown_dataset", "value": dataset}), 400
    invalid_keys = sorted(set(sensor_values.keys()) - allowed)
    if invalid_keys:
        return jsonify({
            "error": "invalid_sensor_keys",
            "invalid": invalid_keys,
            "allowed": sorted(allowed),
        }), 400

    # JS 가 정수로 보낸 슬라이더도 float 로 정규화.
    payload = {k: float(v) for k, v in sensor_values.items()}

    # ─── Sequence (LSTM/DLinear/iTransformer) 경로 ─────────────────────────────
    if dataset in _SEQUENCE_DATASETS:
        model = body.get("model", "bilstm")
        dataset_key = _SEQUENCE_KEYS.get((dataset, model))
        if dataset_key is None:
            return jsonify({
                "error": "unsupported_dataset_model_combo",
                "dataset": dataset,
                "model": model,
            }), 400

        # sensorValues 누락 sensor 는 catalog defaultValue 로 보강.
        sensor_order = _sensor_order_for(dataset)
        for s in sensor_order:
            payload.setdefault(s, _default_value_for(dataset, s))

        raw_len = body.get("sequenceLength", _DEFAULT_SEQUENCE_LENGTH)
        try:
            seq_len = int(raw_len)
        except (TypeError, ValueError):
            return jsonify({"error": "invalid_sequenceLength", "value": raw_len}), 400
        if seq_len < 2 or seq_len > _MAX_SEQUENCE_LENGTH:
            return jsonify({
                "error": "invalid_sequenceLength",
                "value": seq_len,
                "allowed_range": [2, _MAX_SEQUENCE_LENGTH],
            }), 400

        sequence = synthesize_sequence(payload, sensor_order, seq_len)

        try:
            svc = _get_service(dataset_key, risk_method)
            result = svc.run_lstm(
                sequence=sequence,
                feature_names=sensor_order,
                asset_id="SIM",
                dataset_key=dataset_key,
            )
        except Exception:
            logger.exception("analyze lstm failed: dataset=%s model=%s", dataset, model)
            return jsonify({"error": "internal_server_error"}), 500

        return jsonify(result)

    # ─── Scalar (CNN/GBDT) 경로 ───────────────────────────────────────────────
    dataset_key = _SCALAR_KEYS.get(dataset)
    if dataset_key is None:
        return jsonify({"error": "unknown_dataset", "value": dataset}), 400

    try:
        svc = _get_service(dataset_key, risk_method)
        result = svc.run(payload)
    except Exception:
        logger.exception("analyze failed: dataset=%s risk=%s", dataset, risk_method)
        return jsonify({"error": "internal_server_error"}), 500

    return jsonify(result)

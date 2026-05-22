"""
P3-① LlmService.generate_stream() 검증 스크립트.

manual 디렉토리 — pytest 자동 수집 대상 아님. `python -m tests.manual.llm_stream_step` 로 직접 실행.

실행:
    python -m tests.manual.llm_stream_step

동작:
1. Warning 시나리오 기준 mock Prediction/Explanation/Risk 구성
2. generate_stream() 호출 → chunk 단위로 stdout 출력
3. StopIteration.value 로 최종 LlmResult 회수
4. 3-section 구조 검증

API 키가 없으면 fallback 경로(단일 chunk)가 검증된다.
"""
from __future__ import annotations

import sys

from services.llm_service import LlmService
from services.schemas import (
    ExplanationResult,
    FeatureExplanation,
    LlmResult,
    PredictionResult,
    RiskResult,
)


def build_mocks() -> tuple[dict, PredictionResult, ExplanationResult, RiskResult]:
    payload = {
        "air_temperature_k": 303.2,
        "process_temperature_k": 316.5,
        "rotational_speed_rpm": 1380.0,
        "torque_nm": 58.2,
        "tool_wear_min": 180.0,
    }
    pred = PredictionResult(
        dataset_key="ai4i_cnn",
        task_type="classification",
        model_name="cnn",
        model_mode="lite",
        predicted_label="Warning",
        failure_probability=0.42,
        anomaly_score=0.35,
        rul_norm=1.0,
        risk_score=0.48,
        risk_level="Warning",
    )
    risk = RiskResult(
        risk_score=0.48,
        risk_level="Warning",
        method="weighted",
        weights={"failure": 0.5, "anomaly": 0.3, "rul": 0.2},
    )
    exp = ExplanationResult(
        top_features=[
            FeatureExplanation("tool_wear_min", 180.0, 0.55, "up"),
            FeatureExplanation("torque_nm", 58.2, 0.30, "up"),
        ],
        summary="tool_wear와 torque 기여도 상승",
        interaction_summary=None,
        explanation_confidence=0.7,
    )
    return payload, pred, exp, risk


def main() -> int:
    svc = LlmService()
    payload, pred, exp, risk = build_mocks()

    print("=== streaming start ===", flush=True)
    gen = svc.generate_stream(payload, pred, exp, risk)
    accumulated = ""
    result: LlmResult | None = None
    try:
        while True:
            chunk = next(gen)
            accumulated += chunk
            sys.stdout.write(chunk)
            sys.stdout.flush()
    except StopIteration as stop:
        result = stop.value

    print("\n=== streaming end ===")
    assert result is not None, "generator 가 LlmResult 를 반환하지 않음"
    print(f"source:          {result.source}")
    print(f"used_fallback:   {result.used_fallback}")
    print(f"latency_ms:      {result.latency_ms}")
    print(f"prompt_hash:     {result.prompt_hash[:12] if result.prompt_hash else '-'}...")
    print(f"text len:        {len(result.text)}")
    print(f"accumulated len: {len(accumulated)}")

    required = ["상태 요약:", "의심 원인:", "권장 조치:"]
    missing = [s for s in required if s not in result.text]
    if missing:
        print(f"[FAIL] missing sections: {missing}")
        return 1

    print("[OK] 3-section structure verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())

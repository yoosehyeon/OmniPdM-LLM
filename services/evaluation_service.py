from __future__ import annotations

import os
from typing import Optional

from services.schemas import EvaluationResult, LlmResult, PredictionResult, RiskResult


class EvaluationService:
    """
    출력 구조 품질 점검.

    옵션:
        llm_service 주입 + 환경변수 ENABLE_LLM_JUDGE=1 인 경우
        LLM-as-Judge로 factuality_ok 를 보정한다.
        주입이 없거나 환경변수 미설정이면 기존 동작(factuality_ok=True) 유지.
    """

    def __init__(self, llm_service: Optional[object] = None) -> None:
        self.llm_service = llm_service
        self.judge_enabled = os.getenv("ENABLE_LLM_JUDGE", "0").strip() == "1"

    def evaluate(
        self,
        llm_result: LlmResult,
        pred: Optional[PredictionResult] = None,
        risk: Optional[RiskResult] = None,
    ) -> EvaluationResult:
        notes = []

        structure_ok = all(
            key in llm_result.text
            for key in ["상태 요약:", "의심 원인:", "권장 조치:"]
        )
        factuality_ok = True

        if (
            self.judge_enabled
            and self.llm_service is not None
            and pred is not None
            and risk is not None
            and not llm_result.used_fallback
        ):
            try:
                is_factual, judge_notes = self.llm_service.judge(llm_result, pred, risk)
                factuality_ok = bool(is_factual)
                if judge_notes:
                    notes.append(f"judge: {judge_notes}")
            except Exception as e:
                notes.append(f"judge_error:{type(e).__name__}")

        if not structure_ok:
            notes.append("필수 출력 섹션 누락")
        if llm_result.used_fallback:
            notes.append("fallback 응답 사용")
        if not llm_result.passed_guardrail:
            notes.append("guardrail 재작성 또는 fallback 적용")

        score = 1.0
        if not structure_ok:
            score -= 0.5
        if llm_result.used_fallback:
            score -= 0.1
        if not llm_result.passed_guardrail:
            score -= 0.2
        if not factuality_ok:
            score -= 0.2

        return EvaluationResult(
            structure_ok=structure_ok,
            factuality_ok=factuality_ok,
            overall_score=max(0.0, round(score, 4)),
            notes=notes,
        )

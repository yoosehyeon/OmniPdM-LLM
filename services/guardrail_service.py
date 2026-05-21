from __future__ import annotations

import re
from typing import List

from services.schemas import ExplanationResult, LSTMExplainResult, LlmResult, PredictionResult, RiskResult


class GuardrailService:
    """
    LLM 출력 후처리 및 안전 검증.
    """

    REQUIRED_SECTIONS = ["상태 요약:", "의심 원인:", "권장 조치:"]
    FORBIDDEN_PATTERNS = [
        r"즉시 생산 중단",
        r"즉시 전체 교체",
        r"확실히 고장",
        r"반드시 파손",
    ]

    def validate(
        self,
        llm_result: LlmResult,
        pred: PredictionResult,
        exp: "ExplanationResult | LSTMExplainResult",
        risk: RiskResult,
    ) -> LlmResult:
        text = llm_result.text.strip()

        if not self._has_required_sections(text):
            llm_result.text = self._safe_fallback(pred, exp, risk)
            llm_result.used_fallback = True
            llm_result.source = "guardrail_fallback"
            llm_result.passed_guardrail = False
            return llm_result

        if not self._has_valid_action_lines(text):
            llm_result.text = self._safe_fallback(pred, exp, risk)
            llm_result.used_fallback = True
            llm_result.source = "guardrail_fallback"
            llm_result.passed_guardrail = False
            return llm_result

        if self._has_forbidden_phrase(text):
            llm_result.text = self._sanitize_forbidden(text)
            llm_result.passed_guardrail = True
            return llm_result

        llm_result.passed_guardrail = True
        return llm_result

    def _has_required_sections(self, text: str) -> bool:
        return all(section in text for section in self.REQUIRED_SECTIONS)

    def _has_valid_action_lines(self, text: str) -> bool:
        action_lines = re.findall(r"^\s*[1-3]\)\s+.+$", text, flags=re.MULTILINE)
        return len(action_lines) >= 3

    def _has_forbidden_phrase(self, text: str) -> bool:
        return any(re.search(p, text) for p in self.FORBIDDEN_PATTERNS)

    def _sanitize_forbidden(self, text: str) -> str:
        sanitized = text
        for pattern in self.FORBIDDEN_PATTERNS:
            sanitized = re.sub(pattern, "우선 점검 권고", sanitized)
        return sanitized

    def _safe_fallback(
        self,
        pred: PredictionResult,
        exp: "ExplanationResult | LSTMExplainResult",
        risk: RiskResult,
    ) -> str:
        # LSTMExplainResult는 summary 없음 → explanation_text 사용
        cause_text = (
            getattr(exp, "summary", None)
            or getattr(exp, "explanation_text", "설명 정보 없음")
        )
        return (
            f"상태 요약: 현재 위험 수준은 {risk.risk_level}이며 예측 결과는 {pred.predicted_label}입니다.\n\n"
            f"의심 원인: {cause_text}\n\n"
            f"권장 조치:\n"
            f"1) 현재 주요 센서와 설비 상태를 점검합니다.\n"
            f"2) 위험 기여도가 높은 항목을 우선 확인합니다.\n"
            f"3) 다음 점검 주기에 동일 패턴 재발 여부를 확인합니다."
        )
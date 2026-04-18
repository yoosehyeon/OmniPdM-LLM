from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# 기존 AI4I / 공통 Validation
# ---------------------------------------------------------------------
@dataclass
class ValidationIssue:
    field: str
    level: str
    message: str
    original_value: Any = None
    normalized_value: Any = None


@dataclass
class ValidationResult:
    is_valid: bool
    normalized_payload: Dict[str, float]
    issues: List[ValidationIssue] = field(default_factory=list)


# ---------------------------------------------------------------------
# LSTM Sequence Input / Validation
# ---------------------------------------------------------------------
@dataclass
class LSTMSequenceInput:
    """
    LSTM 전용 sequence 입력 계약.

    sequence:
        shape = [timesteps][features]

    feature_names:
        feature 차원 이름 목록 (옵션)
        길이는 sequence[0]의 길이와 같아야 한다.

    asset_id:
        설비 식별자 (옵션)

    dataset_key:
        예: cmapss_lstm, ncmapss_lstm
    """
    sequence: List[List[float]]
    feature_names: Optional[List[str]] = None
    asset_id: str = "UNKNOWN"
    dataset_key: str = "cmapss_lstm"


@dataclass
class LSTMValidationResult:
    """
    LSTM sequence 검증 결과.

    normalized_sequence:
        숫자형 변환 및 shape 검증이 끝난 sequence
    """
    is_valid: bool
    normalized_sequence: List[List[float]]
    issues: List[ValidationIssue] = field(default_factory=list)
    timesteps: int = 0
    feature_dim: int = 0
    feature_names: Optional[List[str]] = None
    dataset_key: str = "cmapss_lstm"


# ---------------------------------------------------------------------
# Prediction / Explain / Risk / LLM / Evaluation
# ---------------------------------------------------------------------
@dataclass
class PredictionResult:
    dataset_key: str
    task_type: str
    model_name: str
    model_mode: str
    predicted_label: str
    failure_probability: float = 0.0
    anomaly_score: float = 0.0
    rul_norm: float = 1.0
    risk_score: float = 0.0
    risk_level: str = ""
    top_contributors: List[tuple[str, float]] = field(default_factory=list)
    raw_output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureExplanation:
    feature: str
    value: float
    importance: float
    direction: str


@dataclass
class ExplanationResult:
    top_features: List[FeatureExplanation]
    summary: str
    interaction_summary: Optional[str] = None
    explanation_confidence: Optional[float] = None
    raw_explanation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LSTMExplainResult:
    """
    LSTM 전용 설명 결과.

    top_features:
        [{"feature": "...", "importance": 0.1234}, ...]

    temporal_summary:
        {
            "most_recent_drivers": [...],
            "largest_variation_features": [...],
            "timesteps": ...,
            "feature_dim": ...
        }
    """
    top_features: List[Dict[str, float]] = field(default_factory=list)
    temporal_summary: Dict[str, Any] = field(default_factory=dict)
    explanation_text: str = ""


@dataclass
class RiskResult:
    risk_score: float
    risk_level: str
    method: str
    weights: Dict[str, float]


@dataclass
class LlmResult:
    text: str
    source: str
    used_fallback: bool
    passed_guardrail: bool = False
    prompt_hash: Optional[str] = None
    latency_ms: Optional[float] = None


@dataclass
class EvaluationResult:
    structure_ok: bool
    factuality_ok: bool
    overall_score: float
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# 최종 분석 결과
# ---------------------------------------------------------------------
@dataclass
class AnalysisResult:
    validation: ValidationResult
    prediction: PredictionResult
    explanation: ExplanationResult
    risk: RiskResult
    llm: LlmResult
    evaluation: EvaluationResult
    summary_text: str
    explanation_text: str
    report_markdown: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LSTMAnalysisResult:
    """
    LSTM 전용 최종 분석 결과.
    기존 AnalysisResult와 유사하지만 validation 타입이 다르다.
    """
    validation: LSTMValidationResult
    prediction: PredictionResult
    explanation: LSTMExplainResult
    risk: RiskResult
    llm: LlmResult
    evaluation: EvaluationResult
    summary_text: str
    explanation_text: str
    report_markdown: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

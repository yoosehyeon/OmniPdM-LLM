from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from services.schemas import (
    ExplanationResult,
    FeatureExplanation,
    LSTMExplainResult,
)


# PdmService top_contributors의 display name → payload dict key 매핑
_FEATURE_NAME_TO_PAYLOAD_KEY: Dict[str, str] = {
    "Tool wear [min]": "tool_wear_min",
    "Torque [Nm]": "torque_nm",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
}


class ExplainService:
    # ------------------------------------------------------------
    # 1) 기존 scalar/tabular explain
    # ------------------------------------------------------------
    def explain(self, payload: Dict[str, float], pred) -> ExplanationResult:
        """
        기존 scalar 입력용 설명 생성.
        AnalyzeService.run()에서 호출한다.
        """
        top_features: List[FeatureExplanation] = []

        contributors = getattr(pred, "top_contributors", []) or []

        for feature_name, importance in contributors[:5]:
            payload_key = _FEATURE_NAME_TO_PAYLOAD_KEY.get(feature_name, feature_name)
            value = float(payload.get(payload_key, 0.0))
            direction = self._infer_direction(value=value, importance=float(importance))
            top_features.append(
                FeatureExplanation(
                    feature=str(feature_name),
                    value=value,
                    importance=round(float(importance), 4),
                    direction=direction,
                )
            )

        # fallback: top_contributors가 없으면 payload 상위 항목으로 대체
        if not top_features:
            for feature_name, value in list(payload.items())[:5]:
                top_features.append(
                    FeatureExplanation(
                        feature=str(feature_name),
                        value=float(value),
                        importance=0.0,
                        direction="neutral",
                    )
                )

        summary = self._build_scalar_summary(pred, top_features)
        interaction_summary = self._build_scalar_interaction_summary(top_features)

        return ExplanationResult(
            top_features=top_features,
            summary=summary,
            interaction_summary=interaction_summary,
            explanation_confidence=0.8,
            raw_explanation={
                "mode": "scalar",
                "feature_count": len(top_features),
            },
        )

    def _build_scalar_summary(
        self,
        pred,
        top_features: List[FeatureExplanation],
    ) -> str:
        feature_text = (
            ", ".join([f"{f.feature}({f.importance})" for f in top_features])
            if top_features
            else "주요 feature 정보 없음"
        )
        return (
            f"예측 라벨은 {pred.predicted_label}이며, "
            f"failure_probability={pred.failure_probability}, "
            f"anomaly_score={pred.anomaly_score}, "
            f"rul_norm={pred.rul_norm} 입니다. "
            f"주요 기여 feature는 {feature_text} 입니다."
        )

    def _build_scalar_interaction_summary(
        self,
        top_features: List[FeatureExplanation],
    ) -> str:
        if not top_features:
            return "feature interaction 정보를 생성할 수 없습니다."
        names = ", ".join([f.feature for f in top_features[:3]])
        return f"상위 feature 조합({names})이 현재 예측에 가장 크게 기여했습니다."

    def _infer_direction(self, value: float, importance: float) -> str:
        if importance == 0:
            return "neutral"
        if value > 0:
            return "positive"
        if value < 0:
            return "negative"
        return "neutral"

    # ------------------------------------------------------------
    # 2) LSTM explain
    # ------------------------------------------------------------
    def explain_lstm_sequence(
        self,
        sequence: Sequence[Sequence[float]],
        feature_names: List[str] | None,
        top_contributors: List[Tuple[str, float]] | None,
        predicted_label: str | None,
        failure_probability: float | None,
        rul_norm: float | None,
    ) -> LSTMExplainResult:
        """
        LSTM 추론 결과를 사람이 읽을 수 있는 설명으로 변환한다.

        입력:
        - sequence: [timesteps, feature_dim]
        - feature_names: feature 이름 리스트. 없으면 f0, f1 ... 자동 생성
        - top_contributors: 모델이 반환한 기여 feature 목록
        - predicted_label / failure_probability / rul_norm: 예측 요약 정보
        """
        arr = np.asarray(sequence, dtype=float)
        if arr.ndim != 2:
            raise ValueError("sequence must be 2D: [timesteps, feature_dim]")

        timesteps, feature_dim = arr.shape

        if not feature_names or len(feature_names) != feature_dim:
            feature_names = [f"f{i}" for i in range(feature_dim)]

        normalized_top = self._normalize_top_contributors(
            top_contributors=top_contributors,
        )

        recent_driver_features = self._find_recent_driver_features(
            arr=arr,
            feature_names=feature_names,
            top_k=3,
        )

        largest_variation_features = self._find_largest_variation_features(
            arr=arr,
            feature_names=feature_names,
            top_k=3,
        )

        explanation_text = self._build_lstm_explanation_text(
            predicted_label=predicted_label,
            failure_probability=failure_probability,
            rul_norm=rul_norm,
            top_features=normalized_top,
            recent_driver_features=recent_driver_features,
            largest_variation_features=largest_variation_features,
            timesteps=timesteps,
        )

        return LSTMExplainResult(
            top_features=normalized_top,
            temporal_summary={
                "most_recent_drivers": recent_driver_features,
                "largest_variation_features": largest_variation_features,
                "timesteps": timesteps,
                "feature_dim": feature_dim,
            },
            explanation_text=explanation_text,
        )

    def _normalize_top_contributors(
        self,
        top_contributors: List[Tuple[str, float]] | None,
    ) -> List[Dict[str, float]]:
        if not top_contributors:
            return []

        rows: List[Dict[str, float]] = []
        for feature, score in top_contributors[:5]:
            rows.append(
                {
                    "feature": str(feature),
                    "importance": round(float(score), 4),
                }
            )
        return rows

    def _find_recent_driver_features(
        self,
        arr: np.ndarray,
        feature_names: List[str],
        top_k: int = 3,
    ) -> List[str]:
        if arr.shape[0] < 2:
            return []

        delta = np.abs(arr[-1] - arr[-2])
        top_idx = np.argsort(delta)[::-1][:top_k]
        return [feature_names[i] for i in top_idx]

    def _find_largest_variation_features(
        self,
        arr: np.ndarray,
        feature_names: List[str],
        top_k: int = 3,
    ) -> List[str]:
        stds = np.std(arr, axis=0)
        top_idx = np.argsort(stds)[::-1][:top_k]
        return [feature_names[i] for i in top_idx]

    def _build_lstm_explanation_text(
        self,
        predicted_label: str | None,
        failure_probability: float | None,
        rul_norm: float | None,
        top_features: List[Dict[str, float]],
        recent_driver_features: List[str],
        largest_variation_features: List[str],
        timesteps: int,
    ) -> str:
        label_text = predicted_label or "Unknown"
        fp_text = (
            f"{float(failure_probability):.4f}"
            if failure_probability is not None
            else "N/A"
        )
        rul_text = f"{float(rul_norm):.4f}" if rul_norm is not None else "N/A"

        top_feature_text = (
            ", ".join([f"{x['feature']}({x['importance']})" for x in top_features])
            if top_features
            else "모델 기여 feature 정보 없음"
        )

        recent_text = ", ".join(recent_driver_features) if recent_driver_features else "식별 불가"
        variation_text = ", ".join(largest_variation_features) if largest_variation_features else "식별 불가"

        return (
            f"[LSTM Explanation]\n"
            f"- Predicted Label: {label_text}\n"
            f"- Failure Probability: {fp_text}\n"
            f"- RUL Norm: {rul_text}\n"
            f"- Observed Timesteps: {timesteps}\n"
            f"- Top Contributing Features: {top_feature_text}\n"
            f"- Most Recent Driver Features: {recent_text}\n"
            f"- Largest Variation Features: {variation_text}\n"
            f"\n"
            f"해석: 최근 timestep 변화와 전체 sequence 변동성을 함께 보면 "
            f"{recent_text}가 단기 이상 징후에 더 직접적으로 연결될 가능성이 있고, "
            f"{variation_text}는 장기적인 상태 변화 추세를 반영할 가능성이 있습니다."
        )
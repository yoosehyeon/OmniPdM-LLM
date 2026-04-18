from __future__ import annotations

from typing import Dict

from services.schemas import PredictionResult, RiskResult
from models_core import risk_score as rs


class RiskService:
    def __init__(self, method: str = "weighted") -> None:
        self.method = method

    def fuse(self, pred: PredictionResult) -> RiskResult:
        """
        기존 risk_score.py 계약을 그대로 사용한다.
        rul_norm은 1.0이면 수명 충분, 0.0이면 수명 소진으로 해석된다.
        """
        weights = self._dynamic_weights(pred)

        if self.method == "weighted":
            score = float(
                rs.weighted_sum(
                    pred.failure_probability,
                    pred.anomaly_score,
                    pred.rul_norm,
                    weights={
                        "failure": weights["failure"],
                        "anomaly": weights["anomaly"],
                        "rul": weights["rul"],
                    },
                )
            )
        elif self.method == "noisy_or":
            score = float(
                rs.noisy_or(
                    pred.failure_probability,
                    pred.anomaly_score,
                    pred.rul_norm,
                    weights={
                        "failure": weights["failure"],
                        "anomaly": weights["anomaly"],
                        "rul": weights["rul"],
                    },
                )
            )
        elif self.method == "max":
            score = max(
                pred.failure_probability,
                pred.anomaly_score,
                1.0 - pred.rul_norm,
            )
        else:
            raise ValueError(f"Unsupported risk method: {self.method}")

        level = rs.to_risk_level(score)

        return RiskResult(
            risk_score=round(score, 4),
            risk_level=level,
            method=self.method,
            weights=weights,
        )

    def _dynamic_weights(self, pred: PredictionResult) -> Dict[str, float]:
        """
        기본 전략:
        - 분류 중심이면 failure weight 우세
        - anomaly 중심이면 anomaly weight 우세
        - RUL 태스크면 rul weight 우세
        """
        if pred.task_type == "regression":
            return {"failure": 0.2, "anomaly": 0.2, "rul": 0.6}
        if pred.anomaly_score >= 0.7:
            return {"failure": 0.3, "anomaly": 0.6, "rul": 0.1}
        if pred.failure_probability >= 0.7:
            return {"failure": 0.7, "anomaly": 0.2, "rul": 0.1}
        return {"failure": 0.5, "anomaly": 0.3, "rul": 0.2}
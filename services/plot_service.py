from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from services.schemas import ExplanationResult, LSTMExplainResult


class PlotService:
    def build_feature_importance(self, explanation: ExplanationResult):
        fig, ax = plt.subplots(figsize=(6, 4))
        names = [f.feature for f in explanation.top_features]
        values = [f.importance for f in explanation.top_features]
        ax.bar(names, values)
        ax.set_title("Feature Importance")
        ax.set_ylabel("Importance")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return fig

    def build_sensor_snapshot(self, payload: dict):
        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(payload.keys())
        values = list(payload.values())
        ax.plot(names, values, marker="o")
        ax.set_title("Sensor Snapshot")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return fig

    def build_lstm_feature_importance(self, explanation: LSTMExplainResult):
        """
        LSTM 경로용 Feature Importance plot.
        LSTMExplainResult.top_features는 List[Dict]이므로 dict 접근 사용.
        """
        fig, ax = plt.subplots(figsize=(7, 4))
        names = [f["feature"] for f in explanation.top_features]
        values = [f["importance"] for f in explanation.top_features]

        if not names:
            names = ["N/A"]
            values = [0.0]

        ax.bar(names, values)
        ax.set_title("LSTM Feature Importance")
        ax.set_ylabel("Importance")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return fig

    def build_lstm_sequence_snapshot(
        self,
        sequence: List[List[float]],
        feature_names: Optional[List[str]] = None,
        top_k: int = 5,
    ):
        """
        마지막 timestep 기준 상위 top_k feature를 bar plot으로 표시한다.
        """
        arr = np.asarray(sequence, dtype=float)
        latest = arr[-1]

        if feature_names is None or len(feature_names) != len(latest):
            feature_names = [f"f{i}" for i in range(len(latest))]

        scores = np.abs(latest)
        top_idx = np.argsort(-scores)[: min(top_k, len(scores))]

        names = [feature_names[i] for i in top_idx]
        values = [float(latest[i]) for i in top_idx]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, values)
        ax.set_title("Sequence Snapshot (Last Timestep Top Features)")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return fig
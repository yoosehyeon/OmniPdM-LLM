from __future__ import annotations

from typing import List, Optional

import numpy as np
import plotly.graph_objects as go

from services.schemas import ExplanationResult, LSTMExplainResult


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, xaxis_visible=False, yaxis_visible=False,
                      annotations=[dict(text="No data", showarrow=False)])
    return fig


class PlotService:
    """Plotly 기반 시각화. 모든 메서드는 plotly Figure 객체를 반환한다."""

    def build_feature_importance(self, explanation: ExplanationResult) -> go.Figure:
        if not explanation.top_features:
            return _empty_figure("Feature Importance")

        names = [f.feature for f in explanation.top_features]
        values = [f.importance for f in explanation.top_features]
        directions = [getattr(f, "direction", "neutral") for f in explanation.top_features]

        colors = [
            "#d62728" if d == "up" else "#1f77b4" if d == "down" else "#7f7f7f"
            for d in directions
        ]

        fig = go.Figure(go.Bar(x=names, y=values, marker_color=colors))
        fig.update_layout(
            title="Feature Importance",
            yaxis_title="Importance",
            xaxis_tickangle=-30,
            margin=dict(l=40, r=20, t=50, b=80),
            height=380,
        )
        return fig

    def build_sensor_snapshot(self, payload: dict) -> go.Figure:
        if not payload:
            return _empty_figure("Sensor Snapshot")

        names = list(payload.keys())
        values = list(payload.values())

        fig = go.Figure(go.Scatter(x=names, y=values, mode="lines+markers",
                                   marker=dict(size=10), line=dict(width=2)))
        fig.update_layout(
            title="Sensor Snapshot",
            xaxis_tickangle=-30,
            margin=dict(l=40, r=20, t=50, b=80),
            height=380,
        )
        return fig

    def build_lstm_feature_importance(self, explanation: LSTMExplainResult) -> go.Figure:
        names = [f["feature"] for f in explanation.top_features] or ["N/A"]
        values = [f["importance"] for f in explanation.top_features] or [0.0]

        fig = go.Figure(go.Bar(x=names, y=values, marker_color="#2ca02c"))
        fig.update_layout(
            title="LSTM Feature Importance",
            yaxis_title="Importance",
            xaxis_tickangle=-30,
            margin=dict(l=40, r=20, t=50, b=80),
            height=380,
        )
        return fig

    def build_lstm_sequence_snapshot(
        self,
        sequence: List[List[float]],
        feature_names: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> go.Figure:
        """마지막 timestep 기준 상위 top_k feature를 bar로 표시."""
        if not sequence:
            return _empty_figure("Sequence Snapshot")

        arr = np.asarray(sequence, dtype=float)
        latest = arr[-1]

        if feature_names is None or len(feature_names) != len(latest):
            feature_names = [f"f{i}" for i in range(len(latest))]

        scores = np.abs(latest)
        top_idx = np.argsort(-scores)[: min(top_k, len(scores))]

        names = [feature_names[i] for i in top_idx]
        values = [float(latest[i]) for i in top_idx]

        fig = go.Figure(go.Bar(x=names, y=values, marker_color="#ff7f0e"))
        fig.update_layout(
            title="Sequence Snapshot (Last Timestep Top Features)",
            yaxis_title="Value",
            xaxis_tickangle=-30,
            margin=dict(l=40, r=20, t=50, b=80),
            height=380,
        )
        return fig

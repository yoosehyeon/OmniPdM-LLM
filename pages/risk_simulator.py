"""Risk Simulator page — 슬라이더로 failure/anomaly/RUL 조정 → 위험도 실시간 시각화."""
from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from models_core import risk_score as rs
from pages._helpers import risk_level_color

dash.register_page(__name__, path="/risk", name="Risk Simulator", order=4)


# -----------------------------------------------------------------------------
# 레이아웃
# -----------------------------------------------------------------------------
def _slider_row(slider_id: str, label: str, value: float, marks_step: float = 0.1) -> dbc.Row:
    return dbc.Row(
        [
            dbc.Label(label, html_for=slider_id, width=4),
            dbc.Col(
                dcc.Slider(
                    id=slider_id,
                    min=0.0, max=1.0, step=0.01, value=value,
                    marks={
                        round(i * marks_step, 1): f"{round(i * marks_step, 1)}"
                        for i in range(int(1 / marks_step) + 1)
                    },
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                width=8,
            ),
        ],
        className="mb-3 align-items-center",
    )


input_card = dbc.Card(
    [
        dbc.CardHeader("Model Outputs (시뮬레이션)"),
        dbc.CardBody(
            [
                _slider_row("rs-failure", "Failure Probability (P)", 0.30),
                _slider_row("rs-anomaly", "Anomaly Score (A)", 0.20),
                _slider_row("rs-rul",     "RUL normalized (1 = full life)", 0.80),
            ]
        ),
    ]
)

config_card = dbc.Card(
    [
        dbc.CardHeader("Fusion Method"),
        dbc.CardBody(
            [
                dcc.RadioItems(
                    id="rs-fusion-method",
                    options=[
                        {"label": " weighted (해석성 우선)", "value": "weighted"},
                        {"label": " noisy_or (FN 최소화)", "value": "noisy_or"},
                        {"label": " max (보수적)", "value": "max"},
                    ],
                    value="weighted",
                    labelStyle={"display": "block", "margin": "6px 0"},
                ),
                html.Hr(),
                dbc.Label("Weights (weighted / noisy_or 만 적용)"),
                html.Small(
                    "기본: failure=0.5, anomaly=0.3, rul=0.2. 합이 1이 아니어도 내부 정규화됨.",
                    className="text-muted d-block mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col([dbc.Label("w_failure"), dbc.Input(id="rs-w-failure", type="number", value=0.5, min=0, max=1, step=0.05)], md=4),
                        dbc.Col([dbc.Label("w_anomaly"), dbc.Input(id="rs-w-anomaly", type="number", value=0.3, min=0, max=1, step=0.05)], md=4),
                        dbc.Col([dbc.Label("w_rul"),     dbc.Input(id="rs-w-rul",     type="number", value=0.2, min=0, max=1, step=0.05)], md=4),
                    ]
                ),
            ]
        ),
    ],
    className="mt-3",
)

result_card = dbc.Card(
    [
        dbc.CardHeader("Risk Result"),
        dbc.CardBody(
            [
                html.Div(id="rs-level-badge", className="mb-3"),
                dcc.Graph(id="rs-gauge", config={"displayModeBar": False}, style={"height": "240px"}),
                html.Div(id="rs-summary", className="mt-2"),
            ]
        ),
    ]
)

method_compare_card = dbc.Card(
    [
        dbc.CardHeader("Method Comparison (같은 입력에 대한 3 방식 동시 비교)"),
        dbc.CardBody(dcc.Graph(id="rs-method-bar", config={"displayModeBar": False}, style={"height": "260px"})),
    ],
    className="mt-3",
)


layout = dbc.Container(
    [
        html.H2("Risk Simulator"),
        html.P("슬라이더로 모델 출력 값을 조정 → 위험 등급이 실시간으로 변동. 가중치/융합 방식 민감도 직관 확인."),
        dbc.Row(
            [
                dbc.Col(
                    [
                        input_card,
                        config_card,
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        result_card,
                        method_compare_card,
                    ],
                    md=6,
                ),
            ]
        ),
    ],
    fluid=True,
)


# -----------------------------------------------------------------------------
# 계산 헬퍼
# -----------------------------------------------------------------------------
def _compute_risk_safe(failure: float, anomaly: float, rul: float,
                       method: str, weights: dict) -> tuple[float, str]:
    """범위 안전 + risk_score.py 호출."""
    f = max(0.0, min(1.0, float(failure)))
    a = max(0.0, min(1.0, float(anomaly)))
    r = max(0.0, min(1.0, float(rul)))

    try:
        if method == "weighted":
            score = float(rs.weighted_sum(f, a, r, weights=weights))
        elif method == "noisy_or":
            score = float(rs.noisy_or(f, a, r, weights=weights))
        elif method == "max":
            score = float(max(f, a, 1.0 - r))
        else:
            return 0.0, "Unknown"
    except Exception:
        return 0.0, "Error"

    level = rs.to_risk_level(score)
    return score, level if isinstance(level, str) else level[0]


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
@callback(
    Output("rs-level-badge", "children"),
    Output("rs-gauge", "figure"),
    Output("rs-summary", "children"),
    Output("rs-method-bar", "figure"),
    Input("rs-failure", "value"),
    Input("rs-anomaly", "value"),
    Input("rs-rul", "value"),
    Input("rs-fusion-method", "value"),
    Input("rs-w-failure", "value"),
    Input("rs-w-anomaly", "value"),
    Input("rs-w-rul", "value"),
)
def update_risk(failure, anomaly, rul, method, w_f, w_a, w_r):
    weights = {
        "failure": float(w_f or 0),
        "anomaly": float(w_a or 0),
        "rul":     float(w_r or 0),
    }

    score, level = _compute_risk_safe(failure, anomaly, rul, method, weights)

    # 1) Badge
    color = risk_level_color(level)
    badge = dbc.Alert(
        html.Span([
            html.Strong(f"{level} "),
            html.Span(f" — score: {score:.4f}"),
        ]),
        color=color,
        className="text-center mb-0 fs-5",
    )

    # 2) Gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"valueformat": ".3f"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#444"},
            "steps": [
                {"range": [0.00, 0.30], "color": "#a3e4a3"},  # Normal
                {"range": [0.30, 0.50], "color": "#cce4f7"},  # Advisory
                {"range": [0.50, 0.80], "color": "#fff1b8"},  # Warning
                {"range": [0.80, 1.00], "color": "#f4a3a3"},  # Critical
            ],
            "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.85, "value": score},
        },
    ))
    gauge.update_layout(margin=dict(l=10, r=10, t=20, b=10))

    # 3) Summary table
    summary = html.Div([
        html.Strong(f"Method: {method}"),
        html.Br(),
        html.Span(f"Inputs: P(fail)={failure:.2f}  A(anomaly)={anomaly:.2f}  RUL={rul:.2f}"),
        html.Br(),
        html.Span(f"Weights: failure={weights['failure']:.2f}  anomaly={weights['anomaly']:.2f}  rul={weights['rul']:.2f}"),
    ])

    # 4) Method comparison
    methods = ["weighted", "noisy_or", "max"]
    scores = []
    levels = []
    for m in methods:
        s, lv = _compute_risk_safe(failure, anomaly, rul, m, weights)
        scores.append(s)
        levels.append(lv)
    bar = go.Figure(go.Bar(
        x=methods, y=scores,
        text=[f"{s:.3f}<br>{lv}" for s, lv in zip(scores, levels)],
        textposition="auto",
        marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    ))
    bar.update_layout(
        title="동일 입력에 대한 3 방식 비교",
        yaxis_title="Risk Score",
        yaxis_range=[0, 1],
        margin=dict(l=40, r=20, t=50, b=40),
    )

    return badge, gauge, summary, bar

"""LSTM Analysis page — Sequence 입력 → RUL 예측 + Attention 설명 + LLM."""
from __future__ import annotations

import json
from typing import List, Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update

from pages._helpers import get_analyze_service, risk_level_color, safe_json

dash.register_page(__name__, path="/lstm", name="LSTM Analysis", order=2)


# -----------------------------------------------------------------------------
# 샘플 시퀀스 (C-MAPSS 14 sensors, 30 timesteps)
# -----------------------------------------------------------------------------
def _make_sample_sequence(case: str = "early") -> List[List[float]]:
    """간단한 sample sequence 생성 (정규화된 형태).

    case:
        early    : 운전 초기 (RUL 큼, 안정 분포)
        critical : degradation 후기 (큰 변동)
    """
    rng = np.random.default_rng(42 if case == "early" else 7)
    base = rng.normal(0, 0.3, (30, 14)).astype(np.float32)
    if case == "critical":
        # 후반 timestep 으로 갈수록 신호 amplitude 증가 → degradation 흉내
        drift = np.linspace(0, 1.5, 30).reshape(-1, 1) * rng.normal(0, 1, (1, 14))
        base = base + drift.astype(np.float32)
    return base.tolist()


SAMPLE_SEQS = {
    "early":    _make_sample_sequence("early"),
    "critical": _make_sample_sequence("critical"),
}

CMAPSS_FEATURE_NAMES = [
    "s2", "s3", "s4", "s7", "s8", "s9", "s11",
    "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]


# -----------------------------------------------------------------------------
# 입력 파싱 헬퍼
# -----------------------------------------------------------------------------
def _parse_sequence_text(text: str) -> tuple[Optional[List[List[float]]], Optional[str]]:
    """입력 textarea (JSON 또는 CSV) → (sequence, error_msg).

    JSON: [[v1, v2, ...], [...], ...]
    CSV : 줄당 한 timestep, 콤마 구분
    """
    text = (text or "").strip()
    if not text:
        return None, "sequence 가 비어있습니다"

    # JSON 시도
    if text.lstrip().startswith("["):
        try:
            data = json.loads(text)
            if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
                return None, "JSON 형식이 2차원 배열이어야 합니다"
            return [[float(v) for v in row] for row in data], None
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            return None, f"JSON 파싱 실패: {e}"

    # CSV 시도 (콤마 / 공백 모두)
    try:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace("\t", ",").replace(" ", ",").split(",") if p.strip()]
            rows.append([float(p) for p in parts])
        if not rows:
            return None, "CSV 데이터가 비어있습니다"
        return rows, None
    except ValueError as e:
        return None, f"CSV 파싱 실패: {e}"


# -----------------------------------------------------------------------------
# 레이아웃
# -----------------------------------------------------------------------------
input_card = dbc.Card(
    [
        dbc.CardHeader("Sequence Input"),
        dbc.CardBody(
            [
                dbc.Label("Dataset"),
                dcc.Dropdown(
                    id="lstm-dataset",
                    options=[
                        {"label": "C-MAPSS (BiLSTM, 14 sensors)", "value": "cmapss_lstm"},
                        {"label": "C-MAPSS (DLinear, baseline)", "value": "cmapss_dlinear"},
                        {"label": "C-MAPSS (iTransformer, cross-channel)", "value": "cmapss_itransformer"},
                        {"label": "N-CMAPSS (BiLSTM, 43 features)", "value": "ncmapss_lstm"},
                    ],
                    value="cmapss_lstm",
                    clearable=False,
                    className="mb-2",
                ),
                dbc.Label("Asset ID"),
                dbc.Input(id="lstm-asset-id", type="text", value="ENGINE_01", className="mb-2"),
                dbc.Label("Sequence (JSON 2D array or CSV — rows=timesteps, cols=features)"),
                dcc.Textarea(
                    id="lstm-sequence-text",
                    style={"width": "100%", "height": "260px", "fontFamily": "monospace", "fontSize": "0.85rem"},
                    placeholder='[[0.1, 0.2, ...], [0.15, 0.22, ...], ...]\n또는\n0.1, 0.2, 0.3, ...\n0.15, 0.22, 0.3, ...',
                ),
            ]
        ),
    ]
)

sample_buttons = dbc.ButtonGroup(
    [
        dbc.Button("Early sample (low RUL risk)", id="lstm-sample-early", color="success", size="sm", outline=True),
        dbc.Button("Critical sample (degradation)", id="lstm-sample-critical", color="danger", size="sm", outline=True),
    ],
    className="mb-3 d-flex flex-wrap",
)

run_button = dbc.Button("Run LSTM Analysis", id="lstm-run-btn", color="primary", className="mb-3 w-100")

results_section = html.Div(
    [
        html.Div(id="lstm-alert-area", className="mb-2"),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="lstm-feature-plot", figure=go.Figure()), md=6),
                dbc.Col(dcc.Graph(id="lstm-sensor-plot", figure=go.Figure()), md=6),
            ]
        ),
        dbc.Card(
            [
                dbc.CardHeader("LLM Report"),
                dbc.CardBody(dcc.Markdown(id="lstm-report-md", children="*분석 결과 미실행*")),
            ],
            className="mt-3",
        ),
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    html.Pre(id="lstm-raw-json", style={"whiteSpace": "pre-wrap", "fontSize": "0.85rem"}),
                    title="Raw Result (JSON)",
                ),
                dbc.AccordionItem(
                    html.Pre(id="lstm-validation-text", style={"whiteSpace": "pre-wrap", "fontSize": "0.85rem"}),
                    title="Validation",
                ),
            ],
            start_collapsed=True,
            className="mt-3",
        ),
    ]
)

layout = dbc.Container(
    [
        html.H2("LSTM / Sequence Analysis"),
        html.P("다변량 시계열 (예: C-MAPSS 30 cycles × 14 sensors) → RUL 예측 + Attention 설명 + LLM 한국어 코멘트"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        sample_buttons,
                        input_card,
                        html.Br(),
                        run_button,
                    ],
                    md=5,
                ),
                dbc.Col(results_section, md=7),
            ]
        ),
    ],
    fluid=True,
)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

@callback(
    Output("lstm-sequence-text", "value"),
    Input("lstm-sample-early", "n_clicks"),
    Input("lstm-sample-critical", "n_clicks"),
    prevent_initial_call=True,
)
def load_sample(_n_early, _n_critical):
    triggered = dash.ctx.triggered_id
    key_map = {
        "lstm-sample-early":    "early",
        "lstm-sample-critical": "critical",
    }
    seq = SAMPLE_SEQS[key_map[triggered]]
    return json.dumps(seq, indent=1)


@callback(
    Output("lstm-alert-area", "children"),
    Output("lstm-feature-plot", "figure"),
    Output("lstm-sensor-plot", "figure"),
    Output("lstm-report-md", "children"),
    Output("lstm-raw-json", "children"),
    Output("lstm-validation-text", "children"),
    Input("lstm-run-btn", "n_clicks"),
    State("lstm-sequence-text", "value"),
    State("lstm-dataset", "value"),
    State("lstm-asset-id", "value"),
    prevent_initial_call=True,
)
def run_lstm(_n_clicks, sequence_text, dataset_key, asset_id):
    sequence, err = _parse_sequence_text(sequence_text)
    if err is not None:
        return (
            dbc.Alert(f"입력 파싱 오류: {err}", color="danger"),
            no_update, no_update, no_update, no_update, no_update,
        )

    service = get_analyze_service(mode="lite", dataset_key=dataset_key)

    # 데이터셋별 기본 feature_names
    if dataset_key.startswith("cmapss_"):
        feat_names = CMAPSS_FEATURE_NAMES if len(sequence[0]) == 14 else None
    else:
        feat_names = None

    try:
        result = service.run_lstm(
            sequence=sequence,
            feature_names=feat_names,
            asset_id=asset_id or "UNKNOWN",
            dataset_key=dataset_key,
        )
    except Exception as e:
        return (
            dbc.Alert(f"실행 오류: {type(e).__name__}: {e}", color="danger"),
            no_update, no_update, no_update, no_update, no_update,
        )

    raw = result.get("raw_result") or {}
    risk = raw.get("risk") or {}
    risk_level = risk.get("risk_level", "N/A")
    risk_score = risk.get("risk_score", 0.0)

    alert = dbc.Alert(
        [
            html.Strong(f"Risk Level: {risk_level} "),
            html.Span(f"(score={risk_score:.3f}) · "),
            html.Span(result.get("alert_text", "")),
        ],
        color=risk_level_color(risk_level),
    )

    feature_plot = result.get("feature_plot") or go.Figure()
    sensor_plot = result.get("sensor_plot") or go.Figure()
    report_md = result.get("report_markdown", "*보고서 생성 실패*")
    raw_json = safe_json(raw)
    validation_text = result.get("validation_text", "")

    return alert, feature_plot, sensor_plot, report_md, raw_json, validation_text

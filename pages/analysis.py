"""Analysis page — Scalar 센서 입력 → 전체 파이프라인 실행."""
from __future__ import annotations

from typing import Any, Dict

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update

from pages._helpers import (
    AI4I_FIELD_LABELS,
    SAMPLE_PAYLOADS,
    get_analyze_service,
    risk_level_color,
    safe_json,
)

dash.register_page(__name__, path="/", name="Analysis", order=1)


def _input_row(field_id: str, label: str, default: float) -> dbc.Row:
    return dbc.Row(
        [
            dbc.Label(label, html_for=field_id, width=4),
            dbc.Col(
                dbc.Input(id=field_id, type="number", value=default, step=0.1),
                width=8,
            ),
        ],
        className="mb-2",
    )


_normal = SAMPLE_PAYLOADS["Normal"]

input_form = dbc.Card(
    [
        dbc.CardHeader("Sensor Input"),
        dbc.CardBody(
            [
                _input_row("air-temp", AI4I_FIELD_LABELS["air_temperature_k"], _normal["air_temperature_k"]),
                _input_row("proc-temp", AI4I_FIELD_LABELS["process_temperature_k"], _normal["process_temperature_k"]),
                _input_row("rpm", AI4I_FIELD_LABELS["rotational_speed_rpm"], _normal["rotational_speed_rpm"]),
                _input_row("torque", AI4I_FIELD_LABELS["torque_nm"], _normal["torque_nm"]),
                _input_row("tool-wear", AI4I_FIELD_LABELS["tool_wear_min"], _normal["tool_wear_min"]),
            ]
        ),
    ]
)

sample_buttons = dbc.ButtonGroup(
    [
        dbc.Button("Normal", id="sample-normal", color="success", size="sm", outline=True),
        dbc.Button("Warning", id="sample-warning", color="warning", size="sm", outline=True),
        dbc.Button("Critical", id="sample-critical", color="danger", size="sm", outline=True),
    ],
    className="mb-3",
)

config_row = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Label("Mode", html_for="mode-select"),
                dcc.Dropdown(
                    id="mode-select",
                    options=[
                        {"label": "lite (rule-based)", "value": "lite"},
                        {"label": "full (checkpoint load)", "value": "full"},
                    ],
                    value="lite",
                    clearable=False,
                ),
            ],
            md=6,
        ),
        dbc.Col(
            [
                dbc.Label("Risk Fusion", html_for="risk-method-select"),
                dcc.Dropdown(
                    id="risk-method-select",
                    options=[
                        {"label": "weighted", "value": "weighted"},
                        {"label": "noisy_or", "value": "noisy_or"},
                        {"label": "max", "value": "max"},
                    ],
                    value="weighted",
                    clearable=False,
                ),
            ],
            md=6,
        ),
    ],
    className="mb-3",
)

run_button = dbc.Button("Run Analysis", id="run-btn", color="primary", className="mb-3 w-100")

results_section = html.Div(
    [
        html.Div(id="alert-area", className="mb-2"),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="feature-plot", figure=go.Figure()), md=6),
                dbc.Col(dcc.Graph(id="sensor-plot", figure=go.Figure()), md=6),
            ]
        ),
        dbc.Card(
            [
                dbc.CardHeader("LLM Report"),
                dbc.CardBody(dcc.Markdown(id="report-md", children="*분석 결과 미실행*")),
            ],
            className="mt-3",
        ),
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    html.Pre(id="raw-json", style={"whiteSpace": "pre-wrap", "fontSize": "0.85rem"}),
                    title="Raw Result (JSON)",
                )
            ],
            start_collapsed=True,
            className="mt-3",
        ),
    ]
)

layout = dbc.Container(
    [
        html.H2("Analysis — Scalar Input"),
        html.P("AI4I 5센서 입력 → DL 추론 → Risk → Explain → LLM → Guardrail → Report"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        sample_buttons,
                        input_form,
                        html.Br(),
                        config_row,
                        run_button,
                    ],
                    md=4,
                ),
                dbc.Col(results_section, md=8),
            ]
        ),
    ],
    fluid=True,
)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

@callback(
    Output("air-temp", "value"),
    Output("proc-temp", "value"),
    Output("rpm", "value"),
    Output("torque", "value"),
    Output("tool-wear", "value"),
    Input("sample-normal", "n_clicks"),
    Input("sample-warning", "n_clicks"),
    Input("sample-critical", "n_clicks"),
    prevent_initial_call=True,
)
def load_sample(_n1, _n2, _n3):
    triggered = dash.ctx.triggered_id
    key_map = {
        "sample-normal": "Normal",
        "sample-warning": "Warning",
        "sample-critical": "Critical",
    }
    sample = SAMPLE_PAYLOADS[key_map[triggered]]
    return (
        sample["air_temperature_k"],
        sample["process_temperature_k"],
        sample["rotational_speed_rpm"],
        sample["torque_nm"],
        sample["tool_wear_min"],
    )


@callback(
    Output("alert-area", "children"),
    Output("feature-plot", "figure"),
    Output("sensor-plot", "figure"),
    Output("report-md", "children"),
    Output("raw-json", "children"),
    Input("run-btn", "n_clicks"),
    State("air-temp", "value"),
    State("proc-temp", "value"),
    State("rpm", "value"),
    State("torque", "value"),
    State("tool-wear", "value"),
    State("mode-select", "value"),
    State("risk-method-select", "value"),
    prevent_initial_call=True,
)
def run_analysis(_n, air_temp, proc_temp, rpm, torque, tool_wear, mode, risk_method):
    payload: Dict[str, Any] = {
        "air_temperature_k": air_temp,
        "process_temperature_k": proc_temp,
        "rotational_speed_rpm": rpm,
        "torque_nm": torque,
        "tool_wear_min": tool_wear,
    }

    service = get_analyze_service(mode=mode, dataset_key="ai4i_cnn")
    service.risk_method = risk_method
    service.risk_service.method = risk_method

    try:
        result = service.run(payload)
    except Exception as e:
        err = dbc.Alert(f"실행 오류: {type(e).__name__}: {e}", color="danger")
        return err, no_update, no_update, no_update, no_update

    raw = result.get("raw_result") or {}
    risk = (raw.get("risk") or {})
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

    return alert, feature_plot, sensor_plot, report_md, raw_json

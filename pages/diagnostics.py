"""Diagnostics page — Analysis 파이프라인 각 단계의 raw JSON 확인 (디버깅용)."""
from __future__ import annotations

import json
from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, no_update

from pages._helpers import SAMPLE_PAYLOADS, get_analyze_service, safe_json

dash.register_page(__name__, path="/diagnostics", name="Diagnostics", order=3)


# -----------------------------------------------------------------------------
# 레이아웃
# -----------------------------------------------------------------------------
sample_buttons = dbc.ButtonGroup(
    [
        dbc.Button("Normal", id="diag-sample-normal", color="success", size="sm", outline=True),
        dbc.Button("Warning", id="diag-sample-warning", color="warning", size="sm", outline=True),
        dbc.Button("Critical", id="diag-sample-critical", color="danger", size="sm", outline=True),
    ],
    className="mb-3",
)


def _input_row(field_id: str, label: str, default: float) -> dbc.Row:
    return dbc.Row(
        [
            dbc.Label(label, html_for=field_id, width=5),
            dbc.Col(
                dbc.Input(id=field_id, type="number", value=default, step=0.1, size="sm"),
                width=7,
            ),
        ],
        className="mb-2",
    )


_normal = SAMPLE_PAYLOADS["Normal"]

input_card = dbc.Card(
    [
        dbc.CardHeader("Scalar Input (AI4I 5 sensors)"),
        dbc.CardBody(
            [
                _input_row("diag-air-temp", "Air Temperature (K)", _normal["air_temperature_k"]),
                _input_row("diag-proc-temp", "Process Temperature (K)", _normal["process_temperature_k"]),
                _input_row("diag-rpm", "Rotational Speed (rpm)", _normal["rotational_speed_rpm"]),
                _input_row("diag-torque", "Torque (Nm)", _normal["torque_nm"]),
                _input_row("diag-tool-wear", "Tool Wear (min)", _normal["tool_wear_min"]),
            ]
        ),
    ]
)

run_button = dbc.Button("Run Pipeline (Diagnostics)", id="diag-run-btn", color="primary", className="mt-3 w-100")


def _section(title: str, json_id: str, default_msg: str = "*아직 실행되지 않음*") -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(title),
            dbc.CardBody(
                html.Pre(
                    id=json_id, children=default_msg,
                    style={"whiteSpace": "pre-wrap", "fontSize": "0.78rem",
                           "maxHeight": "320px", "overflowY": "auto", "margin": 0},
                )
            ),
        ],
        className="mb-3",
    )


layout = dbc.Container(
    [
        html.H2("Diagnostics — 파이프라인 단계별 Raw JSON"),
        html.P(
            "Analysis 페이지와 동일한 입력을 사용하지만, "
            "InputValidation / Prediction / Risk / Explain / LLM / Guardrail / Evaluation 의 "
            "각 단계 결과를 JSON 으로 확인할 수 있습니다 (디버깅 / 개발자 용)."
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        sample_buttons,
                        input_card,
                        run_button,
                        html.Br(),
                        html.Div(id="diag-status"),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        _section("1. Validation", "diag-validation"),
                        _section("2. Prediction", "diag-prediction"),
                        _section("3. Risk", "diag-risk"),
                        _section("4. Explanation", "diag-explanation"),
                        _section("5. LLM Result", "diag-llm"),
                        _section("6. Evaluation", "diag-evaluation"),
                    ],
                    md=8,
                ),
            ]
        ),
    ],
    fluid=True,
)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
@callback(
    Output("diag-air-temp", "value"),
    Output("diag-proc-temp", "value"),
    Output("diag-rpm", "value"),
    Output("diag-torque", "value"),
    Output("diag-tool-wear", "value"),
    Input("diag-sample-normal", "n_clicks"),
    Input("diag-sample-warning", "n_clicks"),
    Input("diag-sample-critical", "n_clicks"),
    prevent_initial_call=True,
)
def load_sample(_n1, _n2, _n3):
    key_map = {
        "diag-sample-normal":   "Normal",
        "diag-sample-warning":  "Warning",
        "diag-sample-critical": "Critical",
    }
    sample = SAMPLE_PAYLOADS[key_map[dash.ctx.triggered_id]]
    return (
        sample["air_temperature_k"],
        sample["process_temperature_k"],
        sample["rotational_speed_rpm"],
        sample["torque_nm"],
        sample["tool_wear_min"],
    )


@callback(
    Output("diag-status", "children"),
    Output("diag-validation", "children"),
    Output("diag-prediction", "children"),
    Output("diag-risk", "children"),
    Output("diag-explanation", "children"),
    Output("diag-llm", "children"),
    Output("diag-evaluation", "children"),
    Input("diag-run-btn", "n_clicks"),
    State("diag-air-temp", "value"),
    State("diag-proc-temp", "value"),
    State("diag-rpm", "value"),
    State("diag-torque", "value"),
    State("diag-tool-wear", "value"),
    prevent_initial_call=True,
)
def run_pipeline(_n_clicks, air_temp, proc_temp, rpm, torque, tool_wear):
    payload = {
        "air_temperature_k":    air_temp,
        "process_temperature_k": proc_temp,
        "rotational_speed_rpm": rpm,
        "torque_nm":            torque,
        "tool_wear_min":        tool_wear,
    }

    service = get_analyze_service(mode="lite", dataset_key="ai4i_cnn")
    try:
        result = service.run(payload)
    except Exception as e:
        return (
            dbc.Alert(f"실행 오류: {type(e).__name__}: {e}", color="danger"),
            no_update, no_update, no_update, no_update, no_update, no_update,
        )

    raw = result.get("raw_result") or {}

    status = dbc.Alert(
        f"파이프라인 실행 완료. Risk Level: {raw.get('risk', {}).get('risk_level', 'N/A')}",
        color="success",
    )

    # 각 단계별 raw JSON 추출 (없는 키는 빈 dict 로 fallback)
    return (
        status,
        safe_json(raw.get("validation", {})),
        safe_json(raw.get("prediction", {})),
        safe_json(raw.get("risk", {})),
        safe_json(raw.get("explanation", {})),
        safe_json(raw.get("llm", {})),
        safe_json(raw.get("evaluation", {})),
    )

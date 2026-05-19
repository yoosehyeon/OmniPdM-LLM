"""Report page — 자동 저장된 분석 보고서 목록 / 렌더링."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dash_table, dcc, html, no_update

from models_core import config

dash.register_page(__name__, path="/reports", name="Report", order=6)


REPORTS_DIR: Path = config.REPORT_DIR


def _scan_reports() -> List[Dict[str, Any]]:
    if not REPORTS_DIR.exists():
        return []

    rows: List[Dict[str, Any]] = []
    for md_path in sorted(REPORTS_DIR.glob("*.md"), reverse=True):
        stat = md_path.stat()
        json_path = md_path.with_suffix(".json")
        rows.append({
            "filename": md_path.name,
            "size": f"{stat.st_size / 1024:.1f} KB",
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "has_json": "✅" if json_path.exists() else "—",
            "_path": str(md_path),
        })
    return rows


layout = dbc.Container(
    [
        html.H2("Report Browser"),
        html.P("자동 저장된 분석 보고서 (Markdown + JSON pair)"),
        dbc.ButtonGroup(
            [
                dbc.Button("Refresh", id="reports-refresh", color="primary", size="sm"),
            ],
            className="mb-3",
        ),
        html.Div(id="reports-summary", className="mb-2"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5("Reports"),
                        dash_table.DataTable(
                            id="reports-table",
                            columns=[
                                {"name": "Filename", "id": "filename"},
                                {"name": "Size", "id": "size"},
                                {"name": "Modified", "id": "modified"},
                                {"name": "JSON", "id": "has_json"},
                            ],
                            row_selectable="single",
                            page_size=15,
                            style_cell={"textAlign": "left", "fontSize": "0.85rem", "padding": "6px"},
                            style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
                            style_table={"overflowX": "auto"},
                        ),
                        # 클릭시 파일 경로를 캐시
                        dcc.Store(id="reports-data-store"),
                    ],
                    md=5,
                ),
                dbc.Col(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    dcc.Markdown(id="report-content", children="*보고서를 선택하세요*"),
                                    label="Markdown",
                                ),
                                dbc.Tab(
                                    html.Pre(
                                        id="report-json",
                                        children="*JSON 데이터 없음*",
                                        style={"whiteSpace": "pre-wrap", "fontSize": "0.8rem"},
                                    ),
                                    label="Raw JSON",
                                ),
                            ]
                        ),
                    ],
                    md=7,
                ),
            ]
        ),
    ],
    fluid=True,
)


@callback(
    Output("reports-table", "data"),
    Output("reports-data-store", "data"),
    Output("reports-summary", "children"),
    Input("reports-refresh", "n_clicks"),
)
def refresh_reports(_n):
    rows = _scan_reports()
    # _path 는 DataTable 표시에서 제외하고 store 에 별도 유지
    display_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    paths = [r["_path"] for r in rows]

    color = "success" if rows else "secondary"
    text = f"{len(rows)}개 보고서" if rows else "저장된 보고서가 없습니다"
    summary = dbc.Alert(text, color=color)

    return display_rows, paths, summary


@callback(
    Output("report-content", "children"),
    Output("report-json", "children"),
    Input("reports-table", "selected_rows"),
    State("reports-data-store", "data"),
)
def render_selected(selected_rows, paths):
    if not selected_rows or not paths:
        return no_update, no_update

    idx = selected_rows[0]
    if idx >= len(paths):
        return no_update, no_update

    md_path = Path(paths[idx])
    if not md_path.exists():
        return "*파일을 찾을 수 없습니다*", "*파일을 찾을 수 없습니다*"

    md_content = md_path.read_text(encoding="utf-8")

    json_path = md_path.with_suffix(".json")
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            json_content = json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            json_content = f"JSON parse error: {type(e).__name__}: {e}"
    else:
        json_content = "*JSON 파일 없음*"

    return md_content, json_content

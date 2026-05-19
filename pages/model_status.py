"""Model Status page — 체크포인트 / 모델 메타 정보 표시."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dash_table, html

from models_core import config

dash.register_page(__name__, path="/status", name="Model Status", order=5)


DATASET_KEYS = [
    ("ai4i_cnn", ".pt", "분류 (밀링머신)"),
    ("ai4i_gbdt", ".pkl", "분류 (GBDT)"),
    ("hydraulic_ae", ".pt", "이상 탐지 (유압)"),
    ("cmapss_lstm", ".pt", "RUL 회귀 (C-MAPSS)"),
    ("ncmapss_lstm", ".pt", "RUL 회귀 (N-CMAPSS)"),
    ("cwru_cnn", ".pt", "베어링 분류 (CWRU)"),
]


def _find_latest_checkpoint(stem: str, suffix: str) -> Path | None:
    candidates = sorted(config.CHECKPOINT_DIR.glob(f"{stem}*{suffix}"))
    return candidates[-1] if candidates else None


def _human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _scan_models() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, suffix, description in DATASET_KEYS:
        ckpt = _find_latest_checkpoint(key, suffix)
        meta_path = _find_latest_checkpoint(key, "_meta.json")

        if ckpt is None:
            rows.append({
                "dataset_key": key,
                "description": description,
                "status": "❌ 없음",
                "file": "-",
                "size": "-",
                "modified": "-",
                "feature_dim": "-",
            })
            continue

        stat = ckpt.stat()
        meta: Dict[str, Any] = {}
        if meta_path and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        rows.append({
            "dataset_key": key,
            "description": description,
            "status": "✅ OK",
            "file": ckpt.name,
            "size": _human_size(stat.st_size),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "feature_dim": meta.get("feature_dim", "-"),
        })
    return rows


layout = dbc.Container(
    [
        html.H2("Model Status"),
        html.P("로컬 체크포인트 / 메타 정보 스캔. HF Hub fallback 사용 시에도 로컬 캐시 우선."),
        dbc.ButtonGroup(
            [
                dbc.Button("Refresh", id="status-refresh", color="primary", size="sm"),
            ],
            className="mb-3",
        ),
        html.Div(id="status-summary", className="mb-2"),
        dash_table.DataTable(
            id="status-table",
            columns=[
                {"name": "Dataset", "id": "dataset_key"},
                {"name": "Description", "id": "description"},
                {"name": "Status", "id": "status"},
                {"name": "File", "id": "file"},
                {"name": "Size", "id": "size"},
                {"name": "Modified", "id": "modified"},
                {"name": "feature_dim", "id": "feature_dim"},
            ],
            style_cell={"textAlign": "left", "padding": "8px", "fontSize": "0.9rem"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
            style_data_conditional=[
                {
                    "if": {"filter_query": '{status} contains "없음"'},
                    "backgroundColor": "#ffe4e4",
                },
            ],
        ),
        html.Hr(),
        html.H5("체크포인트 경로"),
        html.Code(str(config.CHECKPOINT_DIR), style={"fontSize": "0.85rem"}),
        html.Br(),
        html.Br(),
        html.H5("HF Hub Fallback Repo"),
        html.Code(getattr(config, "CHECKPOINT_REPO", "(미설정)"), style={"fontSize": "0.85rem"}),
    ],
    fluid=True,
)


@callback(
    Output("status-table", "data"),
    Output("status-summary", "children"),
    Input("status-refresh", "n_clicks"),
)
def refresh(_n):
    rows = _scan_models()
    ok_count = sum(1 for r in rows if r["status"].startswith("✅"))
    summary = dbc.Alert(
        f"총 {len(rows)}개 모델 중 {ok_count}개 사용 가능",
        color="success" if ok_count == len(rows) else "warning",
    )
    return rows, summary

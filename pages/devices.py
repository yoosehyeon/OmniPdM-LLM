"""Devices page — CMMS Tier 1.5 P1-d. 설비 마스터 list + status badge.

설계:
- before_request 가드가 인증된 사용자만 통과시킴 → 페이지 자체는 단순.
- layout 은 함수 (serve_layout) — 매 요청마다 DB 에서 최신 device 목록 fetch + role 검사.
- admin/operator 만 status 전이 액션 노출 (viewer 는 read-only).
- status 전이는 단순 dropdown + Apply 버튼 (PoC 최소 UX).

audit_log: device_service.set_status 가 내부에서 자동 처리.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dash_table, dcc, html, no_update

from services.auth import create_default_user_service, current_user
from services.realtime.device_service import create_default as create_default_device_service

dash.register_page(__name__, path="/devices", name="Devices", order=7)


# module-level service 인스턴스 (NullBackend 면 빈 결과 반환 → DB OFF 환경에서도 페이지는 부팅).
_device_service = create_default_device_service()
_user_service = create_default_user_service()

# device 상태 → badge 색상 매핑. Bootstrap 5 변수.
_STATUS_COLOR = {
    "operational": "success",
    "warning": "warning",
    "critical": "danger",
    "maintenance": "info",
    "offline": "secondary",
}

# 권한 정책: status 전이는 admin / operator 만.
_STATUS_WRITE_ROLES = {"admin", "operator"}

_STATUS_OPTIONS = [
    {"label": s.title(), "value": s}
    for s in ("operational", "warning", "critical", "maintenance", "offline")
]


def _badge(status: str) -> str:
    color = _STATUS_COLOR.get(status, "secondary")
    return f'<span class="badge bg-{color}">{status}</span>'


def _devices_table_data() -> List[Dict[str, Any]]:
    devices = _device_service.list_active_devices()
    return [
        {
            "device_id": d.device_id,
            "name": d.name,
            "plant_id": d.plant_id,
            "dataset_key": d.dataset_key,
            "current_status": d.current_status,
            "status_badge": _badge(d.current_status),
            "updated_at": d.updated_at.strftime("%Y-%m-%d %H:%M:%S") if d.updated_at else "",
        }
        for d in devices
    ]


def _current_role() -> Optional[str]:
    user = current_user(_user_service)
    return user.role if user else None


def _status_change_panel(can_write: bool) -> Any:
    if not can_write:
        return dbc.Alert(
            "Status changes are restricted to admin / operator roles.",
            color="secondary",
            className="mt-3",
        )
    return dbc.Card(
        [
            dbc.CardHeader("Change device status"),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Dropdown(
                                    id="devices-target-id",
                                    placeholder="Select device_id",
                                    clearable=False,
                                ),
                                md=4,
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="devices-new-status",
                                    options=_STATUS_OPTIONS,
                                    placeholder="New status",
                                    clearable=False,
                                ),
                                md=3,
                            ),
                            dbc.Col(
                                dbc.Input(
                                    id="devices-reason",
                                    placeholder="Reason (optional)",
                                    type="text",
                                ),
                                md=3,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Apply",
                                    id="devices-apply-status",
                                    color="primary",
                                    n_clicks=0,
                                ),
                                md=2,
                            ),
                        ],
                        className="g-2",
                    ),
                    html.Div(id="devices-apply-feedback", className="mt-2"),
                ]
            ),
        ],
        className="mt-3",
    )


def serve_layout() -> Any:
    role = _current_role()
    can_write = role in _STATUS_WRITE_ROLES
    rows = _devices_table_data()
    device_ids = [r["device_id"] for r in rows]

    return dbc.Container(
        [
            html.H3("Devices", className="mt-3"),
            html.P(
                f"Authenticated as role: {role or 'unknown'}. "
                f"Showing {len(rows)} active device(s).",
                className="text-muted small",
            ),
            dash_table.DataTable(
                id="devices-table",
                columns=[
                    {"name": "device_id", "id": "device_id"},
                    {"name": "name", "id": "name"},
                    {"name": "plant_id", "id": "plant_id"},
                    {"name": "dataset_key", "id": "dataset_key"},
                    {"name": "status", "id": "status_badge", "presentation": "markdown"},
                    {"name": "updated_at", "id": "updated_at"},
                ],
                data=rows,
                markdown_options={"html": True},  # badge HTML 렌더
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": "0.9rem", "padding": "0.4rem"},
                page_size=20,
            ),
            # status 전이 패널 — role 에 따라 활성/비활성.
            _status_change_panel(can_write),
            # device_id dropdown 옵션은 콜백 영향 없이 layout 진입 시 결정 (rows 와 동기).
            dcc.Store(id="devices-id-options", data=device_ids),
        ],
        fluid=True,
    )


layout = serve_layout


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(
    Output("devices-target-id", "options"),
    Input("devices-id-options", "data"),
)
def _populate_target_id(device_ids: List[str]) -> List[Dict[str, str]]:
    return [{"label": d, "value": d} for d in (device_ids or [])]


@callback(
    Output("devices-apply-feedback", "children"),
    Input("devices-apply-status", "n_clicks"),
    State("devices-target-id", "value"),
    State("devices-new-status", "value"),
    State("devices-reason", "value"),
    prevent_initial_call=True,
)
def _apply_status(n_clicks: int, target_id: str, new_status: str, reason: str):
    # 권한 재확인 — UI 가 숨기더라도 callback 호출은 직접 가능. defense in depth.
    role = _current_role()
    if role not in _STATUS_WRITE_ROLES:
        return dbc.Alert("Forbidden: status change requires admin/operator role.", color="danger")

    if not target_id or not new_status:
        return dbc.Alert("Select both device_id and new status.", color="warning")

    user = current_user(_user_service)
    actor_id = user.user_id if user else None
    try:
        _device_service.set_status(
            target_id,
            new_status,
            reason=(reason or None),
            actor_id=actor_id,
        )
    except Exception as e:
        return dbc.Alert(f"Update failed: {type(e).__name__}: {e}", color="danger")

    return dbc.Alert(
        f"Updated {target_id} -> {new_status}. Reload the page to see the change.",
        color="success",
    )

"""Maintenance Orders page — CMMS Tier 1.5 P1-d. 정비 주문 list + Open new + Close.

설계:
- before_request 가드 + role 기반 UI (devices.py 패턴 동일).
- list (모든 사용자) + status filter (모든 사용자) + Open new (admin/operator) + Close (admin/operator).
- audit_log: maintenance_order_service 가 자동 처리.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dash_table, dcc, html

from services.auth import create_default_user_service, current_user
from services.realtime.device_service import create_default as create_default_device_service
from services.realtime.maintenance_order_service import (
    ORDER_PRIORITIES,
    ORDER_STATUSES,
    create_default as create_default_order_service,
)

dash.register_page(__name__, path="/orders", name="Orders", order=8)


_order_service = create_default_order_service()
_device_service = create_default_device_service()
_user_service = create_default_user_service()

_WRITE_ROLES = {"admin", "operator"}

_STATUS_FILTER_OPTIONS = [{"label": "All", "value": ""}] + [
    {"label": s.title(), "value": s} for s in ORDER_STATUSES
]
_PRIORITY_OPTIONS = [{"label": p.title(), "value": p} for p in ORDER_PRIORITIES]

_PRIORITY_COLOR = {
    "low": "secondary",
    "normal": "info",
    "high": "warning",
    "critical": "danger",
}
_STATUS_COLOR = {
    "open": "primary",
    "in_progress": "warning",
    "closed": "success",
    "cancelled": "secondary",
}


def _badge(value: str, palette: Dict[str, str]) -> str:
    color = palette.get(value, "secondary")
    return f'<span class="badge bg-{color}">{value}</span>'


def _orders_table_data(status_filter: Optional[str]) -> List[Dict[str, Any]]:
    orders = _order_service.list_orders(
        status=status_filter,
        limit=200,
    )
    return [
        {
            "order_id": o.order_id,
            "device_id": o.device_id,
            "title": o.title,
            "priority_badge": _badge(o.priority, _PRIORITY_COLOR),
            "status_badge": _badge(o.status, _STATUS_COLOR),
            "assigned_to": o.assigned_to or "",
            "created_at": o.created_at.strftime("%Y-%m-%d %H:%M") if o.created_at else "",
            "closed_at": o.closed_at.strftime("%Y-%m-%d %H:%M") if o.closed_at else "",
        }
        for o in orders
    ]


def _current_role() -> Optional[str]:
    user = current_user(_user_service)
    return user.role if user else None


def _current_user_or_none():
    """callback 안에서 role + actor_id 한 번에 — DB 왕복 1회로 정리."""
    return current_user(_user_service)


def _open_new_panel(can_write: bool) -> Any:
    if not can_write:
        return dbc.Alert(
            "Creating new orders is restricted to admin / operator roles.",
            color="secondary",
            className="mt-3",
        )
    return dbc.Card(
        [
            dbc.CardHeader("Open new maintenance order"),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Dropdown(
                                    id="orders-new-device",
                                    placeholder="device_id",
                                    clearable=False,
                                ),
                                md=3,
                            ),
                            dbc.Col(
                                dbc.Input(id="orders-new-title", placeholder="Title", type="text"),
                                md=4,
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="orders-new-priority",
                                    options=_PRIORITY_OPTIONS,
                                    value="normal",
                                    clearable=False,
                                ),
                                md=2,
                            ),
                            dbc.Col(
                                dbc.Input(
                                    id="orders-new-description",
                                    placeholder="Description (optional)",
                                    type="text",
                                ),
                                md=3,
                            ),
                        ],
                        className="g-2",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    "Open order",
                                    id="orders-open-new",
                                    color="primary",
                                    n_clicks=0,
                                ),
                                md=2,
                            ),
                            dbc.Col(
                                html.Div(id="orders-open-feedback"),
                                md=10,
                            ),
                        ],
                        className="g-2 mt-2",
                    ),
                ]
            ),
        ],
        className="mt-3",
    )


def _close_panel(can_write: bool) -> Any:
    if not can_write:
        return None
    return dbc.Card(
        [
            dbc.CardHeader("Close an order"),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Input(
                                    id="orders-close-id",
                                    type="number",
                                    placeholder="order_id",
                                    min=1,
                                ),
                                md=3,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Close",
                                    id="orders-close-btn",
                                    color="success",
                                    n_clicks=0,
                                ),
                                md=2,
                            ),
                            dbc.Col(
                                html.Div(id="orders-close-feedback"),
                                md=7,
                            ),
                        ],
                        className="g-2",
                    ),
                ]
            ),
        ],
        className="mt-3",
    )


def serve_layout() -> Any:
    role = _current_role()
    can_write = role in _WRITE_ROLES
    rows = _orders_table_data(status_filter=None)
    device_ids = [d.device_id for d in _device_service.list_active_devices()]

    return dbc.Container(
        [
            html.H3("Maintenance Orders", className="mt-3"),
            html.P(
                f"Authenticated as role: {role or 'unknown'}. Showing {len(rows)} order(s).",
                className="text-muted small",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="orders-status-filter",
                            options=_STATUS_FILTER_OPTIONS,
                            value="",
                            clearable=False,
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Refresh",
                            id="orders-refresh",
                            color="secondary",
                            n_clicks=0,
                        ),
                        md=2,
                    ),
                ],
                className="g-2 mb-2",
            ),
            dash_table.DataTable(
                id="orders-table",
                columns=[
                    {"name": "order_id", "id": "order_id"},
                    {"name": "device_id", "id": "device_id"},
                    {"name": "title", "id": "title"},
                    {"name": "priority", "id": "priority_badge", "presentation": "markdown"},
                    {"name": "status", "id": "status_badge", "presentation": "markdown"},
                    {"name": "assigned_to", "id": "assigned_to"},
                    {"name": "created_at", "id": "created_at"},
                    {"name": "closed_at", "id": "closed_at"},
                ],
                data=rows,
                markdown_options={"html": True},
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": "0.9rem", "padding": "0.4rem"},
                page_size=25,
            ),
            _open_new_panel(can_write),
            _close_panel(can_write),
            dcc.Store(id="orders-device-options", data=device_ids),
        ],
        fluid=True,
    )


layout = serve_layout


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(
    Output("orders-new-device", "options"),
    Input("orders-device-options", "data"),
)
def _populate_device_options(device_ids: List[str]) -> List[Dict[str, str]]:
    return [{"label": d, "value": d} for d in (device_ids or [])]


@callback(
    Output("orders-table", "data"),
    Input("orders-refresh", "n_clicks"),
    Input("orders-status-filter", "value"),
)
def _refresh_table(_n_clicks: int, status_filter: str):
    return _orders_table_data(status_filter or None)


@callback(
    Output("orders-open-feedback", "children"),
    Input("orders-open-new", "n_clicks"),
    State("orders-new-device", "value"),
    State("orders-new-title", "value"),
    State("orders-new-priority", "value"),
    State("orders-new-description", "value"),
    prevent_initial_call=True,
)
def _open_new(n_clicks: int, device_id: str, title: str, priority: str, description: str):
    user = _current_user_or_none()
    role = user.role if user else None
    if role not in _WRITE_ROLES:
        return dbc.Alert("Forbidden: open order requires admin/operator role.", color="danger")

    if not device_id or not title:
        return dbc.Alert("device_id and title are required.", color="warning")

    actor_id = user.user_id if user else None
    try:
        order = _order_service.create_order(
            device_id=device_id,
            title=title,
            description=(description or None),
            priority=(priority or "normal"),
            actor_id=actor_id,
        )
    except Exception as e:
        return dbc.Alert(f"Create failed: {type(e).__name__}: {e}", color="danger")

    if order is None:
        return dbc.Alert("Backend rejected the order (Null backend?).", color="warning")
    return dbc.Alert(f"Order #{order.order_id} opened for {order.device_id}.", color="success")


@callback(
    Output("orders-close-feedback", "children"),
    Input("orders-close-btn", "n_clicks"),
    State("orders-close-id", "value"),
    prevent_initial_call=True,
)
def _close_order(n_clicks: int, order_id: Optional[int]):
    user = _current_user_or_none()
    role = user.role if user else None
    if role not in _WRITE_ROLES:
        return dbc.Alert("Forbidden: close order requires admin/operator role.", color="danger")
    if not order_id:
        return dbc.Alert("order_id is required.", color="warning")

    actor_id = user.user_id if user else None
    try:
        changed = _order_service.close_order(int(order_id), actor_id=actor_id)
    except Exception as e:
        return dbc.Alert(f"Close failed: {type(e).__name__}: {e}", color="danger")

    if not changed:
        return dbc.Alert("No change (order missing or already closed).", color="warning")
    return dbc.Alert(f"Order #{order_id} closed.", color="success")

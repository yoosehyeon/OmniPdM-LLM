"""
HybridPdM - Dash multi-page application entry point.

사용:
    python app.py

환경변수:
    DASH_HOST   (default: 0.0.0.0)
    DASH_PORT   (default: 8050)
    DASH_DEBUG  (default: false)
"""
from __future__ import annotations

import os

import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc


def create_app() -> Dash:
    app = Dash(
        __name__,
        use_pages=True,
        pages_folder="pages",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="HybridPdM",
    )

    navbar = dbc.NavbarSimple(
        brand="HybridPdM",
        brand_href="/",
        color="dark",
        dark=True,
        fluid=True,
        children=[
            dbc.NavItem(dcc.Link(page["name"], href=page["path"], className="nav-link"))
            for page in dash.page_registry.values()
        ],
    )

    app.layout = dbc.Container(
        [
            navbar,
            html.Div(dash.page_container, className="mt-4"),
        ],
        fluid=True,
    )

    return app


app = create_app()
server = app.server  # WSGI entry for production (gunicorn 등)


if __name__ == "__main__":
    host = os.getenv("DASH_HOST", "0.0.0.0")
    port = int(os.getenv("DASH_PORT", "8050"))
    debug = os.getenv("DASH_DEBUG", "false").lower() == "true"

    app.run(host=host, port=port, debug=debug)

"""GET /api/health — Liveness/Readiness probe.

용도:
- Docker healthcheck (compose `healthcheck:` 또는 ALB target group)
- 프론트엔드 첫 부팅 시 백엔드 도달 가능성 확인
- 모니터링 스크립트

응답은 인증 불필요 (app.py 의 _AUTH_EXEMPT_PREFIXES 에 /api/health 추가).
DB connectivity 까지 검사하지는 않음 — readiness 는 후속 endpoint 분리.
"""
from __future__ import annotations

import time

from flask import Blueprint, jsonify

bp_health = Blueprint("api_health", __name__, url_prefix="/api")

_STARTED_AT = time.time()


@bp_health.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "omnipdm",
        "uptime_seconds": round(time.time() - _STARTED_AT, 2),
    })

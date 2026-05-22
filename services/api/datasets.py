"""GET /api/datasets — Dataset 메타 카탈로그 반환.

React `frontend/src/data.ts` 의 하드코딩 카탈로그를 백엔드 single source of truth 로 교체.
- 인증 미요구 (메타 데이터, 민감 정보 없음 → app.py _AUTH_EXEMPT_PREFIXES 에 등록)
- 정적 카탈로그라 Cache-Control: max-age 짧게 (10분) — 빌드 시 카탈로그가 바뀌어도
  React 가 다음 페이지 진입에서 갱신.
"""
from __future__ import annotations

from flask import Blueprint, Response, jsonify

from services.dataset_catalog import list_datasets

bp_datasets = Blueprint("api_datasets", __name__, url_prefix="/api")


@bp_datasets.route("/datasets", methods=["GET"])
def get_datasets() -> Response:
    resp = jsonify(list_datasets())
    resp.headers["Cache-Control"] = "public, max-age=600"
    return resp

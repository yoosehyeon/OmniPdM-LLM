"""Flask REST API blueprints — React frontend 의 단일 entry.

향후 추가될 endpoint 들 (PRD §14-2-d 로드맵):
- /api/analyze   — services.analyze_service 위임
- /api/datasets  — 데이터셋 메타
- /api/csrf      — CSRF token 발급
- /api/health    — liveness probe (Tier 1.5 P2 현재)

services/* 의 도메인 코드는 변경 없이 재사용. Blueprint 는 thin wrapper 로만 유지.
"""
from .analyze import bp_analyze
from .csrf import bp_csrf
from .datasets import bp_datasets
from .health import bp_health

__all__ = ["bp_analyze", "bp_csrf", "bp_datasets", "bp_health"]

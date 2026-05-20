"""pytest 공통 fixture — Tier 1.5 P0 부터.

DB fixture 정책 (PRD §13.1.b T1.5 P0 — 외부 리뷰 합의안):
- Unit test 는 psycopg connection 을 mock — CI smoke 의 외부 의존 0 원칙 유지.
- Integration test 는 OMNIPDM_TEST_DB_URL 환경변수 설정 시에만 동작.
  미설정 시 pytest.skip 으로 자동 우회 → 기존 CI 영향 0.
- 각 통합 테스트는 autocommit=False + teardown ROLLBACK 으로 DB 상태 격리.
  psycopg v3 는 autocommit=False 일 때 첫 execute 시 implicit BEGIN — 명시 BEGIN 불요.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pytest

if TYPE_CHECKING:
    from psycopg import Connection  # 타입 힌트 전용 — 런타임 import 는 importorskip 으로.

# 프로젝트 루트를 sys.path 맨 앞에 추가 (기존 test 파일들과 동일 패턴).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture(scope="session")
def integration_db_url() -> str:
    """통합 테스트용 DB URL — OMNIPDM_TEST_DB_URL 환경변수에서 읽음.

    예: postgresql://omnipdm:omnipdm@127.0.0.1:5432/omnipdm

    미설정이면 skip — CI smoke 에서는 외부 의존 0 유지.
    """
    url = os.getenv("OMNIPDM_TEST_DB_URL")
    if not url:
        pytest.skip(
            "OMNIPDM_TEST_DB_URL not set — integration test skipped. "
            "Set to a TimescaleDB URL (e.g. postgresql://omnipdm:omnipdm@127.0.0.1:5432/omnipdm) to enable."
        )
    return url


@pytest.fixture
def db_conn(integration_db_url) -> "Iterator[Connection]":
    """통합 테스트용 psycopg connection.

    autocommit=False + 명시 ROLLBACK 으로 테스트 간 DB 상태 격리.
    psycopg 미설치 시 skip — integration_db_url 가 있더라도.
    """
    psycopg = pytest.importorskip("psycopg")
    conn = psycopg.connect(integration_db_url, autocommit=False)
    try:
        yield conn
    finally:
        try:
            conn.rollback()
        finally:
            conn.close()

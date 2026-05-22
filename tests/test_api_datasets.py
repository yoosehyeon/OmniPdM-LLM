"""GET /api/datasets smoke test.

- 200 + JSON array
- 5개 카탈로그 모두 직렬화
- 인증 미요구 (가드 면제)
- 호출 사이에 mutate 시도해도 다음 호출에 영향 없음 (mutate-safe)
"""
from __future__ import annotations

import pytest


@pytest.fixture
def flask_client(monkeypatch):
    # 가드 켜진 상태에서도 /api/datasets 는 통과해야 함 (exempt prefix).
    monkeypatch.setenv("OMNIPDM_AUTH_REQUIRED", "true")
    from app import create_app
    return create_app().test_client()


def test_datasets_returns_array(flask_client):
    resp = flask_client.get("/api/datasets")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert len(data) == 5


def test_datasets_schema(flask_client):
    data = flask_client.get("/api/datasets").get_json()
    for ds in data:
        assert "id" in ds and isinstance(ds["id"], str)
        assert "label" in ds and isinstance(ds["label"], str)
        assert "sensors" in ds and isinstance(ds["sensors"], list)
        assert len(ds["sensors"]) > 0
        for s in ds["sensors"]:
            assert {"name", "defaultValue", "min", "max", "unit"} <= set(s.keys())


def test_datasets_exempt_from_auth(flask_client):
    # 세션 쿠키 없이 호출했을 때 302 redirect 아니어야 함.
    resp = flask_client.get("/api/datasets", follow_redirects=False)
    assert resp.status_code == 200


def test_datasets_mutate_safe(flask_client):
    # 응답을 수정해도 다음 호출에는 영향 없음.
    r1 = flask_client.get("/api/datasets").get_json()
    r1[0]["sensors"][0]["min"] = -999
    r2 = flask_client.get("/api/datasets").get_json()
    assert r2[0]["sensors"][0]["min"] != -999

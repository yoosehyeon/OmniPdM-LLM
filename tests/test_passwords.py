"""services.auth.passwords unit tests — Tier 1.5 P1-a.

CI smoke 외부 의존 0 — werkzeug 는 Dash transitive 의존이라 환경 보장.
"""

from __future__ import annotations

import pytest

from services.auth.passwords import (
    PLACEHOLDER_HASH,
    hash_password,
    is_placeholder_hash,
    verify_password,
)


class TestHashPassword:
    def test_returns_non_empty_string(self):
        h = hash_password("correct horse battery staple")
        assert isinstance(h, str)
        assert len(h) > 20

    def test_different_salt_each_call(self):
        # 같은 평문이라도 salt 가 매번 새로 생성 → hash 결과 달라야 함.
        h1 = hash_password("samepassword")
        h2 = hash_password("samepassword")
        assert h1 != h2

    def test_empty_password_rejected(self):
        with pytest.raises(ValueError):
            hash_password("")

    def test_non_str_rejected(self):
        with pytest.raises(TypeError):
            hash_password(12345)  # type: ignore[arg-type]


class TestVerifyPassword:
    def test_roundtrip_match(self):
        h = hash_password("hunter2")
        assert verify_password(h, "hunter2") is True

    def test_wrong_password(self):
        h = hash_password("hunter2")
        assert verify_password(h, "hunter3") is False

    def test_placeholder_hash_always_rejected(self):
        # placeholder 가 들어 있으면 어떤 평문도 통과시키면 안 됨 — 운영 진입 전 사고 방지.
        assert verify_password(PLACEHOLDER_HASH, "anything") is False
        assert verify_password(PLACEHOLDER_HASH, "admin") is False
        assert verify_password(PLACEHOLDER_HASH, "") is False

    def test_empty_hash_rejected(self):
        assert verify_password("", "anything") is False

    def test_empty_plaintext_rejected(self):
        h = hash_password("hunter2")
        assert verify_password(h, "") is False

    def test_non_str_inputs_rejected(self):
        # DB 에서 NULL 이 올라왔거나 호출자가 잘못 전달한 경우 — robust 하게 False.
        h = hash_password("hunter2")
        assert verify_password(None, "hunter2") is False  # type: ignore[arg-type]
        assert verify_password(h, None) is False  # type: ignore[arg-type]
        assert verify_password(12345, "hunter2") is False  # type: ignore[arg-type]


class TestIsPlaceholder:
    def test_recognizes_placeholder(self):
        assert is_placeholder_hash(PLACEHOLDER_HASH) is True

    def test_real_hash_not_placeholder(self):
        h = hash_password("anything")
        assert is_placeholder_hash(h) is False

    def test_empty_not_placeholder(self):
        assert is_placeholder_hash("") is False

    def test_non_str_not_placeholder(self):
        assert is_placeholder_hash(None) is False  # type: ignore[arg-type]
        assert is_placeholder_hash(12345) is False  # type: ignore[arg-type]

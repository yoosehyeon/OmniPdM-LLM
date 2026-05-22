"""Password hashing — Tier 1.5 P1-a (PRD §13.1.b 후속).

werkzeug.security 의 generate_password_hash / check_password_hash 를 얇게 감싼 wrapper.
이 모듈을 거치는 한 가지 이유는:

1. 다른 곳에서 werkzeug 를 직접 import 하지 않게 — hash 방식 (scrypt → argon2 등) 을
   미래에 바꿔야 할 때 한 곳만 수정.
2. placeholder hash (T1.5 P0 seed admin 의 `PLACEHOLDER_NOT_A_REAL_HASH_...`) 를
   verify_password 가 무조건 False 로 거부하도록 강제.
3. CI smoke 에 외부 의존 없이 호출 가능 (werkzeug 는 dash 의 transitive 의존이라 항상 있음).

werkzeug 기본 hash 방식은 2024 년부터 scrypt — `scrypt:32768:8:1$<salt>$<hash>` 포맷.
이전 버전의 pbkdf2 hash 도 check_password_hash 가 자동 인식해 검증한다 (마이그레이션 자유).
"""

from __future__ import annotations

import secrets

from werkzeug.security import check_password_hash, generate_password_hash

# T1.5 P0 seed admin 의 password_hash 컬럼에 들어있는 placeholder. 운영 진입 전 반드시
# scripts/migrations/bootstrap_admin.py 로 교체해야 한다.
PLACEHOLDER_HASH = "PLACEHOLDER_NOT_A_REAL_HASH_REPLACE_BEFORE_PRODUCTION"

# Timing-flat dummy verify 용 random hash. 모듈 import 시 1회만 생성되며 사용자가 절대
# 알 수 없는 무작위 평문에서 파생되어 어떤 입력에도 match 하지 않는다.
# `dummy_verify` 가 user 미존재 분기에서 호출되어 scrypt 비용을 강제 발생시킨다.
_DUMMY_HASH = generate_password_hash(secrets.token_urlsafe(32))


def hash_password(plaintext: str) -> str:
    """비밀번호 평문 → werkzeug 기본 방식 (현재 scrypt) hash 문자열.

    매 호출마다 salt 가 새로 생성되므로 같은 비밀번호도 결과 hash 가 다르다 (의도).
    빈 문자열은 거부 — 사고성 빈 비밀번호 차단.
    """
    if not isinstance(plaintext, str):
        raise TypeError(f"plaintext password must be str, got {type(plaintext).__name__}")
    if not plaintext:
        raise ValueError("plaintext password must not be empty")
    return generate_password_hash(plaintext)


def is_placeholder_hash(stored_hash: str) -> bool:
    """seed placeholder 인지 확인 — bootstrap_admin 실행 전 인증 시도 차단용.

    None / 비문자열 입력은 placeholder 가 아님 (False) — DB 에서 NULL 이 올라온
    경우 등 비정상 입력에 robust.
    """
    return isinstance(stored_hash, str) and stored_hash == PLACEHOLDER_HASH


def verify_password(stored_hash: str, plaintext: str) -> bool:
    """저장된 hash 와 평문 일치 여부.

    placeholder hash 는 무조건 False — 운영 진입 전 부트스트랩이 누락된 채로 로그인
    되는 사고를 차단. 빈 평문 / 비문자열 입력도 무조건 False (이중 안전).
    """
    if not isinstance(stored_hash, str) or not isinstance(plaintext, str):
        return False
    if not stored_hash or not plaintext:
        return False
    if is_placeholder_hash(stored_hash):
        return False
    return check_password_hash(stored_hash, plaintext)


def dummy_verify(plaintext: str) -> None:
    """Timing-flat dummy — user 미존재 분기에서 호출. 결과는 무시.

    `verify_password(PLACEHOLDER_HASH, ...)` 는 placeholder 분기로 즉시 False 라
    scrypt 가 실행되지 않아 user 존재 분기와 wall-clock 차이가 생긴다. 본 함수는
    실제 random scrypt hash 에 대해 check_password_hash 를 강제로 1회 실행해
    timing 평탄화 — 결과는 항상 False, 호출자는 결과 무시.
    """
    if not isinstance(plaintext, str) or not plaintext:
        # 빈 평문은 어차피 일찍 거부 — scrypt 실행 안 함. 이 분기는 호출자에서
        # 도달하지 않게 가드되어야 하지만, safety net.
        return
    check_password_hash(_DUMMY_HASH, plaintext)

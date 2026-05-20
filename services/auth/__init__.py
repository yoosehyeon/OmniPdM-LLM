"""Auth subpackage — Tier 1.5 P1 진입점.

현재 노출:
- passwords.hash_password / verify_password : werkzeug.security 얇은 wrapper.
- passwords.is_placeholder_hash             : seed admin placeholder 식별.

향후 (P1-b 이후) login session / @role_required 본문 / Flask-Login 통합 추가.
"""

from services.auth.passwords import (
    PLACEHOLDER_HASH,
    hash_password,
    is_placeholder_hash,
    verify_password,
)

__all__ = [
    "PLACEHOLDER_HASH",
    "hash_password",
    "is_placeholder_hash",
    "verify_password",
]

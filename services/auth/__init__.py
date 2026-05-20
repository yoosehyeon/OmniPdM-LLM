"""Auth subpackage — Tier 1.5 P1.

노출:
- passwords.*  : werkzeug hash wrapper + placeholder reject (P1-a)
- users.*      : UserService Protocol + Null/Timescale impls (P1-b)
- sessions.*   : Flask session 기반 login/logout/current_user/csrf (P1-b)
"""

from services.auth.passwords import (
    PLACEHOLDER_HASH,
    hash_password,
    is_placeholder_hash,
    verify_password,
)
from services.auth.sessions import (
    current_user,
    current_user_id,
    get_or_create_csrf_token,
    is_authenticated,
    login_user,
    logout_user,
    record_login_failure,
    rotate_csrf_token,
    verify_csrf_token,
)
from services.auth.users import (
    NullUserService,
    TimescaleUserService,
    User,
    UserService,
    create_default as create_default_user_service,
)

__all__ = [
    # passwords
    "PLACEHOLDER_HASH",
    "hash_password",
    "is_placeholder_hash",
    "verify_password",
    # users
    "User",
    "UserService",
    "NullUserService",
    "TimescaleUserService",
    "create_default_user_service",
    # sessions
    "current_user",
    "current_user_id",
    "get_or_create_csrf_token",
    "is_authenticated",
    "login_user",
    "logout_user",
    "record_login_failure",
    "rotate_csrf_token",
    "verify_csrf_token",
]

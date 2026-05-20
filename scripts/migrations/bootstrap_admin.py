"""Admin seed password 부트스트랩 — Tier 1.5 P1-a.

T1.5 P0 의 001_cmms_schema.sql 은 admin 행을 placeholder hash 로 시드한다 (`PLACEHOLDER_
NOT_A_REAL_HASH_...`). 이 스크립트는 그 hash 를 실제 werkzeug 기반 hash 로 교체한다.

SQL 마이그레이션 안에 hash 를 직접 박을 수 없는 이유: werkzeug 가 salt 를 매 호출 새로
생성하므로 결정적 SQL 이 되지 않고, hash 자체가 secret 이라 PR 에 노출되면 위험.
따라서 1회성 Python 스크립트로 처리한다.

비밀번호 입력 우선순위 (높은 → 낮은):
  1. --password-file PATH     : 파일 첫 줄을 평문 비밀번호로 사용 (k8s secret / docker
                                 secret 의 표준 경로. CI 로그 / 환경변수 / process list
                                 어디에도 노출되지 않는 가장 안전한 경로). POSIX 환경에서는
                                 group/other read 권한이 있으면 거부 (mode 0600 강제) —
                                 Windows 에서는 POSIX bit 가 무의미하므로 검사 생략.
  2. OMNIPDM_ADMIN_BOOTSTRAP_PASSWORD env var
                                : 자동화용. **사용 후 즉시 unset 권장** — env var 은
                                 `ps`, `docker inspect`, k8s `kubectl get pod -o yaml`,
                                 CI 로그 등에서 plaintext 로 노출될 가능성이 크다.
  3. --interactive            : stdin getpass 입력. 운영 콘솔 권장.

Exit codes:
  0 : 성공
  1 : 사용자 행위 오류 (user 미존재, hash 이미 real → --force 필요)
  2 : 환경 설정 오류 (DB 비활성, psycopg 미설치, password-file 권한 위반)
  3 : DB 오류 (connection / SQL 실행 실패)

사용법:
  # 가장 안전: 파일 (chmod 0600 필수, K8s/Docker secret 표준)
  chmod 600 /run/secrets/admin_pw
  python -m scripts.migrations.bootstrap_admin --password-file /run/secrets/admin_pw

  # 자동화 (CI 한정, 즉시 unset)
  set OMNIPDM_ADMIN_BOOTSTRAP_PASSWORD=changeme
  python -m scripts.migrations.bootstrap_admin
  set OMNIPDM_ADMIN_BOOTSTRAP_PASSWORD=

  # 대화형
  python -m scripts.migrations.bootstrap_admin --interactive

  # 미적용 확인 (실제 UPDATE 안 함)
  python -m scripts.migrations.bootstrap_admin --interactive --dry-run

옵션:
  --user-id        교체 대상 user_id (기본 'admin')
  --force          이미 hash 가 placeholder 가 아니어도 덮어씀 (비밀번호 분실 복구).
                   audit_log 에 force=True 가 기록된다 → 사후 추적 가능.
  --interactive    비밀번호를 stdin getpass 로 입력 (다른 입력 경로 없을 때)
  --password-file  비밀번호 평문이 담긴 파일 경로 (첫 줄)
  --dry-run        실제 UPDATE / audit_log INSERT 없이 "would update" 만 출력.
"""

from __future__ import annotations

import argparse
import getpass
import os
import stat
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models_core import config  # noqa: E402
from services import audit_actions  # noqa: E402
from services.audit_actions import AuditAction  # noqa: E402
from services.auth import PLACEHOLDER_HASH, hash_password  # noqa: E402


_MIN_PASSWORD_LEN = 8  # PoC 최소선 — 운영 정책은 P1-b 인증 라우트에서 강화


def _check_file_permissions(p: Path) -> None:
    """POSIX 환경에서만 0600 강제. Windows 는 POSIX bit 가 의미 없어 검사 생략.

    K8s Secret / Docker secret 의 표준 마운트 권한은 0400 또는 0600.
    group/other 에 read 가 열려 있으면 다른 프로세스가 평문을 읽을 수 있어 거부.
    """
    if sys.platform == "win32":
        return
    mode = p.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise SystemExit(
            f"ERROR: --password-file must be 0600 or stricter (no group/other access): "
            f"{p} (current: {oct(mode & 0o777)}). Run `chmod 600 {p}` first."
        )


def _read_password(args: argparse.Namespace) -> str:
    """--password-file > env > --interactive 우선순위."""
    if args.password_file:
        p = Path(args.password_file)
        if not p.is_file():
            raise SystemExit(f"ERROR: --password-file path does not exist: {p}")
        _check_file_permissions(p)
        pw = p.read_text(encoding="utf-8").splitlines()[0].strip()
        if not pw:
            raise SystemExit("ERROR: password file is empty")
        return pw

    env_pw = os.getenv("OMNIPDM_ADMIN_BOOTSTRAP_PASSWORD", "")
    if env_pw:
        return env_pw

    if not args.interactive:
        raise SystemExit(
            "ERROR: no password source. Provide one of:\n"
            "  --password-file PATH\n"
            "  OMNIPDM_ADMIN_BOOTSTRAP_PASSWORD env var\n"
            "  --interactive"
        )
    pw1 = getpass.getpass("New admin password: ")
    pw2 = getpass.getpass("Confirm: ")
    if pw1 != pw2:
        raise SystemExit("ERROR: passwords do not match")
    return pw1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap admin password hash")
    parser.add_argument("--user-id", default="admin", help="대상 user_id (기본 'admin')")
    parser.add_argument(
        "--force",
        action="store_true",
        help="placeholder 가 아닌 hash 도 덮어씀 (비밀번호 분실 복구용). audit_log 에 기록됨.",
    )
    parser.add_argument("--interactive", action="store_true", help="stdin 으로 비밀번호 입력")
    parser.add_argument("--password-file", help="비밀번호 평문이 담긴 파일 경로 (첫 줄)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 UPDATE / audit_log INSERT 없이 'would update' 만 출력.",
    )
    args = parser.parse_args(argv)

    if not config.DB_ENABLED:
        print("[bootstrap_admin] OMNIPDM_DB_ENABLED is false - refusing to run.", flush=True)
        return 2

    try:
        import psycopg
    except ImportError:
        print(
            "[bootstrap_admin] psycopg not installed - install requirements.txt before running.",
            flush=True,
        )
        return 2

    plaintext = _read_password(args)
    if len(plaintext) < _MIN_PASSWORD_LEN:
        raise SystemExit(f"ERROR: password must be at least {_MIN_PASSWORD_LEN} characters")
    new_hash = hash_password(plaintext)

    try:
        with psycopg.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            autocommit=False,
        ) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT password_hash FROM users WHERE user_id = %s",
                (args.user_id,),
            )
            row = cur.fetchone()
            if row is None:
                print(f"[bootstrap_admin] user '{args.user_id}' not found - aborting.", flush=True)
                conn.rollback()
                return 1

            current_hash = row[0]
            was_placeholder = current_hash == PLACEHOLDER_HASH
            if not was_placeholder and not args.force:
                print(
                    f"[bootstrap_admin] user '{args.user_id}' already has a real hash. "
                    f"Use --force to overwrite (e.g. password recovery).",
                    flush=True,
                )
                conn.rollback()
                return 1

            if args.dry_run:
                print(
                    f"[bootstrap_admin] DRY RUN - would update user '{args.user_id}' "
                    f"(was_placeholder={was_placeholder}, force={bool(args.force)}). "
                    f"No changes committed.",
                    flush=True,
                )
                conn.rollback()
                return 0

            cur.execute(
                "UPDATE users SET password_hash = %s WHERE user_id = %s",
                (new_hash, args.user_id),
            )

            # audit_log — bootstrap 추적용. system_event=True 가 자동 추가됨 (actor_id=None).
            audit_actions.log(
                conn,
                actor_id=None,
                action=AuditAction.USER_UPDATED,
                target_type=audit_actions.TARGET_TYPE_USER,
                target_id=args.user_id,
                meta={
                    "reason": "bootstrap_admin",
                    "force": bool(args.force),
                    "was_placeholder": was_placeholder,
                },
            )

            conn.commit()
    except psycopg.Error as e:
        # connection / SQL 실패 — context manager 가 rollback 처리. 사용자에게 명확 메시지.
        msg = getattr(e, "pgerror", None) or str(e)
        print(f"[bootstrap_admin] database error: {type(e).__name__}: {msg}", flush=True)
        return 3

    print(f"[bootstrap_admin] user '{args.user_id}' password_hash updated successfully.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

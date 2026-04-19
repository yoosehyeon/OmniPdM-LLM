"""HF Model Hub repo 생성 + 체크포인트/meta 업로드 (1회용).

업로드 대상: models_core/artifacts/checkpoints/*_050422.{pt,pkl,json}
Repo: yusehyeon/hybridpdm-checkpoints (public, model type)
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "yusehyeon/hybridpdm-checkpoints"
REPO_TYPE = "model"
CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "models_core" / "artifacts" / "checkpoints"
GLOB_PATTERNS = ("*_050422.pt", "*_050422.pkl", "*_050422.json", "*_050422_meta.json")


def main() -> None:
    api = HfApi()

    print(f"[1/3] create_repo {REPO_ID} (public, {REPO_TYPE}) ...")
    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=False,
        exist_ok=True,
    )

    files: list[Path] = []
    for pattern in GLOB_PATTERNS:
        files.extend(sorted(CHECKPOINT_DIR.glob(pattern)))
    files = sorted(set(files))

    if not files:
        print("[abort] no files matched")
        sys.exit(1)

    total_bytes = sum(f.stat().st_size for f in files)
    print(f"[2/3] uploading {len(files)} files ({total_bytes/1024/1024:.2f} MB) ...")
    for f in files:
        print(f"        - {f.name} ({f.stat().st_size/1024:.1f} KB)")

    api.upload_folder(
        folder_path=str(CHECKPOINT_DIR),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=list(GLOB_PATTERNS),
        commit_message="Upload checkpoints + meta.json (20260417_050422)",
    )

    print(f"[3/3] done: https://huggingface.co/{REPO_ID}/tree/main")


if __name__ == "__main__":
    main()

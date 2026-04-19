"""체크포인트 meta.json 추출 (HF Model Hub 업로드용).

각 데이터셋 로더를 호출해 얻은 meta 딕셔너리를
`models_core/artifacts/checkpoints/<stem>_meta.json` 으로 저장한다.

런타임(HF Space) 에서 dataset 없이 모델을 로드할 수 있도록
pdm_service._load_model() 이 이 meta 파일을 읽도록 전환할 예정.

실행: python scripts/export_checkpoint_meta.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models_core import config  # noqa: E402
from models_core import data_pipeline as dp  # noqa: E402

STEM_TO_LOADER = {
    "ai4i_cnn": dp.load_ai4i_cnn,
    "ai4i_gbdt": dp.load_ai4i_gbdt,
    "cwru_cnn": dp.load_cwru_cnn,
    "hydraulic_ae": dp.load_hydraulic_ae,
    "cmapss_lstm": dp.load_cmapss_lstm,
    "ncmapss_lstm": dp.load_ncmapss_lstm,
}


def _find_latest_checkpoint(stem: str) -> Path | None:
    for suffix in (".pt", ".pkl"):
        candidates = sorted(config.CHECKPOINT_DIR.glob(f"{stem}*{suffix}"))
        if candidates:
            return candidates[-1]
    return None


def main() -> None:
    for stem, loader in STEM_TO_LOADER.items():
        ckpt = _find_latest_checkpoint(stem)
        if ckpt is None:
            print(f"[skip] {stem}: no checkpoint found")
            continue

        meta_path = ckpt.with_name(f"{ckpt.stem}_meta.json")
        print(f"[{stem}] loading dataset ...")
        data = loader()
        meta = dict(data["meta"])

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[{stem}] wrote {meta_path.name} ({len(json.dumps(meta))} bytes)")


if __name__ == "__main__":
    main()

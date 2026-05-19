"""HybridPdM - 재평가 스크립트 (학습 없이 기존 체크포인트로 새 metric만 추출).

용도:
    evaluate.py 에 새 metric 추가 (NASA Score, Confusion Matrix, PR-AUC 등) 후,
    재학습 없이 기존 체크포인트로 평가만 다시 돌려 MLflow 에 reeval 태그로 기록.

사용 예:
    python -m scripts.training.reeval                       # 모든 데이터셋
    python -m scripts.training.reeval --datasets ai4i_cnn   # 특정 데이터셋만
    python -m scripts.training.reeval --no-mlflow           # MLflow 미사용
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from models_core import config
from models_core import data_pipeline as dp
from models_core import models
from scripts.training import evaluate as ev
from scripts.training import mlflow_logger as mlf
from scripts.training.main import PIPELINE


def _find_latest_checkpoint(stem: str, suffix: str) -> Path | None:
    candidates = sorted(config.CHECKPOINT_DIR.glob(f"{stem}*{suffix}"))
    return candidates[-1] if candidates else None


def _load_meta(stem: str) -> Dict:
    """체크포인트 meta json 우선 로드. 없으면 빈 dict."""
    meta_path = _find_latest_checkpoint(stem, "_meta.json")
    if meta_path and meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _load_model(name: str, data: Dict) -> object | None:
    """기존 체크포인트에서 모델 복원. PIPELINE 의 build_fn 으로 architecture 만들고
    state_dict 로드. 실패 시 None 반환."""
    if name not in PIPELINE:
        print(f"  [SKIP] unknown dataset_key: {name}")
        return None

    _, _, build_fn = PIPELINE[name]

    if name == "ai4i_gbdt":
        ckpt = _find_latest_checkpoint("ai4i_gbdt", ".pkl")
        if ckpt is None:
            print(f"  [SKIP] no checkpoint: {name}")
            return None
        with open(ckpt, "rb") as f:
            return pickle.load(f)

    # 체크포인트 stem 결정: cmapss_dlinear → cmapss_dlinear*.pt
    stem = name
    ckpt = _find_latest_checkpoint(stem, ".pt")
    if ckpt is None:
        print(f"  [SKIP] no checkpoint: {name} ({stem}*.pt)")
        return None

    try:
        model = build_fn(data)
        state = torch.load(ckpt, map_location=config.get_device())
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as e:
        print(f"  [FAIL] model load: {type(e).__name__}: {e}")
        return None


def reeval_one(name: str, pipeline_run_id: str) -> Dict:
    """단일 데이터셋 재평가 — 학습 없이 evaluate 만 수행하고 MLflow 에 기록."""
    if name not in PIPELINE:
        return {"name": name, "status": "unknown"}
    train_task, eval_task, _ = PIPELINE[name]

    with mlf.start_run(
        dataset_key=name,
        run_id=pipeline_run_id,
        experiment_name="hybridpdm_reeval",
        tags={
            "train_task": train_task,
            "eval_task": eval_task,
            "reeval": "true",
        },
    ):
        # 1) 데이터 로드
        try:
            data = dp.LOADERS[name]()
        except FileNotFoundError as e:
            print(f"  [SKIP] data missing: {e}")
            return {"name": name, "status": "skipped", "reason": str(e)}
        except Exception as e:
            print(f"  [FAIL] data load: {e}")
            return {"name": name, "status": "load_failed", "error": str(e)}

        # 2) 모델 복원
        model = _load_model(name, data)
        if model is None:
            return {"name": name, "status": "no_checkpoint"}

        # 3) 평가 (새 metric 포함)
        try:
            eval_fn = ev.EVALUATORS[eval_task]
            if eval_task == "binary_classification":
                eval_metrics = eval_fn(
                    name, data, model,
                    decision_threshold=config.CNN_CFG["decision_threshold"],
                )
            elif eval_task == "anomaly_detection":
                eval_metrics = eval_fn(name, data, model, use_mahalanobis=False)
            else:
                eval_metrics = eval_fn(name, data, model)
        except Exception as e:
            print(f"  [FAIL] eval: {e}\n{traceback.format_exc()}")
            mlf.set_tag("status", "eval_failed")
            return {"name": name, "status": "eval_failed", "error": str(e)}

        mlf.log_metrics(eval_metrics, prefix="eval")
        mlf.set_tag("status", "ok")

        # 요약 출력
        summary_keys = (
            "rmse", "mae", "r2", "nasa_score_mean", "nasa_score_sum",
            "accuracy", "f1", "precision", "recall",
            "pr_auc", "roc_auc", "false_alarm_rate",
            "test_f1", "best_threshold", "best_percentile",
        )
        summary = {k: v for k, v in eval_metrics.items() if k in summary_keys}
        print(f"  [OK] {summary}")
        return {"name": name, "status": "ok", "eval": eval_metrics}


def main():
    parser = argparse.ArgumentParser(description="HybridPdM 재평가 (학습 없이 새 metric 만 추출)")
    parser.add_argument(
        "--datasets", nargs="+", default=list(PIPELINE.keys()),
        help="평가할 데이터셋 키",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="MLflow 비활성화")
    args = parser.parse_args()

    if args.no_mlflow:
        import os
        os.environ["DISABLE_MLFLOW"] = "1"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_reeval"
    print(f"[reeval] run_id = {run_id}")
    print(f"[reeval] datasets = {args.datasets}")

    results: List[Dict] = []
    for name in args.datasets:
        print(f"\n=== {name} ===")
        res = reeval_one(name, pipeline_run_id=run_id)
        results.append(res)

    out_path = config.REPORT_DIR / f"reeval_report_{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[reeval] report saved -> {out_path}")


if __name__ == "__main__":
    main()

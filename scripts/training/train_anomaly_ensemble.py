"""HybridPdM - 이상 탐지 앙상블 스크립트 (성능 최대화).

3가지 핵심 설계 결정:

1. **Rank-based Fusion** (Z-score normalization 대신):
   AE / VAE / IF 점수의 스케일이 매우 다름. Rank fusion 은 분포에 robust.
   (Tax & Duin 2002, "Combining classifiers")

2. **PR Curve 기반 Threshold 최적화** (F1 percentile grid 대신):
   precision_recall_curve 의 F1-best point 가 연속적 최적값.

3. **Validation Threshold + Test Evaluation** (데이터 leakage 제거) ⭐:
   - Validation set 에서 threshold 결정
   - Test set 에서 최종 평가만 (threshold 고정)
   - test set 으로 threshold 구하면 test leakage → 학술 무효

평가 — 7가지 조합 비교:
   AE 단독, VAE 단독, IF 단독 (3 단독)
   AE+VAE, AE+IF, VAE+IF (2-way 3개)
   AE+VAE+IF (3-way 1개)

사용:
    python -m scripts.training.train_anomaly_ensemble
    python -m scripts.training.train_anomaly_ensemble --seeds 42 43 44
    python -m scripts.training.train_anomaly_ensemble --weights 1.2 1.0 0.8  # AE, VAE, IF
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)

from models_core import config
from models_core import data_pipeline as dp
from models_core import models
from scripts.training import mlflow_logger as mlf
from scripts.training import train as tr


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 7가지 조합 (단독 3 + 2-way 3 + 3-way 1)
# ---------------------------------------------------------------------------
COMBOS: List[str] = [
    "AE",
    "VAE",
    "IF",
    "AE+VAE",
    "AE+IF",
    "VAE+IF",
    "AE+VAE+IF",
]


# ---------------------------------------------------------------------------
# 점수 추출 헬퍼 (모두 batched)
# ---------------------------------------------------------------------------

def _extract_ae_scores(model, X: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    """DenoisingAE reconstruction_error."""
    model.eval()
    model.to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            out.append(model.reconstruction_error(xb).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0,), dtype=np.float32)


def _extract_vae_scores(model, X: np.ndarray, device, mode: str = "elbo_neg",
                       batch_size: int = 256) -> np.ndarray:
    """VAE anomaly_score (batched, OOM-safe)."""
    model.eval()
    model.to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            out.append(model.anomaly_score(xb, mode=mode).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0,), dtype=np.float32)


def _extract_if_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Isolation Forest: -decision_function (높을수록 이상)."""
    return -model.decision_function(X)


# ---------------------------------------------------------------------------
# Rank-based Fusion
# ---------------------------------------------------------------------------

def _to_rank(scores: np.ndarray) -> np.ndarray:
    """0~1 정규화된 rank (높을수록 이상)."""
    if len(scores) == 0:
        return scores
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores))
    return ranks / max(len(scores) - 1, 1)


def rank_fuse(*score_arrays: np.ndarray,
              weights: Optional[List[float]] = None) -> np.ndarray:
    """여러 점수 배열을 rank-based fusion. weights=None 이면 equal weight."""
    if not score_arrays:
        return np.empty((0,), dtype=np.float64)
    n = len(score_arrays[0])
    ranks = [_to_rank(s) for s in score_arrays]
    if weights is None:
        weights = [1.0] * len(score_arrays)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    fused = np.zeros(n, dtype=np.float64)
    for r, wi in zip(ranks, w):
        fused += r * wi
    return fused


# ---------------------------------------------------------------------------
# PR Curve 기반 최적 threshold (Validation 전용)
# ---------------------------------------------------------------------------

def find_best_threshold_pr(scores: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """precision_recall_curve 에서 F1-best threshold + best F1 반환.

    ⭐ Validation set 에서만 호출 — test set 에서 호출하면 data leakage.
    """
    if len(np.unique(y_true)) < 2:
        return float(np.median(scores)), 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall + 1e-12),
            0.0,
        )
    f1 = f1[:-1] if len(f1) > len(thresholds) else f1
    if len(f1) == 0 or len(thresholds) == 0:
        return float(np.median(scores)), 0.0
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def evaluate_at_threshold(scores: np.ndarray, y_true: np.ndarray,
                         threshold: float, prefix: str = "test") -> Dict:
    """고정 threshold 에서 metric 계산."""
    y_pred = (scores > threshold).astype(np.int32)
    metrics: Dict[str, float] = {
        f"{prefix}_threshold": float(threshold),
        f"{prefix}_accuracy":  float(accuracy_score(y_true, y_pred)),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics[f"{prefix}_pr_auc"]  = float(average_precision_score(y_true, scores))
        metrics[f"{prefix}_roc_auc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        pass
    return metrics


# ---------------------------------------------------------------------------
# 모델 학습 / 로드
# ---------------------------------------------------------------------------

def train_or_load_ae(name: str, data: Dict, device, force_retrain: bool = False):
    """기존 체크포인트가 있으면 로드, 없으면 학습 (`{name}_*.pt` 안전 glob)."""
    ck = sorted(config.CHECKPOINT_DIR.glob(f"{name}_*.pt"))
    if ck and not force_retrain:
        ckpt = ck[-1]
        m = models.build_model(
            "ae",
            input_dim=data["meta"]["feature_dim"],
            latent_dim=config.AE_CFG["latent_dim"],
        )
        state = torch.load(ckpt, map_location=device)
        m.load_state_dict(state)
        m.eval()
        logger.info(f"[AE] loaded checkpoint: {ckpt.name}")
        return m
    logger.info(f"[AE] training new (no checkpoint or force_retrain)")
    m = models.build_model(
        "ae",
        input_dim=data["meta"]["feature_dim"],
        latent_dim=config.AE_CFG["latent_dim"],
    )
    tr.train_autoencoder(name, data, m, cfg=config.AE_CFG)
    return m


def train_vae_model(name: str, data: Dict, device):
    """VAE 신규 학습 (beta 등 변동 가능성 → 체크포인트 재사용 안 함)."""
    cfg = config.VAE_CFG
    m = models.build_model(
        "vae",
        input_dim=data["meta"]["feature_dim"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        beta=cfg["beta"],
    )
    tr.train_vae(name, data, m, cfg=cfg)
    return m


def train_if_model(X_train: np.ndarray, seed: int) -> IsolationForest:
    """Isolation Forest — random_state 는 seed 로 (multi-seed 다양성 확보)."""
    cfg = config.IFOREST_CFG
    m = IsolationForest(
        n_estimators=cfg["n_estimators"],
        max_samples=cfg["max_samples"],
        contamination=cfg["contamination"],
        random_state=seed,                       # multi-seed 시 다른 random_state
        n_jobs=cfg["n_jobs"],
    )
    m.fit(X_train)
    return m


# ---------------------------------------------------------------------------
# 메인 실험 (Validation Threshold + Test Evaluation)
# ---------------------------------------------------------------------------

def run_ensemble_experiment(
    dataset: str,
    name_ae: str,
    name_vae: str,
    data: Dict,
    run_id: str,
    seed: int = 42,
    weights: Optional[List[float]] = None,
    force_retrain_ae: bool = False,
) -> Dict:
    """Validation 에서 threshold 결정 → Test 평가 (leakage 제거)."""
    config.set_seed(seed)
    device = torch.device(config.get_device())
    # seed 를 run_id 에 포함 — MLflow UI 가독성 ↑ (run_name 중복 방지)
    seed_run_id = f"{run_id}_seed{seed}"
    logger.info(f"[seed={seed}] Starting ensemble experiment (sub_run_id={seed_run_id})...")

    # 1) 3 모델 학습
    ae_model  = train_or_load_ae(name_ae, data, device, force_retrain=force_retrain_ae)
    vae_model = train_vae_model(name_vae, data, device)
    if_model  = train_if_model(data["X_train"], seed=seed)

    # 2) Validation + Test 점수 추출 (각각 별도)
    X_val,  y_val  = data["X_val"],  np.asarray(data["y_val"]).astype(np.int32)
    X_test, y_test = data["X_test"], np.asarray(data["y_test"]).astype(np.int32)

    vae_mode = config.VAE_CFG.get("anomaly_score_mode", "elbo_neg")

    # Validation 점수
    s_ae_v  = _extract_ae_scores(ae_model, X_val, device)
    s_vae_v = _extract_vae_scores(vae_model, X_val, device, mode=vae_mode)
    s_if_v  = _extract_if_scores(if_model, X_val)

    # Test 점수
    s_ae_t  = _extract_ae_scores(ae_model, X_test, device)
    s_vae_t = _extract_vae_scores(vae_model, X_test, device, mode=vae_mode)
    s_if_t  = _extract_if_scores(if_model, X_test)

    # 3) 7가지 조합별 점수 (validation / test 각각)
    val_scores: Dict[str, np.ndarray] = {
        "AE":        s_ae_v,
        "VAE":       s_vae_v,
        "IF":        s_if_v,
        "AE+VAE":    rank_fuse(s_ae_v, s_vae_v),
        "AE+IF":     rank_fuse(s_ae_v, s_if_v),
        "VAE+IF":    rank_fuse(s_vae_v, s_if_v),
        "AE+VAE+IF": rank_fuse(s_ae_v, s_vae_v, s_if_v, weights=weights),
    }
    test_scores: Dict[str, np.ndarray] = {
        "AE":        s_ae_t,
        "VAE":       s_vae_t,
        "IF":        s_if_t,
        "AE+VAE":    rank_fuse(s_ae_t, s_vae_t),
        "AE+IF":     rank_fuse(s_ae_t, s_if_t),
        "VAE+IF":    rank_fuse(s_vae_t, s_if_t),
        "AE+VAE+IF": rank_fuse(s_ae_t, s_vae_t, s_if_t, weights=weights),
    }

    # 4) Validation 에서 threshold 결정 → Test 평가
    results: Dict[str, Dict] = {}
    logger.info(f"  Evaluating {len(COMBOS)} combinations (threshold from validation):")
    for combo in COMBOS:
        thr, best_val_f1 = find_best_threshold_pr(val_scores[combo], y_val)
        val_metrics  = evaluate_at_threshold(val_scores[combo],  y_val,  thr, prefix="val")
        test_metrics = evaluate_at_threshold(test_scores[combo], y_test, thr, prefix="test")
        combined = {"threshold": thr, "val_f1_best": best_val_f1, **val_metrics, **test_metrics}
        results[combo] = combined
        logger.info(
            f"    {combo:12s}  val_F1={best_val_f1:.4f} | "
            f"test_F1={test_metrics['test_f1']:.4f}  "
            f"P={test_metrics['test_precision']:.3f}  R={test_metrics['test_recall']:.3f}  "
            f"PR-AUC={test_metrics.get('test_pr_auc', -1):.4f}"
        )

    # 5) MLflow 기록 (조합당 별도 run, run_id 에 seed 포함)
    for combo, m in results.items():
        with mlf.start_run(
            dataset_key=f"{dataset}_ensemble_{combo.replace('+', '_')}",
            run_id=seed_run_id,                  # seed 포함된 sub run_id
            experiment_name=f"{dataset}_ensemble",
            tags={
                "combo":            combo,
                "n_models":         str(combo.count("+") + 1),
                "fusion_method":    "rank_fusion",
                "threshold_source": "validation",
                "threshold_opt":    "pr_curve_f1_best",
                "seed":             str(seed),
                "weights":          ",".join(map(str, weights)) if weights else "equal",
            },
        ):
            mlf.log_params({
                "combo":          combo,
                "seed":           seed,
                "n_val":          int(len(y_val)),
                "n_test":         int(len(y_test)),
                "weights":        str(weights or "equal"),
                "vae_score_mode": vae_mode,
            })
            mlf.log_metrics(m)

    return {
        "seed":    seed,
        "results": results,
        "n_val":   int(len(y_val)),
        "n_test":  int(len(y_test)),
        "weights": weights or "equal",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HybridPdM Anomaly Ensemble (Validation Threshold + Rank Fusion)"
    )
    parser.add_argument(
        "--dataset", type=str, default="hydraulic",
        choices=["hydraulic"],
        help="데이터셋 이름 (hydraulic_ae / hydraulic_vae 로더 사용)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="seed 목록 (multi-seed 안정성)")
    parser.add_argument("--weights", type=float, nargs=3, default=None,
                        metavar=("W_AE", "W_VAE", "W_IF"),
                        help="3-way fusion weights (AE VAE IF), 기본 equal")
    parser.add_argument("--force-retrain-ae", action="store_true",
                        help="AE 체크포인트 무시하고 재학습")
    args = parser.parse_args()

    logger.info("=== HybridPdM Anomaly Ensemble ===")
    logger.info(f"dataset: {args.dataset} | seeds: {args.seeds} | weights: {args.weights}")

    loader_key = f"{args.dataset}_ae"
    data = dp.LOADERS[loader_key]()
    logger.info(
        f"Loaded {args.dataset}: train={len(data['X_train'])} "
        f"val={len(data['X_val'])} test={len(data['X_test'])} "
        f"features={data['meta']['feature_dim']}"
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"run_id = {run_id}")

    all_results = []
    for seed in args.seeds:
        res = run_ensemble_experiment(
            dataset=args.dataset,
            name_ae=f"{args.dataset}_ae",
            name_vae=f"{args.dataset}_vae",
            data=data,
            run_id=run_id,
            seed=seed,
            weights=args.weights,
            force_retrain_ae=args.force_retrain_ae,
        )
        all_results.append(res)

    # Multi-seed summary
    if len(args.seeds) > 1:
        logger.info(f"\n=== Multi-seed summary (n={len(args.seeds)}) ===")
        for combo in COMBOS:
            test_f1s = np.array([r["results"][combo]["test_f1"] for r in all_results])
            val_f1s  = np.array([r["results"][combo]["val_f1_best"] for r in all_results])
            logger.info(
                f"  {combo:12s}  val_F1={val_f1s.mean():.4f}+/-{val_f1s.std():.4f}  "
                f"test_F1={test_f1s.mean():.4f}+/-{test_f1s.std():.4f}"
            )

    out_path = config.REPORT_DIR / f"anomaly_ensemble_{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n[ensemble] report saved -> {out_path}")


if __name__ == "__main__":
    main()

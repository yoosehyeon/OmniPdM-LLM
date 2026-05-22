"""AI4I CNN recall variant — fixed-threshold re-evaluation + threshold sweep.

목적 (PRD §13.2 / §16.1 후속):
- `CNN_CFG_RECALL` 의 의도 (`decision_threshold=0.20` 로 recall↑) 가 evaluate_classifier
  의 val_f1 grid search 로 0.60 으로 덮어씌워져 실제 평가 결과 (recall 평균 0.732) 가
  의도된 trade-off 를 반영 못함.
- 추가 학습 없이 multi-seed checkpoint 3개 (seed=42/43/44) 를 load + threshold sweep —
  운영 가능한 (recall ≥ 0.85, false_alarm ≤ 0.20) 임계값 추천.
- baseline (CNN_CFG) checkpoint 도 동일 방식 — apples-to-apples 비교.

Probs caching: checkpoint 당 inference 1회 후 probs 재사용 — sweep 비용을 30배+ 감소.

실행:
    python -m scripts.training.reeval_ai4i_recall_threshold
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models_core import config
from models_core import data_pipeline as dp
from scripts.training.main import PIPELINE


# 운영 권장점 판정 기준 — recall variant 의 의도된 trade-off.
SWEEP_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
TARGET_RECALL = 0.85         # PRD §16.1 KPI
MAX_FALSE_ALARM = 0.20       # 운영 허용선 (1500 row test set 기준 ~300 FP 한도)


def _read_training_threshold(ckpt_path: Path) -> Optional[float]:
    """체크포인트 옆의 metrics json 에서 학습 시점 decision_threshold 를 읽는다.

    "왜 0.60 으로 평가됐는지" 진단 가시화.
    """
    meta_path = ckpt_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        d = json.loads(meta_path.read_text(encoding="utf-8"))
        return float(d.get("decision_threshold", 0.0)) or None
    except Exception:
        return None


def _compute_probs_once(
    ckpt_path: Path, build_fn, data: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """checkpoint 1개에 대해 sigmoid probs 를 한 번만 계산.

    sweep 의 모든 threshold 는 이 probs 만 재사용 — inference 비용 N → 1.
    """
    device = torch.device(config.get_device())
    model = build_fn(data)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    X_test = data["X_test"]
    y_true = np.asarray(data["y_test"]).astype(np.int32)

    with torch.no_grad():
        logits_list = []
        bs = 256
        for i in range(0, len(X_test), bs):
            xb = torch.as_tensor(X_test[i : i + bs], dtype=torch.float32, device=device)
            out = model(xb).detach().cpu().numpy()
            logits_list.append(out)
        logits = np.concatenate(logits_list, axis=0).reshape(-1)

    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs, y_true


def _eval_from_probs(
    probs: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    name: str,
    seed: Optional[int] = None,
) -> Dict:
    """이미 계산된 probs 로 threshold 만 적용해 metric 계산."""
    y_pred = (probs > threshold).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp = cm[0]
    fn, tp = cm[1]

    return {
        "config_name": name,
        "seed": seed,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "confusion_matrix": cm,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "n_test": int(len(y_true)),
    }


def _mean_std(rows: List[Dict], key: str) -> Dict[str, float]:
    vals = [r[key] for r in rows]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def main() -> None:
    print("[reeval] Fixed-threshold evaluation for ai4i_cnn / ai4i_cnn_recall")
    print(f"[reeval] CNN_CFG threshold        = {config.CNN_CFG['decision_threshold']}")
    print(f"[reeval] CNN_CFG_RECALL threshold = {config.CNN_CFG_RECALL['decision_threshold']}")
    print()

    data = dp.LOADERS["ai4i_cnn"]()
    print(f"[reeval] X_test shape = {np.asarray(data['X_test']).shape}, "
          f"y_test positive rate = {np.mean(data['y_test']):.4f}")
    print()

    _, _, build_fn = PIPELINE["ai4i_cnn_recall"]
    results: List[Dict] = []

    # ----- baseline ai4i_cnn (seed=42 단일) — probs caching 적용 -----
    base_ckpts = sorted(config.CHECKPOINT_DIR.glob("ai4i_cnn_2026*.pt"))
    base_ckpts = [p for p in base_ckpts if "ai4i_cnn_recall" not in p.name]
    baseline_probs: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if base_ckpts:
        latest_base = base_ckpts[-1]
        print(f"[reeval] baseline ckpt = {latest_base.name}")
        train_thr = _read_training_threshold(latest_base)
        if train_thr is not None:
            print(f"           training-time threshold (in metrics json) = {train_thr}")
        try:
            baseline_probs = _compute_probs_once(latest_base, build_fn, data)
            r = _eval_from_probs(
                baseline_probs[0], baseline_probs[1],
                threshold=config.CNN_CFG["decision_threshold"],
                name="ai4i_cnn (baseline)",
                seed=42,
            )
            results.append(r)
            print(f"  [baseline @ thr={r['threshold']}] "
                  f"recall={r['recall']:.4f} precision={r['precision']:.4f} "
                  f"f1={r['f1']:.4f} false_alarm={r['false_alarm_rate']:.4f}")
        except Exception as e:
            print(f"  [baseline] load/eval failed: {type(e).__name__}: {e}")
    else:
        print("[reeval] no baseline ai4i_cnn checkpoint found - skipping.")
    print()

    # ----- recall variant — 3-seed multi-seed, probs caching -----
    recall_ckpts = [
        ("ai4i_cnn_recall_20260519_212651.pt",        42),
        ("ai4i_cnn_recall_20260519_212651_seed43.pt", 43),
        ("ai4i_cnn_recall_20260519_212651_seed44.pt", 44),
    ]
    recall_thr = config.CNN_CFG_RECALL["decision_threshold"]
    recall_rows: List[Dict] = []
    probs_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for fname, seed in recall_ckpts:
        ckpt_path = config.CHECKPOINT_DIR / fname
        if not ckpt_path.exists():
            print(f"  [SKIP] missing: {fname}")
            continue
        train_thr = _read_training_threshold(ckpt_path)
        try:
            probs, y_true = _compute_probs_once(ckpt_path, build_fn, data)
            probs_cache[seed] = (probs, y_true)
            r = _eval_from_probs(probs, y_true, recall_thr, name="ai4i_cnn_recall", seed=seed)
            recall_rows.append(r)
            results.append(r)
            trained_at = f" (trained_with={train_thr})" if train_thr is not None else ""
            print(f"  [recall seed={seed} @ thr={recall_thr}{trained_at}] "
                  f"recall={r['recall']:.4f} precision={r['precision']:.4f} "
                  f"f1={r['f1']:.4f} false_alarm={r['false_alarm_rate']:.4f}")
        except Exception as e:
            print(f"  [recall seed={seed}] failed: {type(e).__name__}: {e}")

    # ----- recall variant aggregate at intended threshold -----
    if recall_rows:
        agg = {
            "config_name": "ai4i_cnn_recall (3-seed mean+/-std)",
            "threshold": recall_thr,
            "n_seeds": len(recall_rows),
            "recall": _mean_std(recall_rows, "recall"),
            "precision": _mean_std(recall_rows, "precision"),
            "f1": _mean_std(recall_rows, "f1"),
            "accuracy": _mean_std(recall_rows, "accuracy"),
            "false_alarm_rate": _mean_std(recall_rows, "false_alarm_rate"),
            "pr_auc": _mean_std(recall_rows, "pr_auc"),
            "roc_auc": _mean_std(recall_rows, "roc_auc"),
        }
        results.append(agg)
        print()
        print(f"[aggregate recall @ thr={recall_thr}]")
        for k in ("recall", "precision", "f1", "accuracy", "false_alarm_rate"):
            m, s = agg[k]["mean"], agg[k]["std"]
            print(f"  {k:>16s} = {m:.4f} +/- {s:.4f}")

        ok = agg["recall"]["mean"] >= TARGET_RECALL
        print()
        print(f"[KPI] AI4I CNN recall vs {TARGET_RECALL} target:")
        print(f"  mean recall = {agg['recall']['mean']:.4f}  "
              f"(target {TARGET_RECALL}, baseline 0.706 from PRD)")
        if ok:
            print(f"  STATUS: ACHIEVED (mean recall >= {TARGET_RECALL})")
        else:
            print(f"  STATUS: NOT YET (gap = {TARGET_RECALL - agg['recall']['mean']:+.4f})")

    # ----- threshold sweep — recall variant 3-seed mean (cached probs) -----
    sweep_summary: List[Dict] = []
    if probs_cache:
        print()
        print("[sweep] recall_variant - threshold trade-off (3-seed mean, cached probs)")
        print(f"  {'thr':>5s}  {'recall':>8s}  {'precision':>10s}  {'f1':>8s}  {'false_alarm':>12s}")
        for thr in SWEEP_THRESHOLDS:
            rows = [
                _eval_from_probs(p, y, thr, name=f"recall@{thr}", seed=s)
                for s, (p, y) in probs_cache.items()
            ]
            agg_thr = {
                "threshold": thr,
                "recall_mean": float(np.mean([r["recall"] for r in rows])),
                "precision_mean": float(np.mean([r["precision"] for r in rows])),
                "f1_mean": float(np.mean([r["f1"] for r in rows])),
                "false_alarm_mean": float(np.mean([r["false_alarm_rate"] for r in rows])),
            }
            sweep_summary.append(agg_thr)
            print(f"  {thr:>5.2f}  {agg_thr['recall_mean']:>8.4f}  {agg_thr['precision_mean']:>10.4f}  "
                  f"{agg_thr['f1_mean']:>8.4f}  {agg_thr['false_alarm_mean']:>12.4f}")

        feasible = [
            s for s in sweep_summary
            if s["recall_mean"] >= TARGET_RECALL and s["false_alarm_mean"] <= MAX_FALSE_ALARM
        ]
        print()
        if feasible:
            best = min(feasible, key=lambda s: s["threshold"])
            print(f"[recommended] thr={best['threshold']:.2f}: "
                  f"recall={best['recall_mean']:.4f}, precision={best['precision_mean']:.4f}, "
                  f"f1={best['f1_mean']:.4f}, false_alarm={best['false_alarm_mean']:.4f} "
                  f"(recall>={TARGET_RECALL} & false_alarm<={MAX_FALSE_ALARM})")
        else:
            print(f"[recommended] no threshold satisfies (recall>={TARGET_RECALL} AND "
                  f"false_alarm<={MAX_FALSE_ALARM}). multi-seed baseline retraining recommended.")

    # ----- baseline sweep (single seed) — 비교용 -----
    if baseline_probs is not None:
        print()
        print("[sweep] baseline ai4i_cnn - threshold trade-off (seed=42)")
        print(f"  {'thr':>5s}  {'recall':>8s}  {'precision':>10s}  {'f1':>8s}  {'false_alarm':>12s}")
        for thr in SWEEP_THRESHOLDS:
            r = _eval_from_probs(baseline_probs[0], baseline_probs[1], thr,
                                 name="baseline", seed=42)
            print(f"  {thr:>5.2f}  {r['recall']:>8.4f}  {r['precision']:>10.4f}  "
                  f"{r['f1']:>8.4f}  {r['false_alarm_rate']:>12.4f}")

    results.append({"sweep": sweep_summary})

    out_path = config.REPORT_DIR / f"reeval_ai4i_recall_threshold_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[reeval] saved -> {out_path}")


if __name__ == "__main__":
    main()

"""HybridPdM - Multi-seed 결과 분석 헬퍼.

MLflow 의 hybridpdm experiment 에서 모든 regression run 을 가져와:
  1. 모델(BiLSTM / DLinear / iTransformer) × FD(001~004) × seed 단위 결과 정리
  2. Mean +/- Std 계산 (multi-seed reproducibility)
  3. BiLSTM 기준 격차 % (RMSE / NASA Score / MAE)
  4. 인사이트 자동 추출 (최고/최악 격차, variance 큰 subset 등)
  5. JSON 저장

사용:
    python -m scripts.training.analyze_results
    python -m scripts.training.analyze_results --metric rmse
    python -m scripts.training.analyze_results --metric nasa_score_sum --csv
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from models_core import config


def _normalize_subset(row) -> str:
    """seed=42 baseline 은 cmapss_subset 태그 없이 dataset_key 만 있음 → 보정."""
    sub = row.get("tags.cmapss_subset", "")
    ds = row.get("tags.dataset_key", "")
    if not sub or sub == "-":
        if ds == "cmapss_lstm":
            return "FD001"
        if ds == "cmapss_dlinear":
            return "FD001"
        if ds == "cmapss_itransformer":
            return "FD001"
        if ds.endswith("_fd001"):
            return "FD001"
        if ds.endswith("_fd002"):
            return "FD002"
        if ds.endswith("_fd003"):
            return "FD003"
        if ds.endswith("_fd004"):
            return "FD004"
    return sub or "?"


def _normalize_seed(val) -> int:
    """seed 태그가 '-' 또는 None 이면 42 (default)."""
    if val in (None, "-", ""):
        return 42
    try:
        return int(val)
    except (TypeError, ValueError):
        return 42


def fetch_runs() -> pd.DataFrame:
    """MLflow 에서 모든 regression (FD001~004) full-epoch run 을 DataFrame 으로."""
    import mlflow

    runs = mlflow.search_runs(experiment_names=["hybridpdm"])
    if len(runs) == 0:
        return pd.DataFrame()

    df = runs[
        (runs["tags.smoke"] == "false") &
        (runs["tags.train_task"] == "regression") &
        (runs["tags.status"] == "ok")
    ].copy()

    if len(df) == 0:
        return pd.DataFrame()

    df["subset"] = df.apply(_normalize_subset, axis=1)
    df["seed"] = df["tags.seed"].apply(_normalize_seed)
    df["model"] = df["tags.model_family"]

    keep_cols = {
        "model": "model",
        "subset": "subset",
        "seed": "seed",
        "tags.dataset_key": "dataset_key",
        "metrics.eval_rmse": "rmse",
        "metrics.eval_mae": "mae",
        "metrics.eval_r2": "r2",
        "metrics.eval_nasa_score_sum": "nasa_score_sum",
        "metrics.eval_nasa_score_mean": "nasa_score_mean",
        "metrics.train_best_epoch": "best_epoch",
    }
    avail = {k: v for k, v in keep_cols.items() if k in df.columns}
    out = df[list(avail.keys())].rename(columns=avail)
    out = out.sort_values(["subset", "model", "seed"]).reset_index(drop=True)
    return out


def summarize(df: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    """모델 × subset 단위 mean +/- std 표."""
    if metric not in df.columns:
        raise ValueError(f"metric '{metric}' not in df columns: {list(df.columns)}")

    pivot = df.groupby(["subset", "model"])[metric].agg(["mean", "std", "count"]).reset_index()
    pivot["std"] = pivot["std"].fillna(0.0)
    pivot["display"] = pivot.apply(
        lambda r: f"{r['mean']:.2f} +/- {r['std']:.2f} (n={int(r['count'])})", axis=1,
    )
    table = pivot.pivot(index="subset", columns="model", values="display").fillna("-")
    # 모델 순서 고정
    order = [c for c in ["lstm", "dlinear", "itransformer"] if c in table.columns]
    return table[order]


def compute_gaps(df: pd.DataFrame, baseline_model: str = "lstm", metric: str = "rmse") -> pd.DataFrame:
    """BiLSTM(또는 지정 baseline) 기준 격차 %."""
    if metric not in df.columns:
        raise ValueError(f"metric '{metric}' not in df columns")

    means = df.groupby(["subset", "model"])[metric].mean().unstack("model")
    if baseline_model not in means.columns:
        raise ValueError(f"baseline_model '{baseline_model}' not found")

    gap = pd.DataFrame(index=means.index)
    for m in means.columns:
        if m == baseline_model:
            continue
        gap[f"{m} vs {baseline_model} (%)"] = (
            (means[m] - means[baseline_model]) / means[baseline_model] * 100
        ).round(1)
    return gap


def extract_insights(df: pd.DataFrame) -> List[str]:
    """결과에서 자동 인사이트 추출."""
    insights: List[str] = []

    # 1) 모델별 최고 / 최악 FD
    for model in df["model"].unique():
        sub_df = df[df["model"] == model]
        means_rmse = sub_df.groupby("subset")["rmse"].mean()
        if len(means_rmse) > 0:
            best_sub = means_rmse.idxmin()
            worst_sub = means_rmse.idxmax()
            insights.append(
                f"[{model}] best FD: {best_sub} (RMSE {means_rmse.min():.2f}), "
                f"worst FD: {worst_sub} (RMSE {means_rmse.max():.2f})"
            )

    # 2) 가장 variance 큰 subset
    if "nasa_score_sum" in df.columns:
        std_by_sub = (
            df[df["model"] == "lstm"]
            .groupby("subset")["nasa_score_sum"].std()
            .fillna(0)
        )
        if len(std_by_sub) > 0:
            high_var = std_by_sub.idxmax()
            insights.append(
                f"[BiLSTM] NASA Score variance 가장 큼: {high_var} "
                f"(std {std_by_sub.max():.1f}) - multi-seed 권장"
            )

    # 3) 모델 간 격차 (RMSE 기준)
    means = df.groupby(["subset", "model"])["rmse"].mean().unstack("model")
    if "lstm" in means.columns:
        for m in [c for c in ["dlinear", "itransformer"] if c in means.columns]:
            gaps = ((means[m] - means["lstm"]) / means["lstm"] * 100).round(1)
            max_gap_sub = gaps.idxmax()
            min_gap_sub = gaps.idxmin()
            insights.append(
                f"[{m} vs BiLSTM RMSE] 최대 격차: {max_gap_sub} ({gaps.max():+.1f}%), "
                f"최소 격차: {min_gap_sub} ({gaps.min():+.1f}%)"
            )

    return insights


def main():
    parser = argparse.ArgumentParser(description="HybridPdM Multi-seed 결과 분석")
    parser.add_argument(
        "--metric", default="rmse",
        choices=["rmse", "mae", "r2", "nasa_score_sum", "nasa_score_mean"],
        help="요약 표 기준 metric (기본 rmse)",
    )
    parser.add_argument("--baseline", default="lstm", choices=["lstm", "dlinear", "itransformer"],
                        help="격차 계산용 baseline 모델 (기본 lstm)")
    parser.add_argument("--csv", action="store_true", help="추가로 CSV 파일 저장")
    args = parser.parse_args()

    df = fetch_runs()
    if df.empty:
        print("[analyze_results] No regression runs found in MLflow.")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print(f"\n=== All regression runs (n={len(df)}) ===")
    print(df.to_string(index=False))

    # ── 요약 표 ─────────────────────────────────────────────────────────
    print(f"\n=== {args.metric} (mean +/- std) - model × subset ===")
    summary = summarize(df, metric=args.metric)
    print(summary.to_string())

    # ── 격차 표 ─────────────────────────────────────────────────────────
    available_models = df["model"].unique()
    if args.baseline in available_models and len(available_models) > 1:
        print(f"\n=== Gap (%) vs {args.baseline} on {args.metric} ===")
        gaps = compute_gaps(df, baseline_model=args.baseline, metric=args.metric)
        print(gaps.to_string())

    # ── 인사이트 ────────────────────────────────────────────────────────
    print("\n=== Auto Insights ===")
    for line in extract_insights(df):
        print(f"  - {line}")

    # ── 저장 ────────────────────────────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = config.REPORT_DIR / f"analyze_results_{run_id}.json"
    payload = {
        "generated_at": run_id,
        "metric": args.metric,
        "baseline": args.baseline,
        "n_runs": int(len(df)),
        "raw": df.to_dict(orient="records"),
        "summary": summary.reset_index().to_dict(orient="records"),
        "insights": extract_insights(df),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[analyze_results] JSON saved -> {out_json}")

    if args.csv:
        out_csv = config.REPORT_DIR / f"analyze_results_{run_id}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[analyze_results] CSV  saved -> {out_csv}")


if __name__ == "__main__":
    main()

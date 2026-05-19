"""HybridPdM - 학습/평가 파이프라인 진입점.

사용 예:
    python -m scripts.training.main
    python -m scripts.training.main --datasets ai4i_cnn cmapss_lstm
    python -m scripts.training.main --skip-explain
    python -m scripts.training.main --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from models_core import config
from models_core import data_pipeline as dp
from models_core import models
from scripts.training import evaluate as ev
from scripts.training import mlflow_logger as mlf
from scripts.training import train as tr


# ---------------------------------------------------------------------------
# 데이터셋 → 모델/태스크 매핑
# ---------------------------------------------------------------------------
# 각 항목은 (loader_key, task, build_model_fn) 의 의미.
# build_model_fn은 data dict를 받아 모델을 즉시 빌드한다 (feature_dim 반영).

def _build_ai4i_cnn(data: Dict):
    feat_names = data["meta"]["feature_names"]
    reorder = models.build_reorder_index(feat_names, models.AI4I_REORDER_NAMES)
    return models.build_model(
        "cnn_tabular",
        in_channels=1,
        seq_len=data["meta"]["feature_dim"],
        n_classes=1,
        dropout=config.CNN_CFG["dropout"],
        reorder_index=reorder,
        expected_feature_names=feat_names,
    )


def _build_ai4i_cnn_recall(data: Dict):
    """AI4I CNN — recall 강화 variant. 모델 구조 동일, dropout 만 변경."""
    feat_names = data["meta"]["feature_names"]
    reorder = models.build_reorder_index(feat_names, models.AI4I_REORDER_NAMES)
    return models.build_model(
        "cnn_tabular",
        in_channels=1,
        seq_len=data["meta"]["feature_dim"],
        n_classes=1,
        dropout=config.CNN_CFG_RECALL["dropout"],
        reorder_index=reorder,
        expected_feature_names=feat_names,
    )


def _build_ai4i_ae(data: Dict):
    return models.build_model(
        "ae",
        input_dim=data["meta"]["feature_dim"],
        latent_dim=config.AE_CFG["latent_dim"],
    )


def _build_ai4i_gbdt(data: Dict):
    """AI4I용 HistGradientBoostingClassifier 빌더.

    tabular 데이터에 최적화된 tree-based 모델. CPU에서 빠른 학습 + 클래스 불균형에 강함.
    하이퍼파라미터는 AI4I 규모(10K, 11피처, 3% 양성)에 맞춘 보수적 기본값.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=config.SEED,
    )


def _build_cwru_cnn(data: Dict):
    return models.build_model(
        "cnn_vibration",
        in_channels=1,
        n_classes=data["meta"]["n_classes"],
        dropout=config.CNN_CFG["dropout"],
    )


def _build_hydraulic_vae(data: Dict):
    """Hydraulic VAE 빌더 — VAE+IF 앙상블의 VAE 모델."""
    return models.build_model(
        "vae",
        input_dim=data["meta"]["feature_dim"],
        hidden_dim=config.VAE_CFG["hidden_dim"],
        latent_dim=config.VAE_CFG["latent_dim"],
        beta=config.VAE_CFG["beta"],
    )


def _build_hydraulic_ae(data: Dict):
    return models.build_model(
        "ae",
        input_dim=data["meta"]["feature_dim"],
        latent_dim=config.AE_CFG["latent_dim"],
    )


def _build_lstm(data: Dict):
    return models.build_model(
        "lstm",
        input_dim=data["meta"]["feature_dim"],
        hidden=config.LSTM_CFG["hidden"],
        num_layers=config.LSTM_CFG["num_layers"],
        dropout=config.LSTM_CFG["dropout"],
        input_format="BFL",
    )


def _build_ncmapss_lstm(data: Dict):
    """N-CMAPSS 전용 LSTM 빌더. 43피처에 맞춘 더 큰 hidden을 사용한다."""
    cfg = config.NCMAPSS_LSTM_CFG
    return models.build_model(
        "lstm",
        input_dim=data["meta"]["feature_dim"],
        hidden=cfg["hidden"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        input_format="BFL",
    )


def _build_cmapss_dlinear(data: Dict):
    """C-MAPSS 용 DLinear 빌더. BiLSTM 비교 baseline."""
    return models.build_model(
        "dlinear",
        input_dim=data["meta"]["feature_dim"],
        seq_len=data["meta"]["window"],
        kernel_size=config.DLINEAR_CFG["kernel_size"],
        individual=config.DLINEAR_CFG["individual"],
        input_format="BFL",
    )


def _build_ncmapss_dlinear(data: Dict):
    """N-CMAPSS 용 DLinear 빌더. BiLSTM 비교 baseline (43 채널)."""
    return models.build_model(
        "dlinear",
        input_dim=data["meta"]["feature_dim"],
        seq_len=data["meta"]["window"],
        kernel_size=config.DLINEAR_CFG["kernel_size"],
        individual=config.DLINEAR_CFG["individual"],
        input_format="BFL",
    )


def _build_cmapss_itransformer(data: Dict):
    """C-MAPSS 용 iTransformer 빌더. BiLSTM/DLinear 비교 + cross-channel attention."""
    cfg = config.ITRANSFORMER_CFG
    return models.build_model(
        "itransformer",
        input_dim=data["meta"]["feature_dim"],
        seq_len=data["meta"]["window"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ffn=cfg["d_ffn"],
        dropout=cfg["dropout"],
        input_format="BFL",
    )


def _build_ncmapss_itransformer(data: Dict):
    """N-CMAPSS 용 iTransformer 빌더 (43 채널 → cross-channel attention O(43²))."""
    cfg = config.ITRANSFORMER_CFG
    return models.build_model(
        "itransformer",
        input_dim=data["meta"]["feature_dim"],
        seq_len=data["meta"]["window"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ffn=cfg["d_ffn"],
        dropout=cfg["dropout"],
        input_format="BFL",
    )


# 등록부: name → (task_for_train, task_for_eval, builder)
PIPELINE = {
    "ai4i_cnn":        ("classification",    "binary_classification", _build_ai4i_cnn),
    "ai4i_cnn_recall": ("classification",    "binary_classification", _build_ai4i_cnn_recall),
    "ai4i_gbdt":       ("gbdt_binary",       "gbdt_binary",           _build_ai4i_gbdt),
    "cwru_cnn":        ("classification",    "multiclass",            _build_cwru_cnn),
    "hydraulic_ae":    ("anomaly_detection", "anomaly_detection",     _build_hydraulic_ae),
    "hydraulic_vae":   ("anomaly_detection_vae", "anomaly_detection_vae", _build_hydraulic_vae),
    # ===== C-MAPSS BiLSTM (FD001 기본 + FD001~FD004 명시) =====
    "cmapss_lstm":            ("regression", "regression", _build_lstm),
    "cmapss_lstm_fd001":      ("regression", "regression", _build_lstm),
    "cmapss_lstm_fd002":      ("regression", "regression", _build_lstm),
    "cmapss_lstm_fd003":      ("regression", "regression", _build_lstm),
    "cmapss_lstm_fd004":      ("regression", "regression", _build_lstm),
    # ===== C-MAPSS DLinear (FD001 기본 + FD001~FD004 명시) =====
    "cmapss_dlinear":         ("regression", "regression", _build_cmapss_dlinear),
    "cmapss_dlinear_fd001":   ("regression", "regression", _build_cmapss_dlinear),
    "cmapss_dlinear_fd002":   ("regression", "regression", _build_cmapss_dlinear),
    "cmapss_dlinear_fd003":   ("regression", "regression", _build_cmapss_dlinear),
    "cmapss_dlinear_fd004":   ("regression", "regression", _build_cmapss_dlinear),
    # ===== C-MAPSS iTransformer (cross-channel attention) =====
    "cmapss_itransformer":         ("regression", "regression", _build_cmapss_itransformer),
    "cmapss_itransformer_fd001":   ("regression", "regression", _build_cmapss_itransformer),
    "cmapss_itransformer_fd002":   ("regression", "regression", _build_cmapss_itransformer),
    "cmapss_itransformer_fd003":   ("regression", "regression", _build_cmapss_itransformer),
    "cmapss_itransformer_fd004":   ("regression", "regression", _build_cmapss_itransformer),
    # ===== N-CMAPSS =====
    "ncmapss_lstm":          ("regression", "regression", _build_ncmapss_lstm),
    "ncmapss_dlinear":       ("regression", "regression", _build_ncmapss_dlinear),
    "ncmapss_itransformer":  ("regression", "regression", _build_ncmapss_itransformer),
}


# ---------------------------------------------------------------------------
# 단일 데이터셋 파이프라인
# ---------------------------------------------------------------------------

def _resolve_regression_cfg(name: str) -> Dict:
    """regression 데이터셋명 기준으로 사용 cfg 결정."""
    if "itransformer" in name:
        return config.ITRANSFORMER_CFG
    if "dlinear" in name:
        return config.DLINEAR_CFG
    if name == "ncmapss_lstm":
        return config.NCMAPSS_LSTM_CFG
    # cmapss_lstm / cmapss_lstm_fd00X 등
    return config.LSTM_CFG


def _resolve_classification_cfg(name: str) -> Dict:
    """classification 데이터셋명 기준으로 사용 cfg 결정."""
    if name == "ai4i_cnn_recall":
        return config.CNN_CFG_RECALL
    return config.CNN_CFG


def _get_logged_config(train_task: str, name: str, smoke: bool) -> Dict:
    """MLflow params 로깅용 config 추출."""
    if train_task == "classification":
        cfg = _resolve_classification_cfg(name)
    elif train_task == "anomaly_detection":
        cfg = config.AE_CFG
    elif train_task == "regression":
        cfg = _resolve_regression_cfg(name)
    else:
        cfg = {}
    out = dict(cfg)
    if smoke:
        out["epochs"] = 1
    out["smoke"] = smoke
    out["seed"] = config.SEED
    out["device"] = config.get_device()
    out["dataset_key"] = name
    return out


def run_one(
    name: str,
    smoke: bool,
    skip_explain: bool,
    pipeline_run_id: str = "",
    seed: int = 42,
) -> Dict:
    """name 데이터셋 1개에 대해 load → train → eval (→ explain).

    MLflow가 활성화된 경우 한 데이터셋 = 한 MLflow run 으로 기록.
    seed 인자로 매 실행마다 다른 시드 고정 가능 (multi-seed 재현성 분석용).
    """
    if name not in PIPELINE:
        return {"name": name, "status": "unknown", "error": "no pipeline entry"}

    # 매 실행 시작 시 seed 고정 (multi-seed loop 호환)
    config.set_seed(seed)

    train_task, eval_task, build_fn = PIPELINE[name]
    is_ncmapss = (name == "ncmapss_lstm")
    is_dlinear = "dlinear" in name
    is_itransformer = "itransformer" in name
    is_ncmapss_data = name.startswith("ncmapss_")
    if is_itransformer:
        model_family = "itransformer"
    elif is_dlinear:
        model_family = "dlinear"
    elif train_task == "regression":
        model_family = "lstm"
    else:
        model_family = train_task
    # C-MAPSS subset 추출 (fd001~fd004), 없으면 fd001 가정
    subset = "FD001"
    for fd in ("fd001", "fd002", "fd003", "fd004"):
        if name.endswith(f"_{fd}"):
            subset = fd.upper()
            break

    cmapss_subset_tag = subset if name.startswith("cmapss_") else ""

    with mlf.start_run(
        dataset_key=name,
        run_id=pipeline_run_id,
        tags={
            "train_task": train_task,
            "eval_task": eval_task,
            "model_family": model_family,
            "smoke": "true" if smoke else "false",
            "cmapss_subset": cmapss_subset_tag,
            "seed": str(seed),
        },
    ):
        logged_cfg = _get_logged_config(train_task, name, smoke)
        logged_cfg["seed"] = seed
        mlf.log_params(logged_cfg)

        # 1) 데이터 로드
        try:
            data = dp.LOADERS[name]()
        except FileNotFoundError as e:
            mlf.set_tag("status", "skipped")
            return {"name": name, "status": "skipped", "reason": str(e)}
        except Exception as e:
            mlf.set_tag("status", "load_failed")
            return {
                "name": name, "status": "load_failed",
                "error": str(e), "trace": traceback.format_exc(),
            }

        # 2) 모델 빌드
        try:
            model = build_fn(data)
            # AI4I tabular CNN 등 nn.Module의 컬럼 일치 검증 (sklearn 모델은 hasattr=False)
            if hasattr(model, "verify_data_compatibility"):
                model.verify_data_compatibility(data["meta"]["feature_names"])
        except Exception as e:
            mlf.set_tag("status", "build_failed")
            return {
                "name": name, "status": "build_failed",
                "error": str(e), "trace": traceback.format_exc(),
            }

        # 3) 학습 (smoke 모드는 epoch=1로 패치)
        try:
            train_fn = tr.TRAINERS[train_task]
            # regression / classification 은 데이터셋별로 cfg 가 다르다
            if train_task == "regression":
                base_cfg = _resolve_regression_cfg(name)
                effective_cfg = {**base_cfg, "epochs": 1} if smoke else base_cfg
                train_metrics = train_fn(name, data, model, cfg=effective_cfg)
            elif train_task == "classification":
                base_cfg = _resolve_classification_cfg(name)
                effective_cfg = {**base_cfg, "epochs": 1} if smoke else base_cfg
                train_metrics = train_fn(name, data, model, cfg=effective_cfg)
            elif smoke:
                cfg_map = {
                    "anomaly_detection":     {**config.AE_CFG,  "epochs": 1},
                    "anomaly_detection_vae": {**config.VAE_CFG, "epochs": 1},
                    "gbdt_binary":           {},
                }
                train_metrics = train_fn(name, data, model, cfg=cfg_map[train_task])
            else:
                train_metrics = train_fn(name, data, model)
        except Exception as e:
            mlf.set_tag("status", "train_failed")
            return {
                "name": name, "status": "train_failed",
                "error": str(e), "trace": traceback.format_exc(),
            }

        mlf.log_metrics(train_metrics, prefix="train")

        # 4) 평가
        try:
            eval_fn = ev.EVALUATORS[eval_task]
            if eval_task == "binary_classification":
                classification_cfg = _resolve_classification_cfg(name)
                eval_metrics = eval_fn(
                    name, data, model,
                    decision_threshold=classification_cfg["decision_threshold"],
                )
            elif eval_task == "anomaly_detection":
                eval_metrics = eval_fn(name, data, model, use_mahalanobis=False)
            else:
                eval_metrics = eval_fn(name, data, model)
        except Exception as e:
            mlf.set_tag("status", "eval_failed")
            return {
                "name": name, "status": "eval_failed",
                "train": train_metrics,
                "error": str(e), "trace": traceback.format_exc(),
            }

        mlf.log_metrics(eval_metrics, prefix="eval")
        mlf.set_tag("status", "ok")

        result = {
            "name": name,
            "status": "ok",
            "train": train_metrics,
            "eval":  eval_metrics,
        }

        # 5) 해석 (옵션). GBDT는 captum IG 대상이 아니므로 skip한다.
        if not skip_explain and eval_task != "gbdt_binary":
            try:
                from models_core._archive import explain_captum as ex
                ex_fn = ex.EXPLAINERS[eval_task]
                # 작은 sample로 IG (속도 우선)
                X_sample = data["X_test"][:64]
                feat_names = data["meta"].get("feature_names")
                if eval_task == "regression":
                    result["explain"] = ex_fn(model, X_sample, feature_names=feat_names)
                elif eval_task == "anomaly_detection":
                    result["explain"] = ex_fn(model, X_sample, feature_names=feat_names)
                else:
                    result["explain"] = ex_fn(model, X_sample, feature_names=feat_names)
            except Exception as e:
                result["explain_error"] = str(e)

        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HybridPdM v2.1 학습/평가 파이프라인")
    parser.add_argument(
        "--datasets", nargs="+", default=list(PIPELINE.keys()),
        help=f"실행할 데이터셋 키 (기본: 전체). 가능: {list(PIPELINE.keys())}",
    )
    parser.add_argument("--smoke", action="store_true", help="epoch=1로 빠른 동작 검증")
    parser.add_argument("--skip-explain", action="store_true", help="Captum 해석 단계 건너뛰기")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="실행할 seed 목록 (기본: 42). 여러 개 지정 시 각 데이터셋을 seed 별로 반복 학습.",
    )
    args = parser.parse_args()

    # 실행별 고유 ID — 체크포인트와 리포트 파일에 포함되어 덮어쓰기를 방지한다.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tr.set_run_id(run_id)

    print(f"[HybridPdM] run_id   = {run_id}")
    print(f"[HybridPdM] device   = {config.get_device()}")
    print(f"[HybridPdM] datasets = {args.datasets}")
    print(f"[HybridPdM] seeds    = {args.seeds}")
    if args.smoke:
        print("[HybridPdM] *** SMOKE MODE: epochs=1 ***")
    if len(args.seeds) > 1:
        total = len(args.seeds) * len(args.datasets)
        print(f"[HybridPdM] *** MULTI-SEED: {len(args.seeds)} seeds × {len(args.datasets)} datasets = {total} runs ***")

    results: List[Dict] = []
    for seed in args.seeds:
        # 체크포인트 파일명에 seed 접미사 반영 (seed=42 면 빈 문자열 → backward compat)
        tr.set_seed_suffix(seed)
        for name in args.datasets:
            header = f"=== {name} (seed={seed}) ===" if len(args.seeds) > 1 else f"=== {name} ==="
            print(f"\n{header}")
            res = run_one(
                name, smoke=args.smoke, skip_explain=args.skip_explain,
                pipeline_run_id=run_id, seed=seed,
            )
            res["seed"] = seed
            status = res.get("status")
            if status == "ok":
                ev_m = res.get("eval", {})
                summary = {k: v for k, v in ev_m.items()
                           if k in ("accuracy", "f1", "rmse", "mae", "r2",
                                    "nasa_score_sum", "nasa_score_mean",
                                    "test_f1", "best_threshold", "best_percentile",
                                    "pr_auc", "roc_auc")}
                print(f"  [OK]  {summary}")
            elif status == "skipped":
                print(f"  [--] skipped: {res.get('reason')}")
            else:
                print(f"  [FAIL] {status}: {res.get('error')}")
            results.append(res)

    # 전체 결과를 리포트로 저장 (run_id 포함 → 덮어쓰기 없이 누적)
    out_path = config.REPORT_DIR / f"pipeline_report_{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tr._json_safe(results), f, indent=2, ensure_ascii=False)
    print(f"\n[HybridPdM] report saved → {out_path}")


if __name__ == "__main__":
    main()

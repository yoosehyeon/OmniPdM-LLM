"""MLflow 실험 추적 헬퍼.

설계 원칙:
- MLflow 실패가 학습을 막아서는 안 됨 → 모든 호출은 try-except로 감쌈
- env: MLFLOW_TRACKING_URI 미설정 시 기본 file backend ('./mlruns')
- env: DISABLE_MLFLOW=1 로 완전 비활성화 가능

학습 파이프라인 사용 예:
    with start_run(dataset_key="ai4i_cnn", run_id="20260515_120000"):
        log_params(config_dict)
        log_metrics(train_metrics, prefix="train")
        log_metrics(eval_metrics, prefix="eval")
        log_artifact(checkpoint_path)
"""
from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return os.getenv("DISABLE_MLFLOW", "0") != "1"


@contextlib.contextmanager
def start_run(
    dataset_key: str,
    run_id: str,
    experiment_name: str = "hybridpdm",
    tags: Optional[Dict[str, str]] = None,
) -> Iterator[Optional[Any]]:
    """MLflow run 컨텍스트 매니저.

    Yields: mlflow.ActiveRun 또는 비활성화 시 None.
    """
    if not _enabled():
        yield None
        return

    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        run_name = f"{dataset_key}_{run_id}"

        merged_tags = {"dataset_key": dataset_key, "pipeline_run_id": run_id}
        if tags:
            merged_tags.update(tags)

        with mlflow.start_run(run_name=run_name, tags=merged_tags) as run:
            yield run
    except Exception as e:
        logger.warning("MLflow start_run failed: %s: %s — continuing without tracking",
                       type(e).__name__, e)
        yield None


def log_params(params: Dict[str, Any]) -> None:
    """하이퍼파라미터 로그 (값은 str/int/float만, 그 외는 repr)."""
    if not _enabled() or not params:
        return
    try:
        import mlflow

        # MLflow는 한 호출에 100개 제한 + 값 길이 250자 제한
        safe = {}
        for k, v in params.items():
            if isinstance(v, (int, float, bool, str)):
                safe[k] = v
            else:
                safe[k] = repr(v)[:250]
        mlflow.log_params(safe)
    except Exception as e:
        logger.warning("MLflow log_params failed: %s: %s", type(e).__name__, e)


def log_metrics(metrics: Dict[str, Any], prefix: str = "", step: Optional[int] = None) -> None:
    """평가 메트릭 로그. dict 안의 숫자만 추출."""
    if not _enabled() or not metrics:
        return
    try:
        import mlflow

        flat = {}
        for k, v in metrics.items():
            key = f"{prefix}_{k}" if prefix else k
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                flat[key] = float(v)
            elif isinstance(v, dict):
                # 한 단계만 flatten
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float)) and not isinstance(sub_v, bool):
                        flat[f"{key}_{sub_k}"] = float(sub_v)
        if flat:
            mlflow.log_metrics(flat, step=step)
    except Exception as e:
        logger.warning("MLflow log_metrics failed: %s: %s", type(e).__name__, e)


def log_artifact(path: Path | str) -> None:
    """체크포인트 / 리포트 등 산출물 파일 로그."""
    if not _enabled():
        return
    try:
        import mlflow

        p = Path(path)
        if p.exists():
            mlflow.log_artifact(str(p))
    except Exception as e:
        logger.warning("MLflow log_artifact failed: %s: %s", type(e).__name__, e)


def set_tag(key: str, value: str) -> None:
    if not _enabled():
        return
    try:
        import mlflow

        mlflow.set_tag(key, value)
    except Exception as e:
        logger.warning("MLflow set_tag failed: %s: %s", type(e).__name__, e)

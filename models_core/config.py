"""HybridPdM - 중앙 설정 모듈.

경로, 모델별 하이퍼파라미터, Risk Score 가중치, 재현성(seed) 헬퍼를 한 곳에서 관리한다.
이 모듈을 import하는 것만으로 numpy/torch/random seed가 고정된다.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# torch는 학습/추론 시 필수지만, config 자체는 torch 없이도 import 가능해야 함
try:
    import torch
except ImportError:
    torch = None  # type: ignore

# ---------------------------------------------------------------------------
# 1) 경로 설정
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "dataset"
ARTIFACT_ROOT = ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACT_ROOT / "checkpoints"
REPORT_DIR = ARTIFACT_ROOT / "reports"

for _d in (ARTIFACT_ROOT, CHECKPOINT_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# HF Model Hub repo 에서 체크포인트를 pull 한다. HF Space 에는 binary 를 포함할 수 없기 때문.
# 로컬 CHECKPOINT_DIR 에 동일 stem 이 있으면 우선 사용 (개발 환경).
CHECKPOINT_REPO = os.getenv("CHECKPOINT_REPO", "yusehyeon/hybridpdm-checkpoints")

# 데이터셋별 경로. 키는 모델/스크립트 전반에서 사용되는 데이터셋 식별자.
DATASET_PATHS = {
    "ai4i":      DATA_ROOT / "ai4i2020.csv",
    "cwru":      DATA_ROOT / "10987113",
    "hydraulic": DATA_ROOT / "condition+monitoring+of+hydraulic+systems",
    "cmapss":    DATA_ROOT / "CMAPSSData",
    "ncmapss":   DATA_ROOT / "17. Turbofan Engine Degradation Simulation Data Set 2" / "data_set",
}

# ---------------------------------------------------------------------------
# 2) 재현성(Reproducibility)
# ---------------------------------------------------------------------------
SEED = 42

def set_seed(seed: int = SEED) -> None:
    """python/numpy/torch(CPU+CUDA)의 모든 난수 시드를 일괄 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker별 시드 고정 헬퍼.

    num_workers>0으로 DataLoader를 사용할 때 각 워커의 numpy/random 시드가
    무작위로 바뀌어 비결정적 동작이 되는 문제를 방지한다.
    DataLoader(worker_init_fn=config.worker_init_fn) 형태로 전달.
    """
    seed = SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)


def get_device():
    """GPU가 있으면 'cuda', 아니면 'cpu'를 반환한다."""
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 3) 모델별 하이퍼파라미터
# ---------------------------------------------------------------------------

# 3.1) 1D-CNN (고장 분류)
CNN_CFG = {
    "epochs": 50,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.2,
    "decision_threshold": 0.35,
    # Focal Loss: 극불균형 이진 분류에서 easy negative를 down-weight하여
    # hard positive에 학습 자원을 집중. alpha로 클래스 균형, gamma로 easy 샘플 억제.
    # Focal Loss와 BCE+pos_weight는 배타 사용 (use_focal_loss로 분기).
    "use_focal_loss": True,
    "focal_alpha": 0.85,
    "focal_gamma": 2.0,
}

# 3.1b) 1D-CNN — Recall 강화 variant (PdM 안전성 우선)
# 기존 CNN_CFG 대비 변경 사항:
#   - focal_alpha 0.85 → 0.92 (positive class 가중치 ↑ → recall ↑)
#   - focal_gamma 2.0 → 3.0  (hard sample focus ↑ → 어려운 positive 학습)
#   - decision_threshold 0.35 → 0.20 (예측 임계값 ↓ → recall ↑, precision 일부 trade-off)
#   - epochs 50 → 70 (충분한 학습)
# 목표: AI4I baseline recall 0.706 → 0.85+ (false negative 30% → 15% 이하)
# Trade-off: precision 0.857 → 0.7~0.8 예상, False Alarm Rate 0.41% → 1~2% 예상
CNN_CFG_RECALL = {
    "epochs": 70,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.25,
    "decision_threshold": 0.20,
    "use_focal_loss": True,
    "focal_alpha": 0.92,
    "focal_gamma": 3.0,
}

# 3.2b) Variational Autoencoder (이상 탐지 — ELBO 기반)
# DenoisingAE 와 함께 ensemble 용. rank-based fusion 으로 결합.
# beta-VAE 의 beta 파라미터로 KL 가중 조정.
VAE_CFG = {
    "epochs": 80,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "hidden_dim": 32,
    "latent_dim": 8,
    "beta": 1.0,                       # KL 가중치 (β-VAE 의 β)
    "early_stop_patience": 12,
    "grad_clip": 1.0,
    "anomaly_score_mode": "elbo_neg",  # recon_only / elbo_neg / combined
    "threshold_grid": list(range(85, 100)),
}

# 3.2c) Isolation Forest (이상 탐지 — tree-based, sklearn)
# 다양성 우선: AE 류와 오류 패턴 다르므로 앙상블 효과 ↑
IFOREST_CFG = {
    "n_estimators": 200,
    "max_samples": "auto",
    "contamination": "auto",
    "random_state": 42,
    "n_jobs": -1,
}


# 3.2) Autoencoder (이상 탐지)
AE_CFG = {
    "epochs": 60,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "latent_dim": 16,
    "early_stop_patience": 10,
    "threshold_grid": list(range(85, 100)),  # val F1을 최대화하는 percentile 탐색 구간
}

# 3.3) LSTM (RUL 예측) — C-MAPSS 기본
LSTM_CFG = {
    "epochs": 80,
    "batch_size": 128,
    "lr": 1e-3,
    "dropout": 0.4,
    "weight_decay": 1e-4,
    "early_stop_patience": 10,
    "hidden": 128,
    "num_layers": 2,
    "window": 30,
    "rul_clip": 125,
    "grad_clip": 1.0,
    "use_lr_scheduler": True,
    "use_huber_loss": False,
}

# 3.4) LSTM (RUL 예측) — N-CMAPSS 전용
# 43피처(C-MAPSS 14피처의 3배)에 맞춰 hidden 확대, 에폭·데이터량 증가.
# Huber Loss로 이상치(큰 RUL 오차)에 대한 민감도를 줄인다.
NCMAPSS_LSTM_CFG = {
    "epochs": 120,
    "batch_size": 128,
    "lr": 1e-3,
    "dropout": 0.3,
    "weight_decay": 1e-4,
    "early_stop_patience": 15,
    "hidden": 256,
    "num_layers": 2,
    "window": 30,
    "rul_clip": 125,
    "grad_clip": 1.0,
    "use_lr_scheduler": True,
    "use_huber_loss": True,
    "huber_delta": 5.0,
    # data_pipeline 전용 파라미터
    "stride": 10,
    "max_units_train": 20,
    "max_units_test": 20,
    "max_windows_per_unit": 3000,
}

# 3.6) iTransformer (RUL 회귀 — variate tokenization + cross-channel attention)
# Liu et al. 2024 ICLR, "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
# 토큰 = 변수 → 토큰 수 N = F (C-MAPSS 14, N-CMAPSS 43) → attention O(F²) 매우 저렴
# CPU 환경 정합 설정.
ITRANSFORMER_CFG = {
    "epochs": 80,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "d_model": 128,       # 토큰 임베딩 차원 (BiLSTM 표현력 비교 가능 수준)
    "n_heads": 4,         # multi-head (d_model % n_heads == 0)
    "n_layers": 2,        # encoder block 수 (CPU 비용 고려)
    "d_ffn": 512,         # 4 * d_model (Transformer 표준)
    "dropout": 0.15,
    "early_stop_patience": 15,
    "grad_clip": 1.0,
    "use_lr_scheduler": True,
    "use_huber_loss": True,
    "huber_delta": 5.0,
    # data_pipeline 호환 (LSTM_CFG 와 동일)
    "window": 30,
    "rul_clip": 125,
    "stride": 10,
    "max_units_train": 20,
    "max_units_test": 20,
    "max_windows_per_unit": 3000,
}


# 3.5) DLinear (RUL 회귀 — lightweight Transformer-free baseline)
# Zeng et al. 2023, "Are Transformers Effective for Time Series Forecasting?"
# 파라미터 ~수만개로 매우 작음. CPU 학습 빠름. BiLSTM 대비 baseline 비교용.
DLINEAR_CFG = {
    "epochs": 80,
    "batch_size": 256,
    "lr": 5e-3,           # Linear 모델은 LR 높게
    "weight_decay": 1e-5,
    "kernel_size": 25,
    "individual": True,
    "early_stop_patience": 10,
    "grad_clip": 1.0,
    "use_lr_scheduler": True,
    "use_huber_loss": True,
    "huber_delta": 5.0,
    # data_pipeline 용 (N-CMAPSS) — LSTM 과 동일
    "window": 30,
    "rul_clip": 125,
    "stride": 10,
    "max_units_train": 20,
    "max_units_test": 20,
    "max_windows_per_unit": 3000,
}


# ---------------------------------------------------------------------------
# 4) Risk Score
# ---------------------------------------------------------------------------
# 산식: R = w1*P(failure) + w2*Anomaly + w3*(1 - RUL_norm)
RISK_WEIGHTS = {"failure": 0.5, "anomaly": 0.3, "rul": 0.2}

# 등급 임계값 (내림차순). 점수마다 첫 번째 매칭 등급을 부여한다.
# 계약: 반드시 Critical → Normal 순으로 정렬 — services/realtime/notifier.py 의
# _LEVEL_RANK 가 이 순서를 가정하여 "심각도 ≤ index" 비교에 사용한다. 순서 변경 시 알람 등급 필터가 망가진다.
RISK_LEVELS = [
    ("Critical", 0.80),
    ("Warning",  0.50),
    ("Advisory", 0.30),
    ("Normal",   0.00),
]


# ---------------------------------------------------------------------------
# 5) Tier 1 — Realtime (MQTT / Slack) — PRD v6.2 §10 T1-01, T1-02
# ---------------------------------------------------------------------------
# OmniPdM 실시간 워커 설정. 모두 환경변수 기반 → dev/prod 동일 코드.
# 실제 사용 시 paho-mqtt / requests 가 설치되어 있어야 한다 (requirements.txt).
# Tier 1 미사용(app.py 만 띄울 경우) 시에는 services/realtime/* 를 import 하지 않으므로 영향 없음.

def _env_int(key: str, default: int) -> int:
    """빈 문자열 env var (.env 의 `KEY=`) 도 default 로 처리한다.

    os.getenv(key, default) 는 키가 없을 때만 default 를 반환하고
    빈 문자열은 그대로 돌려주기 때문에 int("") 가 import 시점에 ValueError 를 던진다.
    """
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_str(key: str, default: str) -> str:
    raw = os.getenv(key)
    return default if raw is None else raw


# MQTT broker
MQTT_HOST = _env_str("OMNIPDM_MQTT_HOST", "localhost")
MQTT_PORT = _env_int("OMNIPDM_MQTT_PORT", 1883)  # TLS(8883) 미지원 — 추후 cert 설정 추가 시 보강
# 빈 문자열이면 워커는 username_pw_set 호출 자체를 생략해야 한다 (Mosquitto allow_anonymous 호환).
MQTT_USERNAME = _env_str("OMNIPDM_MQTT_USERNAME", "")
MQTT_PASSWORD = _env_str("OMNIPDM_MQTT_PASSWORD", "")
MQTT_TOPIC_TELEMETRY = _env_str("OMNIPDM_MQTT_TOPIC", "omnipdm/telemetry/+")
# 다중 워커 시 client_id 충돌 주의 — 워커 connect 시점에 PID suffix 를 붙여 사용한다.
MQTT_CLIENT_ID = _env_str("OMNIPDM_MQTT_CLIENT_ID", "omnipdm-worker")
MQTT_QOS = _env_int("OMNIPDM_MQTT_QOS", 1)  # at-least-once: 손실<중복, PdM 알람에 적합
MQTT_KEEPALIVE_SEC = _env_int("OMNIPDM_MQTT_KEEPALIVE_SEC", 60)

# 알람 채널 (Notifier) — 채널 종류와 무관하게 모든 Notifier 가 공유하는 정책.
# 알람 발송 최소 등급. RISK_LEVELS 의 라벨 중 하나여야 한다 (Critical / Warning / Advisory / Normal).
ALERT_MIN_LEVEL = _env_str("OMNIPDM_ALERT_MIN_LEVEL", "Critical")
# Rate limit (초). 동일 device_id 가 이 시간 안에 또 발화하면 중복 발송 억제 (in-memory, 단일 워커 기준).
ALERT_RATE_LIMIT_SEC = _env_int("OMNIPDM_ALERT_RATE_LIMIT_SEC", 300)

# Sanity check — config import 시점에 즉시 발견. 워커 시작 후 무한루프 진입 전에 죽는 게 안전하다.
if not MQTT_HOST.strip():
    raise ValueError("OMNIPDM_MQTT_HOST 가 비어 있습니다. localhost 또는 broker 호스트를 지정하세요.")
_VALID_LEVELS = {label for label, _ in RISK_LEVELS}
if ALERT_MIN_LEVEL not in _VALID_LEVELS:
    raise ValueError(
        f"OMNIPDM_ALERT_MIN_LEVEL='{ALERT_MIN_LEVEL}' 는 유효한 등급이 아닙니다. "
        f"가능: {sorted(_VALID_LEVELS)}"
    )
if MQTT_QOS not in (0, 1, 2):
    raise ValueError(f"OMNIPDM_MQTT_QOS={MQTT_QOS} 는 0/1/2 중 하나여야 합니다.")


# ---------------------------------------------------------------------------
# 6) Tier 1 — TimescaleDB 이력 저장 — PRD v6.3 §10 T1-03
# ---------------------------------------------------------------------------
# docker-compose.yml 의 timescaledb 서비스와 짝. host loopback(127.0.0.1) 만 노출.
# DB_ENABLED=false 이거나 psycopg 미설치 / DB 미기동 시 run_worker.py 가 NullDbWriter 로 fallback.

DB_HOST     = _env_str("OMNIPDM_DB_HOST",     "127.0.0.1")
DB_PORT     = _env_int("OMNIPDM_DB_PORT",     5432)
DB_NAME     = _env_str("OMNIPDM_DB_NAME",     "omnipdm")
DB_USER     = _env_str("OMNIPDM_DB_USER",     "omnipdm")
DB_PASSWORD = _env_str("OMNIPDM_DB_PASSWORD", "omnipdm")
# 명시적 disable — docker-compose 안 띄운 상태로도 워커 실행하고 싶을 때 false 로 두면 NullDbWriter 강제.
DB_ENABLED  = _env_str("OMNIPDM_DB_ENABLED",  "true").lower() in ("1", "true", "yes", "on")


# import 시점에 자동으로 시드를 고정한다.
set_seed()

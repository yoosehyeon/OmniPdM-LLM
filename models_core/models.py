"""HybridPdM - 모델 정의.

3-모델 앙상블 구조:
  · WDCNN1D       : 진동 신호 전용 Wide-Kernel CNN (CWRU 등 1024-length 입력)
  · TabularCNN1D  : 짧은 테이블형 입력 전용 CNN (AI4I 11-피처 등)
  · DenoisingAE   : 비대칭 Denoising Autoencoder (이상 탐지, 임계값 내장)
  · BiLSTMRegressor : BiLSTM + Attention Pooling (RUL 회귀)

설계 원칙:
  - 이종 데이터(진동/테이블)는 백본을 분리하여 각 입력 분포에 최적화한다.
  - AE는 학습 후 정상 데이터 재구성 오차의 percentile로 임계값을 내장하고,
    evaluate.py에서 grid search로 갱신한다.
  - LSTM 입력 형식(BFL/BLF)을 생성자에서 명시받아 축 혼동을 방지한다.
  - TabularCNN1D는 forward 시점에 shape/피처 순서를 검증한다.
"""
from __future__ import annotations

import difflib
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1) 1D-CNN 백본 - 두 변종
# ===========================================================================

class WDCNN1D(nn.Module):
    """Wide-Kernel Deep CNN.

    진동 신호(예: CWRU 1024-length)에 최적화된 백본.
    첫 conv는 wide kernel(64)로 저주파 특징을 추출하고,
    이후 conv는 작은 kernel로 점진적 압축한다.

    출력 / 손실 함수 매핑:
      · n_classes == 1  →  BCEWithLogitsLoss (이진 분류)
      · n_classes >= 2  →  CrossEntropyLoss  (다중 분류)
      두 경우 모두 forward는 logit을 반환한다.
    """
    def __init__(self, in_channels: int = 1, n_classes: int = 1, dropout: float = 0.3):
        super().__init__()
        if n_classes < 1:
            raise ValueError(f"n_classes must be >= 1, got {n_classes}")
        self.n_classes = n_classes
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=64, stride=8, padding=30)
        self.bn1   = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm1d(64)
        # GAP로 입력 길이 변화에 robust + FC 입력 차원 고정
        self.gap = nn.AdaptiveAvgPool1d(4)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * 4, 64)
        self.fc2 = nn.Linear(64, n_classes)  # logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# AI4I 11-피처 reorder 사양
# ---------------------------------------------------------------------------
# 'power', 'temp_diff', 'strain'은 data_pipeline.load_ai4i_cnn()에서 생성되는 파생 피처:
#   power     = "Rotational speed [rpm]" * "Torque [Nm]"
#   temp_diff = "Process temperature [K]" - "Air temperature [K]"
#   strain    = "Tool wear [min]" * "Torque [Nm]"
AI4I_REORDER_NAMES = [
    "temp_diff",                # 열적 스트레스
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "power",                    # 회전 동력
    "Tool wear [min]",
    "strain",                   # 마모 부하
    "Type_L",
    "Type_M",
    "Type_H",
]


def build_reorder_index(source_names: Sequence[str],
                        target_names: Sequence[str]) -> List[int]:
    """source_names → target_names 순서로 재배열할 인덱스 리스트 생성.

    검증 항목:
      1) target에 중복 이름이 있으면 ValueError (같은 피처가 두 번 매핑되는
         무음 오작동을 방지)
      2) target과 source의 길이가 다르면 ValueError
      3) target에 source에 없는 이름이 있으면 ValueError + 누락 목록 출력
    """
    src = list(source_names)
    tgt = list(target_names)

    # (1) 중복 검증
    dups = [n for n in set(tgt) if tgt.count(n) > 1]
    if dups:
        raise ValueError(f"reorder target contains duplicates: {dups}")

    # (2) 길이 검증
    if len(src) != len(tgt):
        raise ValueError(
            f"reorder length mismatch: source={len(src)}, target={len(tgt)}"
        )

    # (3) 누락 이름 검증
    missing = [n for n in tgt if n not in src]
    if missing:
        raise ValueError(
            f"reorder target contains names not present in source: {missing}\n"
            f"  source = {src}"
        )

    return [src.index(n) for n in tgt]


class TabularCNN1D(nn.Module):
    """짧은 테이블형 입력 전용 1D-CNN.

    AI4I(11-피처) 등 짧은 입력에는 kernel=3 small conv를 사용한다.
    forward 시점에 의미 그룹 순서로 피처를 재배열하여 인접 채널이 도메인상
    연관되도록 만들어 CNN의 local receptive field가 의미 있는 패턴을 학습할 수 있게 한다.

    안전장치:
      - 생성자: reorder_index 길이/범위 검증
      - verify_data_compatibility(): 학습 시작 전 컬럼 이름 일치 검증
      - forward(): 입력 shape, conv 출력 길이, fc1 입력 차원 검증
    """
    def __init__(self, in_channels: int = 1, seq_len: int = 11,
                 n_classes: int = 1, dropout: float = 0.3,
                 reorder_index: Optional[Sequence[int]] = None,
                 expected_feature_names: Optional[Sequence[str]] = None):
        super().__init__()
        if n_classes < 1:
            raise ValueError(f"n_classes must be >= 1, got {n_classes}")
        self.n_classes = n_classes
        self.expected_seq_len = int(seq_len)
        self.expected_feature_names = (
            list(expected_feature_names) if expected_feature_names is not None else None
        )

        # reorder_index 검증
        if reorder_index is not None:
            ri = list(reorder_index)
            if len(ri) != self.expected_seq_len:
                raise ValueError(
                    f"reorder_index length ({len(ri)}) != seq_len ({self.expected_seq_len})"
                )
            if len(set(ri)) != len(ri):
                raise ValueError(f"reorder_index contains duplicates: {ri}")
            if max(ri) >= self.expected_seq_len or min(ri) < 0:
                raise ValueError(
                    f"reorder_index out of range for seq_len={self.expected_seq_len}: {ri}"
                )
            self.register_buffer("reorder", torch.tensor(ri, dtype=torch.long))
            self._has_reorder = True
        else:
            self._has_reorder = False

        # padding=1, stride=1 conv 3개 → 출력 길이는 seq_len 그대로 유지된다.
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout)
        # FC 입력 차원: 64채널 * seq_len
        self._fc1_in = 64 * self.expected_seq_len
        self.fc1 = nn.Linear(self._fc1_in, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def verify_data_compatibility(self, source_feature_names: Sequence[str]) -> None:
        """학습 직전에 호출. 데이터 컬럼 순서/이름이 모델 가정과 일치하는지 검증한다."""
        if self.expected_feature_names is None:
            return
        if list(source_feature_names) != self.expected_feature_names:
            raise RuntimeError(
                "TabularCNN1D feature compatibility check failed.\n"
                f"  model expects: {self.expected_feature_names}\n"
                f"  data provides: {list(source_feature_names)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1) 입력 shape: (B, C, L)
        if x.dim() != 3:
            raise RuntimeError(
                f"TabularCNN1D expects (B, C, L), got dim={x.dim()} shape={tuple(x.shape)}"
            )
        if x.shape[-1] != self.expected_seq_len:
            raise RuntimeError(
                f"TabularCNN1D seq_len mismatch: expected {self.expected_seq_len}, "
                f"got {x.shape[-1]}. Check data_pipeline feature count."
            )

        if self._has_reorder:
            x = x.index_select(dim=-1, index=self.reorder)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # conv 출력 길이 가드: padding=1 stride=1 가정이 깨지면 즉시 중단
        if x.shape[-1] != self.expected_seq_len:
            raise RuntimeError(
                f"TabularCNN1D internal shape error: conv output length "
                f"{x.shape[-1]} != expected {self.expected_seq_len}. "
                "Did the conv structure change? Update self._fc1_in accordingly."
            )

        x = x.flatten(1)

        # fc1 입력 차원 가드
        if x.shape[-1] != self._fc1_in:
            raise RuntimeError(
                f"TabularCNN1D fc1 input mismatch: got {x.shape[-1]}, "
                f"expected {self._fc1_in}"
            )

        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ===========================================================================
# 2) Denoising Autoencoder + 내장 임계값
# ===========================================================================

class DenoisingAE(nn.Module):
    """비대칭 Denoising Autoencoder + 임계값 내장.

    학습 시 가우시안 노이즈를 입력에 더해 정상 분포의 본질을 학습하도록 유도한다.
    Encoder/Decoder 모두 BatchNorm을 적용하여 학습 안정성을 확보한다.
    Decoder의 마지막 Linear에는 활성화 함수가 없다. 입력이 StandardScaler로
    표준화된 상태이므로 임의의 실수 출력을 허용해야 하기 때문이다.

    임계값 처리:
      - fit_threshold(): 학습 종료 후 정상 데이터 재구성 오차에서 percentile 기반
                         초기 임계값을 self.threshold에 저장.
      - set_threshold(): evaluate.py가 grid search로 찾은 최적 임계값으로 갱신.
      - threshold가 inf 상태에서 predict/anomaly_score 호출 시 RuntimeError.

    Mahalanobis 보조 점수:
      - fit_mahalanobis(): 정상 데이터의 latent 분포(평균 + 역공분산)를 적합.
      - mahalanobis_score(): latent 공간 Mahalanobis 거리² 반환.
      - combined_score(): 재구성 오차 + Mahalanobis (각 z-score 정규화 후 합).
    """
    def __init__(self, input_dim: int, latent_dim: int = 8, noise_std: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.noise_std = noise_std
        # Encoder: input → 32 → 16 → latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16),        nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        # Decoder: latent → 24 → 32 → input  (BN 추가, 비대칭 capacity)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24), nn.BatchNorm1d(24), nn.ReLU(),
            nn.Linear(24, 32),         nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, input_dim),  # 활성화 없음: 표준화된 입력의 임의 실수 출력 허용
        )
        # 임계값을 buffer로 보관 (state_dict 함께 저장/로드)
        # 초기값 inf → fit_threshold/set_threshold 호출 전 사용 시 RuntimeError
        self.register_buffer("threshold", torch.tensor(float("inf")))

        # Mahalanobis 통계. fit_mahalanobis 호출 전에는 fitted=False.
        self.register_buffer("mahal_mu",      torch.zeros(latent_dim))
        self.register_buffer("mahal_inv_cov", torch.eye(latent_dim))
        self.register_buffer("recon_mu",      torch.zeros(1))
        self.register_buffer("recon_sigma",   torch.ones(1))
        self.register_buffer("mahal_score_mu",    torch.zeros(1))
        self.register_buffer("mahal_score_sigma", torch.ones(1))
        self.register_buffer("mahal_fitted",  torch.tensor(False))

    def _ensure_threshold_set(self):
        """threshold가 inf 상태로 추론에 사용되는 것을 차단."""
        if torch.isinf(self.threshold).item():
            raise RuntimeError(
                "DenoisingAE.threshold is not initialized. "
                "Call model.fit_threshold(normal_train_tensor) after training "
                "and/or model.set_threshold(value) from evaluate.py before "
                "calling predict() / anomaly_score()."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 학습 모드일 때만 노이즈 주입
        if self.training and self.noise_std > 0:
            x_in = x + torch.randn_like(x) * self.noise_std
        else:
            x_in = x
        z = self.encoder(x_in)
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """샘플별 MSE 재구성 오차 (B,)."""
        recon = self.forward(x)
        return ((recon - x) ** 2).mean(dim=1)

    @torch.no_grad()
    def fit_threshold(self, normal_data: torch.Tensor,
                      percentile: float = 95.0,
                      batch_size: int = 256) -> float:
        """정상 데이터 재구성 오차 분포에서 percentile 임계값 계산.

        eval() 모드를 강제하여 BatchNorm running stats를 사용하고,
        batch_size 단위로 청크 처리하여 메모리 폭발을 방지한다.
        """
        self.eval()
        device = self.threshold.device
        n = normal_data.shape[0]
        if n == 0:
            raise ValueError("fit_threshold received empty normal_data")

        errs = []
        for i in range(0, n, batch_size):
            chunk = normal_data[i:i + batch_size].to(device)
            errs.append(self.reconstruction_error(chunk).cpu().numpy())
        errs = np.concatenate(errs)

        thr = float(np.percentile(errs, percentile))
        self.threshold = torch.tensor(thr, dtype=torch.float32, device=device)
        return thr

    def set_threshold(self, value: float) -> None:
        """외부 grid search 결과로 임계값 갱신."""
        self.threshold = torch.tensor(
            float(value), dtype=torch.float32, device=self.threshold.device
        )

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """이상 여부를 0/1로 반환. threshold 미설정 시 RuntimeError."""
        self._ensure_threshold_set()
        err = self.reconstruction_error(x)
        return (err > self.threshold).float()

    @torch.no_grad()
    def fit_mahalanobis(self, normal_data: torch.Tensor,
                        batch_size: int = 256) -> None:
        """정상 데이터로 latent 분포 + 재구성/Mahalanobis 정규화 통계 적합.

        재구성 오차만으로는 잡히지 않는 covariance 이상(피처 간 관계가 깨진 경우)을
        latent 공간 Mahalanobis 거리로 보강한다.
        combined_score()에서 z-score 정규화 후 합산하기 위해 두 분포의
        평균/표준편차도 함께 저장한다.

        ridge regularization: 공분산이 거의 특이행렬일 때 안정적으로 역행렬을
        구하기 위해 trace 비례 항을 더한다.
        """
        self.eval()
        device = self.threshold.device
        n = normal_data.shape[0]
        if n == 0:
            raise ValueError("fit_mahalanobis received empty normal_data")

        zs: List[np.ndarray] = []
        recons: List[np.ndarray] = []
        for i in range(0, n, batch_size):
            chunk = normal_data[i:i + batch_size].to(device)
            z = self.encoder(chunk)
            recon = self.decoder(z)
            err = ((recon - chunk) ** 2).mean(dim=1)
            zs.append(z.cpu().numpy())
            recons.append(err.cpu().numpy())
        Z = np.concatenate(zs, axis=0).astype(np.float64)        # (N, D)
        recon_errs = np.concatenate(recons, axis=0).astype(np.float64)

        mu = Z.mean(axis=0)
        cov = np.cov(Z, rowvar=False)
        if cov.ndim == 0:  # latent_dim == 1 edge case
            cov = np.array([[float(cov)]])
        eps = 1e-4 * max(np.trace(cov) / cov.shape[0], 1e-8)
        inv_cov = np.linalg.inv(cov + eps * np.eye(cov.shape[0]))

        diff = Z - mu                                            # (N, D)
        mahal_train = np.einsum("ni,ij,nj->n", diff, inv_cov, diff)

        self.mahal_mu          = torch.tensor(mu,      dtype=torch.float32, device=device)
        self.mahal_inv_cov     = torch.tensor(inv_cov, dtype=torch.float32, device=device)
        self.recon_mu          = torch.tensor([float(recon_errs.mean())],     dtype=torch.float32, device=device)
        self.recon_sigma       = torch.tensor([float(recon_errs.std() + 1e-8)], dtype=torch.float32, device=device)
        self.mahal_score_mu    = torch.tensor([float(mahal_train.mean())],     dtype=torch.float32, device=device)
        self.mahal_score_sigma = torch.tensor([float(mahal_train.std() + 1e-8)], dtype=torch.float32, device=device)
        self.mahal_fitted      = torch.tensor(True, device=device)

    @torch.no_grad()
    def mahalanobis_score(self, x: torch.Tensor) -> torch.Tensor:
        """Latent 공간 Mahalanobis 거리² (B,). fit_mahalanobis 호출 필수."""
        if not bool(self.mahal_fitted.item()):
            raise RuntimeError(
                "DenoisingAE.mahal_fitted is False. "
                "Call model.fit_mahalanobis(normal_train_tensor) first."
            )
        self.eval()
        z = self.encoder(x)                                      # (B, D)
        diff = z - self.mahal_mu
        return torch.einsum("bi,ij,bj->b", diff, self.mahal_inv_cov, diff)

    @torch.no_grad()
    def combined_score(self, x: torch.Tensor) -> torch.Tensor:
        """재구성 오차 + Mahalanobis (z-score 정규화 후 평균).

        두 신호가 잡는 이상 유형이 다르므로(전자: 점-단위 outlier, 후자:
        피처-간 covariance 깨짐), 정상 분포 통계로 정규화 후 합산하면
        한쪽이 다른 쪽을 압도하지 않는다.
        """
        if not bool(self.mahal_fitted.item()):
            raise RuntimeError(
                "combined_score requires fit_mahalanobis to be called first."
            )
        recon_err = self.reconstruction_error(x)                 # (B,)
        mahal     = self.mahalanobis_score(x)                    # (B,)
        recon_z = (recon_err - self.recon_mu) / self.recon_sigma
        mahal_z = (mahal     - self.mahal_score_mu) / self.mahal_score_sigma
        return 0.5 * (recon_z + mahal_z)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """[0, 1] 정규화된 이상 점수 (Risk Score 합성용).

        score = err / (err + threshold)
          · err == threshold → 0.5
          · err >> threshold → ~1.0
          · err << threshold → ~0.0
        threshold 미설정 시 RuntimeError.
        """
        self._ensure_threshold_set()
        err = self.reconstruction_error(x)
        return err / (err + self.threshold + 1e-8)


# ===========================================================================
# 2b) Variational Autoencoder (이상 탐지, ELBO 기반)
# DenoisingAE 대비 강점:
#   - 확률적 latent → 더 매끄러운 분포 학습
#   - ELBO (recon + KL) 두 가지 이상 신호 동시 사용 가능
#   - Isolation Forest 와 앙상블 시 다양성 ↑ (rank-based fusion)
#
# 구조:
#   x (B, F) → encoder → (mu, logvar) (B, latent)
#           → reparameterize → z = mu + sigma * eps
#           → decoder → x_recon (B, F)
#
# 이상 점수 (3 가지 선택):
#   - recon_only       : ||x - x_recon||^2  (DenoisingAE 와 비교 가능)
#   - elbo_neg         : recon + beta * KL  (ELBO 음수)
#   - combined         : alpha * recon + beta * KL  (가중 결합)
# ===========================================================================

class VariationalAutoencoder(nn.Module):
    """간단한 tabular VAE — Hydraulic 17 sensor 입력 용 (확장 가능)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        beta: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder: x → hidden → (mu, logvar)
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.enc_fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z → hidden → x_recon
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.dec_out = nn.Linear(hidden_dim * 2, input_dim)

        # threshold (anomaly 판정용, fit 후 채워짐)
        self.threshold: Optional[float] = None

    def encode(self, x: torch.Tensor):
        h = torch.relu(self.enc_fc1(x))
        h = torch.relu(self.enc_fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # std 안정성: logvar clamp
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.dec_fc1(z))
        h = torch.relu(self.dec_fc2(h))
        return self.dec_out(h)

    def forward(self, x: torch.Tensor):
        """학습용 forward → (recon, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def vae_loss(
        recon: torch.Tensor, x: torch.Tensor,
        mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0,
    ) -> torch.Tensor:
        """ELBO 음수 (reconstruction MSE + beta * KL divergence)."""
        recon_loss = torch.nn.functional.mse_loss(recon, x, reduction="mean")
        # KL(q(z|x) || N(0,I))
        logvar_c = torch.clamp(logvar, min=-10, max=10)
        kl = -0.5 * torch.mean(1 + logvar_c - mu.pow(2) - logvar_c.exp())
        return recon_loss + beta * kl

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, mode: str = "elbo_neg") -> torch.Tensor:
        """추론용 이상 점수 (값 클수록 이상).

        mode:
            "recon_only" : ||x - recon||^2 (DenoisingAE 와 동일 비교 가능)
            "elbo_neg"   : recon + beta * KL (ELBO 음수)
            "combined"   : 0.7 * recon + 0.3 * KL (가중 합)
        """
        self.eval()
        mu, logvar = self.encode(x)
        # 추론은 deterministic (mu 사용)
        recon = self.decode(mu)
        recon_err = ((x - recon) ** 2).mean(dim=1)
        if mode == "recon_only":
            return recon_err
        logvar_c = torch.clamp(logvar, min=-10, max=10)
        kl_per_sample = -0.5 * (1 + logvar_c - mu.pow(2) - logvar_c.exp()).sum(dim=1)
        if mode == "elbo_neg":
            return recon_err + self.beta * kl_per_sample
        if mode == "combined":
            return 0.7 * recon_err + 0.3 * kl_per_sample
        raise ValueError(f"Unknown anomaly_score mode: {mode}")


# ===========================================================================
# 3) BiLSTM + Self-Attention Pooling (RUL 회귀)
# ===========================================================================

class AttentionPooling(nn.Module):
    """시퀀스 차원에 대한 학습 가능한 attention pooling.

    LSTM 출력 (B, L, H) → (B, H). last-step이나 mean보다 중요 시점에 가중을
    줄 수 있어 RUL 정확도가 향상된다.
    수치 안정성: squeeze 없이 keepdim 형태로 가중치 텐서를 유지하여 L=1
    엣지케이스에서도 broadcast가 안전하게 동작한다.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 3:
            raise RuntimeError(
                f"AttentionPooling expects (B, L, H), got shape={tuple(h.shape)}"
            )
        scores = self.attn(h)                       # (B, L, 1)
        weights = torch.softmax(scores, dim=1)      # (B, L, 1) ← dim=1 = sequence
        return (h * weights).sum(dim=1)             # (B, H)


class BiLSTMRegressor(nn.Module):
    """양방향 LSTM + Attention pooling 기반 RUL 회귀 모델.

    입력 형식 명시 (input_format):
      · "BFL" : (B, F, L) channel-first. forward에서 transpose.
      · "BLF" : (B, L, F) LSTM native. transpose 없음.
      자동 추론 대신 생성자에서 명시받는 이유: F == L인 우연한 케이스에서
      축이 무음으로 뒤집혀 학습되는 것을 막기 위함.
    """
    def __init__(self, input_dim: int, hidden: int = 64,
                 num_layers: int = 2, dropout: float = 0.4,
                 input_format: str = "BFL"):
        super().__init__()
        if input_format not in ("BFL", "BLF"):
            raise ValueError(
                f"input_format must be 'BFL' or 'BLF', got '{input_format}'"
            )
        self.input_dim = input_dim
        self.input_format = input_format
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn_pool = AttentionPooling(hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise RuntimeError(
                f"BiLSTMRegressor expects 3D input, got dim={x.dim()} shape={tuple(x.shape)}"
            )
        if self.input_format == "BFL":
            if x.shape[1] != self.input_dim:
                raise RuntimeError(
                    f"BiLSTMRegressor(input_format=BFL) expected F={self.input_dim} "
                    f"at dim=1, got shape={tuple(x.shape)}"
                )
            x = x.transpose(1, 2)                   # → (B, L, F)
        else:  # "BLF"
            if x.shape[2] != self.input_dim:
                raise RuntimeError(
                    f"BiLSTMRegressor(input_format=BLF) expected F={self.input_dim} "
                    f"at dim=2, got shape={tuple(x.shape)}"
                )

        out, _ = self.lstm(x)                       # (B, L, 2H)
        pooled = self.dropout(self.attn_pool(out))
        # fc 출력은 (B, 1). view(-1)로 (B,) 변환 → 배치 크기 1에서도 안전
        return self.fc(pooled).view(-1)


# ===========================================================================
# DLinear (Zeng et al. 2023, "Are Transformers Effective for Time Series Forecasting?")
# RUL 회귀용 변형: 마지막에 채널 aggregation MLP 추가
# ===========================================================================

class _MovingAverage(nn.Module):
    """Trend 추출용 1D moving average (양 끝 replicate padding)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # 홀수로 강제 (대칭 padding)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)            # (B, L+2pad, F)
        x = x.permute(0, 2, 1)                            # (B, F, L+2pad)
        x = nn.functional.avg_pool1d(x, kernel_size=self.kernel_size, stride=1)
        return x.permute(0, 2, 1)                         # (B, L, F)


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.ma = _MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.ma(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearRegressor(nn.Module):
    """DLinear 기반 RUL 회귀 모델.

    원 논문: 시계열을 trend + seasonal 로 분해, 각 component 에 채널별 linear 적용.
    이 구현은 RUL (scalar) 예측을 위해 마지막에 채널 aggregation linear 를 추가한다.

    구조:
        x (B, F, L) or (B, L, F)
          → SeriesDecomp(kernel) → seasonal, trend
          → channel-individual Linear: L → 1 (각 채널마다)
          → seasonal + trend (B, F)
          → channel aggregation Linear: F → 1
          → (B,)

    파라미터:
        input_dim   : feature 차원 (F)
        seq_len     : 시퀀스 길이 (L). 모델 생성 시 고정.
        kernel_size : moving average kernel (홀수, default 25)
        individual  : 채널별 별도 linear 사용 여부 (True 권장)
        input_format: "BFL" 또는 "BLF"
    """

    def __init__(self, input_dim: int, seq_len: int,
                 kernel_size: int = 25, individual: bool = True,
                 input_format: str = "BFL"):
        super().__init__()
        if input_format not in ("BFL", "BLF"):
            raise ValueError(f"input_format must be 'BFL' or 'BLF', got '{input_format}'")
        # seq_len 보다 큰 kernel 은 양 끝 padding 으로 흐려지므로 보정
        effective_kernel = min(kernel_size, seq_len if seq_len % 2 == 1 else seq_len - 1)
        if effective_kernel < 3:
            effective_kernel = 3

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.input_format = input_format
        self.individual = individual

        self.decomp = _SeriesDecomp(effective_kernel)

        if individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(seq_len, 1) for _ in range(input_dim)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(seq_len, 1) for _ in range(input_dim)]
            )
        else:
            self.linear_seasonal = nn.Linear(seq_len, 1)
            self.linear_trend = nn.Linear(seq_len, 1)

        # 채널 aggregation — 여기서만 cross-channel mixing 발생
        self.channel_proj = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise RuntimeError(
                f"DLinearRegressor expects 3D input, got dim={x.dim()} shape={tuple(x.shape)}"
            )
        if self.input_format == "BFL":
            if x.shape[1] != self.input_dim:
                raise RuntimeError(
                    f"DLinearRegressor(BFL) expected F={self.input_dim} at dim=1, "
                    f"got shape={tuple(x.shape)}"
                )
            x = x.transpose(1, 2)  # → (B, L, F)
        else:  # "BLF"
            if x.shape[2] != self.input_dim:
                raise RuntimeError(
                    f"DLinearRegressor(BLF) expected F={self.input_dim} at dim=2, "
                    f"got shape={tuple(x.shape)}"
                )
        if x.shape[1] != self.seq_len:
            raise RuntimeError(
                f"DLinearRegressor expected L={self.seq_len} at dim=1, "
                f"got shape={tuple(x.shape)}"
            )

        seasonal, trend = self.decomp(x)              # (B, L, F) each
        seasonal = seasonal.permute(0, 2, 1)          # (B, F, L)
        trend = trend.permute(0, 2, 1)                # (B, F, L)

        if self.individual:
            seasonal_outs = []
            trend_outs = []
            for i in range(self.input_dim):
                seasonal_outs.append(self.linear_seasonal[i](seasonal[:, i, :]))  # (B, 1)
                trend_outs.append(self.linear_trend[i](trend[:, i, :]))           # (B, 1)
            seasonal_out = torch.stack(seasonal_outs, dim=1)  # (B, F, 1)
            trend_out = torch.stack(trend_outs, dim=1)        # (B, F, 1)
        else:
            seasonal_out = self.linear_seasonal(seasonal)     # (B, F, 1)
            trend_out = self.linear_trend(trend)              # (B, F, 1)

        out = (seasonal_out + trend_out).squeeze(-1)          # (B, F)
        out = self.channel_proj(out).view(-1)                 # (B,)
        return out


# ===========================================================================
# iTransformer (Liu et al. 2024, ICLR, arXiv:2310.06625) — RUL 회귀 변형
# "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
# 공식 구현: github.com/thuml/iTransformer
#
# 핵심 아이디어:
#   - 전통적 Transformer: 토큰 = 시간 스텝, 토큰 수 = 시퀀스 길이 L
#   - iTransformer  : 토큰 = 각 변수(센서)의 전체 시계열, 토큰 수 = 변수 수 F
#   → cross-channel attention 직접 학습 → multivariate 시계열에 강함
#
# 본 구현이 원논문 대비 채택한 개선 (RUL 회귀 + XAI 정합):
#   1. Pre-Norm 구조 (Xiong et al. 2020 — 학습 안정성)
#   2. FFN 4× expansion (Vaswani et al. 2017 표준)
#   3. Attention Pooling (per-variate projection 대신 RUL scalar 출력 + 변수 중요도)
#   4. forward(x, return_aux=True) 로 attention map + var_importance 노출 (XAI)
#   5. d_model = 128 (BiLSTM 표현력 비교 가능)
#
# 원논문 충실 — 의도적으로 추가하지 않은 것:
#   - Positional Encoding: 원논문 "position embedding is no longer needed here"
#     이유: 토큰이 변수의 전체 시계열을 함축 → 시간 정보가 토큰 안에 이미 존재
#     FFN 이 temporal pattern 을 담당. PE 추가 ablation 은 학습 후 별도 비교.
#   - Output ReLU: gradient saturation 방지 (RUL ≈ 0 영역 학습 유지).
#   - Input LayerNorm: data_pipeline._scale_seq() 가 StandardScaler 적용 중
#     (이중 정규화 회피).
#   - Residual scaling: n_layers=2 얕은 모델에 부적합 (deep ≥ 12 layers 용 기법).
#
# 보류 (학습 후 검토):
#   - RevIN (Reversible Instance Normalization): non-stationary 데이터에 효과적이나
#     C-MAPSS 가 충분히 stationary 한지 데이터 분석 후 결정.
# ===========================================================================

class _ITransformerEncoderLayer(nn.Module):
    """iTransformer encoder block — Pre-Norm + 변수 self-attention + FFN.

    Pre-Norm 구조 (Xiong et al. 2020):
        x = x + Attn(LN(x))
        x = x + FFN(LN(x))
    Post-Norm 대비 deep network 안정성 ↑, warmup 의존도 ↓.
    """

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # Pre-Norm attention
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(
            h, h, h, need_weights=return_attn, average_attn_weights=False
        )
        x = x + self.dropout(attn_out)

        # Pre-Norm FFN
        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))

        if return_attn:
            return x, attn_weights
        return x, None


class _AttentionPool(nn.Module):
    """변수 토큰을 softmax-weighted sum 으로 집계.

    원논문의 per-variate Linear projection (D → pred_len) 대신, RUL scalar 출력을
    위해 학습 가능한 변수 중요도 가중치를 사용. BiLSTMRegressor 의 AttentionPooling
    과 일관된 패턴 + 변수 중요도가 XAI 출력으로 직접 활용 가능.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor):
        # h: (B, F, D)
        weights = torch.softmax(self.score(h), dim=1)   # (B, F, 1)
        pooled = (h * weights).sum(dim=1)               # (B, D)
        return pooled, weights


class iTransformerRegressor(nn.Module):
    """iTransformer 기반 RUL 회귀 모델.

    구조:
        x (B, F, L)                       ← 입력 (channel-first, 이미 정규화됨)
        ↓ Variate Embedding: Linear(L → D) → (B, F, D)
        ↓ Pre-Norm Encoder × n_layers      → (B, F, D)
        ↓ Final LayerNorm                  → (B, F, D)
        ↓ Attention Pooling (over F)       → (B, D), var_importance (B, F, 1)
        ↓ Head: Linear(D → 1)              → (B,)

    forward:
        model(x)                  → tensor (B,)   ← 학습/평가용 (기존 trainer 호환)
        model(x, return_aux=True) → (pred, aux)   ← XAI 용
            aux = {
                "var_importance": (B, F, 1) — softmax weights over variables,
                "attn_maps":      list of (B, n_heads, F, F) — 각 layer attention,
            }
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ffn: Optional[int] = None,
        dropout: float = 0.15,
        input_format: str = "BFL",
    ):
        super().__init__()
        if input_format not in ("BFL", "BLF"):
            raise ValueError(f"input_format must be 'BFL' or 'BLF', got '{input_format}'")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        if d_ffn is None:
            d_ffn = d_model * 4  # Transformer 표준 (Vaswani et al. 2017)

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.input_format = input_format

        # 1) Variate embedding: 각 변수의 시계열(L) → 임베딩 차원(D)
        #    iTransformer 원논문은 PE 미사용 — temporal 정보는 이 Linear weight 와 FFN 이 담당
        self.embedding = nn.Linear(seq_len, d_model)
        self.input_dropout = nn.Dropout(dropout)

        # 2) Pre-Norm Encoder layers
        self.layers = nn.ModuleList([
            _ITransformerEncoderLayer(d_model, n_heads, d_ffn, dropout)
            for _ in range(n_layers)
        ])

        # 3) Final LayerNorm (Pre-Norm 구조의 표준)
        self.norm = nn.LayerNorm(d_model)

        # 4) Attention Pooling — 변수 중요도 학습 + XAI
        self.pool = _AttentionPool(d_model)

        # 5) Output head: scalar RUL (ReLU 사용 안 함 — gradient saturation 방지)
        self.head = nn.Linear(d_model, 1)

    def _validate_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise RuntimeError(
                f"iTransformerRegressor expects 3D input, got dim={x.dim()} "
                f"shape={tuple(x.shape)}"
            )
        if self.input_format == "BFL":
            if x.shape[1] != self.input_dim:
                raise RuntimeError(
                    f"iTransformer(BFL) expected F={self.input_dim} at dim=1, "
                    f"got {tuple(x.shape)}"
                )
        else:  # BLF
            if x.shape[2] != self.input_dim:
                raise RuntimeError(
                    f"iTransformer(BLF) expected F={self.input_dim} at dim=2, "
                    f"got {tuple(x.shape)}"
                )
            x = x.transpose(1, 2)  # → (B, F, L)
        if x.shape[2] != self.seq_len:
            raise RuntimeError(
                f"iTransformer expected L={self.seq_len} at dim=2, "
                f"got {tuple(x.shape)}"
            )
        return x

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = self._validate_shape(x)         # (B, F, L)

        # 1) Variate embedding (PE 없음 — 원논문 충실)
        h = self.embedding(x)               # (B, F, D)
        h = self.input_dropout(h)

        # 2) Encoder layers (Pre-Norm)
        attn_maps: List[torch.Tensor] = []
        for layer in self.layers:
            h, attn = layer(h, return_attn=return_aux)
            if return_aux and attn is not None:
                attn_maps.append(attn)

        # 3) Final norm (Pre-Norm 표준)
        h = self.norm(h)

        # 4) Attention pooling — 변수 중요도 학습
        pooled, var_importance = self.pool(h)  # (B, D), (B, F, 1)

        # 5) Output
        pred = self.head(pooled).view(-1)      # (B,)

        if return_aux:
            return pred, {
                "var_importance": var_importance,  # (B, F, 1)
                "attn_maps": attn_maps,            # list of (B, n_heads, F, F)
            }
        return pred


# ===========================================================================
# 모델 팩토리
# ===========================================================================

ARCHS = ("cnn_vibration", "cnn_tabular", "ae", "vae", "lstm", "dlinear", "itransformer")


def build_model(arch: str, **kwargs) -> nn.Module:
    """문자열 키 기반 모델 팩토리.

    arch:
      "cnn_vibration"  : WDCNN1D                (CWRU 등 진동 신호)
      "cnn_tabular"    : TabularCNN1D           (AI4I 등 짧은 테이블)
      "ae"             : DenoisingAE
      "vae"            : VariationalAutoencoder (이상 탐지, ELBO 기반)
      "lstm"           : BiLSTMRegressor        (RUL 회귀, 시계열)
      "dlinear"        : DLinearRegressor       (RUL 회귀, lightweight baseline)
      "itransformer"   : iTransformerRegressor  (RUL 회귀, cross-channel attention)

    오타 방어: 알려지지 않은 arch는 difflib로 가까운 후보를 추천한다.
    """
    if arch not in ARCHS:
        suggestions = difflib.get_close_matches(arch, ARCHS, n=2, cutoff=0.4)
        hint = f" Did you mean: {suggestions}?" if suggestions else ""
        raise ValueError(
            f"Unknown arch '{arch}'. Available: {ARCHS}.{hint}"
        )
    if arch == "cnn_vibration":
        return WDCNN1D(**kwargs)
    if arch == "cnn_tabular":
        return TabularCNN1D(**kwargs)
    if arch == "ae":
        return DenoisingAE(**kwargs)
    if arch == "vae":
        return VariationalAutoencoder(**kwargs)
    if arch == "lstm":
        return BiLSTMRegressor(**kwargs)
    if arch == "dlinear":
        return DLinearRegressor(**kwargs)
    # arch == "itransformer"
    return iTransformerRegressor(**kwargs)

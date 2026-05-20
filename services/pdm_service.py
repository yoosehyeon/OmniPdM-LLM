from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from services.schemas import PredictionResult

from models_core import config
from models_core import models


class PdmService:
    """
    실제 models_core 자산과 연결되는 추론 서비스.

    지원:
    - ai4i_cnn
    - ai4i_gbdt
    - hydraulic_ae
    - cmapss_lstm
    - ncmapss_lstm

    기본 Gradio 분석 경로는 ai4i_cnn을 사용한다.

    중요:
    - predict(payload): 기존 AI4I scalar 입력용 public API (유지)
    - predict_lstm_sequence(...): 새 LSTM sequence 입력용 public API
    """

    def __init__(self, mode: str = "lite", dataset_key: str = "ai4i_cnn") -> None:
        self.mode = mode
        self.dataset_key = dataset_key
        self.device = torch.device(config.get_device())

        self._torch_model = None
        self._sk_model = None
        self._meta: Dict[str, Any] = {}
        self._input_scaler = None  # 추론 시 학습과 동일한 StandardScaler — 1회 lazy fit, _get_input_scaler 참조

        if self.mode == "full":
            self._load_model()

    # ------------------------------------------------------------------
    # Public API - 기존 scalar 입력용
    # ------------------------------------------------------------------
    def predict(self, payload: Dict[str, float]) -> PredictionResult:
        """
        payload는 기본적으로 5개 AI4I 센서 입력을 받는다.
        full 모드에서는 실제 체크포인트를 사용한다.
        lite 모드에서는 규칙 기반 추론을 사용한다.
        """
        if self.mode == "lite":
            return self._predict_lite(payload)

        if self.dataset_key == "ai4i_cnn":
            return self._predict_ai4i_cnn(payload)
        if self.dataset_key == "ai4i_gbdt":
            return self._predict_ai4i_gbdt(payload)
        if self.dataset_key == "hydraulic_ae":
            return self._predict_hydraulic_ae(payload)
        if self.dataset_key == "cmapss_lstm":
            return self._predict_lstm(payload, dataset_key="cmapss_lstm")
        if self.dataset_key == "ncmapss_lstm":
            return self._predict_lstm(payload, dataset_key="ncmapss_lstm")

        raise ValueError(f"Unsupported dataset_key: {self.dataset_key}")

    # ------------------------------------------------------------------
    # Public API - 새 LSTM sequence 입력용
    # ------------------------------------------------------------------
    def predict_lstm_sequence(
        self,
        sequence: List[List[float]],
        feature_names: Optional[List[str]] = None,
    ) -> PredictionResult:
        """
        LSTM sequence 전용 추론 진입점.

        입력:
            sequence: [timesteps][features]
            feature_names: 옵션. explain용 상위 기여 feature 이름에 사용

        지원 dataset_key:
            - cmapss_lstm
            - ncmapss_lstm

        반환:
            PredictionResult
        """
        if self.dataset_key not in {"cmapss_lstm", "ncmapss_lstm"}:
            raise ValueError(
                f"predict_lstm_sequence is only supported for cmapss_lstm / ncmapss_lstm, "
                f"got dataset_key={self.dataset_key}"
            )

        if self.mode != "full":
            raise NotImplementedError(
                "LSTM sequence inference currently requires mode='full' because "
                "it depends on actual LSTM checkpoints."
            )

        return self._predict_lstm_sequence_full(
            sequence=sequence,
            feature_names=feature_names,
            dataset_key=self.dataset_key,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """
        체크포인트를 로드한다.
        ai4i_cnn / hydraulic_ae / lstm 계열은 torch state_dict,
        ai4i_gbdt는 pickle 모델을 사용한다.
        """
        if self.dataset_key == "ai4i_cnn":
            ckpt_path = self._find_checkpoint("ai4i_cnn", ".pt")
            meta = self._load_checkpoint_meta(ckpt_path)
            feat_names = meta["feature_names"]
            reorder = models.build_reorder_index(feat_names, models.AI4I_REORDER_NAMES)

            model = models.build_model(
                "cnn_tabular",
                in_channels=1,
                seq_len=meta["feature_dim"],
                n_classes=1,
                dropout=config.CNN_CFG["dropout"],
                reorder_index=reorder,
                expected_feature_names=feat_names,
            )
            state = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device).eval()

            self._torch_model = model
            self._meta = meta
            return

        if self.dataset_key == "ai4i_gbdt":
            ckpt_path = self._find_checkpoint("ai4i_gbdt", ".pkl")
            with open(ckpt_path, "rb") as f:
                self._sk_model = pickle.load(f)
            self._meta = self._load_checkpoint_meta(ckpt_path)
            return

        if self.dataset_key == "hydraulic_ae":
            ckpt_path = self._find_checkpoint("hydraulic_ae", ".pt")
            meta = self._load_checkpoint_meta(ckpt_path)
            model = models.build_model(
                "ae",
                input_dim=meta["feature_dim"],
                latent_dim=config.AE_CFG["latent_dim"],
            )
            state = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device).eval()

            self._torch_model = model
            self._meta = meta
            return

        if self.dataset_key in {"cmapss_lstm", "ncmapss_lstm"}:
            ckpt_path = self._find_checkpoint(self.dataset_key, ".pt")
            meta = self._load_checkpoint_meta(ckpt_path)
            cfg = config.LSTM_CFG if self.dataset_key == "cmapss_lstm" else config.NCMAPSS_LSTM_CFG

            model = models.build_model(
                "lstm",
                input_dim=meta["feature_dim"],
                hidden=cfg["hidden"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
                input_format="BFL",
            )
            state = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device).eval()

            self._torch_model = model
            self._meta = meta
            return

        raise ValueError(f"Unsupported dataset_key for loading: {self.dataset_key}")

    def _find_checkpoint(self, stem: str, suffix: str) -> Path:
        """체크포인트 경로 해결.

        1) 로컬 `config.CHECKPOINT_DIR` 에서 정확히 `{stem}_<digits>...` 패턴만 매칭.
           예: stem="ai4i_cnn" 일 때 `ai4i_cnn_recall_*.pt` 같은 longer-stem 변형은 제외하기 위해
           `_` 직후 첫 글자가 숫자(timestamp 시작) 여야 한다.
        2) 없으면 HF Model Hub (`config.CHECKPOINT_REPO`) 에서 동일 규칙으로 매칭 후 download.
           동일 stem 의 `_meta.json` 도 함께 pull 해 `_load_checkpoint_meta()` 가 읽을 수 있게 한다.
        """
        prefix = f"{stem}_"

        def _is_exact_stem_match(name: str) -> bool:
            """name 이 '{stem}_<digit>...{suffix}' 형태인지."""
            if not name.startswith(prefix) or not name.endswith(suffix):
                return False
            tail = name[len(prefix):]
            return len(tail) > 0 and tail[0].isdigit()

        if config.CHECKPOINT_DIR.exists():
            local = sorted(
                p for p in config.CHECKPOINT_DIR.glob(f"{stem}_*{suffix}")
                if _is_exact_stem_match(p.name)
            )
            if local:
                return local[-1]

        from huggingface_hub import hf_hub_download, list_repo_files

        all_files = list_repo_files(config.CHECKPOINT_REPO, repo_type="model")
        matches = sorted(f for f in all_files if _is_exact_stem_match(f))
        if not matches:
            raise FileNotFoundError(
                f"No checkpoint for {stem}{suffix} in {config.CHECKPOINT_REPO}"
            )

        picked = matches[-1]
        ckpt_local = hf_hub_download(
            config.CHECKPOINT_REPO, picked, repo_type="model"
        )

        # sibling meta 도 동일 snapshot 에 내려 _load_checkpoint_meta 가 .with_name() 으로 찾을 수 있게.
        picked_stem = picked[: -len(suffix)]
        meta_filename = f"{picked_stem}_meta.json"
        if meta_filename in all_files:
            hf_hub_download(
                config.CHECKPOINT_REPO, meta_filename, repo_type="model"
            )

        return Path(ckpt_local)

    def _load_checkpoint_meta(self, ckpt_path: Path) -> Dict[str, Any]:
        """체크포인트 옆 `<stem>_meta.json` 을 읽는다.

        로컬/HF 캐시 둘 다에서 ckpt_path.with_name(...) 으로 접근 가능해야 한다.
        (_find_checkpoint 이 HF download 시 sibling meta 도 내려받음)
        """
        meta_path = ckpt_path.with_name(f"{ckpt_path.stem}_meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Checkpoint meta not found: {meta_path}. "
                f"업로드/로컬 모두에 {ckpt_path.stem}_meta.json 이 있는지 확인하세요."
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Lite mode
    # ------------------------------------------------------------------
    def _predict_lite(self, payload: Dict[str, float]) -> PredictionResult:
        wear = payload["tool_wear_min"]
        torque = payload["torque_nm"]
        rpm = payload["rotational_speed_rpm"]

        failure_probability = min(
            1.0,
            (wear / 250.0) * 0.45
            + (torque / 80.0) * 0.35
            + ((1600 - rpm) / 1600.0) * 0.20,
        )
        anomaly_score = min(
            1.0,
            (wear / 300.0) * 0.50 + (torque / 100.0) * 0.50,
        )

        return PredictionResult(
            dataset_key=self.dataset_key,
            task_type="binary_classification",
            model_name="TabularCNN1D-lite",
            model_mode="lite",
            predicted_label="Failure Risk" if failure_probability >= 0.5 else "Normal",
            failure_probability=round(max(0.0, failure_probability), 4),
            anomaly_score=round(max(0.0, anomaly_score), 4),
            rul_norm=1.0,
            top_contributors=[
                ("tool_wear_min", wear),
                ("torque_nm", torque),
                ("rotational_speed_rpm", rpm),
            ],
            raw_output={},
        )

    # ------------------------------------------------------------------
    # Full mode: AI4I CNN
    # ------------------------------------------------------------------
    def _predict_ai4i_cnn(self, payload: Dict[str, float]) -> PredictionResult:
        x = self._build_ai4i_feature_vector(payload)  # raw (11,) — 학습 시 StandardScaler 적용 전 형태

        # 학습 시 _fit_apply_scaler_2d 가 train 분할로 fit 한 StandardScaler 를 동일하게 적용해야
        # 모델 입력 분포가 학습과 일치한다. lazy fit + 인스턴스 캐시.
        scaler = self._get_input_scaler()
        if scaler is not None:
            x = scaler.transform(x.reshape(1, -1)).reshape(-1).astype(np.float32)

        x = x[None, None, :]  # (1, 1, 11)

        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            logits = self._torch_model(x_t).view(-1)
            prob = torch.sigmoid(logits).item()

        predicted_label = "Failure Risk" if prob >= config.CNN_CFG["decision_threshold"] else "Normal"

        top_contributors = [
            ("Tool wear [min]", float(x[0, 0, 4])),
            ("Torque [Nm]", float(x[0, 0, 3])),
            ("Rotational speed [rpm]", float(x[0, 0, 2])),
        ]

        return PredictionResult(
            dataset_key="ai4i_cnn",
            task_type="binary_classification",
            model_name="TabularCNN1D",
            model_mode="full",
            predicted_label=predicted_label,
            failure_probability=round(float(prob), 4),
            anomaly_score=0.0,
            rul_norm=1.0,
            top_contributors=top_contributors,
            raw_output={"logit": float(logits.item())},
        )

    # ------------------------------------------------------------------
    # Full mode: AI4I GBDT
    # ------------------------------------------------------------------
    def _predict_ai4i_gbdt(self, payload: Dict[str, float]) -> PredictionResult:
        x = self._build_ai4i_feature_vector(payload)[None, :]  # (1, 11)

        if hasattr(self._sk_model, "predict_proba"):
            prob = float(self._sk_model.predict_proba(x)[0, 1])
        else:
            pred = int(self._sk_model.predict(x)[0])
            prob = 1.0 if pred == 1 else 0.0

        predicted_label = "Failure Risk" if prob >= 0.5 else "Normal"

        return PredictionResult(
            dataset_key="ai4i_gbdt",
            task_type="binary_classification",
            model_name="HistGradientBoosting",
            model_mode="full",
            predicted_label=predicted_label,
            failure_probability=round(prob, 4),
            anomaly_score=0.0,
            rul_norm=1.0,
            top_contributors=[
                ("Tool wear [min]", payload["tool_wear_min"]),
                ("Torque [Nm]", payload["torque_nm"]),
                ("Rotational speed [rpm]", payload["rotational_speed_rpm"]),
            ],
            raw_output={},
        )

    # ------------------------------------------------------------------
    # Full mode: AE (placeholder contract)
    # ------------------------------------------------------------------
    def _predict_hydraulic_ae(self, payload: Dict[str, float]) -> PredictionResult:
        """
        Hydraulic AE는 실제 입력 계약이 AI4I 5센서와 다르다.
        따라서 Gradio 기본 입력으로는 완전한 현업 입력을 받을 수 없다.
        여기서는 full 연결 인터페이스만 제공한다.
        """
        raise NotImplementedError(
            "hydraulic_ae full inference requires dataset-specific input schema, "
            "not the AI4I 5-sensor payload."
        )

    # ------------------------------------------------------------------
    # 기존 scalar 경로용 placeholder
    # ------------------------------------------------------------------
    def _predict_lstm(self, payload: Dict[str, float], dataset_key: str) -> PredictionResult:
        """
        C-MAPSS/N-CMAPSS LSTM은 시계열 입력이 필요하다.
        기본 Gradio 5센서 폼으로는 단일 시점만 들어오므로,
        별도 sequence 입력 탭이 추가되기 전까지는 full 연결만 명시한다.
        """
        raise NotImplementedError(
            f"{dataset_key} full inference requires sequence input, not AI4I scalar payload."
        )

    # ------------------------------------------------------------------
    # 새 LSTM sequence full inference
    # ------------------------------------------------------------------
    def _predict_lstm_sequence_full(
        self,
        sequence: List[List[float]],
        feature_names: Optional[List[str]],
        dataset_key: str,
    ) -> PredictionResult:
        """
        실제 LSTM 체크포인트를 사용해 sequence 추론.

        현재 LSTM은 RUL regression 성격이므로,
        출력값을 rul_norm / failure_probability 로 변환하여
        기존 RiskService와 연결한다.
        """
        if self._torch_model is None:
            raise RuntimeError("LSTM torch model is not loaded.")

        x_t = self._prepare_lstm_sequence_tensor(sequence, dataset_key=dataset_key)

        with torch.no_grad():
            output = self._torch_model(x_t)

        # 회귀 출력 스칼라화
        raw_pred = float(output.view(-1)[0].item())

        rul_clip = self._get_rul_clip(dataset_key)

        # 학습 시 타깃을 [0, 1] 로 정규화한 경우 (meta["rul_norm"]=True) 모델 출력도 [0, 1] 단위.
        # 정규화 안 한 경우 (cmapss 기본) 출력은 0 ~ rul_clip cycles.
        # 두 경우를 동일하게 다루기 위해 cycles 스케일로 통일한 뒤 rul_norm 을 계산한다.
        # evaluate.py:392 의 역정규화 패턴과 동일 정책.
        if bool(self._meta.get("rul_norm", False)):
            rul_value = max(0.0, raw_pred) * float(rul_clip)  # [0,1] -> cycles
        else:
            rul_value = max(0.0, raw_pred)                    # already cycles
        rul_norm = min(1.0, rul_value / float(rul_clip))

        # failure probability를 RUL 부족 정도로 근사
        failure_probability = min(1.0, max(0.0, 1.0 - rul_norm))

        # anomaly_score는 현재 별도 AE가 아니므로 0.0 고정
        anomaly_score = 0.0

        predicted_label = "Failure Risk" if failure_probability >= 0.5 else "Normal"

        top_contributors = self._build_lstm_top_contributors(
            sequence=sequence,
            feature_names=feature_names,
            top_k=3,
        )

        return PredictionResult(
            dataset_key=dataset_key,
            task_type="regression",
            model_name="BiLSTMRegressor",
            model_mode="full",
            predicted_label=predicted_label,
            failure_probability=round(float(failure_probability), 4),
            anomaly_score=round(float(anomaly_score), 4),
            rul_norm=round(float(rul_norm), 4),
            top_contributors=top_contributors,
            raw_output={
                "raw_regression_output": raw_pred,
                "rul_value": rul_value,
                "rul_clip": rul_clip,
                "input_shape": tuple(x_t.shape),
            },
        )

    # ------------------------------------------------------------------
    # LSTM helpers
    # ------------------------------------------------------------------
    def _prepare_lstm_sequence_tensor(
        self,
        sequence: List[List[float]],
        dataset_key: str,
    ) -> torch.Tensor:
        """
        입력 sequence: [timesteps][features]

        models_core LSTM는 input_format="BFL" 로 로드하고 있으므로
        최종 텐서 shape는 (B=1, F, L) 로 맞춘다.
        """
        arr = np.asarray(sequence, dtype=np.float32)

        if arr.ndim != 2:
            raise ValueError(f"LSTM sequence must be 2D [timesteps][features], got shape={arr.shape}")

        timesteps, feature_dim = arr.shape

        raw_expected_feature_dim = self._meta.get("feature_dim")
        expected_feature_dim = (
            int(raw_expected_feature_dim)
            if raw_expected_feature_dim is not None
            else int(feature_dim)
        )

        if int(feature_dim) != int(expected_feature_dim):
            if dataset_key == "cmapss_lstm":
                expected_hint = (
                    "cmapss_lstm는 한 timestep당 14개 feature가 필요합니다. "
                    "권장 feature_names: s2,s3,s4,s7,s8,s9,s11,s12,s13,s14,s15,s17,s20,s21"
                )
            elif dataset_key == "ncmapss_lstm":
                expected_hint = "ncmapss_lstm는 한 timestep당 43개 feature가 필요합니다."
            else:
                expected_hint = "dataset_key에 맞는 feature_dim을 확인해 주세요."

            raise ValueError(
                f"LSTM feature dimension mismatch for {dataset_key}. "
                f"expected={expected_feature_dim}, got={feature_dim}. "
                f"{expected_hint}"
            )

        # 학습 시 sequence 도 (N*L, F) 통계로 StandardScaler fit → 추론에 동일 적용 (분포 일치).
        scaler = self._get_input_scaler()
        if scaler is not None:
            arr = scaler.transform(arr).astype(np.float32)

        # (L, F) -> (F, L)
        arr = arr.transpose(1, 0)

        # batch 차원 추가: (F, L) -> (1, F, L)
        arr = np.expand_dims(arr, axis=0)

        x_t = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        return x_t

    def _get_input_scaler(self):
        """학습 시 사용한 StandardScaler 를 lazy 로드 + 인스턴스 캐시.

        구현: data_pipeline LOADER 를 1회 호출해 dict["scaler"] 를 받는다.
        LOADER 가 scaler 를 반환하지 않거나 dataset 미등록이면 None — 호출자는
        scaler 없이 raw 값을 그대로 사용 (기존 동작 보존).

        성능: 첫 호출 시 데이터 파일 I/O 비용 1회 발생. 이후는 캐시 hit.
        """
        if self._input_scaler is not None:
            return self._input_scaler
        try:
            from models_core.data_pipeline import LOADERS
        except ImportError:
            return None
        loader = LOADERS.get(self.dataset_key)
        if loader is None:
            return None
        try:
            data = loader()
        except Exception:
            # 데이터 파일 미존재 / 손상 등으로 fit 실패 시 raw 동작으로 fallback.
            return None
        scaler = data.get("scaler") if isinstance(data, dict) else None
        self._input_scaler = scaler
        return scaler

    def _get_rul_clip(self, dataset_key: str) -> float:
        if dataset_key == "ncmapss_lstm":
            return float(config.NCMAPSS_LSTM_CFG["rul_clip"])
        return float(config.LSTM_CFG["rul_clip"])

    def _build_lstm_top_contributors(
        self,
        sequence: List[List[float]],
        feature_names: Optional[List[str]],
        top_k: int = 5,
    ) -> List[tuple[str, float]]:
        """
        LSTM explain과 일관된 heuristic contributor 생성.

        기준:
        - latest magnitude
        - delta magnitude
        - std
        """
        arr = np.asarray(sequence, dtype=np.float32)  # shape=(T, F)
        latest = arr[-1]
        first = arr[0]
        std_vals = arr.std(axis=0)
        delta_vals = latest - first

        if feature_names is None or len(feature_names) != len(latest):
            names = [f"f{i}" for i in range(len(latest))]
        else:
            names = list(feature_names)

        latest_abs = np.abs(latest)
        delta_abs = np.abs(delta_vals)

        def _safe_norm(x: np.ndarray) -> np.ndarray:
            xmax = float(np.max(x)) if len(x) > 0 else 0.0
            if xmax <= 1e-12:
                return np.zeros_like(x, dtype=np.float32)
            return x / xmax

        combined_score = (
            0.5 * _safe_norm(latest_abs)
            + 0.3 * _safe_norm(delta_abs)
            + 0.2 * _safe_norm(std_vals)
        )

        top_idx = np.argsort(-combined_score)[: min(top_k, len(combined_score))]

        # value는 latest 값을 사용
        return [(str(names[i]), float(latest[i])) for i in top_idx]

    # ------------------------------------------------------------------
    # Feature building - 기존 AI4I
    # ------------------------------------------------------------------
    def _build_ai4i_feature_vector(self, payload: Dict[str, float]) -> np.ndarray:
        """
        AI4I 11-feature 계약:
        - 5 numeric
        - 3 type one-hot
        - 3 derived features

        Gradio 기본 입력에는 Type이 없으므로,
        기본값으로 Type_M을 사용한다.
        """
        air = payload["air_temperature_k"]
        proc = payload["process_temperature_k"]
        rpm = payload["rotational_speed_rpm"]
        torque = payload["torque_nm"]
        wear = payload["tool_wear_min"]

        type_l, type_m, type_h = 0.0, 1.0, 0.0
        power = rpm * torque
        temp_diff = proc - air
        strain = wear * torque

        return np.asarray(
            [
                air,
                proc,
                rpm,
                torque,
                wear,
                type_l,
                type_m,
                type_h,
                power,
                temp_diff,
                strain,
            ],
            dtype=np.float32,
        )
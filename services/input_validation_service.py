from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from services.schemas import (
    LSTMSequenceInput,
    LSTMValidationResult,
    ValidationIssue,
    ValidationResult,
)


class InputValidationService:
    """
    입력 검증 서비스.

    지원:
    1) AI4I 5개 센서 scalar 입력 검증
    2) LSTM sequence 입력 검증
    """

    REQUIRED_FIELDS = {
        "air_temperature_k": (250.0, 400.0),
        "process_temperature_k": (250.0, 450.0),
        "rotational_speed_rpm": (1.0, 50000.0),
        "torque_nm": (0.0, 1000.0),
        "tool_wear_min": (0.0, 10000.0),
    }

    SUPPORTED_LSTM_DATASET_KEYS = {"cmapss_lstm", "ncmapss_lstm"}

    # ------------------------------------------------------------------
    # 1) 기존 AI4I scalar validation
    # ------------------------------------------------------------------
    def validate(self, payload: Dict[str, float]) -> ValidationResult:
        issues: List[ValidationIssue] = []
        normalized: Dict[str, float] = {}

        for field, (min_val, max_val) in self.REQUIRED_FIELDS.items():
            if field not in payload:
                issues.append(
                    ValidationIssue(
                        field=field,
                        level="ERROR",
                        message="필수 입력값이 누락되었습니다.",
                    )
                )
                continue

            raw_value = payload[field]

            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                issues.append(
                    ValidationIssue(
                        field=field,
                        level="ERROR",
                        message="숫자형으로 변환할 수 없습니다.",
                        original_value=raw_value,
                    )
                )
                continue

            if value < min_val or value > max_val:
                issues.append(
                    ValidationIssue(
                        field=field,
                        level="ERROR",
                        message=f"허용 범위를 벗어났습니다. 허용 범위: {min_val} ~ {max_val}",
                        original_value=raw_value,
                        normalized_value=value,
                    )
                )
                continue

            normalized[field] = value

        return ValidationResult(
            is_valid=not any(i.level == "ERROR" for i in issues),
            normalized_payload=normalized,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # 2) LSTM sequence validation - public entrypoint
    # ------------------------------------------------------------------
    def validate_lstm_sequence_payload(
        self,
        payload: LSTMSequenceInput,
        min_timesteps: int = 3,
        max_timesteps: int = 1000,
        max_feature_dim: int = 512,
    ) -> LSTMValidationResult:
        """
        LSTM sequence 입력 검증.

        검증 항목:
        - dataset_key 지원 여부
        - sequence 비어있지 않은지
        - 각 timestep row 길이가 동일한지
        - 각 값이 float 변환 가능한지
        - 최소/최대 timestep 범위
        - feature_names 길이 일치 여부
        """
        issues: List[ValidationIssue] = []
        normalized_sequence: List[List[float]] = []

        # 1) dataset_key 검증
        dataset_key = str(payload.dataset_key).strip() if payload.dataset_key else ""
        if dataset_key not in self.SUPPORTED_LSTM_DATASET_KEYS:
            issues.append(
                ValidationIssue(
                    field="dataset_key",
                    level="ERROR",
                    message=(
                        f"지원하지 않는 LSTM dataset_key 입니다: {payload.dataset_key}. "
                        f"지원값: {sorted(self.SUPPORTED_LSTM_DATASET_KEYS)}"
                    ),
                    original_value=payload.dataset_key,
                )
            )

        # 2) sequence 존재 여부
        if payload.sequence is None:
            issues.append(
                ValidationIssue(
                    field="sequence",
                    level="ERROR",
                    message="sequence가 없습니다.",
                    original_value=None,
                )
            )
            return LSTMValidationResult(
                is_valid=False,
                normalized_sequence=[],
                issues=issues,
                timesteps=0,
                feature_dim=0,
                feature_names=payload.feature_names,
                dataset_key=payload.dataset_key,
            )

        if not isinstance(payload.sequence, list) or len(payload.sequence) == 0:
            issues.append(
                ValidationIssue(
                    field="sequence",
                    level="ERROR",
                    message="sequence는 비어 있지 않은 2차원 리스트여야 합니다.",
                    original_value=payload.sequence,
                )
            )
            return LSTMValidationResult(
                is_valid=False,
                normalized_sequence=[],
                issues=issues,
                timesteps=0,
                feature_dim=0,
                feature_names=payload.feature_names,
                dataset_key=payload.dataset_key,
            )

        # 3) row별 숫자형 변환 및 shape 검증
        expected_feature_dim: Optional[int] = None

        for row_idx, row in enumerate(payload.sequence):
            if not isinstance(row, list) or len(row) == 0:
                issues.append(
                    ValidationIssue(
                        field=f"sequence[{row_idx}]",
                        level="ERROR",
                        message="각 timestep row는 비어 있지 않은 리스트여야 합니다.",
                        original_value=row,
                    )
                )
                continue

            normalized_row: List[float] = []

            for col_idx, value in enumerate(row):
                try:
                    f_value = float(value)
                except (TypeError, ValueError):
                    issues.append(
                        ValidationIssue(
                            field=f"sequence[{row_idx}][{col_idx}]",
                            level="ERROR",
                            message="숫자형으로 변환할 수 없습니다.",
                            original_value=value,
                        )
                    )
                    continue

                normalized_row.append(f_value)

            # 이 row에서 변환 실패가 있었어도 길이 비교는 시도
            if expected_feature_dim is None:
                expected_feature_dim = len(row)

            if len(row) != expected_feature_dim:
                issues.append(
                    ValidationIssue(
                        field=f"sequence[{row_idx}]",
                        level="ERROR",
                        message=(
                            f"feature 수가 일치하지 않습니다. "
                            f"첫 row 기준={expected_feature_dim}, 현재 row={len(row)}"
                        ),
                        original_value=row,
                        normalized_value=len(row),
                    )
                )

            normalized_sequence.append(normalized_row)

        timesteps = len(payload.sequence)
        feature_dim = expected_feature_dim or 0

        # 4) timestep 범위 검증
        if timesteps < min_timesteps:
            issues.append(
                ValidationIssue(
                    field="sequence",
                    level="ERROR",
                    message=f"LSTM 입력 timesteps가 너무 짧습니다. 최소 {min_timesteps} 이상이어야 합니다.",
                    normalized_value=timesteps,
                )
            )

        if timesteps > max_timesteps:
            issues.append(
                ValidationIssue(
                    field="sequence",
                    level="ERROR",
                    message=f"LSTM 입력 timesteps가 너무 깁니다. 최대 {max_timesteps} 이하여야 합니다.",
                    normalized_value=timesteps,
                )
            )

        # 5) feature_dim 범위 검증
        if feature_dim == 0:
            issues.append(
                ValidationIssue(
                    field="sequence",
                    level="ERROR",
                    message="feature 수를 결정할 수 없습니다.",
                    normalized_value=feature_dim,
                )
            )

        if feature_dim > max_feature_dim:
            issues.append(
                ValidationIssue(
                    field="sequence",
                    level="ERROR",
                    message=f"feature 수가 너무 큽니다. 최대 {max_feature_dim} 이하여야 합니다.",
                    normalized_value=feature_dim,
                )
            )

        # 6) 각 row의 normalized 길이 재검증
        for row_idx, row in enumerate(normalized_sequence):
            if feature_dim > 0 and len(row) != feature_dim:
                issues.append(
                    ValidationIssue(
                        field=f"normalized_sequence[{row_idx}]",
                        level="ERROR",
                        message=(
                            f"정규화 후 row 길이가 일치하지 않습니다. "
                            f"예상={feature_dim}, 실제={len(row)}"
                        ),
                        normalized_value=len(row),
                    )
                )

        # 7) feature_names 검증
        self._validate_lstm_feature_names(
            feature_names=payload.feature_names,
            feature_dim=feature_dim,
            issues=issues,
        )

        return LSTMValidationResult(
            is_valid=not any(i.level == "ERROR" for i in issues),
            normalized_sequence=normalized_sequence,
            issues=issues,
            timesteps=timesteps,
            feature_dim=feature_dim,
            feature_names=payload.feature_names,
            dataset_key=payload.dataset_key,
        )

    # ------------------------------------------------------------------
    # 3) Helper: feature_names validation
    # ------------------------------------------------------------------
    def _validate_lstm_feature_names(
        self,
        feature_names: Optional[Sequence[str]],
        feature_dim: int,
        issues: List[ValidationIssue],
    ) -> None:
        if feature_names is None:
            return

        if not isinstance(feature_names, (list, tuple)):
            issues.append(
                ValidationIssue(
                    field="feature_names",
                    level="ERROR",
                    message="feature_names는 문자열 리스트여야 합니다.",
                    original_value=feature_names,
                )
            )
            return

        if len(feature_names) != feature_dim:
            issues.append(
                ValidationIssue(
                    field="feature_names",
                    level="ERROR",
                    message=(
                        f"feature_names 길이가 feature 수와 일치하지 않습니다. "
                        f"expected={feature_dim}, actual={len(feature_names)}"
                    ),
                    original_value=list(feature_names),
                    normalized_value=len(feature_names),
                )
            )
            return

        for idx, name in enumerate(feature_names):
            if not isinstance(name, str) or not name.strip():
                issues.append(
                    ValidationIssue(
                        field=f"feature_names[{idx}]",
                        level="ERROR",
                        message="feature 이름은 비어 있지 않은 문자열이어야 합니다.",
                        original_value=name,
                    )
                )
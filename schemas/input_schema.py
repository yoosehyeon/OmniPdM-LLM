from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class LSTMSequenceInput(BaseModel):
    """
    LSTM 전용 시계열 입력 DTO

    sequence:
        2차원 배열 형태
        shape = [timesteps][features]

    feature_names:
        각 feature의 이름 목록
        sequence의 각 row 길이와 일치해야 함

    asset_id:
        선택 입력. 장비/설비 식별자

    threshold:
        이상 판정 threshold
    """

    sequence: List[List[float]] = Field(..., description="2D time-series sequence")
    feature_names: Optional[List[str]] = Field(default=None)
    asset_id: Optional[str] = Field(default="UNKNOWN")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: List[List[float]]) -> List[List[float]]:
        if not v:
            raise ValueError("sequence must not be empty")

        row_len = len(v[0])
        if row_len == 0:
            raise ValueError("sequence rows must not be empty")

        for idx, row in enumerate(v):
            if len(row) != row_len:
                raise ValueError(
                    f"All sequence rows must have the same length. "
                    f"Row 0 has {row_len}, row {idx} has {len(row)}"
                )
        return v

    @field_validator("feature_names")
    @classmethod
    def validate_feature_names(cls, v, info):
        if v is None:
            return v

        sequence = info.data.get("sequence")
        if sequence and len(v) != len(sequence[0]):
            raise ValueError(
                f"feature_names length ({len(v)}) must match feature dimension ({len(sequence[0])})"
            )
        return v
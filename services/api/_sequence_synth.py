"""슬라이더 단일 시점 입력 → LSTM 모델용 시퀀스 합성.

PoC v0: React UI 가 sequence 입력 위젯을 갖지 않으므로, sensor 슬라이더 값을
"현재 시점" 으로 보고 이전 timesteps 를 작은 jitter 로 합성.

설계 메모:
- 마지막 timestep = 슬라이더 값 그대로 (모델 추론의 "현재 상태").
- 이전 timesteps = ±5% 변동을 결정적 시드(asset_id + sensor index) 로 생성.
- 시드 고정 → 같은 입력은 같은 시퀀스 → 디버깅 가능 + LLM 코멘트 안정성.

v1 계획: 실 운전 이력에서 sequence 가져오기 (별도 PR).
"""
from __future__ import annotations

import random
from typing import Dict, List


def synthesize_sequence(
    sensor_values: Dict[str, float],
    sensor_order: List[str],
    timesteps: int,
    *,
    jitter_pct: float = 0.05,
    seed: int = 42,
) -> List[List[float]]:
    """슬라이더 값 → [timesteps][len(sensor_order)] 시퀀스.

    sensor_order 는 dataset_catalog 의 sensors 순서. 백엔드 모델은 학습 시 컬럼 순서를
    가정하므로 호출자가 일관된 순서를 보장해야 함.
    """
    rng = random.Random(seed)
    sequence: List[List[float]] = []
    for t in range(timesteps - 1):
        row = []
        for s in sensor_order:
            v = sensor_values.get(s, 0.0)
            # ±jitter_pct 결정적 변동.
            delta = (rng.random() * 2.0 - 1.0) * jitter_pct * (abs(v) if v != 0 else 1.0)
            row.append(v + delta)
        sequence.append(row)
    # 마지막 step = 슬라이더 값 그대로.
    sequence.append([sensor_values.get(s, 0.0) for s in sensor_order])
    return sequence

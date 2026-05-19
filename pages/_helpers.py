"""페이지 간 공유 헬퍼: 서비스 싱글톤, 샘플 데이터, 포맷터."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict

from services.analyze_service import AnalyzeService


SAMPLE_PAYLOADS: Dict[str, Dict[str, float]] = {
    "Normal": {
        "air_temperature_k": 298.5,
        "process_temperature_k": 308.7,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.5,
        "tool_wear_min": 55.0,
    },
    "Warning": {
        "air_temperature_k": 301.2,
        "process_temperature_k": 311.8,
        "rotational_speed_rpm": 1380.0,
        "torque_nm": 58.2,
        "tool_wear_min": 180.0,
    },
    "Critical": {
        "air_temperature_k": 303.0,
        "process_temperature_k": 314.5,
        "rotational_speed_rpm": 1290.0,
        "torque_nm": 72.1,
        "tool_wear_min": 245.0,
    },
}


AI4I_FIELD_LABELS = {
    "air_temperature_k": "Air Temperature (K)",
    "process_temperature_k": "Process Temperature (K)",
    "rotational_speed_rpm": "Rotational Speed (rpm)",
    "torque_nm": "Torque (Nm)",
    "tool_wear_min": "Tool Wear (min)",
}


@lru_cache(maxsize=8)
def get_analyze_service(mode: str, dataset_key: str) -> AnalyzeService:
    """AnalyzeService 싱글톤 (mode + dataset_key 조합별 캐시)."""
    return AnalyzeService(mode=mode, dataset_key=dataset_key)


def safe_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"<JSON serialize error: {type(e).__name__}: {e}>"


def risk_level_color(level: str) -> str:
    level = (level or "").upper()
    return {
        "CRITICAL": "danger",
        "WARNING": "warning",
        "ADVISORY": "info",
        "NORMAL": "success",
    }.get(level, "secondary")

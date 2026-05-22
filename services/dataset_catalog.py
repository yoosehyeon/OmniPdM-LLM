"""Dataset 메타 카탈로그 — single source of truth.

용도:
- /api/datasets endpoint 가 직렬화해 반환
- React `frontend/src/data.ts` 의 하드코딩을 백엔드 fetch 로 대체

설계:
- 학습용 loader (models_core/data_pipeline.py) 와 분리 — 이 카탈로그는 "UI 가 슬라이더로
  보여줄 센서 명세" 가 목적. dataset_key 는 동일하지만, loader 는 raw column index 를,
  카탈로그는 사람이 읽는 라벨 + 슬라이더 min/max/default 를 다룬다.
- 추후 DB 로 옮길 수도 있으나 현재 5종은 정적이라 Python dict 로 충분.

타입 안전성:
- TypedDict 로 React 의 frontend/src/types.ts `Sensor`/`DatasetConfig` 와 1:1 매핑.
  필드 이름 변경 시 mypy/ruff 가 즉시 잡아준다.
- DATASET_CATALOG 는 Final — "이건 mutate 하면 안 된다" 를 정적 분석기에 명시.
"""
from __future__ import annotations

from typing import Final, List, TypedDict


class Sensor(TypedDict):
    name: str
    defaultValue: float
    min: float
    max: float
    unit: str


class Dataset(TypedDict):
    id: str
    label: str
    sensors: List[Sensor]


# fmt: off  # 정렬된 컬럼 보존 (Black/ruff 가 깨뜨림). 데이터 테이블이라 가독성 우선.
DATASET_CATALOG: Final[List[Dataset]] = [
    {
        "id": "cmapss-fd001",
        "label": "C-MAPSS FD001 (Turbofan)",
        "sensors": [
            {"name": "T30 (HPC Temp)",         "defaultValue": 512,  "min": 480, "max": 560, "unit": "R"},
            {"name": "Ps30 (Static Pressure)", "defaultValue": 47,   "min": 40,  "max": 55,  "unit": "psia"},
            {"name": "BPR (Bypass Ratio)",     "defaultValue": 8.35, "min": 7.5, "max": 9.5, "unit": ""},
            {"name": "T24 (LPC Temp)",         "defaultValue": 641,  "min": 600, "max": 700, "unit": "R"},
        ],
    },
    {
        "id": "cmapss-fd002",
        "label": "C-MAPSS FD002 (Multi-condition)",
        "sensors": [
            {"name": "T30 (HPC Temp)",         "defaultValue": 515,  "min": 480, "max": 580, "unit": "R"},
            {"name": "Ps30 (Static Pressure)", "defaultValue": 46,   "min": 40,  "max": 55,  "unit": "psia"},
            {"name": "BPR (Bypass Ratio)",     "defaultValue": 8.4,  "min": 7.5, "max": 9.5, "unit": ""},
            {"name": "T24 (LPC Temp)",         "defaultValue": 642,  "min": 600, "max": 700, "unit": "R"},
        ],
    },
    {
        "id": "cwru",
        "label": "CWRU Bearing",
        "sensors": [
            {"name": "Vibration RMS",     "defaultValue": 0.5, "min": 0,  "max": 3,   "unit": "g"},
            {"name": "Spectral Kurtosis", "defaultValue": 3.2, "min": 0,  "max": 10,  "unit": ""},
            {"name": "Bearing Temp",      "defaultValue": 65,  "min": 30, "max": 120, "unit": "C"},
        ],
    },
    {
        "id": "n-cmapss",
        "label": "N-CMAPSS (Real-flight)",
        "sensors": [
            {"name": "Altitude",       "defaultValue": 30000, "min": 0,  "max": 42000, "unit": "ft"},
            {"name": "Mach",           "defaultValue": 0.7,   "min": 0,  "max": 0.9,   "unit": ""},
            {"name": "TRA (Throttle)", "defaultValue": 65,    "min": 20, "max": 100,   "unit": "%"},
        ],
    },
    {
        "id": "ai4i",
        "label": "AI4I 2020 (Milling)",
        "sensors": [
            {"name": "Torque [Nm]",             "defaultValue": 40,   "min": 10,   "max": 80,   "unit": "Nm"},
            {"name": "Rotational Speed [rpm]",  "defaultValue": 1510, "min": 1200, "max": 2900, "unit": "rpm"},
            {"name": "Tool Wear [min]",         "defaultValue": 50,   "min": 0,    "max": 260,  "unit": "min"},
            {"name": "Process Temp [K]",        "defaultValue": 308,  "min": 295,  "max": 315,  "unit": "K"},
        ],
    },
]
# fmt: on


def list_datasets() -> List[Dataset]:
    """카탈로그 사본 반환. 호출자가 mutate 해도 원본은 보호.

    Sensor 값은 모두 primitive (str/int/float, immutable) 이므로 dict shallow copy 로 충분.
    lru_cache 는 사용하지 않음 — 캐싱된 객체를 호출자가 mutate 하면 다음 호출 결과까지
    오염되어 mutate-safe 가 깨진다. /api/datasets 는 페이지 진입 시 1회 호출이라 5×N dict
    복사 비용(수 μs)은 무시 가능.
    """
    return [
        {**ds, "sensors": [dict(s) for s in ds["sensors"]]}
        for ds in DATASET_CATALOG
    ]

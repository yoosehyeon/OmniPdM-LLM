"""
P3-② Function Calling tool schemas + dispatcher.

LlmService 가 Groq (OpenAI-compatible) tool calling 을 사용할 때
모델이 호출할 수 있는 5 개의 함수를 정의하고 실행한다.

Scope B:
    1) get_feature_schema(dataset_key)           - 입력 스키마 조회
    2) get_risk_threshold_info(risk_level=None)  - 위험 등급 임계값 조회
    3) get_recent_analysis_history(limit=5)      - 최근 저장된 리포트 요약
    4) perturb_input_and_predict(perturbations)  - what-if 재예측
    5) compute_custom_risk_score(f, a, r, weights) - 사용자 가중치 위험 계산
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from models_core import config
from models_core.data_pipeline import HYDRAULIC_SENSORS
from services.input_validation_service import InputValidationService
from services.pdm_service import PdmService
from services.risk_service import RiskService


_HYDRAULIC_SENSOR_META: Dict[str, Dict[str, Any]] = {
    "PS1":  {"unit": "bar",   "min": 0.0,  "max": 200.0},
    "PS2":  {"unit": "bar",   "min": 0.0,  "max": 200.0},
    "PS3":  {"unit": "bar",   "min": 0.0,  "max": 10.0},
    "PS4":  {"unit": "bar",   "min": 0.0,  "max": 10.0},
    "PS5":  {"unit": "bar",   "min": 0.0,  "max": 10.0},
    "PS6":  {"unit": "bar",   "min": 0.0,  "max": 10.0},
    "EPS1": {"unit": "W",     "min": 0.0,  "max": 3000.0},
    "FS1":  {"unit": "l/min", "min": 0.0,  "max": 15.0},
    "FS2":  {"unit": "l/min", "min": 0.0,  "max": 15.0},
    "TS1":  {"unit": "°C",    "min": 30.0, "max": 70.0},
    "TS2":  {"unit": "°C",    "min": 30.0, "max": 70.0},
    "TS3":  {"unit": "°C",    "min": 30.0, "max": 70.0},
    "TS4":  {"unit": "°C",    "min": 30.0, "max": 70.0},
    "VS1":  {"unit": "mm/s",  "min": 0.0,  "max": 2.0},
    "CE":   {"unit": "%",     "min": 0.0,  "max": 100.0},
    "CP":   {"unit": "kW",    "min": 0.0,  "max": 3.0},
    "SE":   {"unit": "%",     "min": 0.0,  "max": 100.0},
}


logger = logging.getLogger(__name__)


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_feature_schema",
            "description": "현재 dataset_key 의 필수 입력 feature 이름과 허용 범위를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_key": {
                        "type": "string",
                        "description": "dataset 식별자. 예: ai4i_cnn, cmapss_lstm",
                    }
                },
                "required": ["dataset_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_threshold_info",
            "description": "Risk 등급별 임계값을 반환합니다. risk_level 을 지정하면 매칭된 등급도 포함합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_level": {
                        "type": "string",
                        "description": "Critical/Warning/Advisory/Normal 중 하나 (옵션)",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_analysis_history",
            "description": "최근 자동 저장된 분석 리포트의 메타데이터를 최신순으로 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "반환할 최대 건수 (기본 5, 최대 20)",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "perturb_input_and_predict",
            "description": (
                "현재 입력 payload 에 지정한 변동을 적용한 뒤 재예측/재위험계산을 수행합니다. "
                "scalar 입력 경로에서만 동작합니다. what-if 분석용."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "perturbations": {
                        "type": "object",
                        "description": (
                            "feature 이름 → 새 절대값 의 dict. "
                            "예: {\"tool_wear_min\": 200.0, \"torque_nm\": 60.0}"
                        ),
                    }
                },
                "required": ["perturbations"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_custom_risk_score",
            "description": (
                "failure_probability / anomaly_score / rul_norm 와 사용자 지정 weights 로 "
                "risk_score, risk_level 을 계산합니다. (weighted_sum 방식)"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "failure_probability": {"type": "number", "description": "[0,1]"},
                    "anomaly_score": {"type": "number", "description": "[0,1]"},
                    "rul_norm": {"type": "number", "description": "[0,1], 1=수명충분 0=소진"},
                    "weights": {
                        "type": "object",
                        "description": "failure/anomaly/rul 가중치",
                        "properties": {
                            "failure": {"type": "number"},
                            "anomaly": {"type": "number"},
                            "rul": {"type": "number"},
                        },
                    },
                },
                "required": ["failure_probability", "anomaly_score", "rul_norm"],
            },
        },
    },
]


class ToolDispatcher:
    """
    LLM 이 요청한 tool 이름 + arguments 를 받아 실제 결과 dict 를 돌려준다.
    AnalyzeService 가 현재 파이프라인 context (dataset_key, base_payload, 서비스 인스턴스) 를
    주입해서 생성한다.
    """

    def __init__(
        self,
        *,
        dataset_key: str,
        pdm_service: PdmService,
        risk_service: RiskService,
        base_payload: Optional[Dict[str, float]] = None,
    ) -> None:
        self.dataset_key = dataset_key
        self.pdm_service = pdm_service
        self.risk_service = risk_service
        self.base_payload = dict(base_payload) if base_payload else None

    def dispatch(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if name == "get_feature_schema":
                return self._get_feature_schema(args.get("dataset_key") or self.dataset_key)
            if name == "get_risk_threshold_info":
                return self._get_risk_threshold_info(args.get("risk_level"))
            if name == "get_recent_analysis_history":
                return self._get_recent_analysis_history(int(args.get("limit") or 5))
            if name == "perturb_input_and_predict":
                return self._perturb_input_and_predict(args.get("perturbations") or {})
            if name == "compute_custom_risk_score":
                return self._compute_custom_risk_score(
                    args.get("failure_probability"),
                    args.get("anomaly_score"),
                    args.get("rul_norm"),
                    args.get("weights"),
                )
            return {"error": f"unknown tool: {name}"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def _get_feature_schema(self, dataset_key: str) -> Dict[str, Any]:
        key = (dataset_key or "").strip().lower()
        if key.startswith("ai4i"):
            fields = InputValidationService.REQUIRED_FIELDS
            return {
                "dataset_key": dataset_key,
                "input_mode": "scalar",
                "fields": [
                    {"name": k, "min": v[0], "max": v[1]}
                    for k, v in fields.items()
                ],
            }
        if key in {"cmapss_lstm", "ncmapss_lstm"}:
            return {
                "dataset_key": dataset_key,
                "input_mode": "sequence",
                "timesteps_range": [3, 1000],
                "max_feature_dim": 512,
                "note": "2D sequence [timesteps][features] 입력.",
            }
        if key.startswith("hydraulic"):
            return {
                "dataset_key": dataset_key,
                "input_mode": "scalar",
                "note": (
                    "Cycle-averaged 17-dim sensor vector "
                    "(data_pipeline.load_hydraulic_ae 참조). "
                    "min/max 는 UCI Hydraulic 데이터셋 기반 approximate reference 이며, "
                    "실제 검증은 InputValidationService 가 담당합니다."
                ),
                "fields": [
                    {"name": name, **_HYDRAULIC_SENSOR_META[name]}
                    for name in HYDRAULIC_SENSORS
                ],
            }
        return {
            "dataset_key": dataset_key,
            "input_mode": "unknown",
            "note": "지원되지 않는 dataset_key 입니다.",
        }

    def _get_risk_threshold_info(self, risk_level: Optional[str]) -> Dict[str, Any]:
        thresholds = [{"level": name, "min_score": thr} for name, thr in config.RISK_LEVELS]
        if risk_level:
            key = str(risk_level).strip().lower()
            matched = [x for x in thresholds if x["level"].lower() == key]
            return {"query": risk_level, "matched": matched, "all_thresholds": thresholds}
        return {"all_thresholds": thresholds}

    def _get_recent_analysis_history(self, limit: int = 5) -> Dict[str, Any]:
        limit = max(1, min(int(limit), 20))
        reports_dir = Path(config.REPORT_DIR)
        if not reports_dir.exists():
            return {"items": [], "count": 0, "note": "리포트 저장 디렉토리가 없습니다."}

        files = sorted(
            reports_dir.glob("*_report_*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]

        items: List[Dict[str, Any]] = []
        for md_path in files:
            entry: Dict[str, Any] = {
                "file": md_path.name,
                "saved_at": datetime.fromtimestamp(md_path.stat().st_mtime).isoformat(
                    timespec="seconds"
                ),
            }
            json_path = md_path.with_suffix(".json")
            if json_path.exists():
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                    pred = data.get("prediction") or {}
                    risk = data.get("risk") or {}
                    entry.update(
                        {
                            "dataset_key": pred.get("dataset_key"),
                            "predicted_label": pred.get("predicted_label"),
                            "failure_probability": pred.get("failure_probability"),
                            "risk_level": risk.get("risk_level"),
                            "risk_score": risk.get("risk_score"),
                        }
                    )
                except Exception as e:
                    logger.warning(
                        "recent history JSON parse failed (%s): %s: %s",
                        json_path.name, type(e).__name__, e,
                    )
            items.append(entry)

        return {"items": items, "count": len(items)}

    def _perturb_input_and_predict(self, perturbations: Dict[str, Any]) -> Dict[str, Any]:
        if not self.base_payload:
            return {"error": "base_payload 없음. scalar 입력 경로에서만 지원됩니다."}
        if not isinstance(perturbations, dict) or not perturbations:
            return {"error": "perturbations 는 비어 있지 않은 dict 이어야 합니다."}

        new_payload: Dict[str, float] = dict(self.base_payload)
        applied: Dict[str, float] = {}
        for k, v in perturbations.items():
            try:
                new_payload[k] = float(v)
                applied[k] = float(v)
            except (TypeError, ValueError):
                continue

        if not applied:
            return {"error": "적용 가능한 수치 변동이 없습니다."}

        val = InputValidationService().validate(new_payload)
        if not val.is_valid:
            return {
                "error": "perturbed payload 검증 실패",
                "issues": [f"{i.field}: {i.message}" for i in val.issues],
            }

        new_pred = self.pdm_service.predict(val.normalized_payload)
        new_risk = self.risk_service.fuse(new_pred)

        base_pred = self.pdm_service.predict(self.base_payload)
        base_risk = self.risk_service.fuse(base_pred)

        return {
            "applied_perturbations": applied,
            "base": {
                "predicted_label": base_pred.predicted_label,
                "failure_probability": round(float(base_pred.failure_probability), 4),
                "risk_score": round(float(base_risk.risk_score), 4),
                "risk_level": base_risk.risk_level,
            },
            "perturbed": {
                "predicted_label": new_pred.predicted_label,
                "failure_probability": round(float(new_pred.failure_probability), 4),
                "anomaly_score": round(float(new_pred.anomaly_score), 4),
                "rul_norm": round(float(new_pred.rul_norm), 4),
                "risk_score": round(float(new_risk.risk_score), 4),
                "risk_level": new_risk.risk_level,
            },
            "delta": {
                "risk_score": round(float(new_risk.risk_score - base_risk.risk_score), 4),
                "risk_level_changed": new_risk.risk_level != base_risk.risk_level,
            },
        }

    def _compute_custom_risk_score(
        self,
        failure_probability: Any,
        anomaly_score: Any,
        rul_norm: Any,
        weights: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            f = float(failure_probability)
            a = float(anomaly_score)
            r = float(rul_norm)
        except (TypeError, ValueError) as e:
            return {"error": f"숫자형 변환 실패: {e}"}

        for name, val in (
            ("failure_probability", f),
            ("anomaly_score", a),
            ("rul_norm", r),
        ):
            if not (0.0 <= val <= 1.0):
                return {"error": f"{name} 는 [0,1] 범위여야 합니다: got {val}"}

        w = {"failure": 0.5, "anomaly": 0.3, "rul": 0.2}
        if isinstance(weights, dict):
            for k in ("failure", "anomaly", "rul"):
                if k in weights:
                    try:
                        w[k] = float(weights[k])
                    except (TypeError, ValueError):
                        continue

        from models_core import risk_score as rs

        score = float(rs.weighted_sum(f, a, r, weights=w))
        level = rs.to_risk_level(score)
        return {
            "inputs": {
                "failure_probability": f,
                "anomaly_score": a,
                "rul_norm": r,
            },
            "weights": w,
            "risk_score": round(score, 4),
            "risk_level": level,
        }

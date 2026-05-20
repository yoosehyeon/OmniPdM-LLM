from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from models_core import config


# 파일명 컴포넌트(prefix/dataset_key/asset_id) sanitize 용 allowlist.
# alphanumeric + underscore + hyphen + dot 만 통과. 나머지(슬래시, ..,
# NUL byte, Windows 금지 문자 :*?<>|" 등) 는 모두 _ 로 치환.
_SAFE_SEGMENT_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
# 한 segment 최대 길이. Windows MAX_PATH 260 자 제약 + reports_dir prefix 여유.
_MAX_SEGMENT_LEN = 64
from services.schemas import (
    EvaluationResult,
    ExplanationResult,
    LlmResult,
    LSTMExplainResult,
    LSTMValidationResult,
    PredictionResult,
    RiskResult,
    ValidationResult,
)


logger = logging.getLogger(__name__)


class ReportService:
    def generate(
        self,
        payload: dict,
        validation: ValidationResult,
        pred: PredictionResult,
        exp: ExplanationResult,
        risk: RiskResult,
        llm: LlmResult,
        evaluation: EvaluationResult,
    ) -> str:
        issue_lines = []
        for issue in validation.issues:
            issue_lines.append(f"- [{issue.level}] {issue.field}: {issue.message}")

        top_features_md = "\n".join(
            [
                f"- {f.feature}: value={f.value}, importance={f.importance}, direction={f.direction}"
                for f in exp.top_features
            ]
        )

        return f"""# HybridPdM Analysis Report

## 1. Input Payload
{payload}

## 2. Validation
{'정상' if validation.is_valid else '오류 있음'}

{chr(10).join(issue_lines) if issue_lines else '- 없음'}

## 3. Prediction
- dataset_key: {pred.dataset_key}
- task_type: {pred.task_type}
- model_name: {pred.model_name}
- model_mode: {pred.model_mode}
- predicted_label: {pred.predicted_label}
- failure_probability: {pred.failure_probability}
- anomaly_score: {pred.anomaly_score}
- rul_norm: {pred.rul_norm}

## 4. Risk
- risk_score: {risk.risk_score}
- risk_level: {risk.risk_level}
- method: {risk.method}
- weights: {risk.weights}

## 5. Explanation
- summary: {exp.summary}
- interaction_summary: {exp.interaction_summary}
- explanation_confidence: {exp.explanation_confidence}

### Top Features
{top_features_md}

## 6. LLM Recommendation
{llm.text}

## 7. Evaluation
- structure_ok: {evaluation.structure_ok}
- factuality_ok: {evaluation.factuality_ok}
- overall_score: {evaluation.overall_score}
- notes: {evaluation.notes}
"""

    def generate_lstm(
        self,
        asset_id: str,
        validation: LSTMValidationResult,
        pred: PredictionResult,
        exp: LSTMExplainResult,
        risk: RiskResult,
        llm: LlmResult,
        evaluation: EvaluationResult,
    ) -> str:
        issue_lines = []
        for issue in validation.issues:
            issue_lines.append(f"- [{issue.level}] {issue.field}: {issue.message}")

        top_features_md = "\n".join(
            [
                f"- {item['feature']}: importance={item['importance']}"
                for item in exp.top_features
            ]
        ) if exp.top_features else "- 없음"

        temporal_summary = exp.temporal_summary or {}

        return f"""# HybridPdM LSTM Analysis Report

## 1. Input Summary
- asset_id: {asset_id}
- dataset_key: {validation.dataset_key}
- timesteps: {validation.timesteps}
- feature_dim: {validation.feature_dim}
- feature_names: {validation.feature_names}

## 2. Validation
{'정상' if validation.is_valid else '오류 있음'}

{chr(10).join(issue_lines) if issue_lines else '- 없음'}

## 3. Prediction
- dataset_key: {pred.dataset_key}
- task_type: {pred.task_type}
- model_name: {pred.model_name}
- model_mode: {pred.model_mode}
- predicted_label: {pred.predicted_label}
- failure_probability: {pred.failure_probability}
- anomaly_score: {pred.anomaly_score}
- rul_norm: {pred.rul_norm}

## 4. Risk
- risk_score: {risk.risk_score}
- risk_level: {risk.risk_level}
- method: {risk.method}
- weights: {risk.weights}

## 5. Explanation
{exp.explanation_text}

### Top Features
{top_features_md}

### Temporal Summary
- most_recent_drivers: {temporal_summary.get("most_recent_drivers", [])}
- largest_variation_features: {temporal_summary.get("largest_variation_features", [])}
- timesteps: {temporal_summary.get("timesteps", "N/A")}
- feature_dim: {temporal_summary.get("feature_dim", "N/A")}

## 6. LLM Recommendation
{llm.text}

## 7. Evaluation
- structure_ok: {evaluation.structure_ok}
- factuality_ok: {evaluation.factuality_ok}
- overall_score: {evaluation.overall_score}
- notes: {evaluation.notes}
"""

    def save_markdown_report(
        self,
        report_markdown: str,
        dataset_key: str,
        asset_id: str = "UNKNOWN",
        prefix: str = "hybridpdm_report",
    ) -> str:
        """Markdown 보고서를 config.REPORT_DIR 아래에 저장하고 파일 경로를 반환한다."""
        save_path, _ = self._build_save_paths(prefix, dataset_key, asset_id)
        save_path.write_text(report_markdown, encoding="utf-8")
        return str(save_path)

    def save_report_bundle(
        self,
        report_markdown: str,
        raw_result: Optional[Dict[str, Any]],
        dataset_key: str,
        asset_id: str = "UNKNOWN",
        prefix: str = "hybridpdm_report",
    ) -> Tuple[str, Optional[str]]:
        """Markdown + raw JSON 을 같은 타임스탬프로 쌍 저장. (md_path, json_path) 반환."""
        md_path, json_path = self._build_save_paths(prefix, dataset_key, asset_id)
        md_path.write_text(report_markdown, encoding="utf-8")
        json_path_str: Optional[str] = None
        if raw_result is not None:
            try:
                json_path.write_text(
                    json.dumps(raw_result, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                json_path_str = str(json_path)
            except Exception as e:
                logger.warning(
                    "raw JSON bundle write failed (%s): %s: %s",
                    json_path.name, type(e).__name__, e,
                )
                json_path_str = None
        return str(md_path), json_path_str

    @staticmethod
    def _sanitize_segment(value: str, fallback: str) -> str:
        """파일명 컴포넌트를 allowlist 로 정제.

        - alphanumeric / _ / - / . 외 모든 문자는 _ 로 치환
          (슬래시, .., NUL byte, Windows 금지 문자 :*?<>|" 등 일괄 제거).
        - 시작/끝의 ./_ 제거 (.. 단독, 숨김파일화, dangling separator 방지).
        - .. 가 토큰 형태로 남는 경우 추가 치환 (이중 안전망).
        - 길이 제한으로 path 폭주 / FS 한계 회피.
        - 빈 결과는 fallback 사용.
        """
        cleaned = _SAFE_SEGMENT_RE.sub("_", str(value)).strip("._")
        if not cleaned:
            return fallback
        if ".." in cleaned:
            cleaned = cleaned.replace("..", "_")
        return cleaned[:_MAX_SEGMENT_LEN]

    @staticmethod
    def _build_save_paths(prefix: str, dataset_key: str, asset_id: str) -> Tuple[Path, Path]:
        reports_dir = Path(config.REPORT_DIR).resolve()
        reports_dir.mkdir(parents=True, exist_ok=True)
        safe_prefix  = ReportService._sanitize_segment(prefix,      "report")
        safe_dataset = ReportService._sanitize_segment(dataset_key, "unknown")
        safe_asset   = ReportService._sanitize_segment(asset_id,    "UNKNOWN")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = f"{safe_prefix}_{safe_dataset}_{safe_asset}_{timestamp}"
        md_path = reports_dir / f"{stem}.md"
        json_path = reports_dir / f"{stem}.json"
        # 이중 안전망: resolve() 후에도 reports_dir 자손이 아니면 외부 유출.
        # sanitize 가 정상 동작했다면 절대 발생하지 않지만, regex 변경 등 미래의
        # 회귀를 잡기 위해 명시적 가드.
        for p in (md_path, json_path):
            if not p.resolve().is_relative_to(reports_dir):
                raise ValueError(f"report path traversal blocked: {p}")
        return md_path, json_path
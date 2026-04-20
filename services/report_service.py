from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from models_core import config
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
    def _build_save_paths(prefix: str, dataset_key: str, asset_id: str) -> Tuple[Path, Path]:
        reports_dir = Path(config.REPORT_DIR)
        reports_dir.mkdir(parents=True, exist_ok=True)
        safe_dataset = str(dataset_key).replace("/", "_").replace("\\", "_").strip() or "unknown"
        safe_asset = str(asset_id).replace("/", "_").replace("\\", "_").strip() or "UNKNOWN"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = f"{prefix}_{safe_dataset}_{safe_asset}_{timestamp}"
        return reports_dir / f"{stem}.md", reports_dir / f"{stem}.json"
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from services.schemas import (
    EvaluationResult,
    ExplanationResult,
    LlmResult,
    PredictionResult,
    RiskResult,
    ValidationResult,
)


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

    def save_markdown_report(
        self,
        report_markdown: str,
        dataset_key: str,
        asset_id: str = "UNKNOWN",
        prefix: str = "hybridpdm_report",
    ) -> str:
        """
        Markdown 보고서를 artifacts/reports 아래에 저장하고 파일 경로를 반환한다.
        """
        reports_dir = Path("artifacts") / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        safe_dataset = str(dataset_key).replace("/", "_").replace("\\", "_").strip()or "unknown"
        safe_asset = str(asset_id).replace("/", "_").replace("\\", "_").strip() or "UNKNOWN"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{prefix}_{safe_dataset}_{safe_asset}_{timestamp}.md"
        save_path = reports_dir / filename
        
        save_path.write_text(report_markdown, encoding="utf-8")
        return str(save_path)
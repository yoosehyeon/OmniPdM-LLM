from __future__ import annotations

from typing import Dict, Generator, List, Optional

from services.evaluation_service import EvaluationService
from services.explain_service import ExplainService
from services.guardrail_service import GuardrailService
from services.input_validation_service import InputValidationService
from services.llm_service import LlmService
from services.pdm_service import PdmService
from services.plot_service import PlotService
from services.report_service import ReportService
from services.risk_service import RiskService
from services.schemas import (
    AnalysisResult,
    LSTMAnalysisResult,
    LSTMExplainResult,
    LSTMSequenceInput,
)


class AnalyzeService:
    """
    전체 파이프라인 오케스트레이션.

    기존 scalar 경로:
    1) 입력 검증
    2) 추론
    3) Risk 계산
    4) 설명 생성
    5) LLM 생성
    6) Guardrail 검증
    7) Evaluation
    8) Plot 생성
    9) Report 생성

    추가된 LSTM 경로:
    - run_lstm()
    - 기존 서비스들을 최대한 재사용
    """

    def __init__(
        self,
        mode: str = "lite",
        risk_method: str = "weighted",
        dataset_key: str = "ai4i_cnn",
    ) -> None:
        self.validation_service = InputValidationService()
        self.pdm_service = PdmService(mode=mode, dataset_key=dataset_key)
        self.explain_service = ExplainService()
        self.risk_service = RiskService(method=risk_method)
        self.llm_service = LlmService()
        self.guardrail_service = GuardrailService()
        self.evaluation_service = EvaluationService(llm_service=self.llm_service)
        self.plot_service = PlotService()
        self.report_service = ReportService()

        self.mode = mode
        self.risk_method = risk_method
        self.dataset_key = dataset_key

    # ------------------------------------------------------------------
    # 기존 scalar 입력 경로 (유지)
    # ------------------------------------------------------------------
    def run(self, payload: Dict[str, float]) -> dict:
        validation = self.validation_service.validate(payload)

        if not validation.is_valid:
            error_text = self._format_validation_errors(validation.issues)
            return {
                "validation_text": error_text,
                "summary_text": "입력 검증 실패",
                "explanation_text": "",
                "feature_plot": None,
                "sensor_plot": None,
                "report_markdown": f"# Validation Error\n\n{error_text}",
                "raw_result": None,
                "alert_text": "N/A",
            }

        normalized_payload = validation.normalized_payload

        pred = self.pdm_service.predict(normalized_payload)
        risk = self.risk_service.fuse(pred)
        pred.risk_score = risk.risk_score
        pred.risk_level = risk.risk_level

        exp = self.explain_service.explain(normalized_payload, pred)

        llm = self.llm_service.generate(
            payload=normalized_payload,
            pred=pred,
            exp=exp,
            risk=risk,
        )

        llm = self.guardrail_service.validate(
            llm_result=llm,
            pred=pred,
            exp=exp,
            risk=risk,
        )

        evaluation = self.evaluation_service.evaluate(llm, pred=pred, risk=risk)

        feature_plot = self.plot_service.build_feature_importance(exp)
        sensor_plot = self.plot_service.build_sensor_snapshot(normalized_payload)

        report_markdown = self.report_service.generate(
            payload=normalized_payload,
            validation=validation,
            pred=pred,
            exp=exp,
            risk=risk,
            llm=llm,
            evaluation=evaluation,
        )

        result = AnalysisResult(
            validation=validation,
            prediction=pred,
            explanation=exp,
            risk=risk,
            llm=llm,
            evaluation=evaluation,
            summary_text=self._build_summary_text(pred, risk),
            explanation_text=self._build_scalar_explanation_text(exp),
            report_markdown=report_markdown,
        )

        raw_result = result.to_dict()
        saved_md, _ = self._auto_save(
            report_markdown=report_markdown,
            raw_result=raw_result,
            dataset_key=self.dataset_key,
            asset_id="UNKNOWN",
            prefix="scalar_report",
        )

        return {
            "validation_text": self._compose_validation_text("정상", saved_md),
            "summary_text": result.summary_text,
            "explanation_text": result.explanation_text,
            "feature_plot": feature_plot,
            "sensor_plot": sensor_plot,
            "report_markdown": report_markdown,
            "raw_result": raw_result,
            "alert_text": "ALERT TRIGGERED"
            if str(risk.risk_level).upper() in {"CRITICAL", "WARNING"}
            else "NO ALERT",
            "saved_report_path": saved_md,
        }

    # ------------------------------------------------------------------
    # 스트리밍 scalar 입력 경로 (P3-①)
    # - LLM generate_stream() 의 chunk 를 실시간으로 report_markdown 에 누적
    # - 사전 단계 (pred/risk/exp/plot) 는 첫 yield 에 모두 포함
    # - 스트림 종료 후 guardrail/evaluation/report 적용한 최종 dict 를 마지막 yield
    # ------------------------------------------------------------------
    def run_stream(self, payload: Dict[str, float]) -> Generator[dict, None, None]:
        validation = self.validation_service.validate(payload)

        if not validation.is_valid:
            error_text = self._format_validation_errors(validation.issues)
            yield {
                "validation_text": error_text,
                "summary_text": "입력 검증 실패",
                "explanation_text": "",
                "feature_plot": None,
                "sensor_plot": None,
                "report_markdown": f"# Validation Error\n\n{error_text}",
                "raw_result": None,
                "alert_text": "N/A",
            }
            return

        normalized_payload = validation.normalized_payload
        pred = self.pdm_service.predict(normalized_payload)
        risk = self.risk_service.fuse(pred)
        pred.risk_score = risk.risk_score
        pred.risk_level = risk.risk_level

        exp = self.explain_service.explain(normalized_payload, pred)

        feature_plot = self.plot_service.build_feature_importance(exp)
        sensor_plot = self.plot_service.build_sensor_snapshot(normalized_payload)

        summary_text = self._build_summary_text(pred, risk)
        explanation_text = self._build_scalar_explanation_text(exp)
        alert_text = (
            "ALERT TRIGGERED"
            if str(risk.risk_level).upper() in {"CRITICAL", "WARNING"}
            else "NO ALERT"
        )

        # 첫 yield: LLM 생성 전까지 완성된 정보를 UI 에 즉시 표시
        yield {
            "validation_text": "정상",
            "summary_text": summary_text,
            "explanation_text": explanation_text,
            "feature_plot": feature_plot,
            "sensor_plot": sensor_plot,
            "report_markdown": "## LLM 분석 코멘트 (생성 중...)\n\n",
            "raw_result": None,
            "alert_text": alert_text,
        }

        # 토큰 스트리밍
        gen = self.llm_service.generate_stream(normalized_payload, pred, exp, risk)
        accumulated = ""
        llm_result = None
        try:
            while True:
                chunk = next(gen)
                accumulated += chunk
                yield {
                    "validation_text": "정상",
                    "summary_text": summary_text,
                    "explanation_text": explanation_text,
                    "feature_plot": feature_plot,
                    "sensor_plot": sensor_plot,
                    "report_markdown": (
                        "## LLM 분석 코멘트 (생성 중...)\n\n" + accumulated
                    ),
                    "raw_result": None,
                    "alert_text": alert_text,
                }
        except StopIteration as stop:
            llm_result = stop.value

        # 스트림 종료 후: guardrail → evaluation → 최종 report
        llm_result = self.guardrail_service.validate(
            llm_result=llm_result, pred=pred, exp=exp, risk=risk,
        )
        evaluation = self.evaluation_service.evaluate(llm_result, pred=pred, risk=risk)

        report_markdown = self.report_service.generate(
            payload=normalized_payload,
            validation=validation,
            pred=pred,
            exp=exp,
            risk=risk,
            llm=llm_result,
            evaluation=evaluation,
        )

        result = AnalysisResult(
            validation=validation,
            prediction=pred,
            explanation=exp,
            risk=risk,
            llm=llm_result,
            evaluation=evaluation,
            summary_text=summary_text,
            explanation_text=explanation_text,
            report_markdown=report_markdown,
        )

        raw_result = result.to_dict()
        saved_md, _ = self._auto_save(
            report_markdown=report_markdown,
            raw_result=raw_result,
            dataset_key=self.dataset_key,
            asset_id="UNKNOWN",
            prefix="scalar_report",
        )

        yield {
            "validation_text": self._compose_validation_text("정상", saved_md),
            "summary_text": summary_text,
            "explanation_text": explanation_text,
            "feature_plot": feature_plot,
            "sensor_plot": sensor_plot,
            "report_markdown": report_markdown,
            "raw_result": raw_result,
            "alert_text": alert_text,
            "saved_report_path": saved_md,
        }

    # ------------------------------------------------------------------
    # 새 LSTM sequence 입력 경로
    # ------------------------------------------------------------------
    def run_lstm(
        self,
        sequence: List[List[float]],
        feature_names: Optional[List[str]] = None,
        asset_id: str = "UNKNOWN",
        dataset_key: Optional[str] = None,
    ) -> dict:
        """
        LSTM sequence 입력 전용 실행 함수.

        입력:
            sequence: [timesteps][features]
            feature_names: optional
            asset_id: optional
            dataset_key: cmapss_lstm / ncmapss_lstm
                         미지정 시 self.dataset_key 사용

        반환:
            기존 run() 과 최대한 동일한 UI dict 구조
        """
        resolved_dataset_key = dataset_key or self.dataset_key

        lstm_payload = LSTMSequenceInput(
            sequence=sequence,
            feature_names=feature_names,
            asset_id=asset_id,
            dataset_key=resolved_dataset_key,
        )

        validation = self.validation_service.validate_lstm_sequence_payload(lstm_payload)

        if not validation.is_valid:
            error_text = self._format_validation_errors(validation.issues)
            return {
                "validation_text": error_text,
                "summary_text": "LSTM 입력 검증 실패",
                "explanation_text": "",
                "feature_plot": None,
                "sensor_plot": None,
                "report_markdown": f"# LSTM Validation Error\n\n{error_text}",
                "raw_result": None,
                "alert_text": "N/A",
            }

        pred = self.pdm_service.predict_lstm_sequence(
            sequence=validation.normalized_sequence,
            feature_names=validation.feature_names,
        )

        risk = self.risk_service.fuse(pred)
        pred.risk_score = risk.risk_score
        pred.risk_level = risk.risk_level

        explain_payload = self._build_lstm_explain_payload(
            sequence=validation.normalized_sequence,
            feature_names=validation.feature_names,
        )

        # 중요:
        # ExplainService.explain_lstm_sequence()의 실제 시그니처에 맞게 호출해야 한다.
        exp = self.explain_service.explain_lstm_sequence(
            sequence=validation.normalized_sequence,
            feature_names=validation.feature_names,
            top_contributors=pred.top_contributors,
            predicted_label=pred.predicted_label,
            failure_probability=pred.failure_probability,
            rul_norm=pred.rul_norm,
        )

        llm = self.llm_service.generate(
            payload=explain_payload,
            pred=pred,
            exp=exp,
            risk=risk,
        )

        llm = self.guardrail_service.validate(
            llm_result=llm,
            pred=pred,
            exp=exp,
            risk=risk,
        )

        evaluation = self.evaluation_service.evaluate(llm, pred=pred, risk=risk)

        feature_plot = self.plot_service.build_lstm_feature_importance(exp)
        sensor_plot = self.plot_service.build_lstm_sequence_snapshot(
            sequence=validation.normalized_sequence,
            feature_names=validation.feature_names,
        )

        report_markdown = self._generate_lstm_report_markdown(
            asset_id=asset_id,
            validation=validation,
            pred=pred,
            exp=exp,
            risk=risk,
            llm=llm,
            evaluation=evaluation,
        )

        result = LSTMAnalysisResult(
            validation=validation,
            prediction=pred,
            explanation=exp,
            risk=risk,
            llm=llm,
            evaluation=evaluation,
            summary_text=self._build_summary_text(pred, risk),
            explanation_text=exp.explanation_text,
            report_markdown=report_markdown,
        )

        raw_result = result.to_dict()
        saved_md, _ = self._auto_save(
            report_markdown=report_markdown,
            raw_result=raw_result,
            dataset_key=resolved_dataset_key,
            asset_id=asset_id,
            prefix="lstm_report",
        )

        base_validation_text = (
            f"정상\n"
            f"- dataset_key: {validation.dataset_key}\n"
            f"- timesteps: {validation.timesteps}\n"
            f"- feature_dim: {validation.feature_dim}\n"
            f"- asset_id: {asset_id}"
        )

        return {
            "validation_text": self._compose_validation_text(base_validation_text, saved_md),
            "summary_text": result.summary_text,
            "explanation_text": result.explanation_text,
            "feature_plot": feature_plot,
            "sensor_plot": sensor_plot,
            "report_markdown": report_markdown,
            "raw_result": raw_result,
            "alert_text": "ALERT TRIGGERED"
            if str(risk.risk_level).upper() in {"CRITICAL", "WARNING"}
            else "NO ALERT",
            "saved_report_path": saved_md,
        }

    # ------------------------------------------------------------------
    # 스트리밍 LSTM sequence 입력 경로 (P3-①)
    # ------------------------------------------------------------------
    def run_lstm_stream(
        self,
        sequence: List[List[float]],
        feature_names: Optional[List[str]] = None,
        asset_id: str = "UNKNOWN",
        dataset_key: Optional[str] = None,
    ) -> Generator[dict, None, None]:
        resolved_dataset_key = dataset_key or self.dataset_key

        lstm_payload = LSTMSequenceInput(
            sequence=sequence,
            feature_names=feature_names,
            asset_id=asset_id,
            dataset_key=resolved_dataset_key,
        )

        validation = self.validation_service.validate_lstm_sequence_payload(lstm_payload)

        if not validation.is_valid:
            error_text = self._format_validation_errors(validation.issues)
            yield {
                "validation_text": error_text,
                "summary_text": "LSTM 입력 검증 실패",
                "explanation_text": "",
                "feature_plot": None,
                "sensor_plot": None,
                "report_markdown": f"# LSTM Validation Error\n\n{error_text}",
                "raw_result": None,
                "alert_text": "N/A",
            }
            return

        pred = self.pdm_service.predict_lstm_sequence(
            sequence=validation.normalized_sequence,
            feature_names=validation.feature_names,
        )
        risk = self.risk_service.fuse(pred)
        pred.risk_score = risk.risk_score
        pred.risk_level = risk.risk_level

        explain_payload = self._build_lstm_explain_payload(
            sequence=validation.normalized_sequence,
            feature_names=validation.feature_names,
        )
        exp = self.explain_service.explain_lstm_sequence(
            sequence=validation.normalized_sequence,
            feature_names=validation.feature_names,
            top_contributors=pred.top_contributors,
            predicted_label=pred.predicted_label,
            failure_probability=pred.failure_probability,
            rul_norm=pred.rul_norm,
        )

        feature_plot = self.plot_service.build_lstm_feature_importance(exp)
        sensor_plot = self.plot_service.build_lstm_sequence_snapshot(
            sequence=validation.normalized_sequence,
            feature_names=validation.feature_names,
        )

        summary_text = self._build_summary_text(pred, risk)
        explanation_text = exp.explanation_text
        validation_text = (
            f"정상\n"
            f"- dataset_key: {validation.dataset_key}\n"
            f"- timesteps: {validation.timesteps}\n"
            f"- feature_dim: {validation.feature_dim}\n"
            f"- asset_id: {asset_id}"
        )
        alert_text = (
            "ALERT TRIGGERED"
            if str(risk.risk_level).upper() in {"CRITICAL", "WARNING"}
            else "NO ALERT"
        )

        yield {
            "validation_text": validation_text,
            "summary_text": summary_text,
            "explanation_text": explanation_text,
            "feature_plot": feature_plot,
            "sensor_plot": sensor_plot,
            "report_markdown": "## LLM 분석 코멘트 (생성 중...)\n\n",
            "raw_result": None,
            "alert_text": alert_text,
        }

        gen = self.llm_service.generate_stream(explain_payload, pred, exp, risk)
        accumulated = ""
        llm_result = None
        try:
            while True:
                chunk = next(gen)
                accumulated += chunk
                yield {
                    "validation_text": validation_text,
                    "summary_text": summary_text,
                    "explanation_text": explanation_text,
                    "feature_plot": feature_plot,
                    "sensor_plot": sensor_plot,
                    "report_markdown": (
                        "## LLM 분석 코멘트 (생성 중...)\n\n" + accumulated
                    ),
                    "raw_result": None,
                    "alert_text": alert_text,
                }
        except StopIteration as stop:
            llm_result = stop.value

        llm_result = self.guardrail_service.validate(
            llm_result=llm_result, pred=pred, exp=exp, risk=risk,
        )
        evaluation = self.evaluation_service.evaluate(llm_result, pred=pred, risk=risk)

        report_markdown = self._generate_lstm_report_markdown(
            asset_id=asset_id,
            validation=validation,
            pred=pred,
            exp=exp,
            risk=risk,
            llm=llm_result,
            evaluation=evaluation,
        )

        result = LSTMAnalysisResult(
            validation=validation,
            prediction=pred,
            explanation=exp,
            risk=risk,
            llm=llm_result,
            evaluation=evaluation,
            summary_text=summary_text,
            explanation_text=explanation_text,
            report_markdown=report_markdown,
        )

        raw_result = result.to_dict()
        saved_md, _ = self._auto_save(
            report_markdown=report_markdown,
            raw_result=raw_result,
            dataset_key=resolved_dataset_key,
            asset_id=asset_id,
            prefix="lstm_report",
        )

        yield {
            "validation_text": self._compose_validation_text(validation_text, saved_md),
            "summary_text": summary_text,
            "explanation_text": explanation_text,
            "feature_plot": feature_plot,
            "sensor_plot": sensor_plot,
            "report_markdown": report_markdown,
            "raw_result": raw_result,
            "alert_text": alert_text,
            "saved_report_path": saved_md,
        }

    # ------------------------------------------------------------------
    # Helper: auto-save + validation 텍스트 조합
    # ------------------------------------------------------------------
    def _auto_save(
        self,
        report_markdown: str,
        raw_result: Optional[Dict],
        dataset_key: str,
        asset_id: str,
        prefix: str,
    ) -> tuple[Optional[str], Optional[str]]:
        try:
            return self.report_service.save_report_bundle(
                report_markdown=report_markdown,
                raw_result=raw_result,
                dataset_key=dataset_key,
                asset_id=asset_id,
                prefix=prefix,
            )
        except Exception:
            return None, None

    @staticmethod
    def _compose_validation_text(base: str, saved_md: Optional[str]) -> str:
        if not saved_md:
            return base
        from pathlib import Path as _P
        return f"{base}\n- saved: {_P(saved_md).name}"

    # ------------------------------------------------------------------
    # Helper: 기존 summary
    # ------------------------------------------------------------------
    def _build_summary_text(self, pred, risk) -> str:
        return (
            f"Predicted Label: {pred.predicted_label}\n"
            f"Failure Probability: {pred.failure_probability}\n"
            f"Anomaly Score: {pred.anomaly_score}\n"
            f"RUL Norm: {pred.rul_norm}\n"
            f"Risk Score: {risk.risk_score}\n"
            f"Risk Level: {risk.risk_level}\n"
            f"Model: {pred.model_name} ({pred.model_mode})"
        )

    # ------------------------------------------------------------------
    # Helper: 기존 explanation text
    # ------------------------------------------------------------------
    def _build_scalar_explanation_text(self, exp) -> str:
        lines = [
            f"Summary: {exp.summary}",
            f"Interaction: {exp.interaction_summary}",
            f"Confidence: {exp.explanation_confidence}",
            "",
            "Top Features:",
        ]
        for idx, feat in enumerate(exp.top_features, start=1):
            lines.append(
                f"{idx}. {feat.feature} | value={feat.value} | importance={feat.importance} | direction={feat.direction}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helper: validation issues formatting
    # ------------------------------------------------------------------
    def _format_validation_errors(self, issues) -> str:
        return "\n".join(
            [f"- [{issue.level}] {issue.field}: {issue.message}" for issue in issues]
        )

    # ------------------------------------------------------------------
    # Helper: LSTM explain payload
    # ------------------------------------------------------------------
    def _build_lstm_explain_payload(
        self,
        sequence: List[List[float]],
        feature_names: Optional[List[str]],
    ) -> Dict[str, float]:
        """
        ExplainService / LlmService 에 넘길 payload를 생성한다.

        현재 ExplainService는 dict[str, float] 형태 payload에 익숙하므로
        마지막 timestep snapshot을 feature dict로 변환한다.
        """
        if not sequence:
            return {}

        latest = sequence[-1]

        if feature_names is None or len(feature_names) != len(latest):
            feature_names = [f"f{i}" for i in range(len(latest))]

        return {str(name): float(value) for name, value in zip(feature_names, latest)}

    # ------------------------------------------------------------------
    # Helper: LSTM report markdown
    # ------------------------------------------------------------------
    def _generate_lstm_report_markdown(
        self,
        asset_id: str,
        validation,
        pred,
        exp,
        risk,
        llm,
        evaluation,
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
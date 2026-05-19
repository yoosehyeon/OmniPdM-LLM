"""
Smoke tests — checkpoint-free / API-key-free.

Coverage:
  1. Dash app boot           (multi-page 6개 등록 확인)
  2. ToolDispatcher.dispatch (5 tools + unknown tool)
  3. AnalyzeService.run      (lite mode scalar: 정상 / 검증 오류)
  4. LlmService.generate     (GROQ_API_KEY 없음 → fallback)

실행:
    pytest tests/test_smoke.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import pytest

# 프로젝트 루트를 sys.path 맨 앞에 추가
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 외부 서비스 키 제거 — fallback / lite 경로만 통과시킴
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("ENABLE_LLM_TOOLS", None)
os.environ.pop("ENABLE_LLM_STREAM", None)


# ===========================================================================
# 1) Dash app boot
# ===========================================================================

class TestDashAppBoot:
    """app.py 가 정상 부팅하고 6개 페이지가 등록되는지 확인."""

    def test_app_imports(self):
        import app  # noqa: F401
        assert hasattr(app, "app")
        assert hasattr(app, "server")  # WSGI entry

    def test_six_pages_registered(self):
        import app  # noqa: F401
        import dash
        pages = list(dash.page_registry.keys())
        assert len(pages) == 6, f"expected 6 pages, got {len(pages)}: {pages}"

    def test_expected_page_paths(self):
        import app  # noqa: F401
        import dash
        paths = {p["path"] for p in dash.page_registry.values()}
        # 핵심 경로 4개 확인 (Analysis '/', Status, Reports 는 구현됨)
        assert "/" in paths
        assert "/status" in paths
        assert "/reports" in paths
        assert "/lstm" in paths


# ===========================================================================
# 2) ToolDispatcher — 5 tools + unknown
# ===========================================================================

class TestToolDispatcher:
    """ToolDispatcher.dispatch 단독 smoke: 5 tools + unknown."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from services.llm_tools import ToolDispatcher
        from services.pdm_service import PdmService
        from services.risk_service import RiskService

        pdm = PdmService(mode="lite", dataset_key="ai4i_cnn")
        risk = RiskService(method="weighted")

        self._base_payload = {
            "air_temperature_k": 298.5,
            "process_temperature_k": 309.1,
            "rotational_speed_rpm": 1510.0,
            "torque_nm": 36.0,
            "tool_wear_min": 42.0,
        }

        self.dispatcher = ToolDispatcher(
            dataset_key="ai4i_cnn",
            pdm_service=pdm,
            risk_service=risk,
            base_payload=self._base_payload,
        )

    # ── get_feature_schema ──────────────────────────────────────────────

    def test_get_feature_schema_ai4i(self):
        result = self.dispatcher.dispatch("get_feature_schema", {"dataset_key": "ai4i_cnn"})
        assert "error" not in result
        assert result["input_mode"] == "scalar"
        assert isinstance(result["fields"], list)
        assert len(result["fields"]) == 5
        field_names = [f["name"] for f in result["fields"]]
        assert "tool_wear_min" in field_names

    def test_get_feature_schema_cmapss(self):
        result = self.dispatcher.dispatch("get_feature_schema", {"dataset_key": "cmapss_lstm"})
        assert "error" not in result
        assert result["input_mode"] == "sequence"
        assert "timesteps_range" in result

    def test_get_feature_schema_hydraulic(self):
        result = self.dispatcher.dispatch("get_feature_schema", {"dataset_key": "hydraulic_ae"})
        assert "error" not in result
        assert result["input_mode"] == "scalar"
        # hydraulic 스키마에 실제 센서 목록이 포함돼야 한다
        assert "fields" in result
        sensor_names = [f["name"] for f in result["fields"]]
        assert "PS1" in sensor_names
        assert len(sensor_names) == 17

    # ── get_risk_threshold_info ────────────────────────────────────────

    def test_get_risk_threshold_info_all(self):
        result = self.dispatcher.dispatch("get_risk_threshold_info", {})
        assert "error" not in result
        assert "all_thresholds" in result
        levels = [t["level"] for t in result["all_thresholds"]]
        assert "Critical" in levels
        assert "Normal" in levels

    def test_get_risk_threshold_info_with_level(self):
        result = self.dispatcher.dispatch(
            "get_risk_threshold_info", {"risk_level": "Warning"}
        )
        assert "error" not in result
        assert "matched" in result
        assert len(result["matched"]) > 0
        assert result["matched"][0]["level"].lower() == "warning"

    # ── get_recent_analysis_history ────────────────────────────────────

    def test_get_recent_analysis_history_returns_structure(self):
        result = self.dispatcher.dispatch(
            "get_recent_analysis_history", {"limit": 3}
        )
        assert "error" not in result
        assert "items" in result
        assert "count" in result
        assert isinstance(result["items"], list)
        assert result["count"] == len(result["items"])

    # ── perturb_input_and_predict ──────────────────────────────────────

    def test_perturb_input_and_predict_valid(self):
        result = self.dispatcher.dispatch(
            "perturb_input_and_predict",
            {"perturbations": {"tool_wear_min": 200.0, "torque_nm": 65.0}},
        )
        assert "error" not in result, f"unexpected error: {result.get('error')}"
        assert "base" in result
        assert "perturbed" in result
        assert "delta" in result
        assert "risk_level_changed" in result["delta"]

    def test_perturb_input_and_predict_no_base_payload(self):
        from services.llm_tools import ToolDispatcher
        from services.pdm_service import PdmService
        from services.risk_service import RiskService

        dispatcher_no_base = ToolDispatcher(
            dataset_key="cmapss_lstm",
            pdm_service=PdmService(mode="lite", dataset_key="ai4i_cnn"),
            risk_service=RiskService(),
            base_payload=None,
        )
        result = dispatcher_no_base.dispatch(
            "perturb_input_and_predict", {"perturbations": {"tool_wear_min": 200.0}}
        )
        assert "error" in result

    # ── compute_custom_risk_score ──────────────────────────────────────

    def test_compute_custom_risk_score_default_weights(self):
        result = self.dispatcher.dispatch(
            "compute_custom_risk_score",
            {"failure_probability": 0.8, "anomaly_score": 0.5, "rul_norm": 0.2},
        )
        assert "error" not in result
        assert 0.0 <= result["risk_score"] <= 1.0
        assert result["risk_level"] in {"Critical", "Warning", "Advisory", "Normal"}

    def test_compute_custom_risk_score_custom_weights(self):
        result = self.dispatcher.dispatch(
            "compute_custom_risk_score",
            {
                "failure_probability": 0.3,
                "anomaly_score": 0.1,
                "rul_norm": 0.9,
                "weights": {"failure": 0.6, "anomaly": 0.2, "rul": 0.2},
            },
        )
        assert "error" not in result
        assert "weights" in result

    def test_compute_custom_risk_score_out_of_range(self):
        result = self.dispatcher.dispatch(
            "compute_custom_risk_score",
            {"failure_probability": 1.5, "anomaly_score": 0.5, "rul_norm": 0.5},
        )
        assert "error" in result

    # ── unknown tool ───────────────────────────────────────────────────

    def test_dispatch_unknown_tool(self):
        result = self.dispatcher.dispatch("nonexistent_tool", {})
        assert "error" in result
        assert "unknown" in result["error"].lower()


# ===========================================================================
# 3) AnalyzeService — lite mode scalar
# ===========================================================================

_NORMAL_PAYLOAD = {
    "air_temperature_k": 298.5,
    "process_temperature_k": 309.1,
    "rotational_speed_rpm": 1510.0,
    "torque_nm": 36.0,
    "tool_wear_min": 42.0,
}

_INVALID_PAYLOAD = {
    "air_temperature_k": 9999.0,  # 허용 범위 초과
    "process_temperature_k": 309.1,
    "rotational_speed_rpm": 1510.0,
    "torque_nm": 36.0,
    "tool_wear_min": 42.0,
}


class TestAnalyzeServiceLite:
    """AnalyzeService.run() — mode='lite', 체크포인트/API 키 불필요."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from services.analyze_service import AnalyzeService
        self.service = AnalyzeService(mode="lite", risk_method="weighted", dataset_key="ai4i_cnn")

    def test_run_normal_returns_all_keys(self):
        result = self.service.run(_NORMAL_PAYLOAD)
        for key in (
            "validation_text", "summary_text", "explanation_text",
            "feature_plot", "sensor_plot", "report_markdown",
            "raw_result", "alert_text",
        ):
            assert key in result, f"missing key: {key}"

    def test_run_normal_summary_contains_predicted_label(self):
        result = self.service.run(_NORMAL_PAYLOAD)
        assert "Predicted Label" in result["summary_text"]

    def test_run_normal_report_contains_sections(self):
        result = self.service.run(_NORMAL_PAYLOAD)
        report = result["report_markdown"]
        assert "상태 요약" in report or "Prediction" in report

    def test_run_validation_error_returns_error_text(self):
        result = self.service.run(_INVALID_PAYLOAD)
        # 검증 실패 시 summary_text가 오류 메시지를 포함해야 한다
        assert "실패" in result["summary_text"] or "Error" in result["report_markdown"]

    def test_run_alert_text_normal(self):
        result = self.service.run(_NORMAL_PAYLOAD)
        assert result["alert_text"] in {"ALERT TRIGGERED", "NO ALERT"}

    def test_run_stream_yields_multiple_partials(self):
        """run_stream()이 최소 2회 yield 하는지 확인 (첫 partial + 최종)."""
        from services.analyze_service import AnalyzeService
        svc = AnalyzeService(mode="lite", risk_method="weighted", dataset_key="ai4i_cnn")
        partials = list(svc.run_stream(_NORMAL_PAYLOAD))
        assert len(partials) >= 2
        # 마지막 partial에는 raw_result가 있어야 함
        assert partials[-1].get("raw_result") is not None


# ===========================================================================
# 4) LlmService — fallback (no API key)
# ===========================================================================

class TestLlmServiceFallback:
    """GROQ_API_KEY 없을 때 LlmResult(source='fallback', used_fallback=True)를 반환하는지 확인."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from services.llm_service import LlmService
        from services.pdm_service import PdmService
        from services.risk_service import RiskService
        from services.explain_service import ExplainService

        pdm = PdmService(mode="lite", dataset_key="ai4i_cnn")
        risk_svc = RiskService()
        pred = pdm.predict(_NORMAL_PAYLOAD)
        risk = risk_svc.fuse(pred)
        pred.risk_score = risk.risk_score
        pred.risk_level = risk.risk_level
        exp = ExplainService().explain(_NORMAL_PAYLOAD, pred)

        self.svc = LlmService()
        self.pred = pred
        self.risk = risk
        self.exp = exp

    def test_generate_returns_fallback_result(self):
        result = self.svc.generate(
            payload=_NORMAL_PAYLOAD,
            pred=self.pred,
            exp=self.exp,
            risk=self.risk,
        )
        assert result.used_fallback is True
        assert result.source == "fallback"

    def test_generate_fallback_text_has_three_sections(self):
        result = self.svc.generate(
            payload=_NORMAL_PAYLOAD,
            pred=self.pred,
            exp=self.exp,
            risk=self.risk,
        )
        text = result.text
        assert "상태 요약" in text
        assert "의심 원인" in text
        assert "권장 조치" in text

    def test_generate_stream_yields_fallback_chunk(self):
        gen = self.svc.generate_stream(_NORMAL_PAYLOAD, self.pred, self.exp, self.risk)
        chunks = []
        llm_result = None
        try:
            while True:
                chunks.append(next(gen))
        except StopIteration as stop:
            llm_result = stop.value

        assert len(chunks) > 0
        assert llm_result is not None
        assert llm_result.used_fallback is True

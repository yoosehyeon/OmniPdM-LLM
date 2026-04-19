from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Generator

from openai import OpenAI

from services.schemas import ExplanationResult, LSTMExplainResult, LlmResult, PredictionResult, RiskResult


_DEFAULT_SYSTEM_PROMPT = (
    "당신은 산업설비 예지보전 분석 보조 AI입니다. "
    "입력된 정보만 기반으로 한국어로 답하십시오. "
    "입력에 없는 수치, 장비, 원인을 추정해서 쓰지 마십시오. "
    "반드시 '상태 요약', '의심 원인', '권장 조치' 3개 섹션을 사용하십시오."
)


_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LlmService:
    """
    Groq Llama 실연동 (OpenAI 호환 endpoint) + fallback.
    출력 계약은 반드시 3섹션 구조를 유지한다.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.temperature = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
        self.max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "800"))
        self.base_url = os.getenv("GROQ_BASE_URL", _GROQ_BASE_URL)
        self.prompt_dir = Path(os.getenv("PROMPT_DIR", "prompts"))
        self.logs_path = Path("logs")
        self.logs_path.mkdir(parents=True, exist_ok=True)

        self.system_prompt = self._load_system_prompt()
        self.few_shot_messages = self._build_few_shot_messages()

        self.client = (
            OpenAI(api_key=self.api_key, base_url=self.base_url)
            if self.api_key
            else None
        )

    def _load_system_prompt(self) -> str:
        try:
            path = self.prompt_dir / "system_pdm_assistant.txt"
            if path.exists():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text
        except Exception:
            pass
        return _DEFAULT_SYSTEM_PROMPT

    def _build_few_shot_messages(self) -> list[dict]:
        normal_user = json.dumps({
            "sensor_payload": {"torque": 40.5, "tool_wear": 55, "rotational_speed": 1500},
            "prediction": {"predicted_label": "Normal", "failure_probability": 0.05, "risk_level": "Normal"},
            "explanation": {"summary": "주요 센서가 정상 범위 내", "top_features": []},
        }, ensure_ascii=False)
        normal_assistant = (
            "상태 요약: 현재 설비 상태는 Normal 수준이며 주요 센서는 정상 범위 내에 있습니다.\n\n"
            "의심 원인: 비정상 징후가 관찰되지 않으며 예측 확률도 낮은 수준입니다.\n\n"
            "권장 조치:\n"
            "1) 현재 운전 조건을 유지합니다.\n"
            "2) 다음 점검 주기에 주요 센서 값을 재확인합니다.\n"
            "3) 이력 데이터를 누적해 장기 추세를 모니터링합니다."
        )

        warning_user = json.dumps({
            "sensor_payload": {"torque": 58.2, "tool_wear": 180, "rotational_speed": 1380},
            "prediction": {"predicted_label": "Warning", "failure_probability": 0.42, "risk_level": "Warning"},
            "explanation": {"summary": "tool_wear와 torque 기여도 상승", "top_features": []},
        }, ensure_ascii=False)
        warning_assistant = (
            "상태 요약: 현재 위험 수준은 Warning이며 일부 센서 지표가 정상 범위를 벗어날 가능성이 있습니다.\n\n"
            "의심 원인: tool_wear와 torque의 기여도가 상승하여 마모와 부하 증가 추세를 지켜볼 필요가 있습니다.\n\n"
            "권장 조치:\n"
            "1) 다음 운전 전 tool_wear와 torque 추이를 재확인합니다.\n"
            "2) 운전 조건 변화 여부를 점검합니다.\n"
            "3) 점검 주기 단축을 검토합니다."
        )

        critical_user = json.dumps({
            "sensor_payload": {"torque": 72.1, "tool_wear": 245, "rotational_speed": 1290},
            "prediction": {"predicted_label": "Failure", "failure_probability": 0.87, "risk_level": "Critical"},
            "explanation": {"summary": "tool_wear 임계 근접, torque 급등", "top_features": []},
        }, ensure_ascii=False)
        critical_assistant = (
            "상태 요약: 현재 위험 수준은 Critical이며 예측 결과는 Failure입니다. 주요 지표가 임계치에 근접해 있습니다.\n\n"
            "의심 원인: tool_wear가 임계 근처이며 torque 급등이 동반되어 마모 및 부하 이상 가능성이 큽니다.\n\n"
            "권장 조치:\n"
            "1) 우선 점검 권고에 따라 현장 점검을 수행합니다.\n"
            "2) 토크 및 마모 관련 부품 상태를 확인합니다.\n"
            "3) 정비 이력과 반복 패턴을 기준으로 재발 방지 조치를 수립합니다."
        )

        return [
            {"role": "user", "content": normal_user},
            {"role": "assistant", "content": normal_assistant},
            {"role": "user", "content": warning_user},
            {"role": "assistant", "content": warning_assistant},
            {"role": "user", "content": critical_user},
            {"role": "assistant", "content": critical_assistant},
        ]

    def _get_temperature_for_risk(self, risk_level: str) -> float:
        level = (risk_level or "").strip().lower()
        mapping = {
            "critical": 0.1,
            "warning": 0.2,
            "advisory": 0.25,
            "normal": 0.3,
        }
        return mapping.get(level, self.temperature)

    def _get_domain_context(self, dataset_key: str) -> str:
        key = (dataset_key or "").strip().lower()
        if key.startswith("ncmapss"):
            return (
                "NASA N-CMAPSS 터보팬 엔진 데이터셋. C-MAPSS 대비 비행 조건 변동이 크고 "
                "실측 특성이 반영되어 있으며 RUL 예측이 주 과제. 다채널 시계열 센서(운전 조건·성능·건전성 관련)가 포함됨."
            )
        if key.startswith("cmapss"):
            return (
                "NASA C-MAPSS 터보팬 엔진 시뮬레이션 데이터셋. 다수의 센서 및 운전 조건 신호가 시계열로 포함되며 "
                "RUL(잔여수명) 예측 및 degradation 추적이 주 과제."
            )
        if key.startswith("ai4i"):
            return (
                "AI4I 2020 밀링머신 데이터셋. 주요 센서: air_temperature, process_temperature, "
                "rotational_speed, torque, tool_wear. 고장 유형: TWF(공구 마모)/HDF(열 발산)/PWF(전력)/OSF(과부하)/RNF(임의)."
            )
        return ""

    def judge(
        self,
        llm_result: LlmResult,
        pred: PredictionResult,
        risk: RiskResult,
    ) -> tuple[bool, str]:
        """
        LLM-as-Judge. 옵트인(env ENABLE_LLM_JUDGE=1)일 때만 호출되도록 외부에서 제어.
        실패/미설정 시 (True, "") 반환하여 기존 동작을 유지한다.
        """
        if not self.client:
            return True, "judge_skipped_no_client"

        judge_prompt = json.dumps({
            "task": "evaluate_pdm_output",
            "prediction": {
                "predicted_label": pred.predicted_label,
                "failure_probability": pred.failure_probability,
                "risk_level": risk.risk_level,
            },
            "llm_output": llm_result.text,
            "evaluation_criteria": [
                "입력에 없는 수치/장비/원인을 추가했는지 여부",
                "risk_level에 맞는 톤으로 작성됐는지 여부",
                "3섹션(상태 요약/의심 원인/권장 조치) 구조 준수 여부",
            ],
            "output_format": {
                "is_factual": "true | false",
                "notes": "간단한 한국어 설명",
            },
        }, ensure_ascii=False)

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 산업설비 예지보전 출력 품질 검증자입니다. "
                            "주어진 LLM 출력이 근거 범위를 벗어났거나 톤이 부적절한지 판단하십시오. "
                            "결과는 반드시 JSON 객체로 반환하십시오: "
                            '{"is_factual": bool, "notes": string}'
                        ),
                    },
                    {"role": "user", "content": judge_prompt},
                ],
            )
            raw = completion.choices[0].message.content.strip()
            data = json.loads(raw)
            is_factual = bool(data.get("is_factual", True))
            notes = str(data.get("notes", "") or "")
            return is_factual, notes
        except Exception as e:
            return True, f"judge_error:{type(e).__name__}"

    def generate(
        self,
        payload: Dict[str, float],
        pred: PredictionResult,
        exp: "ExplanationResult | LSTMExplainResult",
        risk: RiskResult,
    ) -> LlmResult:
        prompt = self._build_prompt(payload, pred, exp, risk)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        if not self.client:
            result = LlmResult(
                text=self._generate_fallback(pred, exp, risk),
                source="fallback",
                used_fallback=True,
                prompt_hash=prompt_hash,
            )
            self._log(prompt_hash, 0.0, "fallback_no_key")
            return result

        started = time.perf_counter()
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.few_shot_messages)
            messages.append({"role": "user", "content": prompt})

            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self._get_temperature_for_risk(risk.risk_level),
                max_tokens=self.max_tokens,
                messages=messages,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            text = completion.choices[0].message.content.strip()

            result = LlmResult(
                text=text,
                source="groq",
                used_fallback=False,
                prompt_hash=prompt_hash,
                latency_ms=round(latency_ms, 2),
            )
            self._log(prompt_hash, latency_ms, "ok")
            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._log(prompt_hash, latency_ms, f"error:{type(e).__name__}")
            return LlmResult(
                text=self._generate_fallback(pred, exp, risk),
                source="fallback",
                used_fallback=True,
                prompt_hash=prompt_hash,
                latency_ms=round(latency_ms, 2),
            )

    def generate_stream(
        self,
        payload: Dict[str, float],
        pred: PredictionResult,
        exp: "ExplanationResult | LSTMExplainResult",
        risk: RiskResult,
    ) -> Generator[str, None, LlmResult]:
        """
        토큰 단위 스트리밍 generator.

        - yield: OpenAI delta content 조각 (fallback 경로는 단일 chunk)
        - return (StopIteration.value): 누적 텍스트로 구성된 LlmResult

        호출 예:
            gen = svc.generate_stream(...)
            try:
                while True:
                    chunk = next(gen)
                    ...
            except StopIteration as stop:
                result: LlmResult = stop.value
        """
        prompt = self._build_prompt(payload, pred, exp, risk)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        if not self.client:
            fallback_text = self._generate_fallback(pred, exp, risk)
            self._log(prompt_hash, 0.0, "fallback_no_key_stream")
            yield fallback_text
            return LlmResult(
                text=fallback_text,
                source="fallback",
                used_fallback=True,
                prompt_hash=prompt_hash,
            )

        started = time.perf_counter()
        accumulated = ""
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.few_shot_messages)
            messages.append({"role": "user", "content": prompt})

            stream = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self._get_temperature_for_risk(risk.risk_level),
                max_tokens=self.max_tokens,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    accumulated += delta
                    yield delta

            latency_ms = (time.perf_counter() - started) * 1000.0
            self._log(prompt_hash, latency_ms, "ok_stream")
            return LlmResult(
                text=accumulated.strip(),
                source="groq",
                used_fallback=False,
                prompt_hash=prompt_hash,
                latency_ms=round(latency_ms, 2),
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._log(prompt_hash, latency_ms, f"error_stream:{type(e).__name__}")
            # 이미 일부 chunk 가 소비됐을 수 있으므로 fallback 은 별도 전체 텍스트로 재시작
            fallback_text = self._generate_fallback(pred, exp, risk)
            yield fallback_text
            return LlmResult(
                text=fallback_text,
                source="fallback",
                used_fallback=True,
                prompt_hash=prompt_hash,
                latency_ms=round(latency_ms, 2),
            )

    def _build_prompt(
        self,
        payload: Dict[str, float],
        pred: PredictionResult,
        exp: "ExplanationResult | LSTMExplainResult",
        risk: RiskResult,
    ) -> str:
        # ExplanationResult(FeatureExplanation 객체 리스트) / LSTMExplainResult(dict 리스트) 모두 처리
        top_features = []
        for f in exp.top_features:
            if isinstance(f, dict):
                top_features.append({
                    "feature": f.get("feature", ""),
                    "value": f.get("value", None),
                    "importance": f.get("importance", 0.0),
                    "direction": f.get("direction", "neutral"),
                })
            else:
                top_features.append({
                    "feature": f.feature,
                    "value": f.value,
                    "importance": f.importance,
                    "direction": f.direction,
                })

        # ExplanationResult: summary/interaction_summary/confidence 필드 보유
        # LSTMExplainResult: explanation_text 필드 보유
        is_lstm = isinstance(exp, LSTMExplainResult)
        summary = exp.explanation_text if is_lstm else exp.summary
        interaction_summary = None if is_lstm else exp.interaction_summary
        explanation_confidence = None if is_lstm else exp.explanation_confidence

        domain_context = self._get_domain_context(pred.dataset_key)

        content = {
            "sensor_payload": payload,
            "prediction": {
                "dataset_key": pred.dataset_key,
                "task_type": pred.task_type,
                "predicted_label": pred.predicted_label,
                "failure_probability": pred.failure_probability,
                "anomaly_score": pred.anomaly_score,
                "rul_norm": pred.rul_norm,
                "risk_score": risk.risk_score,
                "risk_level": risk.risk_level,
            },
            "explanation": {
                "summary": summary,
                "interaction_summary": interaction_summary,
                "explanation_confidence": explanation_confidence,
                "top_features": top_features,
            },
            "domain_context": domain_context,
            "output_format": {
                "required_sections": ["상태 요약", "의심 원인", "권장 조치"],
                "language": "ko",
            },
        }
        return json.dumps(content, ensure_ascii=False, indent=2)

    def _generate_fallback(
        self,
        pred: PredictionResult,
        exp: "ExplanationResult | LSTMExplainResult",
        risk: RiskResult,
    ) -> str:
        actions = self._build_actions(risk.risk_level)

        # LSTMExplainResult는 summary 대신 explanation_text 사용
        if isinstance(exp, LSTMExplainResult):
            cause_text = exp.explanation_text or "설명 정보 없음"
        else:
            cause_text = f"{exp.summary} {exp.interaction_summary or ''}".strip()

        return (
            f"상태 요약: 현재 설비 상태는 {risk.risk_level} 수준으로 평가되며, "
            f"고장 위험 예측 결과는 {pred.predicted_label}입니다.\n\n"
            f"의심 원인: {cause_text}\n\n"
            f"권장 조치:\n"
            f"1) {actions[0]}\n"
            f"2) {actions[1]}\n"
            f"3) {actions[2]}"
        )

    def _build_actions(self, risk_level: str) -> list[str]:
        if risk_level in {"Critical", "CRITICAL"}:
            return [
                "즉시 현장 점검을 수행하고 현재 운전 조건을 확인합니다.",
                "토크 및 마모 관련 부품을 우선 점검합니다.",
                "정비 이력과 반복 패턴을 기준으로 재발 방지 조치를 수립합니다.",
            ]
        if risk_level in {"Warning", "WARNING"}:
            return [
                "다음 운전 전 센서 상태와 운전 조건 변화를 확인합니다.",
                "토크와 마모 수치의 변동 추이를 재점검합니다.",
                "점검 주기 조정을 검토합니다.",
            ]
        return [
            "현재 상태를 유지하되 기준값 이탈 여부를 모니터링합니다.",
            "다음 점검 주기에 주요 센서를 재확인합니다.",
            "이력 데이터를 누적해 위험 상승 패턴을 추적합니다.",
        ]

    def _log(self, prompt_hash: str, latency_ms: float, status: str) -> None:
        try:
            line = {
                "prompt_hash": prompt_hash,
                "latency_ms": round(latency_ms, 2),
                "status": status,
            }
            with open(self.logs_path / "llm_calls.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception:
            # 로깅 실패가 전체 파이프라인을 멈춰서는 안 된다
            pass
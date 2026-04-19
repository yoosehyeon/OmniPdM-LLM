"""
HybridPdM - Gradio 단독 UI 엔트리포인트

역할:
- 사용자 입력 수집
- AnalyzeService 호출
- 결과 시각화
- Risk Simulator 제공
- Model Status 표시
- Report / Raw JSON 확인

실행:
    python gradio_app.py

선택:
    set GRADIO_SHARE=true
    set OPENAI_API_KEY=sk-...
    python gradio_app.py
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any, Dict, Generator, List, Tuple

import gradio as gr

# 서비스 계층
from services.analyze_service import AnalyzeService

# 기존 models_core 자산 재사용
from models_core import config
from models_core import risk_score as rs


# ---------------------------------------------------------------------
# 샘플 케이스
# PRD/README에 명시된 3종 고정값을 사용
# ---------------------------------------------------------------------
SAMPLE_CASES: Dict[str, Dict[str, float]] = {
    "Normal": {
        "air_temperature_k": 298.5,
        "process_temperature_k": 309.1,
        "rotational_speed_rpm": 1510.0,
        "torque_nm": 36.0,
        "tool_wear_min": 42.0,
    },
    "Warning": {
        "air_temperature_k": 303.2,
        "process_temperature_k": 316.5,
        "rotational_speed_rpm": 1320.0,
        "torque_nm": 54.0,
        "tool_wear_min": 122.0,
    },
    "High Risk": {
        "air_temperature_k": 307.8,
        "process_temperature_k": 321.4,
        "rotational_speed_rpm": 1180.0,
        "torque_nm": 68.0,
        "tool_wear_min": 210.0,
    },
}


# ---------------------------------------------------------------------
# 헬퍼 함수
# ---------------------------------------------------------------------
def load_sample(case_name: str) -> Tuple[float, float, float, float, float]:
    """
    드롭다운에서 선택한 샘플 케이스 값을 입력칸에 채운다.
    """
    item = SAMPLE_CASES[case_name]
    return (
        item["air_temperature_k"],
        item["process_temperature_k"],
        item["rotational_speed_rpm"],
        item["torque_nm"],
        item["tool_wear_min"],
    )


def _safe_json(data: Any) -> str:
    """
    raw_result를 Gradio Code 컴포넌트에 보기 좋게 표시하기 위한 JSON 변환.
    """
    try:
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return "{}"


def _extract_report_text(report_markdown: str) -> str:
    """
    Markdown 컴포넌트와 별도로 텍스트 박스에도 보고서를 보여주고 싶을 때 사용 가능.
    현재는 Markdown만 사용하므로 보조 함수로만 둔다.
    """
    return report_markdown if isinstance(report_markdown, str) else ""


def list_saved_reports(limit: int = 50) -> List[List[str]]:
    """config.REPORT_DIR 의 `*_report_*.md` 목록 (최신순)."""
    report_dir = config.REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(
        report_dir.glob("*_report_*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]

    from datetime import datetime as _dt
    rows: List[List[str]] = []
    for path in md_files:
        stat = path.stat()
        mtime_str = _dt.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        rows.append([path.name, mtime_str, f"{stat.st_size:,}"])
    return rows


def load_report_preview(evt: gr.SelectData) -> Tuple[str, str, Any]:
    """
    Report tab Dataframe 행 클릭 시: markdown + raw json + 다운로드 파일 반환.
    """
    try:
        row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        rows = list_saved_reports()
        if not rows or row_idx is None or row_idx >= len(rows):
            return "선택된 파일이 없습니다.", "{}", None

        filename = rows[row_idx][0]
        md_path = config.REPORT_DIR / filename

        if not md_path.exists():
            return f"파일을 찾을 수 없습니다: {filename}", "{}", None

        markdown_text = md_path.read_text(encoding="utf-8")

        json_path = md_path.with_suffix(".json")
        if json_path.exists():
            raw_json = json_path.read_text(encoding="utf-8")
        else:
            raw_json = "(해당 리포트의 raw JSON 이 없습니다. 구버전 저장본일 수 있습니다.)"

        return markdown_text, raw_json, str(md_path)
    except Exception as e:
        return f"리포트 로드 실패: {type(e).__name__}: {e}", "{}", None


def _latest_pipeline_reports(limit: int = 5) -> List[List[str]]:
    """
    models_core/main.py가 저장한 pipeline_report_*.json 목록을 최근순으로 보여준다.
    """
    report_dir = config.REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    reports = sorted(
        report_dir.glob("pipeline_report*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]

    rows: List[List[str]] = []
    for path in reports:
        rows.append([
            path.name,
            str(path.stat().st_size),
            path.stat().st_mtime_ns.__str__(),
        ])
    return rows


def get_model_status() -> List[List[str]]:
    """
    기존 Streamlit app.py의 모델 상태 확인 로직을 Gradio용으로 변환.
    """
    model_list = [
        ("CNN (AI4I)", "ai4i_cnn", ".pt"),
        ("GBDT (AI4I)", "ai4i_gbdt", ".pkl"),
        ("CNN (CWRU)", "cwru_cnn", ".pt"),
        ("AE (Hydraulic)", "hydraulic_ae", ".pt"),
        ("LSTM (C-MAPSS)", "cmapss_lstm", ".pt"),
        ("LSTM (N-CMAPSS)", "ncmapss_lstm", ".pt"),
    ]

    # HF Model Hub 에 어떤 체크포인트가 있는지 1회 조회 (Space 환경에서는 로컬이 비어 있음).
    hub_files: List[str] = []
    try:
        from huggingface_hub import list_repo_files

        hub_files = list_repo_files(config.CHECKPOINT_REPO, repo_type="model")
    except Exception:
        hub_files = []

    rows: List[List[str]] = []
    for label, key, ext in model_list:
        found = sorted(config.CHECKPOINT_DIR.glob(f"{key}*{ext}"))
        if found:
            rows.append([label, "Ready", found[-1].name])
            continue

        hub_match = sorted(f for f in hub_files if f.startswith(f"{key}_") and f.endswith(ext))
        if hub_match:
            rows.append([label, "Ready (HF)", hub_match[-1]])
        else:
            rows.append([label, "N/A", "-"])
    return rows


def refresh_status_and_reports() -> Tuple[List[List[str]], List[List[str]]]:
    """
    Model Status 탭과 최근 리포트 목록을 같이 새로고침.
    """
    return get_model_status(), _latest_pipeline_reports()


def simulate_risk(
    failure_prob: float,
    anomaly_score: float,
    rul_norm: float,
    method: str,
) -> Tuple[float, float, float, str]:
    """
    Risk Simulator 탭 전용.
    기존 risk_score.py의 weighted_sum / noisy_or / compute_risk / to_risk_level를 그대로 사용한다.
    """
    weighted = float(rs.weighted_sum(failure_prob, anomaly_score, rul_norm))
    noisy = float(rs.noisy_or(failure_prob, anomaly_score, rul_norm))
    final_risk = float(rs.compute_risk(failure_prob, anomaly_score, rul_norm, method))
    level = rs.to_risk_level(final_risk)

    return round(weighted, 4), round(noisy, 4), round(final_risk, 4), str(level)

def get_lstm_dataset_info(dataset_key: str) -> Tuple[int, List[str], str]:
    """
    dataset_key별 LSTM 입력 계약 정보를 반환한다.

    반환:
    - expected_feature_dim
    - default_feature_names
    - human-readable description
    """
    if dataset_key == "cmapss_lstm":
        feature_names = [
            "s2", "s3", "s4", "s7", "s8", "s9", "s11",
            "s12", "s13", "s14", "s15", "s17", "s20", "s21",
        ]
        description = (
            "cmapss_lstm는 한 timestep당 14개 feature가 필요합니다.\n"
            "권장 feature_names: s2,s3,s4,s7,s8,s9,s11,s12,s13,s14,s15,s17,s20,s21"
        )
        return 14, feature_names, description

    if dataset_key == "ncmapss_lstm":
        feature_names = [f"f{i}" for i in range(43)]
        description = (
            "ncmapss_lstm는 한 timestep당 43개 feature가 필요합니다.\n"
            "현재 프로젝트에서는 기본적으로 f0~f42 형식으로 입력 예시를 제공합니다."
        )
        return 43, feature_names, description

    # fallback
    feature_names = ["f0", "f1", "f2"]
    description = "지원하지 않는 dataset_key 입니다."
    return 3, feature_names, description


def build_lstm_sequence_placeholder(dataset_key: str, timesteps: int = 3) -> str:
    """
    dataset_key에 맞는 예시 sequence placeholder를 생성한다.
    """
    feature_dim, _, _ = get_lstm_dataset_info(dataset_key)

    lines = []
    for t in range(timesteps):
        row = [f"{0.1 * (i + 1) + 0.01 * t:.2f}" for i in range(feature_dim)]
        lines.append(",".join(row))
    return "\n".join(lines)


def build_lstm_feature_names_placeholder(dataset_key: str) -> str:
    """
    dataset_key에 맞는 feature_names placeholder를 생성한다.
    """
    _, feature_names, _ = get_lstm_dataset_info(dataset_key)
    return ",".join(feature_names)

def build_lstm_sample_values(dataset_key: str, timesteps: int = 4) -> Tuple[str,str,str]:
    """
    dataset_key에 맞는 LSTM 샘플 입력을 생성한다.

    반환:
    1) asset_id
    2) sequence_text
    3) feature_names_text
    """
    feature_dim, feature_names, _ = get_lstm_dataset_info(dataset_key)

    lines = []
    for t in range(timesteps):
        row = []
        for i in range(feature_dim):
            # timestep별로 조금씩 변화하는 더미 시계열
            value = round(0.1 * (i + 1) + 0.01 * t, 4)
            row.append(str(value))
        lines.append(",".join(row))

    sequence_text = "\n".join(lines)
    feature_names_text = ",".join(feature_names)

    if dataset_key == "cmapss_lstm":
        asset_id = "CMAPSS-ENGINE-001"
    elif dataset_key == "ncmapss_lstm":
        asset_id = "NCMAPSS-ENGINE-001"
    else:
        asset_id = "ENGINE-001"

    return asset_id, sequence_text, feature_names_text


def load_lstm_sample(dataset_key: str) -> Tuple[str, str, str]:
    """
    LSTM Sample 버튼 클릭 시 dataset_key에 맞는 샘플을 반환한다.

    반환 순서:
    1) asset_id
    2) sequence_text
    3) feature_names_text
    """
    return build_lstm_sample_values(dataset_key)

def update_lstm_input_guide(dataset_key: str):
    """
    LSTM dataset 변경 시:
    - 안내 문구
    - sequence placeholder
    - feature_names placeholder
    를 함께 갱신한다.
    """
    feature_dim, feature_names, description = get_lstm_dataset_info(dataset_key)

    guide_md = (
        f"### LSTM Input Guide\n"
        f"- dataset_key: `{dataset_key}`\n"
        f"- expected feature_dim: **{feature_dim}**\n"
        f"- 한 줄 = 1 timestep\n"
        f"- 각 줄은 쉼표(,)로 구분\n"
        f"- 최소 3 timestep 이상 권장\n\n"
        f"{description}"
    )

    seq_placeholder = build_lstm_sequence_placeholder(dataset_key)
    feature_placeholder = build_lstm_feature_names_placeholder(dataset_key)

    return (
        gr.update(value=guide_md),
        gr.update(placeholder=seq_placeholder),
        gr.update(placeholder=feature_placeholder),
    )

# ---------------------------------------------------------------------
# LSTM 입력 파싱 헬퍼
# ---------------------------------------------------------------------
def parse_sequence_text(sequence_text: str) -> List[List[float]]:
    """
    멀티라인 텍스트를 LSTM sequence 2차원 리스트로 변환한다.

    입력 예:
        0.1, 0.2, 0.3
        0.11, 0.21, 0.31
        0.12, 0.22, 0.32
    """
    if sequence_text is None or not str(sequence_text).strip():
        raise ValueError("sequence 입력이 비어 있습니다.")

    lines = [line.strip() for line in str(sequence_text).strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("유효한 sequence line이 없습니다.")

    sequence: List[List[float]] = []

    for line_no, line in enumerate(lines, start=1):
        # 쉼표 구분
        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        if not parts:
            raise ValueError(f"{line_no}번째 줄에 값이 없습니다.")

        row: List[float] = []
        for part in parts:
            try:
                row.append(float(part))
            except ValueError as e:
                raise ValueError(
                    f"{line_no}번째 줄의 값 '{part}' 을 숫자로 변환할 수 없습니다."
                ) from e

        sequence.append(row)

    return sequence


def parse_feature_names_text(feature_names_text: str) -> List[str] | None:
    """
    콤마 구분 feature name 문자열을 리스트로 변환한다.
    비어 있으면 None 반환.
    """
    if feature_names_text is None or not str(feature_names_text).strip():
        return None

    names = [x.strip() for x in str(feature_names_text).split(",") if x.strip()]
    return names if names else None


# ---------------------------------------------------------------------
# 기존 Analysis 실행 함수
# ---------------------------------------------------------------------
_LOADING_TUPLE: Tuple[str, str, Any, Any, str, str, str, str] = (
    "모델 로딩 중... (첫 실행 시 체크포인트 다운로드로 최대 30초 소요 가능)",
    "", None, None, "", "", "", "N/A",
)


def run_analysis(
    mode: str,
    dataset_key: str,
    risk_method: str,
    air_temperature_k: float,
    process_temperature_k: float,
    rotational_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
) -> Generator[Tuple[str, str, Any, Any, str, str, str, str], None, None]:
    """Analysis 탭 실행 — 첫 yield 로 로딩 표시 후 결과 yield (generator)."""
    payload = {
        "air_temperature_k": air_temperature_k,
        "process_temperature_k": process_temperature_k,
        "rotational_speed_rpm": rotational_speed_rpm,
        "torque_nm": torque_nm,
        "tool_wear_min": tool_wear_min,
    }

    yield _LOADING_TUPLE

    try:
        service = AnalyzeService(
            mode=mode, risk_method=risk_method, dataset_key=dataset_key,
        )
        result = service.run(payload)
        raw_json = _safe_json(result.get("raw_result", {}))
        yield (
            result.get("validation_text", ""),
            result.get("summary_text", ""),
            result.get("feature_plot", None),
            result.get("sensor_plot", None),
            result.get("explanation_text", ""),
            result.get("report_markdown", ""),
            raw_json,
            result.get("alert_text", "N/A"),
        )
    except NotImplementedError as e:
        msg = f"현재 선택한 dataset_key는 이 입력 폼으로는 실행할 수 없습니다.\n\n{e}"
        yield (msg, "실행 불가", None, None, "", f"# Execution Error\n\n{msg}", "{}", "N/A")
    except FileNotFoundError as e:
        msg = f"체크포인트 또는 필수 파일이 없습니다.\n\n{e}"
        yield (msg, "실행 실패", None, None, "", f"# File Error\n\n{msg}", "{}", "N/A")
    except Exception as e:
        msg = f"분석 중 예외가 발생했습니다.\n\n{type(e).__name__}: {e}"
        yield (msg, "실행 실패", None, None, "", f"# Runtime Error\n\n{msg}", "{}", "N/A")


# ---------------------------------------------------------------------
# Streaming 실행 함수 (ENABLE_LLM_STREAM=1 시 사용)
# ---------------------------------------------------------------------
_STREAM_OUTPUT_KEYS = (
    "validation_text",
    "summary_text",
    "feature_plot",
    "sensor_plot",
    "explanation_text",
    "report_markdown",
    "raw_result",
    "alert_text",
)


def _partial_to_tuple(partial: Dict[str, Any]) -> Tuple[str, str, Any, Any, str, str, str, str]:
    raw = partial.get("raw_result")
    raw_json = _safe_json(raw) if raw is not None else ""
    return (
        partial.get("validation_text", ""),
        partial.get("summary_text", ""),
        partial.get("feature_plot", None),
        partial.get("sensor_plot", None),
        partial.get("explanation_text", ""),
        partial.get("report_markdown", ""),
        raw_json,
        partial.get("alert_text", "N/A"),
    )


def run_analysis_stream(
    mode: str,
    dataset_key: str,
    risk_method: str,
    air_temperature_k: float,
    process_temperature_k: float,
    rotational_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
) -> Generator[Tuple[str, str, Any, Any, str, str, str, str], None, None]:
    """AnalyzeService.run_stream() 를 소비하여 Gradio 출력 tuple 을 점진 yield."""
    payload = {
        "air_temperature_k": air_temperature_k,
        "process_temperature_k": process_temperature_k,
        "rotational_speed_rpm": rotational_speed_rpm,
        "torque_nm": torque_nm,
        "tool_wear_min": tool_wear_min,
    }

    yield _LOADING_TUPLE

    try:
        service = AnalyzeService(
            mode=mode, risk_method=risk_method, dataset_key=dataset_key,
        )
        for partial in service.run_stream(payload):
            yield _partial_to_tuple(partial)
    except NotImplementedError as e:
        msg = f"현재 선택한 dataset_key는 이 입력 폼으로는 실행할 수 없습니다.\n\n{e}"
        yield (msg, "실행 불가", None, None, "", f"# Execution Error\n\n{msg}", "{}", "N/A")
    except FileNotFoundError as e:
        msg = f"체크포인트 또는 필수 파일이 없습니다.\n\n{e}"
        yield (msg, "실행 실패", None, None, "", f"# File Error\n\n{msg}", "{}", "N/A")
    except Exception as e:
        msg = f"분석 중 예외가 발생했습니다.\n\n{type(e).__name__}: {e}"
        yield (msg, "실행 실패", None, None, "", f"# Runtime Error\n\n{msg}", "{}", "N/A")


def run_lstm_analysis_stream(
    mode: str,
    dataset_key: str,
    risk_method: str,
    asset_id: str,
    sequence_text: str,
    feature_names_text: str,
) -> Generator[Tuple[str, str, Any, Any, str, str, str, str], None, None]:
    """AnalyzeService.run_lstm_stream() 을 소비하여 Gradio 출력 tuple 을 점진 yield."""
    yield _LOADING_TUPLE
    try:
        sequence = parse_sequence_text(sequence_text)
        feature_names = parse_feature_names_text(feature_names_text)

        service = AnalyzeService(
            mode=mode, risk_method=risk_method, dataset_key=dataset_key,
        )
        for partial in service.run_lstm_stream(
            sequence=sequence,
            feature_names=feature_names,
            asset_id=asset_id or "UNKNOWN",
            dataset_key=dataset_key,
        ):
            yield _partial_to_tuple(partial)
    except ValueError as e:
        msg = f"LSTM 입력 파싱 오류입니다.\n\n{e}"
        yield (msg, "실행 실패", None, None, "", f"# Input Parse Error\n\n{msg}", "{}", "N/A")
    except NotImplementedError as e:
        msg = f"현재 LSTM 설정으로는 실행할 수 없습니다.\n\n{e}"
        yield (msg, "실행 불가", None, None, "", f"# Execution Error\n\n{msg}", "{}", "N/A")
    except FileNotFoundError as e:
        msg = f"LSTM 체크포인트 또는 필수 파일이 없습니다.\n\n{e}"
        yield (msg, "실행 실패", None, None, "", f"# File Error\n\n{msg}", "{}", "N/A")
    except Exception as e:
        msg = f"LSTM 분석 중 예외가 발생했습니다.\n\n{type(e).__name__}: {e}"
        yield (msg, "실행 실패", None, None, "", f"# Runtime Error\n\n{msg}", "{}", "N/A")


# ---------------------------------------------------------------------
# 새 LSTM Analysis 실행 함수
# ---------------------------------------------------------------------
def run_lstm_analysis(
    mode: str,
    dataset_key: str,
    risk_method: str,
    asset_id: str,
    sequence_text: str,
    feature_names_text: str,
) -> Generator[Tuple[str, str, Any, Any, str, str, str, str], None, None]:
    """LSTM Analysis 탭 실행 — 첫 yield 로 로딩 표시 후 결과 yield (generator)."""
    yield _LOADING_TUPLE
    try:
        sequence = parse_sequence_text(sequence_text)
        feature_names = parse_feature_names_text(feature_names_text)

        service = AnalyzeService(
            mode=mode, risk_method=risk_method, dataset_key=dataset_key,
        )
        result = service.run_lstm(
            sequence=sequence,
            feature_names=feature_names,
            asset_id=asset_id or "UNKNOWN",
            dataset_key=dataset_key,
        )
        raw_json = _safe_json(result.get("raw_result", {}))
        yield (
            result.get("validation_text", ""),
            result.get("summary_text", ""),
            result.get("feature_plot", None),
            result.get("sensor_plot", None),
            result.get("explanation_text", ""),
            result.get("report_markdown", ""),
            raw_json,
            result.get("alert_text", "N/A"),
        )
    except ValueError as e:
        msg = f"LSTM 입력 파싱 오류입니다.\n\n{e}"
        yield (msg, "실행 실패", None, None, "", f"# Input Parse Error\n\n{msg}", "{}", "N/A")
    except NotImplementedError as e:
        msg = f"현재 LSTM 설정으로는 실행할 수 없습니다.\n\n{e}"
        yield (msg, "실행 불가", None, None, "", f"# Execution Error\n\n{msg}", "{}", "N/A")
    except FileNotFoundError as e:
        msg = f"LSTM 체크포인트 또는 필수 파일이 없습니다.\n\n{e}"
        yield (msg, "실행 실패", None, None, "", f"# File Error\n\n{msg}", "{}", "N/A")
    except Exception as e:
        msg = f"LSTM 분석 중 예외가 발생했습니다.\n\n{type(e).__name__}: {e}"
        yield (msg, "실행 실패", None, None, "", f"# Runtime Error\n\n{msg}", "{}", "N/A")

def save_lstm_report(report_markdown: str, dataset_key: str, asset_id: str) -> Tuple[str, str]:
    """
    현재 화면의 LSTM report_markdown을 파일로 저장한다.

    반환:
    1) 상태 메시지
    2) 저장된 파일 경로 (gr.File 출력용)
    """
    try:
        from services.report_service import ReportService

        if report_markdown is None or not str(report_markdown).strip():
            return "저장할 보고서 내용이 없습니다. 먼저 LSTM Analysis를 실행해 주세요.", ""

        service = ReportService()
        saved_path = service.save_markdown_report(
            report_markdown=report_markdown,
            dataset_key=dataset_key,
            asset_id=asset_id or "UNKNOWN",
            prefix="lstm_report",
        )
        return f"보고서 저장 완료: {saved_path}", saved_path

    except Exception as e:
        return f"보고서 저장 실패: {type(e).__name__}: {e}", ""

# ---------------------------------------------------------------------
# Gradio UI 정의
# ---------------------------------------------------------------------
class ExecutionMode(str, Enum):
    """
    LLM 실행 경로 모드. 현재 2 값.
    향후 mid-stream tool 호출 등 하이브리드가 필요하면 여기에 추가한다.
    """
    STREAM = "stream"
    NON_STREAM = "non_stream"


def resolve_execution_mode(
    *, stream_env: bool, tools_env: bool
) -> Tuple[ExecutionMode, str]:
    """
    Env 플래그 조합 → (mode, reason) 으로 결정.
    이후 요청 단위 resolver 가 필요해지면 입력을 (request, config) 로 확장.
    """
    if tools_env:
        # tool calling 은 여러 번의 round-trip 이 필요해 mid-stream 호출이 복잡함
        return ExecutionMode.NON_STREAM, "tools_enabled_disables_stream"
    if stream_env:
        return ExecutionMode.STREAM, "stream_env_enabled"
    return ExecutionMode.NON_STREAM, "default_non_stream"


_ENABLE_LLM_STREAM = os.getenv("ENABLE_LLM_STREAM", "0") == "1"
_ENABLE_LLM_TOOLS = os.getenv("ENABLE_LLM_TOOLS", "0") == "1"
_execution_mode, _mode_reason = resolve_execution_mode(
    stream_env=_ENABLE_LLM_STREAM,
    tools_env=_ENABLE_LLM_TOOLS,
)
print(
    f"[HybridPdM] execution mode resolved: "
    f"mode={_execution_mode.value} "
    f"stream_env={_ENABLE_LLM_STREAM} tools_env={_ENABLE_LLM_TOOLS} "
    f"reason={_mode_reason}",
    flush=True,
)
_use_stream = _execution_mode == ExecutionMode.STREAM
_analysis_handler = run_analysis_stream if _use_stream else run_analysis
_lstm_analysis_handler = run_lstm_analysis_stream if _use_stream else run_lstm_analysis

with gr.Blocks(title="HybridPdM") as demo:
    gr.Markdown("# HybridPdM")
    gr.Markdown(
        "센서 입력 → 예측 → 설명 → Risk 판단 → 조치 제안 → 검증 → 보고서 생성까지 수행합니다."
    )

    with gr.Tabs():
        # ============================================================
        # 1) Analysis (기존 AI4I scalar 입력)
        # ============================================================
        with gr.Tab("Analysis"):
            with gr.Row():
                mode_dd = gr.Dropdown(
                    choices=["lite", "full"],
                    value="lite",
                    label="Execution Mode",
                    info="lite는 경량 규칙 기반, full은 실제 체크포인트 사용",
                )
                dataset_dd = gr.Dropdown(
                    choices=[
                        "ai4i_cnn",
                        "ai4i_gbdt",
                        "hydraulic_ae",
                    ],
                    value="ai4i_cnn",
                    label="Dataset Key",
                    info="이 탭은 scalar 입력 전용입니다. LSTM 계열은 'LSTM Analysis' 탭에서 실행합니다.",
                )
                risk_method_dd = gr.Dropdown(
                    choices=["weighted", "noisy_or", "max"],
                    value="weighted",
                    label="Risk Method",
                )

            with gr.Row():
                sample_dd = gr.Dropdown(
                    choices=list(SAMPLE_CASES.keys()),
                    value="Normal",
                    label="Sample Case",
                )
                load_btn = gr.Button("Load Sample")

            with gr.Row():
                air_input = gr.Number(label="Air Temperature (K)", value=298.5)
                proc_input = gr.Number(label="Process Temperature (K)", value=309.1)
                rpm_input = gr.Number(label="Rotational Speed (RPM)", value=1510.0)
                torque_input = gr.Number(label="Torque (Nm)", value=36.0)
                wear_input = gr.Number(label="Tool Wear (min)", value=42.0)

            run_btn = gr.Button("Run Analysis", variant="primary")

            with gr.Row():
                validation_box = gr.Textbox(label="Validation Result", lines=6)
                alert_box = gr.Textbox(label="Alert Status", lines=2)

            with gr.Row():
                summary_box = gr.Textbox(label="Prediction Summary", lines=10)
                explanation_box = gr.Textbox(label="Explanation Summary", lines=12)

            with gr.Row():
                feature_plot = gr.Plot(label="Feature Importance")
                sensor_plot = gr.Plot(label="Sensor Snapshot")

            report_md = gr.Markdown(label="Report + Recommendation")
            raw_json_box = gr.Code(label="Raw Result JSON", language="json")

            load_btn.click(
                fn=load_sample,
                inputs=[sample_dd],
                outputs=[air_input, proc_input, rpm_input, torque_input, wear_input],
            )

            run_btn.click(
                fn=_analysis_handler,
                inputs=[
                    mode_dd,
                    dataset_dd,
                    risk_method_dd,
                    air_input,
                    proc_input,
                    rpm_input,
                    torque_input,
                    wear_input,
                ],
                outputs=[
                    validation_box,
                    summary_box,
                    feature_plot,
                    sensor_plot,
                    explanation_box,
                    report_md,
                    raw_json_box,
                    alert_box,
                ],
            )

        # ============================================================
        # 2) LSTM Analysis (새 sequence 입력 탭)
        # ============================================================
        with gr.Tab("LSTM Analysis"):
            lstm_guide_md = gr.Markdown(
                value=(
                    "### LSTM Input Guide\n"
                    "- dataset_key: `cmapss_lstm`\n"
                    "- expected feature_dim: **14**\n"
                    "- 한 줄 = 1 timestep\n"
                    "- 각 줄은 쉼표(,)로 구분\n"
                    "- 최소 3 timestep 이상 권장\n\n"
                    "cmapss_lstm는 한 timestep당 14개 feature가 필요합니다."
                )
            )

            with gr.Row():
                lstm_mode_dd = gr.Dropdown(
                    choices=["full"],
                    value="full",
                    label="Execution Mode",
                    info="LSTM sequence 추론은 현재 full 모드만 지원합니다.",
                )
                lstm_dataset_dd = gr.Dropdown(
                    choices=["cmapss_lstm", "ncmapss_lstm"],
                    value="cmapss_lstm",
                    label="LSTM Dataset Key",
                )
                lstm_risk_method_dd = gr.Dropdown(
                    choices=["weighted", "noisy_or", "max"],
                    value="weighted",
                    label="Risk Method",
                )

            with gr.Row():
                lstm_asset_id = gr.Textbox(
                    label="Asset ID",
                    value="ENGINE-001",
                )
            
            with gr.Row():
                lstm_load_sample_btn = gr.Button("Load LSTM Sample")

            lstm_sequence_text = gr.Textbox(
                label="Sequence Input",
                lines=12,
                placeholder=build_lstm_sequence_placeholder("cmapss_lstm"),
            )

            lstm_feature_names_text = gr.Textbox(
                label="Feature Names (optional)",
                lines=2,
                placeholder=build_lstm_feature_names_placeholder("cmapss_lstm"),
            )
            lstm_run_btn = gr.Button("Run LSTM Analysis", variant="primary")

            with gr.Row():
                lstm_validation_box = gr.Textbox(label="Validation Result", lines=8)
                lstm_alert_box = gr.Textbox(label="Alert Status", lines=2)

            with gr.Row():
                lstm_summary_box = gr.Textbox(label="Prediction Summary", lines=10)
                lstm_explanation_box = gr.Textbox(label="Explanation Summary", lines=12)

            with gr.Row():
                lstm_feature_plot = gr.Plot(label="Feature Importance")
                lstm_sensor_plot = gr.Plot(label="Sequence Snapshot")

            lstm_report_md = gr.Markdown(label="LSTM Report + Recommendation")
            lstm_raw_json_box = gr.Code(label="Raw Result JSON", language="json")
            
            with gr.Row():
                lstm_save_report_btn = gr.Button("Save LSTM Report")
                lstm_save_status_box = gr.Textbox(label="Save Status", lines=2)
                lstm_report_file = gr.File(label="Saved Report File")
            
            lstm_dataset_dd.change(
                fn=update_lstm_input_guide,
                inputs=[lstm_dataset_dd],
                outputs=[lstm_guide_md, lstm_sequence_text, lstm_feature_names_text],
            )

            lstm_load_sample_btn.click(
                fn=load_lstm_sample,
                inputs=[lstm_dataset_dd],
                outputs=[lstm_asset_id, lstm_sequence_text, lstm_feature_names_text],
            )
            
            lstm_run_btn.click(
                fn=_lstm_analysis_handler,
                inputs=[
                    lstm_mode_dd,
                    lstm_dataset_dd,
                    lstm_risk_method_dd,
                    lstm_asset_id,
                    lstm_sequence_text,
                    lstm_feature_names_text,
                ],
                outputs=[
                    lstm_validation_box,
                    lstm_summary_box,
                    lstm_feature_plot,
                    lstm_sensor_plot,
                    lstm_explanation_box,
                    lstm_report_md,
                    lstm_raw_json_box,
                    lstm_alert_box,
                ],
            )
            
            lstm_save_report_btn.click(
                fn=save_lstm_report,
                inputs=[lstm_report_md, lstm_dataset_dd, lstm_asset_id],
                outputs=[lstm_save_status_box, lstm_report_file],
            )

        # ============================================================
        # 3) Diagnostics
        # ============================================================
        with gr.Tab("Diagnostics"):
            gr.Markdown("최근 파이프라인 리포트와 체크포인트 상태를 확인합니다.")

            refresh_diag_btn = gr.Button("Refresh Diagnostics")

            with gr.Row():
                model_status_df = gr.Dataframe(
                    headers=["Model", "Status", "Latest File"],
                    datatype=["str", "str", "str"],
                    label="Model Status",
                    value=get_model_status(),
                )
                reports_df = gr.Dataframe(
                    headers=["Report File", "Size(bytes)", "mtime_ns"],
                    datatype=["str", "str", "str"],
                    label="Recent Pipeline Reports",
                    value=_latest_pipeline_reports(),
                )

            refresh_diag_btn.click(
                fn=refresh_status_and_reports,
                inputs=[],
                outputs=[model_status_df, reports_df],
            )

        # ============================================================
        # 4) Risk Simulator
        # ============================================================
        with gr.Tab("Risk Simulator"):
            gr.Markdown("모델 출력값을 직접 조절하여 Risk Engine을 검증합니다.")

            with gr.Row():
                sim_fp = gr.Slider(0.0, 1.0, value=0.30, step=0.01, label="Failure Probability")
                sim_an = gr.Slider(0.0, 1.0, value=0.20, step=0.01, label="Anomaly Score")
                sim_rul = gr.Slider(0.0, 1.0, value=0.70, step=0.01, label="RUL Norm")

            sim_method = gr.Dropdown(
                choices=["weighted", "noisy_or", "max"],
                value="weighted",
                label="Fusion Method",
            )
            sim_btn = gr.Button("Simulate Risk")

            with gr.Row():
                weighted_box = gr.Number(label="Weighted Sum")
                noisy_box = gr.Number(label="Noisy-OR")
                final_box = gr.Number(label="Final Risk")
                level_box = gr.Textbox(label="Risk Level")

            sim_btn.click(
                fn=simulate_risk,
                inputs=[sim_fp, sim_an, sim_rul, sim_method],
                outputs=[weighted_box, noisy_box, final_box, level_box],
            )

        # ============================================================
        # 5) Model Status
        # ============================================================
        with gr.Tab("Model Status"):
            gr.Markdown("체크포인트 존재 여부를 확인합니다.")

            refresh_btn = gr.Button("Refresh Model Status")
            status_df = gr.Dataframe(
                headers=["Model", "Status", "Latest File"],
                datatype=["str", "str", "str"],
                value=get_model_status(),
                label="Checkpoint Status",
            )

            refresh_btn.click(
                fn=get_model_status,
                inputs=[],
                outputs=[status_df],
            )

        # ============================================================
        # 6) Report
        # ============================================================
        with gr.Tab("Report"):
            gr.Markdown(
                "분석 실행 시 자동 저장된 리포트 목록입니다. 행을 클릭하면 미리보기와 다운로드가 가능합니다."
            )

            with gr.Row():
                reports_refresh_btn = gr.Button("Refresh")
                reports_dir_md = gr.Markdown(
                    value=f"- 저장 위치: `{config.REPORT_DIR}`"
                )

            saved_reports_df = gr.Dataframe(
                headers=["File", "Saved At", "Size(bytes)"],
                datatype=["str", "str", "str"],
                label="Saved Reports",
                value=list_saved_reports(),
                interactive=False,
            )

            with gr.Row():
                with gr.Column():
                    saved_report_md = gr.Markdown(label="Report Preview")
                with gr.Column():
                    saved_report_json = gr.Code(label="Raw Result JSON", language="json")

            saved_report_file = gr.File(label="Download Markdown")

            reports_refresh_btn.click(
                fn=list_saved_reports,
                inputs=[],
                outputs=[saved_reports_df],
            )

            saved_reports_df.select(
                fn=load_report_preview,
                inputs=[],
                outputs=[saved_report_md, saved_report_json, saved_report_file],
            )

    gr.Markdown(
        f"""
**Runtime Info**
- Device: `{config.get_device()}`
- Checkpoint Dir: `{config.CHECKPOINT_DIR}`
- Report Dir: `{config.REPORT_DIR}`
- Default Port: `7861`
"""
    )

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    _port_env = os.getenv("GRADIO_SERVER_PORT")
    server_port = int(_port_env) if _port_env else None  # None → Gradio가 7860부터 자동 탐색
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        inbrowser=True,
    )
---
title: HybridPdM
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
python_version: "3.10"
---

# HybridPdM — Hybrid Predictive Maintenance with LLM Analysis

> 본 저장소는 Hugging Face Spaces (Gradio SDK) 배포에 맞춰 구성되어 있습니다.
> Space 진입 파일: [app.py](app.py) (내부적으로 `gradio_app.py` 의 `demo` 재노출).
> 로컬 개발 시에는 `python gradio_app.py` 또는 `python app.py` 모두 사용 가능합니다.

산업설비 예지보전(Predictive Maintenance)을 위한 하이브리드 파이프라인입니다.
정형 센서 데이터(스칼라·시계열)를 입력받아 딥러닝 기반 고장 예측을 수행하고,
위험도 평가·설명(Feature Importance / Attention)을 종합해 **LLM이 현장 정비 담당자용 한국어 분석 코멘트를 생성**합니다.

Gradio 기반 UI로 즉시 결과를 확인할 수 있습니다.

---

## 주요 기능

- **스칼라 입력 분석**: 단일 시점 센서 벡터 → 고장 확률 + 위험 등급 + 설명 + LLM 코멘트
- **시계열(LSTM) 분석**: 다변량 시계열 → RUL/고장 예측 + Attention 설명 + LLM 코멘트
- **Risk Scoring**: 예측 확률과 도메인 규칙을 결합한 위험도 산정 (`weighted` / `rule-based`)
- **LLM 분석 코멘트**: 상태 요약 / 의심 원인 / 권장 조치 3섹션으로 구성된 한국어 리포트
- **Guardrail**: 금지 표현 필터링 및 출력 구조 검증, 실패 시 재작성 / 안전 fallback
- **Evaluation**: 출력 구조 점검 및 (옵션) LLM-as-Judge 기반 사실성 검증
- **Report**: Markdown 리포트 자동 생성 및 저장

### LLM 파이프라인 고도화 (본 브랜치 `update`)

| 단계 | 내용 |
|-----|-----|
| P1 | 프롬프트 파일 로더 / Few-shot 예시 3종 / 위험등급별 temperature / `max_tokens` 환경변수화 |
| P2 | Chain-of-Thought 내부 추론 지시 / 데이터셋별 domain context 분기 / LLM-as-Judge (opt-in) |

---

## 아키텍처

```text
[User Input]
     ↓
InputValidationService   ← 범위/타입 검증
     ↓
PdmService               ← 딥러닝 추론 (scalar / LSTM)
     ↓
RiskService              ← 위험도 점수 산정
     ↓
ExplainService           ← Feature Importance / Attention 추출
     ↓
LlmService               ← 시스템 프롬프트 + Few-shot + CoT + domain_context 로 코멘트 생성
     ↓
GuardrailService         ← 금지 표현/구조 검증, 재작성 또는 fallback
     ↓
EvaluationService        ← 구조 점수 + (옵션) LLM-as-Judge
     ↓
PlotService / ReportService  ← 시각화 + Markdown 리포트
```

---

## 프로젝트 구조

```text
hybridpdm_gradio/
├── gradio_app.py              # Gradio UI 진입점
├── run_ngrok.py               # ngrok 터널링 실행 스크립트
├── requirements.txt
├── .env.example               # 환경변수 템플릿
│
├── services/                  # 도메인 서비스 계층
│   ├── analyze_service.py     # 전체 파이프라인 오케스트레이션
│   ├── input_validation_service.py
│   ├── pdm_service.py
│   ├── risk_service.py
│   ├── explain_service.py
│   ├── llm_service.py         # OpenAI 호출 + Few-shot + Judge
│   ├── guardrail_service.py
│   ├── evaluation_service.py
│   ├── plot_service.py
│   ├── report_service.py
│   └── schemas.py             # 공용 dataclass
│
├── models_core/               # 딥러닝 모델·학습 코드
│   ├── models.py              # CNN / LSTM 정의
│   ├── train.py
│   ├── evaluate.py
│   ├── data_pipeline.py
│   ├── risk_score.py
│   ├── explain.py
│   └── config.py
│
├── prompts/                   # LLM 프롬프트 자원
│   ├── system_pdm_assistant.txt
│   └── user_context_template.json
│
├── schemas/
│   └── input_schema.py
│
├── utils/
│   └── sample_cases.py
│
└── tests (루트)
    ├── test_validation_step.py
    ├── test_pdm_lstm_step.py
    └── test_analyze_lstm_step.py
```

> 데이터셋 원본(`models_core/dataset/`), 학습된 체크포인트(`artifacts/checkpoints/`), 생성된 리포트(`artifacts/reports/`), 런타임 로그(`logs/`)는 저장소에 포함되지 않습니다. 아래 [데이터셋](#데이터셋) 섹션 참고.

---

## 설치

요구사항: Python 3.10+ (권장)

```bash
git clone https://github.com/yoosehyeon/hybridpdm-LLM.git
cd hybridpdm-LLM

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 환경변수

`.env.example` 을 `.env` 로 복사 후 값을 채웁니다.

```bash
cp .env.example .env
```

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `GROQ_API_KEY` | Groq Cloud 에서 발급한 API 키 (미설정 시 LLM 호출 없이 fallback 텍스트 사용) | - |
| `GROQ_MODEL` | 사용할 모델 | `llama-3.3-70b-versatile` |
| `GROQ_TEMPERATURE` | 기본 temperature (위험등급별로 동적 조정됨) | `0.2` |
| `GROQ_MAX_TOKENS` | 응답 최대 토큰 | `800` |
| `GROQ_BASE_URL` | OpenAI 호환 endpoint (커스텀 프록시용) | `https://api.groq.com/openai/v1` |
| `PROMPT_DIR` | 시스템 프롬프트 디렉터리 | `prompts` |
| `ENABLE_LLM_JUDGE` | `1` 설정 시 LLM-as-Judge 활성화 (추가 API 호출) | `0` |
| `ENABLE_LLM_STREAM` | `1` 설정 시 UI 계층에서 `generate_stream()` 사용 (서비스 메서드는 값 무관하게 항상 제공) | `0` |
| `CHECKPOINT_REPO` | 런타임에 체크포인트를 pull 할 HF Model Hub repo. 로컬 `models_core/artifacts/checkpoints/` 에 동일 stem 이 있으면 우선 사용 | `yusehyeon/hybridpdm-checkpoints` |

> LLM 프로바이더로 **Groq (Llama 3.3 70B Versatile)** 를 사용합니다. OpenAI 호환 endpoint 를 통해 `openai` SDK 그대로 호출하며, 무료 티어 한도(30 RPM / ~14.4K RPD) 는 본 프로젝트의 시연 수요를 충분히 감당합니다. API 키는 [Groq Cloud Console](https://console.groq.com/keys) 에서 발급.

---

## 실행

### Gradio UI

```bash
python gradio_app.py
```

기본적으로 `http://localhost:7860` 에서 UI 가 열립니다.

### ngrok 터널 (외부 접속)

```bash
python run_ngrok.py
```

---

## Hugging Face Spaces 배포

본 프로젝트는 **Hugging Face Spaces (Gradio SDK, free CPU basic)** 를 1순위 시연 URL 로 사용합니다.

### 1. Space 생성

1. [huggingface.co/new-space](https://huggingface.co/new-space) 에서 새 Space 생성
   - SDK: **Gradio**
   - Hardware: **CPU basic (free)**
2. 생성된 Space 저장소 clone 후 본 저장소 내용을 push (또는 기존 저장소를 Space remote 로 추가)

### 2. Secrets 등록

Space 의 `Settings → Variables and secrets → New secret` 메뉴에서 다음을 등록합니다.
(값이 없으면 LLM 호출은 fallback 경로로 동작합니다.)

| Key | 필수 | 설명 |
|-----|------|------|
| `GROQ_API_KEY` | 권장 | [Groq Cloud Console](https://console.groq.com/keys) 에서 발급 (무료). 미설정 시 LLM 코멘트는 규칙 기반 fallback 으로 대체됩니다 |
| `GROQ_MODEL` | 선택 | 미설정 시 `llama-3.3-70b-versatile` |
| `ENABLE_LLM_JUDGE` | 선택 | `1` 로 설정 시 Judge 파이프라인 활성화 (API 호출 증가) |
| `ENABLE_LLM_STREAM` | 선택 | `1` 로 설정 시 UI 스트리밍 경로 사용 (P3-① Option B 적용 이후) |

### 3. 체크포인트 전략

Space 저장소는 binary 를 포함할 수 없으므로 체크포인트는 **별도 HF Model Hub repo** 에서 런타임에 pull 합니다.

- 기본 repo: [`yusehyeon/hybridpdm-checkpoints`](https://huggingface.co/yusehyeon/hybridpdm-checkpoints) (public, model type)
- 포함 파일: 6개 모델의 `*.pt` / `*.pkl` 가중치 + `*_meta.json` (feature_names / feature_dim / task 등 데이터셋 의존성 제거용 메타) + 학습 history JSON
- 로드 순서: 로컬 `models_core/artifacts/checkpoints/<stem>*.{pt,pkl}` → 없으면 `huggingface_hub.hf_hub_download(CHECKPOINT_REPO, …)` fallback (캐시: `~/.cache/huggingface/hub/`)
- 다른 repo 로 교체하려면 Space Secret 에 `CHECKPOINT_REPO=<org>/<repo>` 추가
- 업로드/재생성 스크립트: [scripts/export_checkpoint_meta.py](scripts/export_checkpoint_meta.py) (데이터셋 있는 개발 환경에서 meta.json 추출) + [scripts/upload_checkpoints_to_hf.py](scripts/upload_checkpoints_to_hf.py) (repo 생성 + 업로드)
- 결과: `Analysis` / `LSTM Analysis` / `Diagnostics` / `Model Status` 4 탭 모두 Space 에서 정상 동작 (첫 추론 시 cold-pull 지연 수 초 발생)

### 4. 진입 파일

- HF Spaces 는 `app.py` 의 `demo` 객체를 자동으로 launch 합니다.
- 본 저장소 `app.py` 는 `from gradio_app import demo` 로 재노출하는 얇은 래퍼이므로, 기존 `gradio_app.py` 엔트리 구조가 그대로 유지됩니다.

### 5. 배포 백업

Space 가 불안정하거나 빌드 오류가 장기화될 경우 백업으로 **ngrok paid (고정 도메인)** 을 사용할 수 있습니다. `run_ngrok.py` 참조.

---

## 데이터셋

본 프로젝트는 아래 공개 데이터셋을 사용합니다. 저장소에는 포함되어 있지 않으므로 직접 다운로드하여 `models_core/dataset/` 하위에 배치해야 합니다.

> **배포 환경 (HF Space) 에서는 데이터셋이 필요 없습니다.** 체크포인트와 함께 업로드된 `*_meta.json` 이 feature 정의를 대신 제공합니다. 데이터셋은 **재학습 / meta.json 재생성 시에만** 필요합니다.

| 데이터셋 | 설명 | 다운로드 |
|---------|------|---------|
| **NASA C-MAPSS** | Turbofan Engine Degradation Simulation (RUL 예측) | [NASA PCoE Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| **NASA N-CMAPSS** | Turbofan Engine Degradation Simulation-2 (실제 비행 조건 반영) | [NASA PCoE Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| **AI4I 2020** | Milling machine predictive maintenance | [UCI ML Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) |
| **Condition Monitoring of Hydraulic Systems** | 유압 시스템 상태 모니터링 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) |
| **PHM 2012 Bearing** | IEEE PHM 2012 Prognostic Challenge (FEMTO-ST) | [FEMTO-ST / IEEE PHM 2012](https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset) |
| **CWRU Bearing Data** | Case Western Reserve University Bearing Data Center | [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) |

> 각 데이터셋의 라이선스·인용 조건은 원 제공처 정책을 따르십시오.

### 배치 예시

```text
models_core/dataset/
├── CMAPSSData/
├── 17. Turbofan Engine Degradation Simulation Data Set 2/   # N-CMAPSS
├── ai4i2020.csv
├── condition+monitoring+of+hydraulic+systems/
├── PHM2012/
└── 10987113/                                                # CWRU
```

---

## 테스트

```bash
python test_validation_step.py
python test_pdm_lstm_step.py
python test_analyze_lstm_step.py
```

---

## 브랜치 전략

- `update`: 현재 개발 브랜치 (LLM 파이프라인 P1·P2 고도화 반영)
- 이후 기능은 별도 브랜치에서 작업 후 머지 예정 (예: P3 — RAG / multiturn / function calling / streaming)

---

## 라이선스

사용한 공개 데이터셋의 라이선스는 각 제공처 정책을 따릅니다. 코드 라이선스는 추후 결정.

# OmniPdM — All-in-One Predictive Maintenance System with LLM Analysis

> 이전 코드네임 **HybridPdM**. 2026-05-19 부로 GitHub 저장소(`OmniPdM-LLM`) + 로컬 디렉터리(`OmniPdM/`) 모두 새 이름으로 통합. 제품/문서 표기는 **OmniPdM** 으로 통일.
> - 의미: "Omni-" = 모든 것/전체. 다종 산업 설비(Milling / Bearing / Hydraulic / Turbofan 등) 센서 데이터를 단일 플랫폼으로 통합, 예측~LLM 설명까지 풀스택 PdM.
> - 로고: [assets/omnipdm_logo.png](assets/omnipdm_logo.png) — Dark Gray + Neon Green, ∞(인피니티) 회로 라인.
> - 브랜드/네이밍 정당성은 [PRD.md](PRD.md) §0 참고.

산업설비 예지보전(Predictive Maintenance)을 위한 하이브리드 파이프라인.
정형 센서 데이터(스칼라 / 시계열)를 입력받아 딥러닝 기반 고장·이상·RUL 예측을 수행하고,
위험도 평가 + 설명(Feature Importance / Attention) + **LLM 한국어 분석 코멘트**까지 단일 흐름으로 제공합니다.

UI는 **Dash multi-page**, 실험 추적은 **MLflow**, 학습 진입점은 `scripts/training/main.py`로 분리되어 있습니다.

---

## 주요 기능

- **스칼라 입력 분석**: 단일 시점 센서 벡터 → 고장 확률 + 위험 등급 + 설명 + LLM 코멘트
- **시계열 분석 (BiLSTM / DLinear / iTransformer)**: 다변량 시계열 → RUL 예측 + Attention/Temporal 설명
- **이상 탐지 (Denoising AE)**: 정상 데이터 기반 재구성 오차로 이상 점수 산출
- **Risk Scoring**: `weighted` / `noisy_or` / `max` 3가지 융합 + 동적 가중치
- **LLM 분석 코멘트**: 상태 요약 / 의심 원인 / 권장 조치 3섹션 한국어 리포트 (Groq Llama 3.3 70B)
- **Guardrail**: 금지 표현 정규식 치환 + 3섹션 구조 검증 + 안전 fallback
- **Evaluation**: 구조 점수 + (옵션) LLM-as-Judge
- **학술 metric 보강**: NASA PHM Score (RUL asymmetric), Confusion Matrix, PR-AUC, ROC-AUC, False Alarm Rate
- **Multi-seed 재현성**: `--seeds 42 43 44` CLI 지원, MLflow 태그로 자동 비교
- **Report**: Markdown + JSON 보고서 자동 저장 / 브라우저 조회
- **MLflow 실험 추적**: 학습 파라미터 + 평가 메트릭 + 모델 family / FD subset / seed 태그 자동 기록

---

## 아키텍처

```text
[User Input]
     ↓
InputValidationService     ← 범위 / 타입 검증
     ↓
PdmService                 ← 모델 추론 (CNN / GBDT / AE / BiLSTM / DLinear / iTransformer)
     ↓
RiskService                ← 위험도 융합 (weighted / noisy_or / max)
     ↓
ExplainService             ← Feature Importance / Attention 추출
     ↓
LlmService                 ← System + Few-shot + CoT + Domain context
     ↓
GuardrailService           ← 금지 표현 치환 + 구조 검증 + fallback
     ↓
EvaluationService          ← 구조 점수 + (옵션) LLM-as-Judge
     ↓
PlotService / ReportService ← Plotly 차트 + Markdown 보고서
     ↓
[Dash UI (6 pages)]
```

서비스 계층(`services/`)은 UI 의존성이 없어 Dash / FastAPI / CLI 어디에서도 재사용 가능합니다.

---

## 프로젝트 구조

```text
OmniPdM/
├── app.py                              # Dash multi-page 진입점
├── requirements.txt
├── .env.example
├── PRD.md
│
├── pages/                              # Dash 6 페이지
│   ├── _helpers.py                     # 공유 헬퍼 (싱글톤, 샘플 데이터)
│   ├── analysis.py                     # Scalar 분석 (구현 완료)
│   ├── lstm_analysis.py                # Sequence 분석 (placeholder)
│   ├── diagnostics.py                  # 파이프라인 진단 (placeholder)
│   ├── risk_simulator.py               # 위험도 슬라이더 (placeholder)
│   ├── model_status.py                 # 체크포인트 상태 (구현 완료)
│   └── report.py                       # 보고서 브라우저 (구현 완료)
│
├── services/                           # 도메인 서비스 계층 (UI 무관)
│   ├── analyze_service.py              # 9 단계 오케스트레이션
│   ├── input_validation_service.py
│   ├── pdm_service.py
│   ├── risk_service.py
│   ├── explain_service.py
│   ├── llm_service.py                  # Groq + Streaming + Function Calling
│   ├── llm_tools.py                    # 5 tools + ToolDispatcher
│   ├── guardrail_service.py
│   ├── evaluation_service.py
│   ├── plot_service.py                 # Plotly Figure 반환
│   ├── report_service.py
│   └── schemas.py
│
├── models_core/                        # 모델 / 데이터 런타임
│   ├── config.py                       # LSTM_CFG, DLINEAR_CFG, ITRANSFORMER_CFG, NCMAPSS_LSTM_CFG
│   ├── models.py                       # WDCNN1D / TabularCNN1D / AE / BiLSTM / DLinear / iTransformer
│   ├── data_pipeline.py                # 6+ dataset_key + FD001~004 별칭 loaders
│   ├── risk_score.py                   # weighted / noisy_or / max 융합
│   ├── _archive/                       # 미사용 코드 보관 (Captum 등)
│   └── artifacts/
│       ├── checkpoints/                # 학습된 가중치 (gitignored)
│       └── reports/                    # 자동 저장 보고서 (gitignored)
│
├── scripts/
│   ├── run_ngrok.py                    # (선택) ngrok 외부 공유
│   ├── export_checkpoint_meta.py
│   ├── upload_checkpoints_to_hf.py
│   └── training/                       # 학습 / 평가 / 분석 (런타임 분리)
│       ├── main.py                     # 학습 진입점 (MLflow + multi-seed)
│       ├── train.py                    # TRAINERS, EarlyStopping, set_seed_suffix
│       ├── evaluate.py                 # EVALUATORS + NASA Score + Confusion + PR-AUC
│       ├── reeval.py                   # 학습 없이 기존 체크포인트 재평가
│       ├── analyze_results.py          # MLflow 결과 mean±std + 3-way 비교
│       ├── compute_efficiency.py       # latency / memory / param 벤치마크
│       └── mlflow_logger.py            # MLflow wrapper
│
├── prompts/
│   └── system_pdm_assistant.txt
│
├── notebooks/
│   ├── sensor_correlation_analysis.ipynb   # iTransformer 정당화 분석
│   └── multiseed_3way_comparison.ipynb     # 3-way 모델 비교 시각화 (7개 차트)
│
└── tests/
    ├── test_smoke.py                   # Dash boot + Services smoke
    ├── test_validation_step.py
    ├── test_auth_sessions.py           # Flask login + RBAC
    ├── test_passwords.py               # werkzeug hash wrapper
    ├── realtime/                       # device_service / maintenance_order_service
    └── manual/                         # 수동 실행 step 스크립트 (pytest 미수집)
        ├── pdm_lstm_step.py
        ├── llm_stream_step.py
        └── analyze_lstm_step.py
```

데이터셋 원본, 체크포인트, 보고서, 로그, MLflow DB는 저장소에 포함되지 않습니다.

---

## 설치

요구사항: Python 3.10+

```bash
git clone https://github.com/yoosehyeon/OmniPdM-LLM.git OmniPdM
cd OmniPdM

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 환경변수

`.env.example` → `.env` 복사 후 값 입력.

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `GROQ_API_KEY` | - | Groq Cloud 발급. 미설정 시 LLM 코멘트는 rule-based fallback |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | 사용할 모델 |
| `GROQ_TEMPERATURE` | `0.2` | 기본 temperature (위험등급별 동적 조정됨) |
| `GROQ_MAX_TOKENS` | `800` | 응답 최대 토큰 |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI 호환 endpoint |
| `PROMPT_DIR` | `prompts` | 시스템 프롬프트 디렉터리 |
| `ENABLE_LLM_JUDGE` | `0` | `1` 설정 시 LLM-as-Judge 활성화 |
| `ENABLE_LLM_STREAM` | `0` | `1` 설정 시 토큰 스트리밍 사용 |
| `ENABLE_LLM_TOOLS` | `0` | `1` 설정 시 Function Calling 5 tools 활성화 |
| `CHECKPOINT_REPO` | `yusehyeon/hybridpdm-checkpoints` | HF Hub fallback (로컬 우선) |
| `DASH_HOST` | `0.0.0.0` | Dash 서버 호스트 |
| `DASH_PORT` | `8050` | Dash 서버 포트 |
| `DASH_DEBUG` | `false` | Dash debug 모드 |
| `DISABLE_MLFLOW` | `0` | `1` 설정 시 MLflow 추적 비활성화 |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow 백엔드 (PostgreSQL 가능) |

---

## 실행

### Dash UI

```bash
python app.py
```

기본 접속: `http://localhost:8050`

페이지:
- `/` — Analysis (Scalar 입력, 구현 완료)
- `/lstm` — LSTM Analysis (구현 예정)
- `/diagnostics` — Diagnostics (구현 예정)
- `/risk` — Risk Simulator (구현 예정)
- `/status` — Model Status (구현 완료)
- `/reports` — Report Browser (구현 완료)

### 학습

```bash
# 전체 데이터셋 1 epoch smoke
python -m scripts.training.main --smoke --skip-explain

# 특정 데이터셋만 본 학습 (default seed=42)
python -m scripts.training.main --datasets ai4i_cnn cmapss_lstm

# C-MAPSS FD별 비교 (BiLSTM)
python -m scripts.training.main \
    --datasets cmapss_lstm_fd001 cmapss_lstm_fd002 cmapss_lstm_fd003 cmapss_lstm_fd004 \
    --skip-explain

# Multi-seed 재현성 학습 (seed 42, 43, 44)
python -m scripts.training.main \
    --datasets cmapss_lstm cmapss_dlinear cmapss_itransformer \
    --seeds 42 43 44 \
    --skip-explain

# 모든 데이터셋 + 모든 모델 (장시간)
python -m scripts.training.main
```

### 재평가 (학습 없이 새 metric만)

evaluate.py에 새 metric (NASA Score 등)을 추가한 후 기존 체크포인트로 재계산:

```bash
python -m scripts.training.reeval
# 또는 특정 데이터셋만
python -m scripts.training.reeval --datasets cmapss_lstm cmapss_dlinear
```

별도 MLflow experiment `hybridpdm_reeval` 에 기록 (기존 학습 run 보존).

### 결과 분석 (Multi-seed mean±std + 3-way 비교)

```bash
python -m scripts.training.analyze_results
# CSV 저장
python -m scripts.training.analyze_results --csv
# 다른 metric 기준
python -m scripts.training.analyze_results --metric nasa_score_sum
```

출력:
- 모든 run 표
- `model × subset` mean±std (n=seed 개수)
- BiLSTM 기준 격차 % (DLinear / iTransformer)
- 자동 인사이트 (best/worst FD, variance 큰 subset)
- JSON 저장

### Compute Efficiency 벤치마크 (latency / memory / params)

```bash
python -m scripts.training.compute_efficiency
# 재현성 우선 (단일 스레드)
python -m scripts.training.compute_efficiency --single-thread
# GPU
python -m scripts.training.compute_efficiency --device cuda
```

3개 RUL 모델(BiLSTM / DLinear / iTransformer)을 C-MAPSS (F=14) + N-CMAPSS (F=43) 차원에서 측정. MLflow `hybridpdm_efficiency` experiment 기록.

### MLflow UI

```bash
mlflow ui
```

기본 접속: `http://localhost:5000`. 모든 학습 run 의 파라미터 / 메트릭 / 태그 비교 가능.

### ngrok 외부 공유 (선택)

```bash
python scripts/run_ngrok.py
```

---

## 데이터셋

본 프로젝트는 아래 공개 데이터셋을 사용합니다. 저장소에 포함되어 있지 않으므로 직접 다운로드 후 `models_core/dataset/` 하위 배치.

| 데이터셋 | 설명 | 다운로드 |
|---------|------|---------|
| **NASA C-MAPSS** | Turbofan Engine Degradation Simulation (RUL, FD001~FD004) | [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| **NASA N-CMAPSS** | Turbofan Degradation Simulation-2 (실측 비행 조건, 43 features) | [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| **AI4I 2020** | Milling machine predictive maintenance | [UCI ML Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) |
| **Hydraulic Systems** | 유압 시스템 상태 모니터링 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) |
| **PHM 2012 Bearing** | IEEE PHM 2012 (FEMTO-ST) | [GitHub: wkzs111/phm-ieee-2012](https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset) |
| **CWRU Bearing** | Case Western Reserve Bearing Data | [CWRU Data Center](https://engineering.case.edu/bearingdatacenter) |

배치 예시:

```text
models_core/dataset/
├── CMAPSSData/                                                # FD001~FD004
├── 17. Turbofan Engine Degradation Simulation Data Set 2/    # N-CMAPSS
├── ai4i2020.csv (또는 AI4I-PMDI.csv)
├── condition+monitoring+of+hydraulic+systems/
├── PHM2012/
└── 10987113/                                                   # CWRU
```

각 데이터셋의 라이선스·인용 조건은 원 제공처 정책을 따르십시오.

---

## 모델

### Dataset Keys (PIPELINE)

| dataset_key | Task | 모델 | 파라미터 (F=14, L=30) |
|---|---|---|---|
| `ai4i_cnn` | 이진 분류 | TabularCNN1D | — |
| `ai4i_gbdt` | 이진 분류 | GBDT (sklearn HistGradientBoosting) | — |
| `cwru_cnn` | 다중 분류 (10-class) | WDCNN1D (Wide-kernel CNN) | — |
| `hydraulic_ae` | 이상 탐지 | Denoising AE | — |
| `cmapss_lstm` / `cmapss_lstm_fd001~fd004` | RUL 회귀 | BiLSTM + Attention | **551,234** |
| `cmapss_dlinear` / `cmapss_dlinear_fd001~fd004` | RUL 회귀 | **DLinear** | **883** |
| `cmapss_itransformer` / `cmapss_itransformer_fd001~fd004` | RUL 회귀 | **iTransformer** | **401,026** |
| `ncmapss_lstm` | RUL 회귀 | BiLSTM + Attention (43 features, hidden=256) | — |
| `ncmapss_dlinear` | RUL 회귀 | DLinear | — |
| `ncmapss_itransformer` | RUL 회귀 | iTransformer | — |

### RUL 회귀 모델 비교 (C-MAPSS FD001~FD004, multi-seed)

| 모델 | 설계 철학 | 강점 | 약점 |
|---|---|---|---|
| **BiLSTM + Attention** | Temporal recurrence + 시점 가중 | dynamic change 강한 FD (003, 001) 우수 | 파라미터 많음 |
| **DLinear** | Trend + Seasonal 분해 + 채널 독립 | 매우 작음 (883), 안정적 (std 0.07~0.15) | 모든 FD에서 +22~32% RMSE 손해 |
| **iTransformer** | Variate tokenization + Cross-channel attention | mean \|corr\| 강한 FD (002, 004) 기대 | 시변 강한 FD에선 BiLSTM에 근소 열세 |

논문:
- **DLinear**: [Zeng et al. 2023, "Are Transformers Effective for Time Series Forecasting?"](https://arxiv.org/abs/2205.13504)
- **iTransformer**: [Liu et al. 2024 ICLR, "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"](https://arxiv.org/abs/2310.06625)

---

## 평가 메트릭

### 분류 (AI4I, CWRU)
- Accuracy, Precision, Recall, F1
- **Confusion Matrix** (TP/FP/FN/TN, False Alarm Rate)
- **PR-AUC, ROC-AUC** (불균형 데이터 평가)

### RUL 회귀 (C-MAPSS, N-CMAPSS)
- RMSE, MAE, R²
- **NASA PHM Score** (asymmetric scoring function, late prediction 강한 패널티 — RUL 의사결정 비용 metric)
  - d = pred - true
  - d ≥ 0 (late): exp(d/10) - 1
  - d < 0 (early): exp(-d/13) - 1

### 이상 탐지 (Hydraulic)
- F1, Precision, Recall
- Percentile grid search (85~99) + Mahalanobis distance 임계값 최적화

---

## 분석 노트북

- `notebooks/sensor_correlation_analysis.ipynb` — C-MAPSS FD001~FD004 sensor 간 cross-correlation + dynamic change + RUL Mutual Information 분석. iTransformer 도입 정당화 근거.
- `notebooks/multiseed_3way_comparison.ipynb` — BiLSTM / DLinear / iTransformer 3-way 비교 시각화 (7개 차트: RMSE/NASA bar, multi-seed box plot, correlation 예측 검증 산점도, parameter trade-off, model selection heatmap 등).

---

## 테스트

```bash
# 자동 (pytest 전체)
pytest tests/ -v

# Smoke (Dash boot + Services fallback 경로) — CI 외부 의존 0
pytest tests/test_smoke.py -v
pytest tests/test_validation_step.py -v

# 인증 / RBAC
pytest tests/test_auth_sessions.py -v
pytest tests/test_passwords.py -v

# 실시간 service (mock 단위 + DB integration — OMNIPDM_TEST_DB_URL 설정 시)
pytest tests/realtime/ -v

# 수동 step (pytest 미수집 — 직접 실행)
python -m tests.manual.pdm_lstm_step
python -m tests.manual.analyze_lstm_step
python -m tests.manual.llm_stream_step
```

---

## 향후 로드맵

- **남은 UI 페이지**: LSTM Analysis / Diagnostics / Risk Simulator 완성
- **N-CMAPSS 전체 비교**: BiLSTM + DLinear + iTransformer 학습 + 분석
- **AI4I CNN recall 개선**: Focal loss tuning, class-balanced sampling (현재 recall 0.706 → 0.85+ 목표)
- **이상 탐지 강화**: VAE + Isolation Forest 앙상블 (Hydraulic F1 0.83 → 0.90+ 기대)
- **추가 모델**: PatchTST + channel-mixing, TCN, TFT (외부 GPU 환경)
- **인프라**: Docker Compose (Dash + Postgres) → PostgreSQL 마이그레이션 → FastAPI 분리 → 실시간 워커 (MQTT/Kafka)
- **CPU 가속**: Intel Extension for PyTorch (IPEX), `torch.compile()`

---

## 라이선스

사용한 공개 데이터셋의 라이선스는 각 제공처 정책을 따릅니다. 코드 라이선스는 추후 결정.

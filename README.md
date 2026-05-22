# OmniPdM — All-in-One Predictive Maintenance System

> 스마트팩토리 다종 설비를 단일 플랫폼으로 통합하여 사전 예측 + XAI + LLM 한국어 의사결정 지원을 제공하는 예지보전 시스템.

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-84%2F84%20passed-22ff88)](#테스트)
[![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ed?logo=docker&logoColor=white)](./Dockerfile)
[![PRD](https://img.shields.io/badge/PRD-v7.0-22ff88)](./PRD.md)

<p align="center">
  <strong>∞ OmniPdM</strong> · <em>Dark Gray + Neon Green</em> · 옴니 피디엠
</p>

---

## Overview

OmniPdM은 스마트팩토리 SME(중소기업)를 위한 **온프레미스 친화적 예지보전 플랫폼**입니다. 6종 이상의 산업 데이터셋(밀링·베어링·유압·터보팬 엔진·음향)을 단일 파이프라인으로 통합하고, **RUL/Anomaly/Fault 사전 예측 + XAI 투명성 + LLM 한국어 코멘트** 3축을 모든 예측에 첨부하여 비숙련 운영자도 즉시 의사결정을 내릴 수 있도록 지원합니다.

### 3대 차별점

| 차별점 | 내용 |
|---|---|
| **사전 예측 중심** | RUL · Anomaly · Fault — 고장 발생 전 탐지 (NASA Score 의사결정 비용 metric) |
| **XAI 투명성** | Feature Importance + Attention + Temporal — 모든 예측에 기여도 첨부 |
| **LLM 한국어** | 상태 · 의심 원인 · 권장 조치 3 섹션 한국어 보고서 (Groq Llama 3.3 70B) |

---

## Core Architecture

OmniPdM은 4 계층으로 구성됩니다.

1. **Realtime Layer** — MQTT 워커 (paho-mqtt QoS=1) + NotifierChain + DbWriter
2. **Services Layer** — `analyze · explain · llm · risk · guardrail · evaluation · report` (UI 무관 도메인)
3. **API Layer** — Flask `/api/*` Blueprint 4종 (health · csrf · datasets · analyze) + 세션 인증 + RBAC
4. **Presentation Layer** — React SPA (Vite + TypeScript + Tailwind + Recharts) 정적 서빙

```text
[설비 IoT]
   │ MQTT / Kafka / OPC-UA
   ▼
[Realtime Worker] ──► [Services] ──┬──► [TimescaleDB]   telemetry · predictions · alerts · CMMS
                                   ├──► [Notifier]       Slack / Email / Telegram
                                   └──► [Flask /api/*] ──► [React SPA]
                                                              ▲
                                                       운영자 / 유지보수자 / 관리자
```

**핵심 설계 원칙**
- **services/ 순수성** — UI 의존성 0, React/FastAPI/CLI 어디서나 재사용 가능
- **API-first** — 모든 화면은 `/api/*` REST 만 호출, 모바일·서드파티 확장 용이
- **단일 origin** — React build → Flask static serve, CORS·CSRF 단순화
- **환경변수 중심** — `OMNIPDM_*` 로 dev/prod 전환 (Docker 단일 이미지)

---

## Key Features

| 기능 | 목적 | Endpoint / 모듈 |
|---|---|---|
| **Health Probe** | Liveness/Readiness (Docker healthcheck) | `GET /api/health` |
| **Dataset Catalog** | 5종 dataset 메타 (SSOT, slider 명세) | `GET /api/datasets` |
| **CSRF Token** | per-session 토큰 발급 (OWASP Synchronizer) | `GET /api/csrf` |
| **Analyze Pipeline** | 8 단계 오케스트레이션 (scalar + sequence) | `POST /api/analyze` |
| **Auth & RBAC** | Flask session + werkzeug scrypt + role guard | `POST /login`, `GET /logout` |
| **Realtime Worker** | MQTT subscribe → predict → DB + alarm | `services/realtime/mqtt_worker.py` |
| **LLM + Guardrail** | Groq Llama 3.3 70B + 금지표현 치환 + Judge | `services/llm_service.py` |
| **XAI Explanation** | Integrated Gradients + Attention + Temporal | `services/explain_service.py` |
| **CMMS** | devices · orders · status_history · audit_log | `services/realtime/*_service.py` |
| **MLflow Tracking** | 학습 run 파라미터/메트릭/태그 비교 | `mlflow ui` (:5000) |

---

## Technology Stack

**Frontend**
![React](https://img.shields.io/badge/React-18-61dafb?logo=react&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-6-646cff?logo=vite&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178c6?logo=typescript&logoColor=white)
![Tailwind](https://img.shields.io/badge/Tailwind-3-06b6d4?logo=tailwindcss&logoColor=white)
![Recharts](https://img.shields.io/badge/Recharts-2-22ff88)

**Backend**
![Flask](https://img.shields.io/badge/Flask-3-000000?logo=flask&logoColor=white)
![Gunicorn](https://img.shields.io/badge/Gunicorn-22-499848?logo=gunicorn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2-ee4c2c?logo=pytorch&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3-f7931e?logo=scikitlearn&logoColor=white)

**LLM / Tracking**
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-f55036)
![MLflow](https://img.shields.io/badge/MLflow-2-0194e2?logo=mlflow&logoColor=white)

**Data / Realtime**
![TimescaleDB](https://img.shields.io/badge/TimescaleDB-Postgres_16-fdb515?logo=postgresql&logoColor=white)
![Mosquitto](https://img.shields.io/badge/Mosquitto-MQTT_v2-3c5280)
![psycopg](https://img.shields.io/badge/psycopg-3-336791)

**Infra**
![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ed?logo=docker&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-9-0a9edc?logo=pytest&logoColor=white)

---

## Quick Start

### Prerequisites

- Python **3.10+**
- Node.js **20+** (frontend 빌드)
- Docker & Docker Compose (선택, 운영 환경)
- (선택) Groq API key — [console.groq.com](https://console.groq.com/keys)

### 1) Clone & 환경 설정

```bash
git clone https://github.com/yoosehyeon/OmniPdM-LLM.git OmniPdM
cd OmniPdM

# Python venv
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate         # macOS/Linux

pip install -r requirements.txt

# .env 생성 (Groq key 등 채움)
cp .env.example .env
```

### 2) 개발 모드 (Vite HMR + Flask)

```bash
# 터미널 1 — Flask 백엔드 (:8050)
python app.py

# 터미널 2 — React dev server (HMR, :5173)
cd frontend
npm install      # 최초 1회
npm run dev
```

브라우저: **http://localhost:5173** — Vite 가 `/api/*` 를 Flask 로 프록시.

### 3) 프로덕션 빌드 (단일 origin)

```bash
cd frontend && npm run build
cd .. && python app.py
# 접속: http://localhost:8050
```

### 4) Docker (DB + MQTT 포함)

```bash
docker compose up -d              # app + timescaledb + mosquitto
docker compose logs -f app
docker compose down -v            # 정리
```

---

## Project Structure

```text
OmniPdM/
├── app.py                              # Flask 진입점 + SPA 정적 서빙
├── Dockerfile                          # multi-stage: node:20 → python:3.11
├── docker-compose.yml                  # app + timescaledb + mosquitto
├── PRD.md                              # Product Requirements v7.0
│
├── frontend/                           # React SPA (Vite + TS + Tailwind)
│   ├── package.json
│   ├── vite.config.ts                  # /api/* → :8050 proxy
│   └── src/
│       ├── App.tsx · main.tsx
│       ├── csrf.ts                     # CSRF singleton + postWithCsrf
│       ├── data.ts                     # fetchDatasets(), CORE_MODELS, SAMPLE_HISTORY
│       ├── types.ts                    # 도메인 타입
│       ├── components/                 # Navbar · KpiCard · LlmCard · RiskGauge
│       └── pages/                      # Home · Analysis · Diagnostics · Simulator · Reports
│
├── services/                           # 도메인 서비스 (UI 무관)
│   ├── api/                            # Flask Blueprint
│   │   ├── health.py · csrf.py · datasets.py · analyze.py
│   │   └── _sequence_synth.py
│   ├── analyze_service.py              # 8 단계 오케스트레이션
│   ├── pdm_service.py                  # CNN / GBDT / AE / BiLSTM / DLinear / iTransformer
│   ├── risk_service.py                 # weighted / noisy_or / max 융합
│   ├── explain_service.py              # XAI (Importance · Attention · Temporal)
│   ├── llm_service.py                  # Groq + streaming + function calling
│   ├── llm_tools.py                    # 5 tools + ToolDispatcher
│   ├── guardrail_service.py · evaluation_service.py · report_service.py
│   ├── input_validation_service.py · schemas.py
│   ├── dataset_catalog.py              # SSOT — UI 슬라이더 메타
│   ├── auth/                           # passwords · sessions · routes · users
│   └── realtime/                       # mqtt_worker · db_writer · notifier · device · order
│
├── models_core/                        # 모델 / 데이터 런타임
│   ├── config.py                       # LSTM_CFG, DLINEAR_CFG, ITRANSFORMER_CFG
│   ├── models.py                       # WDCNN1D / TabularCNN1D / AE / BiLSTM / DLinear / iTransformer
│   ├── data_pipeline.py                # 6+ dataset_key loaders
│   └── risk_score.py
│
├── scripts/
│   └── training/                       # main · train · evaluate · reeval · analyze_results
│
├── infra/
│   ├── timescaledb/                    # init.sql + 001_cmms_schema.sql
│   └── mosquitto/                      # mosquitto.conf
│
└── tests/                              # 84 tests, 외부 의존 0
    ├── test_smoke.py · test_validation_step.py
    ├── test_api_health.py · test_api_csrf.py · test_api_datasets.py · test_api_analyze.py
    ├── test_auth_sessions.py · test_passwords.py
    ├── test_e2e_pipeline.py            # 통합 E2E
    └── realtime/                       # mock + DB integration
```

---

## Configuration

`.env.example` 을 `.env` 로 복사 후 값 입력. **상세 변수는 `.env.example` 참조.**

### Flask + 인증

| 변수 | 기본 | 설명 |
|---|---|---|
| `OMNIPDM_HOST` | `0.0.0.0` | Flask bind host |
| `OMNIPDM_PORT` | `8050` | Flask bind port |
| `OMNIPDM_DEBUG` | `false` | Flask debug 모드 |
| `OMNIPDM_SECRET_KEY` | (random) | 세션 쿠키 서명. **운영 필수** |
| `OMNIPDM_AUTH_REQUIRED` | `true` | `false` 시 가드 비활성 (dev/test) |
| `OMNIPDM_SESSION_COOKIE_SECURE` | `false` | HTTPS 운영 시 `true` |

### LLM (Groq)

| 변수 | 기본 | 설명 |
|---|---|---|
| `GROQ_API_KEY` | — | 미설정 시 rule-based fallback |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | |
| `GROQ_TEMPERATURE` | `0.2` | 위험등급별 동적 조정됨 |
| `ENABLE_LLM_JUDGE` | `0` | `1` 시 LLM-as-Judge 활성 |
| `ENABLE_LLM_STREAM` | `0` | `1` 시 토큰 스트리밍 |
| `ENABLE_LLM_TOOLS` | `0` | `1` 시 Function Calling 5 tools |

### Data / Realtime

| 변수 | 기본 | 설명 |
|---|---|---|
| `OMNIPDM_DB_HOST` | `127.0.0.1` | docker 사용 시 `timescaledb` |
| `OMNIPDM_MQTT_HOST` | `localhost` | docker 사용 시 `mosquitto` |
| `OMNIPDM_ALERT_MIN_LEVEL` | `Critical` | Critical/Warning 만 외부 알람 |
| `DISABLE_MLFLOW` | `0` | `1` 시 MLflow 비활성 |

---

## Datasets

`models_core/dataset/` 하위 배치. 라이선스는 각 제공처 정책을 따름.

| 데이터셋 | 설명 | 다운로드 |
|---|---|---|
| **NASA C-MAPSS** | Turbofan RUL (FD001~FD004) | [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| **NASA N-CMAPSS** | 실측 비행 조건, 43 features | [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| **AI4I 2020** | Milling machine PdM | [UCI ML](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) |
| **Hydraulic Systems** | 유압 상태 모니터링 | [UCI ML](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) |
| **PHM 2012 Bearing** | IEEE PHM 2012 (FEMTO-ST) | [GitHub](https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset) |
| **CWRU Bearing** | Case Western Reserve | [CWRU Data Center](https://engineering.case.edu/bearingdatacenter) |

---

## Models

### Dataset Keys × 모델

| dataset_key | Task | 모델 | 파라미터 (F=14, L=30) |
|---|---|---|---|
| `ai4i_cnn` / `ai4i_gbdt` | 이진 분류 | TabularCNN1D / HistGradientBoosting | — |
| `cwru_cnn` | 10-class 분류 | WDCNN1D (Wide-kernel) | — |
| `hydraulic_ae` | 이상 탐지 | Denoising AE | — |
| `cmapss_lstm_*` | RUL 회귀 | **BiLSTM + Attention** | **551,234** |
| `cmapss_dlinear_*` | RUL 회귀 | **DLinear** | **883** |
| `cmapss_itransformer_*` | RUL 회귀 | **iTransformer** | **401,026** |
| `ncmapss_*` | RUL 회귀 | 위 3종 (43 features, hidden=256) | — |

### RUL 모델 비교

| 모델 | 설계 철학 | 강점 | 약점 |
|---|---|---|---|
| **BiLSTM + Attention** | Temporal recurrence + 시점 가중 | dynamic FD (003/001) 우수 | 파라미터 많음 |
| **DLinear** | Trend+Seasonal 분해, 채널 독립 | 매우 가벼움 (883), 안정 (std 0.07~0.15) | +22~32% RMSE 손해 |
| **iTransformer** | Variate token + Cross-channel attn | corr 강한 FD (002/004) 기대 | 시변 강한 FD 근소 열세 |

📄 [Zeng et al. 2023, DLinear](https://arxiv.org/abs/2205.13504) · [Liu et al. 2024 ICLR, iTransformer](https://arxiv.org/abs/2310.06625)

---

## Evaluation Metrics

| 카테고리 | 메트릭 |
|---|---|
| **분류** (AI4I, CWRU) | Accuracy · Precision · Recall · F1 · Confusion Matrix · **PR-AUC / ROC-AUC** |
| **RUL 회귀** (C-MAPSS, N-CMAPSS) | RMSE · MAE · R² · **NASA PHM Score** (asymmetric, late prediction 강 패널티) |
| **이상 탐지** (Hydraulic) | F1 · Precision · Recall · Percentile grid search (85~99) · Mahalanobis |

NASA PHM Score 공식:
```
d = pred - true
d ≥ 0 (late):  exp(d/10)  - 1
d <  0 (early): exp(-d/13) - 1
```

---

## Training

```bash
# Smoke (전체 dataset, 1 epoch)
python -m scripts.training.main --smoke --skip-explain

# 본 학습 (특정 dataset)
python -m scripts.training.main --datasets ai4i_cnn cmapss_lstm

# Multi-seed 재현성
python -m scripts.training.main \
    --datasets cmapss_lstm cmapss_dlinear cmapss_itransformer \
    --seeds 42 43 44 --skip-explain

# 학습 없이 새 metric 재계산
python -m scripts.training.reeval --datasets cmapss_lstm

# 결과 분석 (mean±std + 3-way 비교)
python -m scripts.training.analyze_results --csv

# Compute Efficiency 벤치마크
python -m scripts.training.compute_efficiency

# MLflow UI
mlflow ui   # http://localhost:5000
```

---

## 테스트

```bash
# 전체 (84 tests)
pytest tests/ -v --ignore=tests/manual --ignore=tests/realtime

# 카테고리별
pytest tests/test_smoke.py -v               # Flask boot + Services
pytest tests/test_api_*.py -v               # /api/health · csrf · datasets · analyze
pytest tests/test_auth_sessions.py -v       # 로그인 + RBAC
pytest tests/test_e2e_pipeline.py -v        # E2E (login → CSRF → analyze)

# 실시간 (DB integration — OMNIPDM_TEST_DB_URL 설정 시)
pytest tests/realtime/ -v

# 수동 step (직접 실행)
python -m tests.manual.pdm_lstm_step
python -m tests.manual.analyze_lstm_step
python -m tests.manual.llm_stream_step
```

**CI 외부 의존 0** — DB/MQTT/모델 체크포인트 부재 환경에서도 통과.

---

## Security Standards

| 항목 | 구현 |
|---|---|
| **CSRF** | per-session token (`secrets.token_urlsafe(32)`) + `X-CSRF-Token` 헤더 + Cache-Control no-store |
| **Session** | HttpOnly · SameSite=Lax · Secure (운영) — OWASP Session Mgmt 준수 |
| **Password** | werkzeug scrypt (3.x 기본) |
| **RBAC** | `role_required` decorator — admin / operator / viewer |
| **Audit Log** | 로그인 성공/실패, RBAC reject 전부 기록 (`audit_log` 테이블) |
| **Validation** | Sensor range/type + SSOT key whitelist (`dataset_catalog`) |

---

## Roadmap

**Phase 2** (트리거 조건 충족 시)
- Grafana 재도입 (운영자 5명+ 또는 외부 시스템 연동 필요)
- WebSocket 실시간 푸시 (현재 polling 충분)
- FastAPI 분리 (Flask 처리량 한계 도달 시)
- ONNX Edge 추론 (제조 현장 PLC 직배포)

**Phase 3** (상용화)
- SaaS 구독제 (설비당 월 과금)
- 온프레미스 라이선스 + ERP/MES 연동 컨설팅
- 다국어 (영/일/중) 확장

**기술 부채 백로그**
- N-CMAPSS 전체 비교 (BiLSTM + DLinear + iTransformer)
- AI4I CNN recall 0.706 → 0.85+ (Focal loss + class-balanced sampling)
- Hydraulic 이상 탐지 F1 0.83 → 0.90+ (VAE + Isolation Forest 앙상블)
- CPU 가속 (Intel Extension for PyTorch, `torch.compile()`)

---

## Documentation

- [`PRD.md`](./PRD.md) — Product Requirements v7.0
- [`docs/DEV_GUIDE_REACT_MIGRATION.md`](./docs/DEV_GUIDE_REACT_MIGRATION.md) — Dash → React 마이그레이션 가이드
- [`docs/PRESENTATION.md`](./docs/PRESENTATION.md) — 20-slide 발표자료

---

## License

코드 라이선스는 추후 결정. 사용한 공개 데이터셋의 라이선스는 각 제공처 정책을 따릅니다.

---

<p align="center">
  <sub><strong>OmniPdM</strong> — All-in-One PdM System · 유세현 (YU SEHYEON) · 2026</sub>
</p>

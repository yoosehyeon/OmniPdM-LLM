# OmniPdM — Product Requirements Document (v6.3)

> **상태**: 발표 종료 후 개인 프로젝트 (2026-05-15) → 실무 도입 가능성 강화 (2026-05-19) → 학술 1차 완료 + Tier 1 절반 완료 (2026-05-19)
> **본질적 동기**: 스마트팩토리 설비의 **사전 예측 + 고장 손해 최소화**
> **최종 갱신**: 2026-05-19 (v6.3 — 학술 결과 §8 신설, Tier 1 T1-01/T1-02 완료 반영)
> **이전 코드네임**: HybridPdM (~v6.1까지) / 본 문서부터 **OmniPdM** 으로 통합

---

# 0. 브랜드 아이덴티티 (OmniPdM)

## 0.1 제품명: **OmniPdM** (옴니 피디엠)
**All-in-One PdM System**

## 0.2 네이밍 정당성
- **Omni-** (라틴어 접두사): "모든 것 / 전체"
- 분산된 개별 산업 설비(Milling / Bearing / Hydraulic / Turbofan / Acoustic 등)의 센서 데이터를 **하나의 시스템으로 통합**
- 데이터 수집 → 예측(RUL/Anomaly/Fault) → XAI → LLM 한국어 설명 → 위험 등급 → 알람 → 의사결정 지원 **전 과정을 아우르는 전방위적 예지보전 플랫폼**
- 기존 코드네임 "HybridPdM" 의 "Hybrid (LSTM+LLM)" 가치는 그대로 계승하되, **다종 설비 통합** + **풀스택 PdM 파이프라인** 의 외연을 더 정확히 표현

## 0.3 로고 디자인 컨셉
- **심볼**: 무한대(∞) 기호를 디지털 회로 라인 형태로 각지게 변형 → 공장 전체 설비가 끊임없이 하나로 연결되어 모니터링되는 모습
- **컬러**: Dark Gray (정밀함) + Neon Green (스마트팩토리의 생동감)
- **타이포**: "OMNIPDM" 굵은 산세리프 + 하단 "All-in-One PdM System" 서브카피
- **파일 위치 (권장)**: [assets/omnipdm_logo.png](assets/omnipdm_logo.png) — Dash app/README 에서 동일 경로 참조

## 0.4 브랜드 사용 가이드 (요약)
| 위치 | 표기 |
|---|---|
| 문서 제목 / 헤더 | **OmniPdM** |
| 한글 표기 | 옴니 피디엠 |
| 풀네임 | OmniPdM — All-in-One PdM System |
| 학술 인용 시 | OmniPdM (formerly HybridPdM) — 첫 등장에만 병기 |
| GitHub 저장소 | `OmniPdM-LLM` (2026-05-19 `hybridpdm-LLM` → 이 이름으로 리네이밍, GitHub redirect 자동) |
| 로컬 디렉터리 | `OmniPdM/` (2026-05-19 `hybridpdm_gradio` 에서 리네이밍) |
| Python 패키지 네임스페이스 | `models_core`, `services`, `scripts` 그대로 유지 — 디렉터리/리포 이름과 무관 |

---

# 1. Executive Summary

## 1.1 제품명
**OmniPdM** — All-in-One Predictive Maintenance System with LLM Analysis

## 1.2 한 줄 정의
**스마트팩토리의 다종 설비를 하나의 플랫폼으로 통합하여, 상태를 사전 예측하고 고장 발생 전 운영자에게 한국어 의사결정 지원을 제공하는 전방위적 예지보전 시스템.**

## 1.3 시장 Pain Point — 본 시스템이 정조준하는 3대 문제

2026년 한국 스마트팩토리 SME 가 PdM 도입에서 실제로 막히는 지점은 다음 셋이다. OmniPdM 의 모든 설계 결정은 이 3개를 직접 해소하기 위한 것이다.

1. **숙련공 부족 — "고장 신호를 알아볼 수 있는 사람"의 절대 부족**
   대응: 위험 등급 자동 분류(Critical/Warning/Advisory/Normal) + LLM 한국어 코멘트로 비숙련 운영자도 즉시 조치 가능.
2. **블랙박스 AI — 현장이 이유 없이 울리는 알람을 신뢰하지 않음**
   대응: XAI 기여도(Feature Importance / Attention) 를 모든 예측에 첨부 → "왜 위험한가" 가 알람과 같이 도착.
3. **한국어 실무 갭 — 영어 논문/도구 중심, 현장 보고서 직결 미비**
   대응: LLM 응답·UI 라벨·알람 본문 전부 한국어. 보고서는 그대로 사내 결재에 첨부 가능 수준.

## 1.4 비즈니스 가치 제안

> **"사람 투입 감소 + 지속 생산"을 추구하는 스마트팩토리 환경에서, 다양한 설비를 단일 플랫폼(Omni-)으로 통합 모니터링하여 고장 발생 후 수리 손해를 사전 예측으로 최소화한다.**

### 핵심 가치 흐름
```
[설비 센서 데이터]
      ↓ (실시간 또는 batch)
[딥러닝 사전 예측]     ← failure probability, anomaly score, RUL
      ↓
[XAI 설명]            ← Feature Importance / Attention / Temporal
      ↓
[LLM 한국어 코멘트]    ← 현장 작업자가 즉시 이해 가능
      ↓
[위험 등급 자동 분류]  ← Critical / Warning / Advisory / Normal
      ↓
[운영자 통지 + 의사결정 지원]
```

### 기존 시스템 대비 차별점
1. **사전 예측**: 사후 대응이 아닌 고장 전 예측 (RUL / Anomaly)
2. **고장 손해 직결**: **NASA Score** (asymmetric, late prediction 강하게 패널티) — RMSE 가 아닌 의사결정 비용 metric
3. **현장 작업자 친화**: LLM 한국어 (상태 / 의심 원인 / 권장 조치 3섹션)
4. **자동 의사결정 지원**: Risk Level 자동 등급 분류
5. **다양한 설비 대응**: 6+ 데이터셋 (밀링 / 베어링 / 유압 / 엔진)

---

# 2. 실무 도입 시나리오 (Use Cases) — 신규 섹션

## 2.1 페르소나 1 — 운영자 (24시간 모니터링)

**시나리오**:
- 통합 대시보드에서 모든 설비 위험도 한눈에 확인
- Critical 등급 발생 → **Slack/모바일 알림** 즉시
- LLM 코멘트로 즉시 조치 가능 여부 판단
- 의심 원인 Top-3 + 권장 조치 우선순위 제공

**요구 시스템**:
- 실시간 데이터 수신 (현재 갭)
- 알람 채널 (Slack/이메일/푸시) (현재 갭)
- Dash 대시보드 (✅ 구현 완료)

## 2.2 페르소나 2 — 유지보수자 (현장 대응)

**시나리오**:
- 알람 받고 현장 출동
- 모바일에서 LLM 보고서로 의심 원인 + 조치 우선순위 확인
- 작업 완료 후 결과 보고 → 모델 학습 데이터 누적
- 부품 재고 확인 + 자동 발주 제안 받음

**요구 시스템**:
- 모바일 친화 UI (Dash responsive 가능)
- 작업 이력 추적 (현재 갭)
- ERP/CMMS 연동 (현재 갭)

## 2.3 페르소나 3 — 관리자 (장기 트렌드)

**시나리오**:
- 주간/월간 리포트: 설비별 위험도 추이
- MTBF (Mean Time Between Failures) 개선 추적
- 모델 정확도 모니터링 → 재학습 의사결정
- 부서별 KPI: 알람 정확도, 작업 지시 자동화율

**요구 시스템**:
- 시계열 이력 (현재 갭)
- 트렌드 시각화 (현재 갭, Grafana 권장)
- 모델 드리프트 모니터링 (현재 갭)

---

# 3. 현장 도입 시 필요한 통합 요구사항 — 신규 섹션

## 3.1 데이터 수신 인터페이스
| 프로토콜 | 우선순위 | PoC 가능성 |
|---|---|---|
| MQTT (산업 IoT 표준) | ⭐⭐⭐⭐⭐ | ✅ Mosquitto 로컬 (무료) |
| Kafka (대용량 스트리밍) | ⭐⭐⭐ | ✅ Docker 로컬 |
| OPC-UA (PLC 표준) | ⭐⭐⭐⭐ | ✅ Open62541 시뮬레이터 |
| Modbus (레거시 PLC) | ⭐⭐ | ⚠️ 시뮬레이터 가능하나 제한적 |
| REST API (외부 IoT 플랫폼) | ⭐⭐⭐ | ✅ FastAPI |

## 3.2 알람 채널
| 채널 | 비용 | 구현 난이도 |
|---|---|---|
| Slack Webhook | 무료 | 매우 쉬움 |
| 이메일 (SMTP) | 무료 | 쉬움 |
| Pushover / Pushbullet (모바일) | 무료 tier | 중간 |
| Discord Webhook | 무료 | 매우 쉬움 |
| SMS | 유료 | 비싸므로 보류 |

## 3.3 ERP/MES/CMMS 연동
- 실제 상용 시스템 (SAP, Oracle) 연동은 라이선스/비용 제약 → **Mock REST API 로 PoC**
- 오픈소스 MES (OpenMES) 도 옵션
- 작업 지시 자동 발행 / 부품 재고 / 작업 이력 동기화

## 3.4 권한 / 인증
- 운영자 / 유지보수자 / 관리자 / 관리책임자 RBAC
- JWT + FastAPI Security 무료 구현 가능
- Keycloak 로컬 (선택)

## 3.5 모델 운영
- **드리프트 모니터링**: 시간 지나면 모델 성능 변동 → 재학습 트리거
  - Evidently AI (오픈소스) 활용
- **Confidence 핸들링**: 모델 신뢰도 낮을 때 인간 검토 단계
- **A/B 테스트**: 새 모델 vs 구 모델 점진 교체

## 3.6 Edge Deployment (선택)
- 네트워크 단절 시 로컬 추론
- ONNX 양자화 → CPU/Raspberry Pi 추론
- 추후 검토

---

# 4. 프로젝트 단계 (2026-05-19 기준)

## 4.1 완료된 마일스톤

| 마일스톤 | 완료 |
|---|---|
| 발표 (Gradio 단일 앱) | 2026-05-15 |
| Gradio → Dash 6 페이지 전환 | 2026-05-15 |
| MLflow 실험 추적 | 2026-05-15 |
| 학술 metric 보강 (NASA Score / Confusion / PR-AUC) | 2026-05-18 |
| C-MAPSS FD002~FD004 baseline | 2026-05-18 |
| Multi-seed (3 seeds) BiLSTM + DLinear | 2026-05-18~19 |
| iTransformer 모델 추가 + FD001~004 학습 | 2026-05-19 |
| Sensor correlation 분석 (H3 가설 수립) | 2026-05-19 |
| **H3 가설 반증 발견** (iTransformer FD002/004 압도적 열세) | 2026-05-19 |
| iTransformer multi-seed (seed 42·43·44) 완료 + H3 multi-seed 확정 반증 | 2026-05-19 |
| **Phase B (Hydraulic 7-combo 앙상블)** — AE/VAE/IF + 페어/트리오 (seed 42, 43) | 2026-05-19 |
| **Phase C (AI4I CNN recall variant)** — 3 seeds 학습 완료 | 2026-05-19 |
| **OmniPdM 리브랜딩** (PRD v6.2 §0 브랜드 아이덴티티 신설) | 2026-05-19 |
| **Tier 1 T1-01/T1-02 1차 구현** — MQTT 워커 + Notifier protocol + e2e smoke | 2026-05-19 |

## 4.2 현재 진행 중 (2026-05-20)
- Tier 1 T1-01 ~ T1-04 완료 (실시간 워커 + 알람 + TimescaleDB + Grafana 2종 대시보드)
- 외부 코드 리뷰 항목 #6 (Report path 경화) + #8 (CI 스코프 확장) 처리 완료
- 외부 PRD 개선 보고서 B-1 (Grafana KMPRO 톤) + B-2 (Pain Point 프레이밍) + B-3 (overfit_score) 흡수 완료
- 다음: **CMMS Tier 1.5** (설비 마스터 + 상태 머신 + 수리 워크플로 + 인증/권한) — Dash 강화 방향으로 진행

## 4.3 명시적 제외
| 항목 | 이유 |
|---|---|
| HF Spaces 배포 | 발표 종료, 시연 목적 없음 |
| 발표 시연 안정성 우선 | 학술 검증 + 실무 도입으로 전환 |
| 상용 ERP/MES 라이선스 통합 | 비용 / 산학협력 필요 |
| 실제 공장 raw 데이터 확보 | NDA / 보안 제약 |

---

# 5. 사용자 환경 및 제약

- **하드웨어**: Galaxy Book 4 Pro (Intel Core Ultra 7 155H, CPU only)
- **OS**: Windows 11
- **Python**: 3.10+
- **메모리**: 16~32GB
- **비용 정책**: 모든 도구 **무료** (Dash / Plotly / MLflow / PostgreSQL / Mosquitto / Grafana 등)

---

# 6. 시스템 아키텍처 (현재 + 목표)

## 6.1 현재 (Batch / 분석 도구)
```
[User Input (수동)]
     ↓
InputValidationService → PdmService → RiskService → ExplainService
     ↓
LlmService → GuardrailService → EvaluationService
     ↓
PlotService / ReportService → [Dash UI]
                            → [MLflow Tracking]
                            → [File Reports]
```

## 6.2 목표 (실시간 PdM 시스템)
```
[설비 IoT (시뮬레이터)] → MQTT broker → [실시간 Worker]
                                              ↓
                              [services/ 그대로 호출]
                                              ↓
                              [PostgreSQL (TimescaleDB)]
                                              ↓
                          ┌─────────┴─────────┐
                          ↓                   ↓
                  [Slack/Email 알람]    [Grafana 대시보드]
                                              ↓
                                       [Dash UI (분석)]
```

**핵심**: `services/` 계층은 그대로 재사용 — UI/Worker/CLI 어디에서나 동일 코드.

---

# 7. 모델 및 데이터셋

## 7.1 현재 사용 데이터셋 (6종)

| dataset_key | 설비 | Task |
|---|---|---|
| `ai4i_cnn` / `ai4i_gbdt` | 밀링머신 | 이진 분류 |
| `cwru_cnn` | 베어링 진동 | 10-class 분류 |
| `hydraulic_ae` | 유압 시스템 | 이상 탐지 |
| `cmapss_lstm` × 4 FDs | 터보팬 엔진 | RUL 회귀 |
| `cmapss_dlinear` × 4 FDs | (동일) | RUL baseline |
| `cmapss_itransformer` × 4 FDs | (동일) | RUL 비교 |
| `ncmapss_*` | N-CMAPSS (43 features) | RUL 확장 |

## 7.2 추가 도입 가능 데이터셋 (Tier별 우선순위)

| 데이터셋 | 도메인 | 가치 | Tier |
|---|---|---|---|
| **MIMII** | 산업 음향 (펌프/팬/슬라이더/밸브) | 공장 설비 직접 | **T1** |
| **Bosch Production Line** | 제조 라인 불량 (Kaggle) | 실제 공장 라인 | **T2** |
| **SECOM** | 반도체 (1:14 극단 불균형) | 불균형 검증 | T2 |
| **TEP** | 화학 공정 이상 | 화학공장 벤치마크 | T2 |
| **NASA Battery** | 리튬이온 RUL | 다른 도메인 검증 | T3 |
| **Wind Turbine SCADA** | 풍력발전 | 발전 설비 | T3 |
| **AI Hub 산업공정 불량** | 한국어 도메인 | 국내 활용 | T3 (신청 필요) |

---

# 8. 학술 결과 종합 (2026-05-19)

## 8.1 RUL 회귀 (C-MAPSS) — 3-way 비교
| 모델 | Params (F=14) | FD001 RMSE | 평가 |
|---|---|---|---|
| BiLSTM + Attention | 551,234 | 13.75 ± 0.68 (n=3) | 가장 robust |
| DLinear | **883** | 16.77 ± 0.07 (n=3) | baseline, 안정적 |
| iTransformer (PE-free) | 401,026 | 14.02 ± 변동 (n=3) | FD001 OK, FD002/004 압도적 열세 |

**H3 가설 결론**: multi-seed (n=3) 로 확정 반증. FD002/FD004 에서 iTransformer RMSE 가 BiLSTM 대비 +100% 안팎. operating condition diversity 가 sensor correlation 보다 중요 변수.

## 8.2 Hydraulic 이상 탐지 — 7-combo 앙상블 (Phase B-7)
multi-seed n=2 (seed 42, 43). PR-Curve 기반 validation threshold + Rank fusion.

| 조합 | test F1 (seed 42 / 43) | mean | PR-AUC mean |
|---|---|---|---|
| **IF (단독)** | **0.9194 / 0.9106** | **0.9150** | 0.932 |
| AE+IF | 0.9106 / 0.9114 | 0.9110 | 0.935 |
| AE+VAE+IF | 0.8747 / 0.8585 | 0.8666 | 0.926 |
| AE (단독) | 0.8793 / 0.8793 | 0.8793 | 0.928 |
| VAE+IF | 0.8441 / 0.8197 | 0.8319 | 0.905 |
| AE+VAE | 0.8199 / 0.8151 | 0.8175 | 0.902 |
| VAE (단독) | 0.7949 / 0.7875 | 0.7912 | 0.819 |

**결론**: 목표 F1 ≥ 0.90 **달성** (IF 0.915, AE+IF 0.911). 단일 IF 가 가장 효율적 — 앙상블의 모델 다양성 vs 단일 강모델의 trade-off 에서 후자가 우세.

## 8.3 AI4I CNN recall variant (Phase C)
focal loss α=0.92 γ=3.0, decision threshold 0.20, 70 epochs, 3 seeds.

| seed | accuracy | precision | recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| 42 | 0.976 | 0.653 | 0.627 | 0.640 | 0.704 | 0.974 |
| 43 | 0.974 | 0.594 | 0.745 | 0.661 | 0.633 | 0.969 |
| 44 | 0.977 | 0.627 | 0.824 | 0.712 | 0.783 | 0.977 |
| **mean ± std** | 0.976 ± 0.001 | 0.625 ± 0.024 | **0.732 ± 0.082** | 0.671 ± 0.030 | 0.707 ± 0.061 | 0.973 ± 0.003 |

baseline (recall 0.706) 대비 **+3.7%p**. 목표 0.85+ 미달, variance 큼 (0.627 ~ 0.824). focal loss 강도 조정 효과는 marginal, 다음 단계는 클래스 reweighting + Optuna 탐색 (향후 작업).

---

# 9. 평가 메트릭

## 9.1 학술 metric (완료)
- 분류: Accuracy / Precision / Recall / F1 / **PR-AUC / ROC-AUC / Confusion Matrix / False Alarm Rate**
- RUL 회귀: RMSE / MAE / R² / **NASA PHM Score (asymmetric)**
- 이상 탐지: F1 + Percentile grid search + Mahalanobis

## 9.2 실무 KPI (신규 — 도입 후 측정 가능)
- **MTBF 개선율** (Mean Time Between Failures)
- **Critical 알람 응답 시간** (목표 < 5분)
- **고장 방지율** (전체 alert 중 실제 고장 예방)
- **작업 지시 자동화율** (수동 vs 자동)
- **False Alarm Rate** (현장 무시되는 비율, 목표 < 5%)
- **모델 드리프트 발견 → 재학습 lead time**

---

# 10. PoC 도달 가능 범위 (Tier 우선순위) — 신규 섹션

## 🥇 Tier 1 — 무료 + 큰 임팩트 (1~2주)

**핵심: "분석 도구 → 실시간 예지보전 시스템" 격 변경**

| ID | 작업 | 시간 | 동기 정합도 | 상태 (2026-05-19) |
|---|---|---|---|---|
| T1-01 | **MQTT 시뮬레이터 + 실시간 워커** (Mosquitto + Python) | 3~4일 | 최상 | 완료 — paho-mqtt + DI worker, broker 무관 e2e 검증 |
| T1-02 | **알람 시스템** (Critical 등급 발생 시, Notifier protocol) | 1일 | 최상 | 완료 — StdoutNotifier 기본, 채널 pluggable |
| T1-03 | **PostgreSQL + TimescaleDB 이력 저장** | 2일 | 상 | 완료 — psycopg v3 + autocommit + 3 hypertable (telemetry/predictions/alerts) e2e 검증 |
| T1-04 | **Grafana 대시보드** (Fleet Overview + Device Detail 2종) | 2일 | 상 | 완료 — datasource/dashboard provisioning 자동화, 색상 일관성, drill-down 연동 |

## 🥈 Tier 2 — 데이터 확장 + 신뢰성 (3~4주)

| ID | 작업 | 시간 | 동기 정합도 |
|---|---|---|---|
| T2-01 | **MIMII 음향 데이터셋 추가** (공장 설비 직접) | 4~5일 | ⭐⭐⭐⭐⭐ |
| T2-02 | **Bosch Production Line** (Kaggle) | 3일 | ⭐⭐⭐⭐ |
| T2-03 | **Evidently AI 드리프트 모니터링** | 3일 | ⭐⭐⭐⭐ |
| T2-04 | **모델 confidence 임계값 + 인간 검토** | 2일 | ⭐⭐⭐ |

## 🥉 Tier 3 — 통합성 (2~3개월, 선택)

| ID | 작업 | 시간 | 동기 정합도 |
|---|---|---|---|
| T3-01 | Mock ERP/MES API + 자동 작업 지시 | 1주 | ⭐⭐⭐ |
| T3-02 | JWT 권한 시스템 | 4~5일 | ⭐⭐ |
| T3-03 | OPC-UA 시뮬레이터 (Open62541) | 1주 | ⭐⭐⭐⭐ |
| T3-04 | ONNX 양자화 + Edge 시뮬 | 1주 | ⭐⭐⭐ |
| T3-05 | TEP / SECOM 데이터셋 추가 | 4~5일 | ⭐⭐⭐ |

## ❌ 추구할 가치 낮음 (개인 프로젝트 범위 외)
- 실제 공장 raw 데이터 확보 (NDA)
- 상용 ERP/MES 연동 (라이선스 비용)
- 인증 평가 (ISO 13374 등)
- 대규모 동시 트래픽 (인프라 비용)

---

# 11. 기능 요구사항 (Functional Requirements)

## 11.1 Must Have (P0) — 완료 또는 진행 중

| ID | 기능 | 상태 |
|---|---|---|
| FR-001 | Scalar 입력 → 고장 예측 + 위험등급 + 설명 + LLM | ✅ |
| FR-002 | Sequence 입력 → RUL 회귀 (3 모델군) | ✅ |
| FR-003 | Guardrail (구조 + 금지표현) | ✅ |
| FR-004 | NASA Score + Confusion + PR-AUC | ✅ |
| FR-005 | Multi-seed 재현성 학습 | ✅ |
| FR-006 | Dash 6 페이지 UI | ✅ 코드 완성 (검증 대기) |
| FR-007 | MLflow 실험 추적 | ✅ |
| FR-008 | Markdown + JSON 자동 보고서 | ✅ |
| FR-009 | reeval (학습 없이 재평가) | ✅ |
| FR-010 | compute_efficiency 벤치마크 | ✅ |
| FR-011 | LLM 스트리밍 + Function Calling | ✅ |

## 11.2 Should Have (P1) — 진행 중 또는 대기

| ID | 기능 | 상태 |
|---|---|---|
| FR-101 | iTransformer 모델 | 완료 (multi-seed 검증 완료, H3 반증 확정) |
| FR-102 | AI4I CNN recall 강화 variant | 완료 (3 seeds, mean recall 0.732) |
| FR-103 | Hydraulic 7-combo 앙상블 (Phase B-7) | 완료 (IF F1 0.915, 목표 0.90+ 달성) |
| **FR-104** | **MQTT 실시간 워커** (T1-01) | 완료 (1차) — broker 무관 e2e 검증 |
| **FR-105** | **알람 Notifier** (T1-02) | 완료 (1차) — Stdout 기본, 외부 채널 pluggable |
| **FR-106** | **PostgreSQL 이력 저장** (T1-03) | 완료 — TimescaleDbWriter 통합, e2e 검증 |
| **FR-107** | **Grafana 대시보드** (T1-04) | 완료 — Fleet Overview + Device Detail 2종 |
| FR-108 | MIMII 데이터셋 추가 (T2-01) | 대기 (Tier 2) |
| FR-109 | Evidently 드리프트 모니터링 (T2-03) | 대기 (Tier 2) |

## 11.3 Later (P2~P3) — Tier 3 / 향후 검토

| ID | 기능 | Tier |
|---|---|---|
| FR-201 | FastAPI 분리 + REST API | T3 prerequisite |
| FR-202 | Mock ERP/MES API | T3-01 |
| FR-203 | JWT 권한 | T3-02 |
| FR-204 | OPC-UA 시뮬레이터 | T3-03 |
| FR-205 | ONNX 양자화 | T3-04 |
| FR-206 | Docker Compose | T3 prerequisite |
| FR-207 | Multi-LLM 프로바이더 | 장기 |
| FR-208 | RAG 정비 매뉴얼 | 장기 |
| FR-209 | 멀티턴 대화 | 장기 |

---

# 12. 비기능 요구사항

## 12.1 성능 (실시간 시스템 도입 후)
- 메시지 수신 → 추론 → 알람: **< 5초** (T1 도입 후 측정)
- 동시 설비 모니터링: 50대 (개인 프로젝트 범위)
- LLM 응답: 1~3초 (Groq 의존)

## 12.2 신뢰성
- API 키 미설정 시 rule-based fallback
- 모든 학습 metric MLflow 기록 (실패도 status 태그)
- Multi-seed 재현성 (n≥3 seeds)
- Engine-wise train/val split (data leakage 방지)

## 12.3 유지보수성
- 9개 서비스 클래스 책임 분리 (UI 무관)
- Dataclass DTO
- 환경변수 중심 설정
- 자동 인사이트 (analyze_results.py)

---

# 13. 학술적 정당성

## 13.1 검증 완료
- Data leakage 방지 (engine-wise split)
- 평가 metric 완전성 (NASA Score / PR-AUC / Confusion Matrix)
- Multi-seed 재현성 (BiLSTM × DLinear × iTransformer, n=3)
- Sensor correlation 분석 기반 모델 선택 가설 수립
- H3 가설 (iTransformer 우위) — multi-seed n=3 으로 **확정 반증**: operating condition diversity 가 sensor correlation 보다 우세 변수
- Hydraulic 7-combo 앙상블 비교 — IF 단독 F1 0.915 가 모든 페어/트리오 fusion 대비 동급 이상
- AI4I CNN recall variant — focal loss 강도 조정으로 recall +3.7%p (0.706 → 0.732), variance 크다는 부수 관찰 확보

## 13.1.b Tier 1 회고 (2026-05-20) — 실시간 PdM 트랙

학술 검증과 별개의 **운영 트랙** (T1-01 ~ T1-04 + 후속 보강).

### 구현 완료
| 작업 | 산출물 | 검증 |
|---|---|---|
| T1-01 MQTT 워커 | [services/realtime/mqtt_worker.py](services/realtime/mqtt_worker.py) — paho v2, DI 기반 | broker 무관 process_message smoke + 실 broker e2e |
| T1-02 알람 Notifier | [services/realtime/notifier.py](services/realtime/notifier.py) — Protocol + StdoutNotifier 기본, 채널 pluggable | NotifyResult 4종 (SENT/FILTERED/RATE_LIMITED/FAILED) |
| T1-03 TimescaleDB 저장 | [services/realtime/db_writer.py](services/realtime/db_writer.py) + [infra/timescaledb/init.sql](infra/timescaledb/init.sql) — 3 hypertable, 압축/보존 정책 | 72/72/72 + 90/90/90 row 적재 e2e |
| T1-04 Grafana | [infra/grafana/dashboards/](infra/grafana/dashboards/) — Fleet Overview + Device Detail 2종 | datasource health OK, drill-down 동작 |
| 보안 #6 | Report path 경화 — sanitize allowlist + REPORTS_DIR 자손 검증 | path traversal / 시스템 파일 read 모두 차단 |
| 분석 B-3 | overfit_score 자동 로깅 ([scripts/training/train.py](scripts/training/train.py)) | metrics.json 에 자동 포함 (향후 학습부터) |
| CI #8 | smoke.yml 에 test_validation_step.py 추가 | 6 tests 통과, 외부 의존 0 |

### 발견된 회귀 (모두 수정 완료)
1. **paho v2 ReasonCode TypeError** — int(reason_code) 가 TypeError. `.value` fallback 으로 v1/v2 양립. broker 없는 smoke 단계에서는 발현 안 되어 실 broker 첫 연결 시 발견.
2. **psycopg 미설치 시 워커 사망** — fallback 메시지의 em-dash 가 Windows cp949 환경에서 UnicodeEncodeError 발생. NullDbWriter 로 살아남아야 하는 운영 안정성이 깨져 있었음. ASCII hyphen 으로 교체.
3. **dcc.Store path injection 가능성** — 외부 노출 직전 발견, S5 에서 경화.

### 의도적 미진행 (CMMS Tier 1.5 로 이관)
- LLM 한국어 코멘트의 실시간 워커 통합 (현재 batch analyze_service 만 지원)
- 익명 viewer + Grafana home dashboard 설정 (인증 통합 전 임시 우회 회피)
- Tier 2 데이터셋 확장 (MIMII 등)

### T1.5 사전 의사결정 v1.0 (2026-05-20 확정)

CMMS Tier 1.5 진입 전 결정해야 하는 7개 항목 + 외부 리뷰 보완 흡수 결과를 명시하여 다음 세션의 단일 진입 참조점으로 사용한다.

**근거 출처**:
- ISO 55001:2024 §5.3 (자산관리시스템 역할/책임/권한)
- NIST SP 800-53 AC-2 / AC-3 / AC-5 / AC-6 (Least Privilege, Separation of Duties, RBAC, Audit)
- TimescaleDB 공식 가이드 (hypertable → regular table FK 완전 지원)
- Flask-SQLAlchemy 베스트 프랙티스
- CMMS 산업 사례 (LLumin, ClickMaint, Fiix, Maximo, UpKeep)

**의사결정 7건**:

| # | 항목 | 결정 | 한 줄 요약 |
|---|---|---|---|
| 1 | device_id 외래키 | 강결합 + ON DELETE RESTRICT + TEXT PK 유지 + soft delete | hypertable → regular table FK 공식 지원, 기존 'milling-01' 식별자 호환 |
| 2 | 인증 사용자 DB | TimescaleDB 동일 인스턴스 (users 테이블) | Flask-SQLAlchemy 표준, FK 연계 + 백업 단순 |
| 3 | PDF export | xhtml2pdf 우선, 품질 부족 시 weasyprint 전환 | Windows GTK/Pango/Cairo 의존성 회피, 한국어 @font-face 즉시 가능 |
| 4 | Grafana 인증 | PoC: 별도 로그인 유지. SSO 는 외부 사용자 도입 시점 | Nginx auth_proxy 도입 비용 > admin/admin 2회 로그인 비용 |
| 5 | RBAC 구현 깊이 | 3-role + hard-coded @role_required (PoC) | Flask 공식 RBAC 튜토리얼 패턴, middleware 는 사용자 2명+ 시 |
| 6 | Scoping | devices 컬럼만 추가 (plant_id, equipment_group_id). 쿼리 필터링은 사용자 5명+ 시 | 데이터 모델 미리 준비, middleware 는 트리거 충족 시 |
| 7 | Audit log | 도입 (NIST AC-6) | 비용 작고 디버깅에도 유리 |

**외부 리뷰 보완 흡수 (9 채택 / 1 거부 / 1 조정)**:

| 보완 | 판정 | 사유 |
|---|---|---|
| 인덱스 추가 (4개 선별) | 채택 | maintenance_orders(device_id,status), (assigned_to,status), audit_log(occurred_at DESC), devices(current_status). 나머지 3개는 사용자 2명+ 시 추가 |
| soft delete `active_filter` 공통 메서드 | 채택 | device_service.py 작성 시 자연스럽게 도입 |
| FK 추가 순서 (테이블 → backfill → FK) | 채택 | backfill 없이 FK 추가 시 기존 데이터 무결성 위반 |
| backfill SQL | 채택 (수정) | 리뷰의 `dataset_key='legacy'` 하드코딩은 부정확 — 실제 telemetry.dataset_key 그대로 사용해야 워커 publish 와 일치 |
| scope_required 데코레이터 placeholder | 채택 | PoC 는 NULL 우회, 정의만 둬서 미래 확장 비용 0 |
| simulator device upsert 추상화 (`device_service.upsert_device`) | 채택 | 한 곳만 수정으로 scoping 컬럼 확장 흡수 |
| xhtml2pdf link_callback + Noto Sans KR @font-face | 채택 (조정) | 리뷰의 `app.root_path` 는 Flask app context 의존, Dash 환경에서는 `Path(__file__).parent / "static" / "fonts"` 직접 경로 계산 |
| PRD 회고 한 문장 추가 | 채택 | 비용 0, 가독성↑ |
| 의사결정 큐 각 항목 한 줄 요약 | 채택 | 위 표에 반영 |
| **audit_log.action CHECK constraint** | **거부** | PoC 단계에서 action 종류 빠르게 증가. CHECK 추가 시 매번 ALTER TABLE 필요 → 유연성 저하. 대안: `services/audit_actions.py` Python 상수로 enum 관리. CHECK 는 사용자 5+ / production 시점에 |

**P0 진입 직전 산출물 명세 (다음 세션 시작점)**:

`infra/timescaledb/migrations/001_cmms_schema.sql` (또는 init.sql 확장):

1. 신규 5 테이블: `users`, `devices`, `maintenance_orders`, `device_status_history`, `audit_log`
2. 인덱스 4개:
   - `maintenance_orders(device_id, status)`
   - `maintenance_orders(assigned_to, status)`
   - `audit_log(occurred_at DESC)`
   - `devices(current_status)`
3. backfill: `INSERT INTO devices SELECT DISTINCT ON (device_id) device_id, dataset_key, device_id, 'plant-01', TRUE FROM telemetry ORDER BY device_id, time DESC ON CONFLICT DO NOTHING`
4. 기존 hypertable 3종에 FK 추가:
   - `telemetry.device_id → devices.device_id ON DELETE RESTRICT`
   - `predictions.device_id → devices.device_id ON DELETE RESTRICT`
   - `alerts.device_id → devices.device_id ON DELETE RESTRICT`
5. `services/realtime/device_service.py` 신규 (upsert_device 추상화)
6. `services/audit_actions.py` 신규 (action 상수 enum 관리)

P0 작업량: 1.5d + 1h (당초 1.5d 에서 외부 리뷰 흡수로 +1h).

### T1.5 P0 구현 완료 (2026-05-20)

| 산출물 | 비고 |
|---|---|
| [infra/timescaledb/migrations/001_cmms_schema.sql](infra/timescaledb/migrations/001_cmms_schema.sql) | 5 테이블 + 4 인덱스 + backfill + FK 3종 + `updated_at`/`closed_at` 트리거. 멱등 (`IF NOT EXISTS`, `pg_constraint`/`pg_trigger` 가드) |
| [docker-compose.yml](docker-compose.yml) volumes | `init.sql → 00_init.sql`, `migrations/001_cmms_schema.sql → 01_cmms_schema.sql` 명시 마운트 (postgres entrypoint 가 알파벳순 실행) |
| [services/audit_actions.py](services/audit_actions.py) | `AuditAction` enum + `TARGET_TYPE_*` 상수 + `log()` (psycopg `Jsonb` adapter, system_event 자동 추가, PII 금지 docstring) |
| [services/realtime/device_service.py](services/realtime/device_service.py) | `DeviceService` Protocol + `NullDeviceService` + `TimescaleDeviceService` (upsert 캐시, `set_status`/`mark_deleted` 트랜잭션, audit_log 자동 통합) + `scope_required` placeholder |
| [services/realtime/mqtt_worker.py](services/realtime/mqtt_worker.py) | `devices` DI + `process_message` 에 `upsert_device` 추가 (FK 충족용, hot path 캐시 hit) |
| [scripts/realtime/run_worker.py](scripts/realtime/run_worker.py) | `_resolve_device_service` 추가 — `DB_ENABLED` / `--no-db` 보고 dispatch |
| [tests/conftest.py](tests/conftest.py) + [tests/realtime/test_device_service.py](tests/realtime/test_device_service.py) | unit (mock, 4 pass) + integration (`OMNIPDM_TEST_DB_URL` 설정 시만, 미설정 자동 skip → CI smoke 영향 0) |

**마운트 컨벤션 (확정)**: postgres 공식 entrypoint 는 `/docker-entrypoint-initdb.d/` **직속 파일**만 알파벳순 실행 (하위 디렉토리 무시). 따라서 신규 마이그레이션 추가 시 `docker-compose.yml` 의 `timescaledb.volumes` 에 `02_*.sql`, `03_*.sql` 형식으로 1줄씩 추가하는 컨벤션 채택. PRD `migrations/001_*.sql` 디렉토리 명명은 유지.

**의도된 atomicity 비대칭** (device_service 구현 결정): `upsert_device` 의 device INSERT 와 audit_log INSERT 는 **별도 트랜잭션**이다. audit 실패가 device 생성을 rollback 시키면 telemetry FK 위반으로 워커 hot path 가 깨진다. audit_log 는 부가, device 는 필수 — 의도된 비대칭. `set_status` / `mark_deleted` 는 사용자 명시 행동이라 audit 까지 한 트랜잭션으로 묶음.

**외부 리뷰 추가 흡수 (P0 작성 중 11건 평가 → 6건 채택)**:
- 채택: `updated_at` 자동 트리거, `closed_at` 자동 트리거, backfill `RAISE NOTICE`, psycopg `Jsonb` adapter, `TARGET_TYPE_*` 상수, `mark_deleted` 캐시 discard 를 트랜잭션 내부로 이동
- 거부: `target_type` Enum 강제 (T1.5 결정 #7 정책 동일), `tenacity` retry (PoC), `row_factory` namedtuple (YAGNI), 명시 `BEGIN` (psycopg v3 autocommit=False 가 implicit), `logging.getLogger` (services/realtime/* 전체 `print` 일관성)

**검증 결과**:
- `pytest tests/realtime/test_device_service.py -v` → **4 passed, 4 skipped** (integration 은 DB URL 없어 자동 skip)
- `pytest tests/test_smoke.py tests/test_validation_step.py` → **30 passed** (회귀 0)

**다음 단계 (P1, 예상 2~3d)**: Flask 인증 라우트 + `@role_required` 데코레이터 본문 + Dash devices/orders 페이지 + werkzeug 비밀번호 hash 마이그레이션 (seed admin placeholder 교체).

### T1.5 P1-a 구현 완료 (2026-05-20) — werkzeug hash + admin seed 부트스트랩

| 산출물 | 비고 |
|---|---|
| [services/auth/passwords.py](services/auth/passwords.py) | `hash_password` / `verify_password` / `is_placeholder_hash` (werkzeug.security 얇은 wrapper). placeholder hash 는 verify 시 무조건 reject — 부트스트랩 누락 사고 차단. 비문자열 입력 robust (`isinstance` 가드) |
| [scripts/migrations/bootstrap_admin.py](scripts/migrations/bootstrap_admin.py) | CLI: `--password-file` > env > `--interactive` 우선순위 (K8s/Docker secret 표준 준수). `--dry-run` / `--force` / POSIX 0600 권한 검사 (Windows 가드). 성공 시 `AuditAction.USER_UPDATED` + meta(reason/force/was_placeholder) 자동 INSERT. Exit codes 0/1/2/3 명시 |
| [tests/test_passwords.py](tests/test_passwords.py) | 13 unit tests — hash idempotency 거부, salt 무작위성, placeholder reject 무조건, 비문자열 입력 robust |
| [requirements.txt](requirements.txt) | `werkzeug>=3.0` 직접 의존 명시 (Dash transitive 이지만 명확화) |

**외부 리뷰 흡수 (P1-a 작성 중 8건 평가 → 4건 채택)**:
- 채택: `--password-file` 최우선 (K8s/Docker secret 표준), `--dry-run`, `psycopg.Error` 명시 catch + exit code 3, POSIX 권한 검사 (Windows 가드)
- 거부: env var 제거 (CI 자동화 필수), logging 마이그레이션 (services/realtime/* 일관성), `PLACEHOLDER_HASH = "!"` (이미 ship 된 SQL 과 호환 파괴), password_policy 모듈 분리 (YAGNI)

**사용법**:
```bash
# 첫 부트스트랩 (대화형)
python -m scripts.migrations.bootstrap_admin --interactive

# CI (env var, 사용 후 즉시 unset)
export OMNIPDM_ADMIN_BOOTSTRAP_PASSWORD=changeme
python -m scripts.migrations.bootstrap_admin
unset OMNIPDM_ADMIN_BOOTSTRAP_PASSWORD

# K8s/Docker (secret 파일)
chmod 600 /run/secrets/admin_pw
python -m scripts.migrations.bootstrap_admin --password-file /run/secrets/admin_pw
```

**다음 단계 (P1-b, 예상 0.5~1d)**: Flask 인증 라우트 (login/logout/session) + `services/auth/sessions.py` (Flask-Login 또는 직접 session 관리 결정).

### T1.5 P1-b 구현 완료 (2026-05-20) — Flask 인증 라우트 + Dash 페이지 가드

**설계 결정 4건 (외부 리뷰 합의)**:
1. session 라이브러리: **직접 Flask session** (Flask-Login 미도입 — PoC 단일 사용자, 의존성 최소)
2. CSRF 방어: **수동 csrf token** (login 폼만 — Flask-WTF 미도입)
3. 가드 위치: **`@server.before_request` 단일 진입점** (페이지 decorator 대신)
4. `OMNIPDM_SECRET_KEY` 미설정 시: **random fallback + 경고** (개발 편의 vs hard error)

| 산출물 | 비고 |
|---|---|
| [models_core/config.py](models_core/config.py) §7 | `OMNIPDM_SECRET_KEY` 환경변수 + `secrets.token_urlsafe(32)` fallback + 경고 |
| [services/auth/users.py](services/auth/users.py) | `UserService` Protocol + Null/Timescale impls. `verify_credentials` 가 모든 실패 경로에서 `verify_password` 1회 호출 → timing 평탄화 (NIST/OWASP 권장). password_hash 는 `User` dataclass 에서 제외 |
| [services/auth/sessions.py](services/auth/sessions.py) | `login_user` / `logout_user` / `current_user` / `is_authenticated` / `current_user_id` + CSRF (`get_or_create_csrf_token` / `verify_csrf_token` / `rotate_csrf_token` — session fixation 방어). audit_log 자동 INSERT (`LOGIN_SUCCESS` / `LOGIN_FAILURE` / `LOGOUT`) |
| [services/auth/routes.py](services/auth/routes.py) | Flask Blueprint (`/login` GET/POST, `/logout` GET/POST). Bootstrap CDN HTML 임베드, `_is_safe_next` 로 open redirect 방어. Dash page 로 등록 안 함 (test_smoke 의 6 페이지 카운트 유지) |
| [app.py](app.py) | `server.secret_key` 설정 + blueprint 등록 + `@server.before_request` 가드 + `OMNIPDM_AUTH_REQUIRED` toggle + `_dash` / `/static` / `/assets` exempt + 동적 navbar (logout 링크). `audit_conn_provider` 는 closure 캐시로 connection leak 방지 |
| [tests/test_auth_sessions.py](tests/test_auth_sessions.py) | 14 unit tests (CSRF, 인증 실패/성공, ?next= 보존, open redirect 차단, 가드 exempt, logout) — Mock UserService 주입, CI smoke 외부 의존 0 |

**외부 리뷰 흡수 (P1-b 작성 중 12건 평가 → 4건 채택)**:
- 채택: `verify_credentials` 의 inactive user dummy verify (timing 평탄화), `PLACEHOLDER_HASH` module-level import, `audit_conn_provider` closure 캐시, `_is_safe_next` open redirect 방어
- 거부: `row_factory=dict_row` (positional tuple 일관성), `print` → `logging` 마이그레이션 (전체 일관성), `dcc.Link` 로 logout (Flask 라우트 호출 안 됨), Flask-Login/Flask-WTF 도입 (PoC 과잉)

**환경변수 운영 가이드**:
```bash
# 운영 필수
export OMNIPDM_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

# 개발 편의 (인증 비활성)
export OMNIPDM_AUTH_REQUIRED=false
```

**Future migration 메모 (사용자 5명+ 시점)**:
- Flask-Login + Flask-WTF 도입 검토 — multi-factor / "remember me" / session protection 강화
- psycopg `ConnectionPool` 로 audit_conn_provider 교체
- RBAC `@role_required` 본문 활성화 (P1-c)
- Row-Level Security (RLS) — devices/maintenance_orders 에 role 기반 정책

**검증 결과**:
- `pytest tests/test_auth_sessions.py -v` → **14 passed**
- `pytest tests/` → **62 passed, 4 skipped, 0 failed** (P1-a 의 48 + 신규 14)
- `app.server.test_client()` smoke: `/login` 200 / `/logout` 302 / `/` 302→`/login?next=/` / `/status?foo=bar` 302→`/login?next=/status?foo=bar`

**다음 단계 (P1-c, 예상 2~3h)**: `services/auth/sessions.py` 에 `role_required` 데코레이터 본문 활성화 + `services/realtime/device_service.scope_required` placeholder 와 통합. 기존 Dash 페이지 callback 에 부착.

## 13.2 부분 검증 / 향후 작업
- AI4I recall 0.85+ 목표 미달 — 클래스 reweighting + Optuna 탐색은 향후 작업
- FD002 iTransformer 열세 원인 정밀 분석 (operating regime 별 잔차 분포)
- N-CMAPSS 전체 모델 비교 (현재 미수행, 우선순위 낮음)
- Compute efficiency 정량 보고 (compute_efficiency.py 결과 PRD 본문 인용)

---

# 14. 진화 로드맵

## 14.1 단기 (1~2개월) — Tier 1 우선

| Phase | 작업 | Tier | 상태 |
|---|---|---|---|
| 14-1-a | 학술 검증 마무리 (iT multi-seed, AI4I recall, Hydraulic 앙상블) | — | 완료 (2026-05-19) |
| 14-1-b | **MQTT 실시간 워커 + Notifier** (T1-01, T1-02) | T1 | 1차 완료 (2026-05-19) — broker 무관 e2e 검증 |
| **14-1-c** | **PostgreSQL + TimescaleDB 이력 저장** (T1-03) | T1 | 완료 (2026-05-20) |
| **14-1-d** | **Grafana 대시보드** (T1-04) | T1 | 완료 (2026-05-20) — Fleet Overview + Device Detail 2종, drill-down 연동 |
| 14-1-e | Report path 경화 (path traversal / 임의 파일 read 차단) | T1 후속 (보안) | 완료 (2026-05-20) — allowlist sanitize + REPORTS_DIR 자손 검증 |
| 14-1-f | overfit_score 자동 로깅 (외부 PRD 개선 보고서 B-3 흡수) | T1 후속 (분석) | 완료 (2026-05-20) — train.py metrics dict 에 자동 포함 |
| 14-1-g | CI 스코프 확장 (test_validation_step.py 추가) | T1 후속 (품질) | 완료 (2026-05-20) |
| 14-1-h | 외부 알람 채널 1종 도입 (Telegram / Email 중 택일) | T1 후속 | 채널 결정 후 |
| 14-1-i | CMMS Tier 1.5 (설비 마스터 + 상태 머신 + 수리 워크플로 + 인증/권한) | T1.5 (신규) | 설계 단계 — Dash 강화 방향 확정 (2026-05-20) |
| 14-1-j | N-CMAPSS 학습 (선택) | — | 우선순위 낮음 |

## 14.2 중기 (3~6개월) — Tier 2 도입

| Phase | 작업 | Tier |
|---|---|---|
| 14-2-a | MIMII / Bosch / TEP 데이터셋 통합 | T2-01~02 |
| 14-2-b | Evidently AI 드리프트 모니터링 | T2-03 |
| 14-2-c | Docker Compose 운영 환경 | T3 prereq |
| 14-2-d | FastAPI 분리 (services 재사용) | T3 prereq |

## 14.3 장기 (6~12개월) — Tier 3 통합

| Phase | 작업 | Tier |
|---|---|---|
| 14-3-a | Mock ERP/MES + JWT 권한 | T3-01~02 |
| 14-3-b | OPC-UA 시뮬레이터 + ONNX Edge | T3-03~04 |
| 14-3-c | 멀티 LLM / RAG / 멀티턴 | 장기 |

---

# 15. 핵심 설계 원칙

1. **`services/` 순수성 유지** — UI/HTTP/Queue 의존성 금지 → 진입점 무관 재사용
2. **`schemas.py` Dataclass DTO** — 모든 경계의 계약
3. **환경변수 중심 설정** — dev/prod/batch 같은 코드 다른 동작
4. **상태 외부화** — 파일 I/O → DB 어댑터 교체 쉽게
5. **LLM 호출 옵셔널** — 실시간에서 Critical/Warning만 호출 등 비용 제어
6. **모든 도구 무료** — 학생/개인 개발자 범위 유지

---

# 16. 성공 기준 (KPI)

## 16.1 학술 KPI (2026-05-19 기준)
- Multi-seed mean±std 보고 (n=3) — 완료
- NASA Score + Confusion + PR-AUC 자동 계산 — 완료
- C-MAPSS FD001~FD004 3-way 비교 (BiLSTM/DLinear/iT) — 완료, H3 가설 multi-seed 반증
- Hydraulic 이상 탐지 F1 ≥ 0.90 — 달성 (IF 0.915)
- AI4I CNN recall 0.706 → 0.85+ — 미달성 (현재 0.732, +3.7%p), 향후 보강

## 16.2 실무 KPI (Tier 1 도입 후 측정)
- **알람 응답 시간** < 5초 (메시지 수신 → Slack 알림)
- **동시 모니터링 설비** 50대 이상
- **드리프트 발견 → 재학습 lead time** < 1주
- **False Alarm Rate** < 5%

## 16.3 PoC 완성도 KPI
- Tier 1 완료 (실시간 + 알람 + DB + Grafana)
- Tier 2 50% 진행 (데이터셋 확장 + 드리프트)
- Tier 3 검토 완료 (필요성 판단)

---

# 17. 라이선스

사용한 공개 데이터셋의 라이선스는 각 제공처 정책을 따릅니다. 코드 라이선스는 추후 결정.

---

# 부록 A. 변경 이력

| 버전 | 일자 | 주요 변경 |
|---|---|---|
| v6.0 | 2026-05-19 | 발표 후 첫 PRD (학술 검증 중심) |
| v6.1 | 2026-05-19 | 스마트팩토리 동기 명시 + Tier 우선순위 + Use Cases + 통합 요구사항 |
| v6.2 | 2026-05-19 | 브랜드 리네이밍: HybridPdM → OmniPdM (All-in-One PdM System) + 로고 컨셉 + 네이밍 정당성 (§0) 추가 |
| **v6.3** | **2026-05-19** | **§8 학술 결과 종합 신설 (RUL 3-way / Hydraulic 7-combo / AI4I recall variant). Tier 1 T1-01 / T1-02 1차 완료 반영 (broker 무관 e2e 검증). FR / 로드맵 / KPI / 학술 정당성 섹션 상태 동기화.** |

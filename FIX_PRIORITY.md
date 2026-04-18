# HybridPdM 런타임 오류 수정 우선순위 보고서

**작성일**: 2026-04-18  
**검토 방법**: 각 항목을 실제 파일 내용 확인(`ast.parse` 문법 검증, 체크포인트 디렉토리 조회)으로 이중 검증

---

## 검증 결과 요약

초기 탐색에서 10개 이슈가 제기되었으나, 실제 파일 내용·문법을 검증한 결과 **3개는 오보**, **7개는 실제 버그**로 확정되었다.

| 초기 주장 | 실제 검증 결과 |
|----------|-------------|
| `report_service.py:95` 문법 오류 (`strip()or`) | **오보** — `)` 뒤 `or`는 Python이 정상 파싱 (`ast.parse` 통과 확인) |
| `llm_service.py:72` `completion.choices[0]` 크래시 | **오보** — 이미 `try/except` 블록(51–92번째 줄) 안에 있어 폴백으로 정상 처리 |
| `pdm_service.py:344` lite 모드 + LSTM 불일치 | **오보** — 101~105번째 줄에서 `NotImplementedError`로 선제 차단됨 |

---

## P0 — CRITICAL (즉시 수정 필요)

### ① `requirements.txt`가 실질적으로 비어 있음

**파일**: [requirements.txt](requirements.txt)  
**검증**: `Read` 도구로 확인 — 파일 전체가 1줄(빈 줄)뿐

**증상**  
사용자가 `pip install -r requirements.txt` 실행 시 아무것도 설치되지 않음 → 첫 실행에서 `ModuleNotFoundError: No module named 'gradio'` 즉시 크래시

**수정안** — 아래 내용으로 완전 교체
```
torch>=2.0
gradio>=4.0
openai>=1.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
```

**영향도**: 모든 신규 사용자가 앱을 실행조차 할 수 없음

---

### ② `ncmapss_lstm` 체크포인트 부재 — UI에서 선택 시 크래시

**파일**: [gradio_app.py:688](gradio_app.py#L688)  
**검증**: `ls models_core/artifacts/checkpoints/` 실행 결과
```
ai4i_cnn_*.pt     ✓
ai4i_gbdt_*.pkl   ✓
cwru_cnn_*.pt     ✓
hydraulic_ae_*.pt ✓
cmapss_lstm_*.pt  ✓
ncmapss_lstm_*.pt ✗  ← 없음
```

**증상**  
LSTM Analysis 탭에서 `ncmapss_lstm` 선택 후 실행 → `FileNotFoundError` 발생. 현재 예외 처리가 있긴 하지만 사용자는 "왜 되지 않는지" 알 수 없음

**수정안 (2가지 중 택1)**

(a) 드롭다운에서 임시로 제거
```python
# gradio_app.py:688
lstm_dataset_dd = gr.Dropdown(
    choices=["cmapss_lstm"],  # ncmapss_lstm 일시 제거
    value="cmapss_lstm",
    ...
)
```

(b) 체크포인트 존재 여부로 동적 choices 구성
```python
def _available_lstm_datasets():
    avail = ["cmapss_lstm"] if list(config.CHECKPOINT_DIR.glob("cmapss_lstm*.pt")) else []
    if list(config.CHECKPOINT_DIR.glob("ncmapss_lstm*.pt")):
        avail.append("ncmapss_lstm")
    return avail or ["cmapss_lstm"]
```

**영향도**: UI에서 선택 가능하지만 실행 불가능한 옵션을 노출

---

## P1 — HIGH (운영 중 크래시 위험)

### ③ `pdm_service.py`에 디버깅 print 문이 프로덕션에 남음

**파일**: [services/pdm_service.py:416-422](services/pdm_service.py#L416-L422)  
**검증**: 직접 읽어서 확인

```python
# 디버깅용 출력
print("[DEBUG] dataset_key =", dataset_key)
print("[DEBUG] arr.shape =", arr.shape)
print("[DEBUG] timesteps =", timesteps)
print("[DEBUG] feature_dim =", feature_dim, type(feature_dim))
print("[DEBUG] raw_expected_feature_dim =", raw_expected_feature_dim, type(raw_expected_feature_dim))
print("[DEBUG] expected_feature_dim =", expected_feature_dim, type(expected_feature_dim))
print("[DEBUG] self._meta =", self._meta)
```

**증상**  
- LSTM 실행마다 콘솔에 7줄씩 출력 → HF Spaces 로그 오염
- `self._meta`가 dict이므로 민감 정보가 포함될 경우 보안 이슈

**수정안**: 전체 7줄 삭제 또는 `logging.debug()`로 교체

**영향도**: 프로덕션 품질 저하, 로그 가독성 저하

---

### ④ `update_lstm_input_guide()`가 Textbox value를 초기화

**파일**: [gradio_app.py:270-297](gradio_app.py#L270-L297)

```python
return (
    gr.Markdown(value=guide_md),
    gr.Textbox(placeholder=seq_placeholder),   # ← value= 없이 placeholder만
    gr.Textbox(placeholder=feature_placeholder),
)
```

**증상**  
사용자가 `cmapss_lstm`에서 `ncmapss_lstm`으로 드롭다운을 바꾸면 이미 입력한 시퀀스 텍스트가 **사라짐**. 구현 의도상 placeholder 변경이 목적이지만 Gradio는 새 component 인스턴스를 반환하면 기존 value를 치환함

**수정안**: `gr.update()` 사용으로 변경
```python
return (
    gr.update(value=guide_md),
    gr.update(placeholder=seq_placeholder),
    gr.update(placeholder=feature_placeholder),
)
```

**영향도**: UX 손상 + 사용자가 데이터셋 잘못 선택 후 되돌리면 입력 재작성 필요

---

### ⑤ 로그 파일 쓰기 실패 시 파이프라인 전체 중단

**파일**: [services/llm_service.py:195-202](services/llm_service.py#L195-L202)

```python
def _log(self, prompt_hash: str, latency_ms: float, status: str) -> None:
    line = {...}
    with open(self.logs_path / "llm_calls.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
```

**증상**  
HF Spaces 재시작 직후 `logs/` 디렉토리가 재생성되기 전, 또는 권한 문제 발생 시 `PermissionError`·`OSError` 발생 → LLM 호출 성공했어도 전체 분석 파이프라인 중단

**수정안**: try/except로 감싸기
```python
def _log(self, prompt_hash: str, latency_ms: float, status: str) -> None:
    try:
        line = {"prompt_hash": prompt_hash, "latency_ms": round(latency_ms, 2), "status": status}
        with open(self.logs_path / "llm_calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 로깅 실패가 전체 파이프라인을 멈추면 안 됨
```

**영향도**: HF Spaces 배포 직후 첫 호출에서 크래시 가능성

---

## P2 — MEDIUM (안정성·일관성 개선)

### ⑥ `_meta` dict에 무방비 접근

**파일**: [services/pdm_service.py:412](services/pdm_service.py#L412)

```python
raw_expected_feature_dim = self._meta.get("feature_dim", feature_dim)
expected_feature_dim = int(raw_expected_feature_dim)
```

**증상**  
체크포인트 `json` 메타가 `feature_dim` 키를 가지지 않거나 값이 None이면 `int(None)` → `TypeError`

**수정안**
```python
raw = self._meta.get("feature_dim")
expected_feature_dim = int(raw) if raw is not None else int(feature_dim)
```

---

### ⑦ 빈 프롬프트 파일 — Dead code

**파일**: [prompts/system_pdm_assistant.txt](prompts/system_pdm_assistant.txt), [prompts/user_context_template.json](prompts/user_context_template.json)  
**검증**: 둘 다 0 바이트

**증상**  
파일은 존재하지만 어느 코드도 로드하지 않음 → 존재 이유 없음  
추후 누군가 `open()`으로 로드하려 하면 빈 문자열 반환 → LLM 출력 붕괴

**수정안 (택1)**
- (a) 파일 삭제
- (b) 실제 프롬프트 내용 작성 후 `llm_service.py`의 하드코딩된 프롬프트(58–63번째 줄)를 파일 로드 방식으로 교체

---

### ⑧ 체크포인트 디렉토리 접근성 미검증

**파일**: [services/pdm_service.py:_find_checkpoint](services/pdm_service.py)

```python
candidates = sorted(config.CHECKPOINT_DIR.glob(f"{stem}*{suffix}"))
if not candidates:
    raise FileNotFoundError(...)
```

**증상**  
`CHECKPOINT_DIR` 자체가 존재하지 않을 때 `glob()`은 예외 없이 빈 리스트 반환 → "체크포인트 없음" 메시지로 오해 유발. 실제로는 디렉토리 자체가 부재

**수정안**
```python
if not config.CHECKPOINT_DIR.exists():
    raise FileNotFoundError(
        f"체크포인트 디렉토리가 존재하지 않습니다: {config.CHECKPOINT_DIR}. "
        f"HF Spaces 사용자는 Git LFS 또는 HF Hub 다운로드 코드가 포함됐는지 확인하세요."
    )
```

---

## P3 — LOW (장기 개선)

### ⑨ `ExplainService`의 tuple unpacking 가정

**파일**: [services/explain_service.py:37](services/explain_service.py#L37)

```python
for feature_name, importance in contributors[:5]:
```

**증상**  
`top_contributors` 요소가 tuple이 아니거나 길이가 2가 아닌 경우 `ValueError: too many values to unpack` 발생. 현재는 모델이 항상 tuple을 반환하므로 실질 발생 확률 낮음

**수정안**: 방어적 코드 추가
```python
for item in contributors[:5]:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        continue
    feature_name, importance = item[0], item[1]
```

---

### ⑩ `AnalyzeService.__init__`에서 모든 서비스 즉시 초기화

**파일**: [services/analyze_service.py:48-56](services/analyze_service.py#L48-L56)

9개 서비스 객체를 생성자에서 모두 생성 → Gradio 요청마다 `AnalyzeService(...)`가 새로 만들어짐 → 매 요청 9개 서비스 재초기화 + 모델 체크포인트 재로드 → **응답 지연 누적**

**수정안**: 싱글톤 또는 모듈 수준 캐싱
```python
# gradio_app.py 모듈 레벨
_SERVICE_CACHE: Dict[Tuple, AnalyzeService] = {}

def get_service(mode, risk_method, dataset_key):
    key = (mode, risk_method, dataset_key)
    if key not in _SERVICE_CACHE:
        _SERVICE_CACHE[key] = AnalyzeService(mode, risk_method, dataset_key)
    return _SERVICE_CACHE[key]
```

---

## 수정 우선순위 실행 계획

| 순서 | 항목 | 예상 작업시간 | 선결 조건 |
|-----|------|------------|----------|
| 1 | ① requirements.txt 채우기 | 5분 | 없음 |
| 2 | ② ncmapss_lstm 드롭다운에서 제거 | 10분 | 없음 |
| 3 | ③ DEBUG print 삭제 | 5분 | 없음 |
| 4 | ⑤ `_log()` try/except 추가 | 10분 | 없음 |
| 5 | ④ `gr.update()` 로 교체 | 15분 | Gradio 버전 확인 |
| 6 | ⑥ `_meta.get()` None-safe 처리 | 10분 | 없음 |
| 7 | ⑧ 체크포인트 디렉토리 존재 검증 | 10분 | 없음 |
| 8 | ⑦ 빈 프롬프트 파일 처리 (삭제 or 채우기) | 30분~2시간 | 프롬프트 설계 결정 |
| 9 | ⑨ ExplainService 방어 코드 | 15분 | 없음 |
| 10 | ⑩ 서비스 캐싱 도입 | 30분 | 테스트 필요 |

**P0 + P1 단계만 수정하면 ≈ 45분 이내에 프로덕션 배포 가능한 안정성 확보**

---

## 배포 전 필수 체크리스트

```
[P0 완료]
□ requirements.txt 내용 확인 (torch, gradio, openai 포함)
□ ncmapss_lstm 드롭다운 제거 또는 체크포인트 추가

[P1 완료]
□ pdm_service.py DEBUG print 전체 제거
□ gradio_app.py update_lstm_input_guide → gr.update() 전환
□ llm_service.py _log() try/except 래핑

[P2 권장]
□ _meta.get() None-safe 처리
□ 빈 prompts/ 파일 삭제 또는 내용 작성
□ CHECKPOINT_DIR 존재 검증 추가
```

---

*이 보고서는 Explore 에이전트 결과를 `Read`·`Bash(ast.parse)`·`ls` 로 재검증하여 오보 3건을 제거하고 실제 버그 10건을 우선순위별로 분류한 결과입니다.*

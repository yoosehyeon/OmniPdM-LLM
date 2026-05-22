# OmniPdM Frontend

React + Vite + TypeScript + Tailwind. Flask (`/api/*`) 백엔드와 분리된 SPA.

## 개발

```bash
cd frontend
npm install
npm run dev     # http://localhost:5173 — /api/* 는 Flask(:8050) 로 프록시
```

별도 터미널에서 Flask:
```bash
python app.py   # http://localhost:8050
```

## 프로덕션 빌드

```bash
npm run build   # dist/ 생성, Flask 가 static serve
```

## 환경변수
- `VITE_API_PROXY_TARGET` — 개발 시 Flask 주소 override (기본 `http://localhost:8050`)
- API 호출은 항상 상대 경로(`/api/*`)로 작성. 절대 URL 금지.

## 인증
- Flask 세션 쿠키(HttpOnly) 기반. `fetch` 시 `credentials: 'include'` 필수.
- CSRF: 상태 변경 요청은 `X-CSRF-Token` 헤더 동봉 (`GET /api/csrf` 로 발급).

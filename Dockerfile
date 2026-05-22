# OmniPdM — multi-stage build (React frontend + Python backend).
#
# Stage 1 (frontend-builder): npm install + vite build → dist/
# Stage 2 (runtime): python:3.11-slim, gunicorn, frontend dist 복사
#
# 빌드:
#   docker build -t omnipdm:latest .
# 또는 docker-compose 의 services.app.build 가 자동 호출.

# -----------------------------------------------------------------------------
# Stage 1 — Frontend build
# -----------------------------------------------------------------------------
FROM node:20-alpine AS frontend-builder

WORKDIR /build/frontend

# package*.json 만 먼저 복사 → layer cache 활용 (소스 변경 시 deps 재설치 회피).
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2 — Python runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libpq (psycopg binary 가 포함하나, postgres client 진단용으로 minimal tools 추가).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements 먼저 복사 → cache.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스.
COPY . .

# Stage 1 빌드 결과물을 frontend/dist 로 복사 (Flask static serve 경로).
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8050/api/health || exit 1

# Dash 의 Flask app 객체를 직접 노출 — app.py 의 `server = app.server`.
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "2", "--timeout", "120", "app:server"]

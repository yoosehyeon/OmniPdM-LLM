"""OmniPdM Realtime Worker 진입점 — PRD v6.2 §10 T1-01 + v6.3 §10 T1-03.

CLI:
  python -m scripts.realtime.run_worker
  python -m scripts.realtime.run_worker --no-db   # NullDbWriter 강제 (DB 없이도 실행)

전제:
- MQTT broker (Mosquitto) 동작 중. docker-compose 사용 시 자동.
- DB_ENABLED=true (기본) + TimescaleDB 동작 중이면 TimescaleDbWriter 자동 주입.
- 둘 중 하나라도 실패 시 명확한 메시지 후 종료 (broker) 또는 NullDbWriter fallback (DB).
"""

from __future__ import annotations

import argparse

from models_core import config
from services.realtime import MqttWorker, NullDbWriter, TimescaleDbWriter


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="OmniPdM realtime MQTT worker")
    p.add_argument("--host", default=config.MQTT_HOST)
    p.add_argument("--port", type=int, default=config.MQTT_PORT)
    p.add_argument("--topic", default=config.MQTT_TOPIC_TELEMETRY)
    p.add_argument("--client-id", default=config.MQTT_CLIENT_ID)
    p.add_argument("--no-db", action="store_true",
                   help="DB 비활성 (NullDbWriter 강제). DB 미기동 환경에서 즉시 검증용")
    return p.parse_args(argv)


def _resolve_db_writer(no_db: bool):
    """config + CLI 플래그 보고 DbWriter 인스턴스 결정. 실패 시 NullDbWriter fallback."""
    if no_db or not config.DB_ENABLED:
        print("[worker] DB disabled — using NullDbWriter")
        return NullDbWriter()
    try:
        writer = TimescaleDbWriter(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
        )
        print(f"[worker] DB connected: {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}")
        return writer
    except Exception as e:
        # psycopg 미설치 / DB 미기동 / 인증 실패 등 — 워커 자체는 살린다 (운영 안정성).
        # ASCII hyphen 사용 — Windows cp949 환경에서 em-dash 가 UnicodeEncodeError 유발.
        print(f"[worker] DB unavailable ({type(e).__name__}: {e}) - falling back to NullDbWriter")
        return NullDbWriter()


def main() -> None:
    args = parse_args()
    db = _resolve_db_writer(no_db=args.no_db)
    worker = MqttWorker(
        host=args.host,
        port=args.port,
        topic=args.topic,
        client_id=args.client_id,
        db=db,
    )
    try:
        worker.start(block=True)
    except KeyboardInterrupt:
        print("[worker] interrupted by user")
    finally:
        worker.stop()
        print(f"[worker] final stats: {worker.stats}")


if __name__ == "__main__":
    main()

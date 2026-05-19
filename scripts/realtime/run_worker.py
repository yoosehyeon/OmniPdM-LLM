"""OmniPdM Realtime Worker 진입점 — PRD v6.2 §10 T1-01.

CLI:
  python -m scripts.realtime.run_worker
  python -m scripts.realtime.run_worker --host localhost --port 1883 --topic 'omnipdm/telemetry/+'

브로커가 동작 중이어야 한다 (Mosquitto 등). 브로커 없이 워커 로직만 검증하려면
services.realtime.MqttWorker().process_message(topic, payload) 를 직접 호출.
"""

from __future__ import annotations

import argparse

from models_core import config
from services.realtime import MqttWorker


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="OmniPdM realtime MQTT worker")
    p.add_argument("--host", default=config.MQTT_HOST)
    p.add_argument("--port", type=int, default=config.MQTT_PORT)
    p.add_argument("--topic", default=config.MQTT_TOPIC_TELEMETRY)
    p.add_argument("--client-id", default=config.MQTT_CLIENT_ID)
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    worker = MqttWorker(
        host=args.host,
        port=args.port,
        topic=args.topic,
        client_id=args.client_id,
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

"""OmniPdM MQTT 시뮬레이터 — PRD v6.2 §10 T1-01.

목적:
- 실제 공장 PLC/IoT 게이트웨이 대신 모의 센서 데이터를 publish.
- 동일 토픽 컨벤션 (`omnipdm/telemetry/<device_id>`) 으로 worker 와 e2e 검증.

기본 동작:
- N 개 가상 device 가 일정 주기로 정상 + 가끔 이상 데이터 발행.
- 각 device 의 tool_wear_min 이 시간에 따라 증가 (열화 시뮬).

CLI:
  python -m scripts.realtime.mqtt_simulator --devices 3 --interval 1.0 --duration 30
  python -m scripts.realtime.mqtt_simulator --host localhost --port 1883 --topic omnipdm/telemetry
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import paho.mqtt.client as mqtt

from models_core import config


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_device_state(device_id: str, initial_wear: float = 0.0) -> Dict[str, Any]:
    """device 의 누적 마모(tool_wear_min) 만 외부에서 추적하면 됨."""
    return {"device_id": device_id, "wear": initial_wear}


def sample_sensors(state: Dict[str, Any], rng: random.Random, anomaly_prob: float = 0.15) -> Dict[str, float]:
    """정상 분포 ± 가끔 이상 spike. AI4I 5-sensor 계약 따름.

    rng: 외부에서 주입된 Random 인스턴스 — global random 상태 오염 방지.
    """
    is_anomaly = rng.random() < anomaly_prob
    state["wear"] += rng.uniform(0.5, 1.5)  # 점진적 마모

    if is_anomaly:
        return {
            "air_temperature_k": rng.gauss(300.0, 1.5),
            "process_temperature_k": rng.gauss(315.0, 2.0),
            "rotational_speed_rpm": rng.gauss(1200, 60),     # 낮은 RPM
            "torque_nm": rng.gauss(75.0, 5.0),               # 높은 torque
            "tool_wear_min": min(300.0, state["wear"] + rng.uniform(30, 60)),  # 급격한 마모 spike
        }
    return {
        "air_temperature_k": rng.gauss(298.0, 0.5),
        "process_temperature_k": rng.gauss(308.5, 1.0),
        "rotational_speed_rpm": rng.gauss(1530, 80),
        "torque_nm": rng.gauss(40.0, 3.0),
        "tool_wear_min": state["wear"],
    }


def build_payload(sensors: Dict[str, float]) -> bytes:
    return json.dumps({
        "timestamp": _now_iso(),
        "sensors": {k: round(float(v), 3) for k, v in sensors.items()},
    }).encode("utf-8")


def run(
    host: str,
    port: int,
    topic_prefix: str,
    devices: int,
    interval_sec: float,
    duration_sec: float,
    anomaly_prob: float,
    seed: int,
) -> None:
    rng = random.Random(seed)  # 인스턴스 분리 — global random 오염 방지

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"omnipdm-simulator-{seed}",
        clean_session=True,
    )
    if config.MQTT_USERNAME:
        client.username_pw_set(config.MQTT_USERNAME, config.MQTT_PASSWORD or None)

    try:
        client.connect(host, port, keepalive=config.MQTT_KEEPALIVE_SEC)
    except OSError as e:
        # 브로커 미기동 / 호스트 오타 → traceback 대신 한 줄 안내. 종료는 그대로.
        print(f"[simulator] MQTT broker 연결 실패 ({host}:{port}): {e}")
        raise

    client.loop_start()

    states: List[Dict[str, Any]] = [
        make_device_state(f"milling-{i+1:02d}", initial_wear=rng.uniform(0, 50))
        for i in range(devices)
    ]

    start = time.time()
    n_sent = 0
    try:
        while time.time() - start < duration_sec:
            for s in states:
                sensors = sample_sensors(s, rng=rng, anomaly_prob=anomaly_prob)
                payload = build_payload(sensors)
                topic = f"{topic_prefix.rstrip('/')}/{s['device_id']}"
                client.publish(topic, payload, qos=config.MQTT_QOS)
                n_sent += 1
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("[simulator] interrupted by user")
    finally:
        client.loop_stop()
        client.disconnect()
        print(f"[simulator] published {n_sent} messages over {time.time()-start:.1f}s")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="OmniPdM MQTT telemetry simulator")
    p.add_argument("--host", default=config.MQTT_HOST)
    p.add_argument("--port", type=int, default=config.MQTT_PORT)
    p.add_argument("--topic", default="omnipdm/telemetry",
                   help="토픽 prefix. device_id 가 자동 append 됨")
    p.add_argument("--devices", type=int, default=3)
    p.add_argument("--interval", type=float, default=1.0, help="device 1회 발행 주기(초)")
    p.add_argument("--duration", type=float, default=30.0, help="총 실행 시간(초)")
    p.add_argument("--anomaly-prob", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    run(
        host=args.host,
        port=args.port,
        topic_prefix=args.topic,
        devices=args.devices,
        interval_sec=args.interval,
        duration_sec=args.duration,
        anomaly_prob=args.anomaly_prob,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

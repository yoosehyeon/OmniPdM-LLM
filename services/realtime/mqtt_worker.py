"""MqttWorker — Tier 1 T1-01.

설계:
- MQTT 브로커에서 telemetry 를 구독하고, 메시지 1건마다 PdmService→RiskService→Notifier 파이프라인 실행.
- broker 가 없어도 process_message() 를 직접 호출하면 동일 파이프라인이 돈다 → smoke 테스트 가능.
- 콜백 내부에서 어떤 예외가 발생해도 워커 루프를 죽이지 않는다 (운영 안정성).

메시지 계약:
- Topic 형식  : `omnipdm/telemetry/<device_id>`
- Payload (JSON):
    {
      "timestamp": "2026-05-19T21:50:00Z",
      "sensors": {
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.5,
        "rotational_speed_rpm": 1500,
        "torque_nm": 40.0,
        "tool_wear_min": 105
      }
    }
  현재 PoC 는 AI4I (5-sensor scalar) 만 지원. LSTM sequence 입력은 추후 확장.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import paho.mqtt.client as mqtt

from models_core import config
from services.pdm_service import PdmService
from services.realtime.db_writer import DbWriter, NullDbWriter
from services.realtime.device_service import DeviceService, NullDeviceService
from services.realtime.notifier import Notifier, NotifyResult, StdoutNotifier
from services.risk_service import RiskService


@dataclass
class WorkerStats:
    """워커 누적 통계 — smoke 테스트와 운영 모니터링 양쪽에 유용."""

    received: int = 0
    processed_ok: int = 0
    parse_errors: int = 0
    inference_errors: int = 0
    sent: int = 0
    filtered: int = 0
    rate_limited: int = 0
    failed: int = 0
    # DbWriter — 통합 후 별도 카운트 (writer 자체가 fail_count 보유하지만 워커 측 집계도 유지)
    db_writes: int = 0
    db_errors: int = 0


class MqttWorker:
    """MQTT telemetry 구독 → 추론 → 위험 등급 → 알람 파이프라인.

    Dependency Injection 으로 PdmService / RiskService / Notifier 를 받는다 →
    smoke 테스트에서 lite mode PdmService + StdoutNotifier 로 broker 없이도 검증 가능.
    """

    def __init__(
        self,
        pdm: Optional[PdmService] = None,
        risk: Optional[RiskService] = None,
        notifier: Optional[Notifier] = None,
        db: Optional[DbWriter] = None,
        devices: Optional[DeviceService] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        topic: Optional[str] = None,
        client_id: Optional[str] = None,
        keepalive_sec: Optional[int] = None,
        qos: Optional[int] = None,
    ) -> None:
        self.pdm = pdm if pdm is not None else PdmService(mode="lite", dataset_key="ai4i_cnn")
        self.risk = risk if risk is not None else RiskService(method="weighted")
        self.notifier: Notifier = notifier if notifier is not None else StdoutNotifier()
        # DbWriter 기본은 NullDbWriter — 외부 DB 없이도 워커 동작. TimescaleDbWriter 는
        # 호출자가 명시적으로 주입 (run_worker.py 가 config 보고 결정).
        self.db: DbWriter = db if db is not None else NullDbWriter()
        # DeviceService — telemetry FK (devices.device_id) 충족을 위해 publish 직전 upsert.
        # T1.5 P0. 기본은 NullDeviceService — DB 없이도 워커 hot path 동작.
        self.devices: DeviceService = devices if devices is not None else NullDeviceService()

        self.host = host if host is not None else config.MQTT_HOST
        self.port = port if port is not None else config.MQTT_PORT
        self.topic = topic if topic is not None else config.MQTT_TOPIC_TELEMETRY
        self.keepalive_sec = keepalive_sec if keepalive_sec is not None else config.MQTT_KEEPALIVE_SEC
        self.qos = qos if qos is not None else config.MQTT_QOS

        # 다중 워커 시 client_id 충돌 방지를 위해 PID suffix 추가.
        base_id = client_id if client_id is not None else config.MQTT_CLIENT_ID
        self.client_id = f"{base_id}-{os.getpid()}"

        self.stats = WorkerStats()
        self._client: Optional[mqtt.Client] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_message(self, topic: str, payload_bytes: bytes) -> NotifyResult:
        """한 건의 telemetry 메시지를 처리한다 (broker 무관).

        에러는 모두 잡아서 stats 에 누적하고 NotifyResult.FAILED 를 반환한다 —
        워커 메인 루프는 절대 죽지 않는다.
        """
        self.stats.received += 1

        # 1. Topic → device_id 추출 (마지막 토큰)
        device_id = self._extract_device_id(topic)
        if not device_id:
            self.stats.parse_errors += 1
            self.stats.failed += 1
            print(f"[MqttWorker] invalid topic (no device_id): {topic}", flush=True)
            return NotifyResult.FAILED

        # 2. Payload 파싱
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
            sensors = payload.get("sensors")
            if not isinstance(sensors, dict):
                raise ValueError("payload['sensors'] must be a dict")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
            self.stats.parse_errors += 1
            self.stats.failed += 1
            print(f"[MqttWorker] payload parse failed (device={device_id}): {e}", flush=True)
            return NotifyResult.FAILED

        # 2.4 DeviceService: 첫 publish 인 device 만 INSERT. 캐시 hit 시 DB 왕복 0회 (T1.5 P0).
        # FK (telemetry.device_id → devices.device_id) 충족용. 실패는 _db_safe 가 흡수.
        self._db_safe(
            lambda: self.devices.upsert_device(device_id, self.pdm.dataset_key),
            op="upsert_device",
        )

        # 2.5 DB: telemetry 저장 (추론과 무관하게 raw 보존 — 파싱 통과만 하면 기록).
        # DbWriter 실패는 워커 루프 죽이지 않음, _db_safe 가 내부에서 catch.
        self._db_safe(
            lambda: self.db.write_telemetry(device_id, self.pdm.dataset_key, sensors),
            op="write_telemetry",
        )

        # 3. 추론 → 위험 등급
        try:
            pred = self.pdm.predict(sensors)
            risk_result = self.risk.fuse(pred)
        except Exception as e:
            # PdmService 가 던지는 KeyError(센서 누락) / ValueError 등을 모두 흡수.
            self.stats.inference_errors += 1
            self.stats.failed += 1
            print(f"[MqttWorker] inference failed (device={device_id}): {e}", flush=True)
            return NotifyResult.FAILED

        self.stats.processed_ok += 1

        # 3.5 DB: prediction 저장 (모든 정상 추론 보존 → Grafana 시계열 트렌드 소스)
        self._db_safe(
            lambda: self.db.write_prediction(device_id, self.pdm.dataset_key, pred, risk_result),
            op="write_prediction",
        )

        # 4. 알람 (필터/rate limit 은 Notifier 가 결정)
        try:
            result = self.notifier.notify(device_id, pred, risk_result)
        except Exception as e:
            # Notifier 가 외부 API 호출하다 죽어도 워커는 살아남는다.
            self.stats.failed += 1
            print(f"[MqttWorker] notifier error (device={device_id}): {e}", flush=True)
            return NotifyResult.FAILED

        # 4.5 DB: alert 발송 시도 결과 저장 (SENT/FILTERED/RATE_LIMITED/FAILED 모두 기록 → 감사 추적)
        self._db_safe(
            lambda: self.db.write_alert(
                device_id=device_id,
                dataset_key=self.pdm.dataset_key,
                risk=risk_result,
                channel=type(self.notifier).__name__,
                notify_result=result.name,
            ),
            op="write_alert",
        )

        # 5. 결과 카운터
        if result is NotifyResult.SENT:
            self.stats.sent += 1
        elif result is NotifyResult.FILTERED:
            self.stats.filtered += 1
        elif result is NotifyResult.RATE_LIMITED:
            self.stats.rate_limited += 1
        else:
            self.stats.failed += 1

        return result

    def _db_safe(self, fn, op: str) -> None:
        """DbWriter 호출을 절대 외부로 누설하지 않는 wrapper.

        DbWriter 자체 _execute 도 예외 catch 하지만, 여기서 한 번 더 가드해
        Notifier 와 동일한 운영 안정성 보장.
        """
        try:
            fn()
            self.stats.db_writes += 1
        except Exception as e:
            self.stats.db_errors += 1
            print(f"[MqttWorker] {op} failed: {type(e).__name__}: {e}", flush=True)

    def start(self, block: bool = True) -> None:
        """브로커 연결 + 구독 + 루프 시작.

        block=True (기본): loop_forever() — 메인 스레드가 워커가 됨.
        block=False        : loop_start() — 백그라운드 스레드에서 동작, 호출자가 stop() 호출 책임.
        """
        if self._client is not None:
            raise RuntimeError("MqttWorker is already started.")

        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.client_id,
            clean_session=True,
        )
        # 인증 정보가 비어 있으면 username_pw_set 호출 자체를 생략 → allow_anonymous 호환.
        if config.MQTT_USERNAME:
            client.username_pw_set(config.MQTT_USERNAME, config.MQTT_PASSWORD or None)

        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.on_disconnect = self._on_disconnect

        print(f"[MqttWorker] connecting to {self.host}:{self.port} as '{self.client_id}'", flush=True)
        client.connect(self.host, self.port, keepalive=self.keepalive_sec)

        self._client = client
        if block:
            client.loop_forever()
        else:
            client.loop_start()

    def stop(self) -> None:
        """클라이언트 정리. start(block=False) 와 함께 사용. DbWriter 도 함께 종료."""
        if self._client is not None:
            try:
                self._client.loop_stop()
            except Exception:
                pass
            try:
                self._client.disconnect()
            except Exception:
                pass
        # DB connection 정리 — start() 호출 안 했어도 안전 (NullDbWriter.close() = no-op).
        try:
            self.db.close()
        except Exception:
            pass
        # DeviceService 도 별도 connection 일 수 있음 — 함께 정리.
        try:
            self.devices.close()
        except Exception:
            pass
        self._client = None

    # ------------------------------------------------------------------
    # MQTT callbacks  (paho-mqtt v2 CallbackAPIVersion.VERSION2 시그니처)
    # ------------------------------------------------------------------
    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
        # paho v1: int / v2: ReasonCode 객체. int(ReasonCode) 는 TypeError 라 .value 사용.
        # getattr fallback 으로 양쪽 모두 호환.
        rc_value = getattr(reason_code, "value", reason_code)
        if rc_value == 0:
            print(f"[MqttWorker] connected, subscribing topic='{self.topic}' qos={self.qos}", flush=True)
            client.subscribe(self.topic, qos=self.qos)
        else:
            print(f"[MqttWorker] connect failed (reason_code={reason_code})", flush=True)

    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        # process_message 내부에서 모든 예외를 잡지만, 콜백 자체 한 번 더 가드.
        try:
            self.process_message(msg.topic, msg.payload)
        except Exception as e:
            print(f"[MqttWorker] unexpected callback error: {e}", flush=True)

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, disconnect_flags: Any, reason_code: Any, properties: Any = None) -> None:
        # paho-mqtt 가 자동 reconnect 를 시도하므로 로그만 남긴다.
        print(f"[MqttWorker] disconnected (reason_code={reason_code})", flush=True)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_device_id(topic: str) -> str:
        # 기대: omnipdm/telemetry/<device_id>
        parts = topic.split("/")
        if not parts:
            return ""
        return parts[-1].strip()

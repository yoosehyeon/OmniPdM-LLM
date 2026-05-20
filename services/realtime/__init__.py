"""OmniPdM Realtime worker — PRD v6.2 §10 Tier 1.

이 패키지는 services/ 의 기존 PdmService/RiskService 를 그대로 재사용하여
MQTT telemetry 를 구독하고 위험도가 임계 이상이면 외부 채널로 알람을 보낸다.

진입점 (예정): scripts/realtime/run_worker.py (CLI)
시뮬레이터 (예정): scripts/realtime/mqtt_simulator.py
"""

from services.realtime.db_writer import DbWriter, NullDbWriter, TimescaleDbWriter
from services.realtime.device_service import (
    Device,
    DeviceService,
    NullDeviceService,
    TimescaleDeviceService,
    scope_required,
)
from services.realtime.mqtt_worker import MqttWorker, WorkerStats
from services.realtime.notifier import (
    Notifier,
    NotifyResult,
    StdoutNotifier,
    format_alert,
)

__all__ = [
    "MqttWorker",
    "WorkerStats",
    "Notifier",
    "NotifyResult",
    "StdoutNotifier",
    "format_alert",
    "DbWriter",
    "NullDbWriter",
    "TimescaleDbWriter",
    "Device",
    "DeviceService",
    "NullDeviceService",
    "TimescaleDeviceService",
    "scope_required",
]

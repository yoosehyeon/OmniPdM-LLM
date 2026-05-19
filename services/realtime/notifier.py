"""Pluggable Notifier — Tier 1 T1-02.

설계 의도:
- 워커는 채널 종류를 모른다. Notifier protocol 만 받아서 호출한다.
- 기본 구현은 StdoutNotifier — 외부 의존성 0, broker 없이도 smoke 가능.
- 추후 TelegramNotifier / EmailNotifier / SlackNotifier 등을 같은 protocol 로 추가하면
  워커 코드는 한 줄도 바뀌지 않는다.

채널 후보 (추후 선택, 현재 미구현):
- Email (SMTP)   — 산업현장 표준에 근접, stdlib smtplib 만으로 가능, 즉시성 보통.
- Telegram Bot   — 무료, 모바일 푸시 직접. MarkdownV2 사용 시 '.', '_', '*', '(' escape 필수.
- SMS (Aligo 등) — 한국 현장 정합성 높지만 건당 비용 발생.
- PLC 알람 릴레이 — 현장 표준이지만 OPC-UA/Modbus 인터록 필요 (PRD T3-03 범위).

Thread safety:
- paho-mqtt 의 loop_forever() / loop_start() 는 콜백을 백그라운드 스레드에서 실행한다.
- 같은 워커 인스턴스에서 publish 등 다른 스레드와 RateLimiter 가 공유될 수 있으므로
  RLock 으로 보호한다 (단일 스레드 환경에서는 비용 거의 0).
"""

from __future__ import annotations

import time
from enum import Enum, auto
from threading import RLock
from typing import Dict, Optional, Protocol, runtime_checkable

from models_core import config
from services.schemas import PredictionResult, RiskResult


class NotifyResult(Enum):
    """notify() 호출 결과를 명시적으로 구분 — 워커가 메트릭/로깅을 카테고리별로 집계할 수 있다."""

    SENT = auto()          # 실제 채널로 발송됨
    FILTERED = auto()      # min_level 미달로 발송 안 함
    RATE_LIMITED = auto()  # 동일 device_id 가 rate_limit_sec 내 재발화로 차단
    FAILED = auto()        # 채널 발송 자체가 실패 (네트워크 / API 오류 등)


# RISK_LEVELS 는 (label, threshold) 내림차순 정렬 — config.py 의 명시적 계약.
# label 의 "심각도 순위" 를 사전 계산: Critical=0, Warning=1, Advisory=2, Normal=3.
_LEVEL_RANK: Dict[str, int] = {label: idx for idx, (label, _) in enumerate(config.RISK_LEVELS)}


@runtime_checkable
class Notifier(Protocol):
    """알람 채널 추상화. 새 채널을 추가하려면 이 protocol 을 따르기만 하면 된다."""

    def notify(
        self,
        device_id: str,
        pred: PredictionResult,
        risk: RiskResult,
        extra_text: Optional[str] = None,
    ) -> NotifyResult:
        """발송 시도. 결과를 NotifyResult enum 으로 반환."""
        ...


class _RateLimiter:
    """동일 device_id 가 짧은 시간 안에 또 발화하면 중복 발송 억제 (thread-safe)."""

    def __init__(self, rate_limit_sec: int) -> None:
        self.rate_limit_sec = rate_limit_sec
        self._last_sent_at: Dict[str, float] = {}
        self._lock = RLock()

    def allow(self, device_id: str, now: Optional[float] = None) -> bool:
        """rate_limit_sec <= 0 이면 항상 True (rate limit off — 테스트/긴급 모드용)."""
        if self.rate_limit_sec <= 0:
            return True
        now_ts = time.time() if now is None else now
        with self._lock:
            last = self._last_sent_at.get(device_id)
            if last is not None and (now_ts - last) < self.rate_limit_sec:
                return False
            self._last_sent_at[device_id] = now_ts
            return True


def _passes_level_filter(level: str, min_level: str) -> bool:
    """level 이 min_level 이상 심각도인지."""
    if level not in _LEVEL_RANK or min_level not in _LEVEL_RANK:
        return False
    return _LEVEL_RANK[level] <= _LEVEL_RANK[min_level]


def format_alert(
    device_id: str,
    pred: PredictionResult,
    risk: RiskResult,
    extra_text: Optional[str] = None,
) -> str:
    """모든 Notifier 구현이 공통으로 쓸 수 있는 plain-text 포맷.

    채널별 마크업(Slack mrkdwn / Telegram MarkdownV2 등) 이 필요하면
    각 Notifier 구현에서 별도 변환 함수를 사용하라.
    """
    lines = [
        f"[OmniPdM] {risk.risk_level} - device={device_id}",
        f"  risk_score = {risk.risk_score:.3f} (method={risk.method})",
        f"  failure_prob = {pred.failure_probability:.3f}, "
        f"anomaly = {pred.anomaly_score:.3f}, rul_norm = {pred.rul_norm:.3f}",
        f"  model = {pred.model_name} ({pred.dataset_key})",
    ]
    if pred.top_contributors:
        top = ", ".join(f"{name}={val:.2f}" for name, val in pred.top_contributors[:3])
        lines.append(f"  top_contributors = {top}")
    if extra_text:
        lines.append(f"  note = {extra_text}")
    return "\n".join(lines)


class StdoutNotifier:
    """기본 채널 — 표준출력으로 알람 출력. 외부 의존성 0.

    워커 검증 / broker 없는 smoke / 채널 미정 단계에서 사용.
    실제 채널 선정 후에도 디버깅용으로 계속 유용.
    """

    def __init__(
        self,
        min_level: Optional[str] = None,
        rate_limit_sec: Optional[int] = None,
    ) -> None:
        self.min_level = min_level if min_level is not None else config.ALERT_MIN_LEVEL
        rl = rate_limit_sec if rate_limit_sec is not None else config.ALERT_RATE_LIMIT_SEC

        if self.min_level not in _LEVEL_RANK:
            raise ValueError(
                f"min_level='{self.min_level}' 는 유효한 등급이 아닙니다. "
                f"가능: {sorted(_LEVEL_RANK)}"
            )

        self._rl = _RateLimiter(rl)

    def notify(
        self,
        device_id: str,
        pred: PredictionResult,
        risk: RiskResult,
        extra_text: Optional[str] = None,
    ) -> NotifyResult:
        if not _passes_level_filter(risk.risk_level, self.min_level):
            return NotifyResult.FILTERED
        if not self._rl.allow(device_id):
            return NotifyResult.RATE_LIMITED

        text = format_alert(device_id, pred, risk, extra_text)
        # 한 메시지가 여러 줄이라도 묶어서 출력 → 로그 파싱 친화.
        print(text, flush=True)
        return NotifyResult.SENT

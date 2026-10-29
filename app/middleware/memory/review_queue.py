"""
===============================================================================
Idle Review Queue — 세션 종료 감지 후 에피소드 추출 실행
===============================================================================
채팅 세션에는 명시적인 종료가 없으므로 "대화가 멈춘 시점"을 세션 종료로 봅니다.
세션 종료 조건은 session_end_reason() 한 곳에서 판단하고, 조건을 만족한 세션의
에피소드 추출(finalize_session)을 실행합니다.

Hermes 대응: 외부 메모리 provider의 on_session_end는 /new·종료·압축 같은 명시적 경계와
게이트웨이 1시간 유휴(gateway/run_agent_cache.py _AGENT_CACHE_IDLE_TTL_SECS)에서 호출됩니다.
이 저장소는 결과를 바로 확인할 수 있도록 유휴 기준을 짧게(기본 15초) 둔 교육용 설정입니다.
(메모리·스킬 리뷰는 이 큐가 아니라 턴 종료 시 nudge 조건으로 실행됩니다 — MemoryMiddleware 참고)

규칙 (기본값: settle 15초 / max age 30분):
- 세션당 대기 슬롯 1개. 새로 예약하면 최신 예약만 남깁니다 (원문 전체를 다시 읽으므로 손실 없음).
- 마지막 턴 종료 후 settle_s 동안 새 입력이 없으면 실행합니다.
- 대화가 계속 이어져도 최초 예약 후 max_age_s가 지나면 실행합니다.
- 새 턴이 시작되면(mark_active) 대기 슬롯은 "사용 중"으로 바뀌고, 대기 시간 만료로 실행 중이던
  리뷰는 취소됩니다. 이때 최초 예약 시각은 유지되어 max_age 보장이 깨지지 않습니다.
- max_age로 실행된 리뷰는 취소하지 않고 끝까지 진행합니다.
===============================================================================
"""

from __future__ import annotations

import inspect
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class _Slot:
    payload: Dict[str, Any]
    first_at: float
    last_activity: float
    busy: bool = False


@dataclass
class _Running:
    cancel: Callable[[], None]
    reason: str
    first_at: float
    payload: Dict[str, Any] = field(default_factory=dict)


class IdleReviewQueue:
    """세션별 에피소드 추출을 세션 종료 조건(session_end_reason)이 될 때까지 미뤘다가 실행하는 큐.

    Args:
        dispatch: (session_id, payload, reason) -> cancel 함수. 실제 리뷰를 스레드 풀에 제출합니다.
        settle_s: 유휴로 판단할 대기 시간(초). 0 이하면 유휴 즉시 실행합니다.
        max_age_s: 최초 예약 후 이 시간이 지나면 유휴 여부와 무관하게 실행합니다.
        poll_s: 감시 스레드의 최대 대기 간격(초).
    """

    def __init__(
        self,
        dispatch: Optional[Callable[..., Any]] = None,
        settle_s: float = 15.0,
        max_age_s: float = 1800.0,
        poll_s: float = 1.0,
        dispatch_fn: Optional[Callable[..., Any]] = None,
    ):
        self._dispatch = dispatch or dispatch_fn
        if self._dispatch is None:
            raise ValueError("IdleReviewQueue requires dispatch or dispatch_fn")
        self.settle_s = float(settle_s)
        self.max_age_s = float(max_age_s)
        self.poll_s = float(poll_s)
        self._slots: Dict[str, _Slot] = {}
        self._running: Dict[str, _Running] = {}
        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._stopped = False

    # ── Public API ──

    def schedule(
        self,
        session_id: str,
        payload_or_user_id: Any = None,
        payload: Optional[Dict[str, Any]] = None,
        *,
        user_id: Optional[str] = None,
    ) -> None:
        """턴 종료 시 호출: 리뷰를 예약합니다 (최신 예약만 유지)."""
        if payload is None:
            actual_payload: Dict[str, Any] = (
                dict(payload_or_user_id) if isinstance(payload_or_user_id, dict) else {}
            )
            if isinstance(payload_or_user_id, str) and "user_id" not in actual_payload:
                actual_payload["user_id"] = payload_or_user_id
        else:
            actual_payload = dict(payload)
            if isinstance(payload_or_user_id, str) and "user_id" not in actual_payload:
                actual_payload["user_id"] = payload_or_user_id
        if user_id and "user_id" not in actual_payload:
            actual_payload["user_id"] = user_id
        actual_payload.setdefault("session_id", session_id)

        now = time.monotonic()
        with self._cond:
            slot = self._slots.get(session_id)
            if slot is None:
                slot = _Slot(payload=actual_payload, first_at=now, last_activity=now)
                self._slots[session_id] = slot
            else:
                slot.payload = actual_payload
                slot.last_activity = now
            slot.busy = False
            self._ensure_thread()
            self._cond.notify_all()
        if self.settle_s <= 0:
            self._dispatch_due()

    def mark_active(self, session_id: str) -> None:
        """새 턴 시작 시 호출: 유휴 상태를 해제하고, 유휴로 실행 중이던 리뷰를 취소합니다."""
        now = time.monotonic()
        cancel = None
        with self._cond:
            slot = self._slots.get(session_id)
            if slot is not None:
                slot.busy = True
                slot.last_activity = now
            running = self._running.get(session_id)
            if running is not None and running.reason != "max_age":
                self._running.pop(session_id, None)
                cancel = running.cancel
                if slot is None:
                    # 취소된 리뷰를 다시 대기열에 올리되 최초 예약 시각은 유지 (max_age 보장)
                    self._slots[session_id] = _Slot(payload=running.payload, first_at=running.first_at,
                                                    last_activity=now, busy=True)
        if cancel is not None:
            try:
                cancel()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("cancel callback failed: %s", e)

    def finished(self, session_id: str, cancel_event: Any = None) -> None:
        """리뷰 완료/취소 시 dispatch 측에서 호출합니다 (같은 세션의 더 새 리뷰 기록은 지우지 않음)."""
        with self._cond:
            running = self._running.get(session_id)
            if running is None:
                return
            if cancel_event is not None and running.cancel != getattr(cancel_event, "set", None):
                return
            self._running.pop(session_id, None)

    def flush(self) -> None:
        """대기 중인(비활성) 모든 슬롯을 즉시 실행합니다 (테스트·종료 시 사용)."""
        with self._cond:
            due = [(sid, slot) for sid, slot in self._slots.items() if not slot.busy]
            for sid, _ in due:
                self._slots.pop(sid, None)
        for sid, slot in due:
            self._start(sid, slot, "flush")

    def pending(self, session_id: Optional[str] = None) -> Union[bool, int]:
        with self._cond:
            if session_id is not None:
                slot = self._slots.get(session_id)
                return bool(slot is not None and not slot.busy)
            return sum(1 for s in self._slots.values() if not s.busy)

    def shutdown(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    # ── Internals ──

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="memory-idle-review", daemon=True)
            self._thread.start()

    def session_end_reason(self, slot: _Slot, now: float) -> Optional[str]:
        """세션 종료 조건: 세션이 끝난 것으로 볼 이유("idle" / "max_age"), 아니면 None.

        - idle: 마지막 턴 종료 후 settle_s 동안 새 입력이 없음
        - max_age: 대화가 계속 이어져도 최초 예약 후 max_age_s가 지남 (긴 세션도 주기적으로 추출)
        """
        if now - slot.first_at >= self.max_age_s:
            return "max_age"
        if not slot.busy and now - slot.last_activity >= self.settle_s:
            return "idle"
        return None

    def _collect_due(self) -> list:
        now = time.monotonic()
        due = []
        for sid, slot in list(self._slots.items()):
            reason = self.session_end_reason(slot, now)
            if reason:
                due.append((sid, slot, reason))
        for sid, _, _ in due:
            self._slots.pop(sid, None)
        return due

    def _dispatch_due(self) -> None:
        with self._cond:
            due = self._collect_due()
        for sid, slot, reason in due:
            self._start(sid, slot, reason)

    def _start(self, session_id: str, slot: _Slot, reason: str) -> None:
        params = list(inspect.signature(self._dispatch).parameters.values())
        try:
            if len(params) == 2:
                # 실행 중 등록을 먼저 해야, 리뷰가 곧바로 끝나 finished()가 먼저 불려도 기록이 남지 않습니다
                cancel_event = threading.Event()
                running = _Running(cancel=cancel_event.set, reason=reason, first_at=slot.first_at,
                                   payload=slot.payload)
                with self._cond:
                    self._running[session_id] = running
                res = self._dispatch(slot.payload, cancel_event)
                if callable(res):
                    running.cancel = res
            else:
                res = self._dispatch(session_id, slot.payload, reason)
                cancel = res if callable(res) else (lambda: None)
                with self._cond:
                    self._running[session_id] = _Running(cancel=cancel, reason=reason, first_at=slot.first_at,
                                                         payload=slot.payload)
        except Exception as e:
            with self._cond:
                self._running.pop(session_id, None)
            logger.warning("[IdleReviewQueue] dispatch failed for %s: %s", session_id, e)
            return
        logger.info("[IdleReviewQueue] review dispatched (session=%s, reason=%s)", session_id, reason)

    def _loop(self) -> None:
        while True:
            with self._cond:
                if self._stopped:
                    return
                self._cond.wait(timeout=self.poll_s)
                if self._stopped:
                    return
                due = self._collect_due()
            for sid, slot, reason in due:
                self._start(sid, slot, reason)


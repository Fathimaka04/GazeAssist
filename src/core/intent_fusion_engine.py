"""
src/core/intent_fusion_engine.py

Week 9/10/11: Multimodal Fusion Engine
------------------------------------
Combines the two independent outputs of GazeBlinkEngine (zone + blink)
into a single "intent" signal:

    - A stable gaze zone + a LONG blink -> AWAITING_CONFIRM (not yet
      spoken). A second LONG blink in the SAME zone within
      CONFIRM_WINDOW_SEC -> ZONE_SELECTED (confirmed, speak it). No
      repeat in time -> silently cancels back to normal, no event,
      no side effects.
    - Single-hold SOS with a confirm step: one extra-long hold
      (>= EXTRA_LONG_BLINK_MIN_FRAMES) ARMS SOS; a second extra-long
      hold within SOS_ARM_CONFIRM_WINDOW_SEC CONFIRMS it
      (SOS_TRIGGERED). No confirm in time -> silently cancels back to
      normal, no side effects.

WEEK 11 CHANGE (repeat-to-confirm): ZONE_SELECTED no longer fires
immediately off a single qualifying long blink. The first qualifying
blink now only arms a per-zone "awaiting confirm" state
(AWAITING_CONFIRM), mirroring the existing SOS arm/confirm pattern.
IMPORTANT: cooldown is NOT applied after AWAITING_CONFIRM (only after
a final confirmed ZONE_SELECTED, or an SOS event) — otherwise the
confirming blink itself would be blocked by COOLDOWN_SEC and the
confirm step would be permanently unreachable.

An extra-long (SOS-length) hold always takes priority: it cancels any
pending zone confirm, matching the existing "SOS overrides pending
selection" rule from Week 9/10.

CONFIRM_WINDOW_SEC=3.0 is an untuned starting guess (not yet
live-measured) — expect to retune this from real hold-to-hold timing
data the same way SOS_ARM_CONFIRM_WINDOW_SEC was tuned (2.0s -> 5.0s
-> 7.0s -> 10.0s) in Week 10. It is deliberately shorter than the SOS
window since confirming a phrase selection is a lower-stakes, more
frequent action than confirming SOS.
"""

from dataclasses import dataclass
from enum import Enum
import time


class IntentType(Enum):
    NONE = "none"
    AWAITING_CONFIRM = "awaiting_confirm"
    ZONE_SELECTED = "zone_selected"
    SOS_ARMED = "sos_armed"
    SOS_TRIGGERED = "sos_triggered"


@dataclass
class IntentEvent:
    type: IntentType
    zone: str = None
    long_blink_count: int = 0
    timestamp: float = 0.0


class IntentFusionEngine:
    ZONE_STABLE_TIME_SEC = 0.5
    EXTRA_LONG_BLINK_MIN_FRAMES = 55
    SOS_ARM_CONFIRM_WINDOW_SEC = 10.0
    CONFIRM_WINDOW_SEC = 3.0  # untuned starting guess, see module docstring
    COOLDOWN_SEC = 1.5

    def __init__(self):
        self._current_zone = None
        self._zone_stable_since = None
        self._armed_at = None
        self._pending_confirm_zone = None
        self._pending_confirm_at = None
        self._cooldown_until = 0.0

    def update(self, zone: str, blink_type: str, blink_duration_frames: int = 0,
               now: float = None) -> IntentEvent:
        if now is None:
            now = time.time()

        if zone != self._current_zone:
            self._current_zone = zone
            self._zone_stable_since = now
        zone_stable_duration = now - self._zone_stable_since
        # print(f"DEBUG: zone={zone}, stable_for={zone_stable_duration:.2f}s, blink_type={blink_type}, dur={blink_duration_frames}")
        in_cooldown = now < self._cooldown_until

        if self._armed_at is not None and (now - self._armed_at) > self.SOS_ARM_CONFIRM_WINDOW_SEC:
            self._armed_at = None

        if (self._pending_confirm_zone is not None
                and (now - self._pending_confirm_at) > self.CONFIRM_WINDOW_SEC):
            self._clear_pending_confirm()

        if blink_type == "long" and not in_cooldown:

            if blink_duration_frames >= self.EXTRA_LONG_BLINK_MIN_FRAMES:
                self._clear_pending_confirm()  # SOS overrides any pending zone confirm
                if self._armed_at is not None:
                    event = IntentEvent(
                        type=IntentType.SOS_TRIGGERED,
                        long_blink_count=2,
                        timestamp=now,
                    )
                    self._reset_sequence()
                    self._cooldown_until = now + self.COOLDOWN_SEC
                    return event
                else:
                    self._armed_at = now
                    return IntentEvent(
                        type=IntentType.SOS_ARMED,
                        long_blink_count=1,
                        timestamp=now,
                    )

            if self._armed_at is None:

                # Second qualifying blink in the SAME zone -> confirmed
                if self._pending_confirm_zone == zone and zone is not None:
                    event = IntentEvent(
                        type=IntentType.ZONE_SELECTED,
                        zone=zone,
                        long_blink_count=2,
                        timestamp=now,
                    )
                    self._clear_pending_confirm()
                    self._cooldown_until = now + self.COOLDOWN_SEC
                    return event

                # First qualifying blink -> arm the pending confirm
                if zone_stable_duration >= self.ZONE_STABLE_TIME_SEC and zone != "CENTER":
                    self._pending_confirm_zone = zone
                    self._pending_confirm_at = now
                    # No cooldown set here — the confirming blink must
                    # be allowed to register, possibly in quick succession.
                    return IntentEvent(
                        type=IntentType.AWAITING_CONFIRM,
                        zone=zone,
                        long_blink_count=1,
                        timestamp=now,
                    )

        if self._armed_at is not None:
            return IntentEvent(type=IntentType.SOS_ARMED, long_blink_count=1, timestamp=now)

        if self._pending_confirm_zone is not None:
            return IntentEvent(
                type=IntentType.AWAITING_CONFIRM,
                zone=self._pending_confirm_zone,
                long_blink_count=1,
                timestamp=now,
            )

        return IntentEvent(type=IntentType.NONE, timestamp=now)

    def _clear_pending_confirm(self):
        self._pending_confirm_zone = None
        self._pending_confirm_at = None

    def _reset_sequence(self):
        self._armed_at = None
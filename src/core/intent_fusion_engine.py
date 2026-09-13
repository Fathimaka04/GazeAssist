"""
src/core/intent_fusion_engine.py
"""

from dataclasses import dataclass
from enum import Enum
import time


class IntentType(Enum):
    NONE = "none"
    PENDING = "pending"
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
    LONG_BLINK_SEQUENCE_WINDOW_SEC = 1.5
    EXTRA_LONG_BLINK_MIN_FRAMES = 60

    # How long the user has, after the first extra-long hold arms SOS,
    # to do a SECOND extra-long hold to confirm it. Starting guess -
    # not yet measured/tuned against real usage.
    SOS_ARM_CONFIRM_WINDOW_SEC = 10.0

    COOLDOWN_SEC = 1.5

    def __init__(self):
        self._current_zone = None
        self._zone_stable_since = None

        self._last_long_blink_time = None
        self._pending_zone = None

        self._armed_at = None  # timestamp of the arming hold, or None if not armed

        self._cooldown_until = 0.0

    def update(self, zone: str, blink_type: str, blink_duration_frames: int = 0,
               now: float = None) -> IntentEvent:
        if now is None:
            now = time.time()

        if zone != self._current_zone:
            self._current_zone = zone
            self._zone_stable_since = now
        zone_stable_duration = now - self._zone_stable_since

        in_cooldown = now < self._cooldown_until

        # --- If armed but the confirm window has passed, cancel silently ---
        if self._armed_at is not None and (now - self._armed_at) > self.SOS_ARM_CONFIRM_WINDOW_SEC:
            print(f"DEBUG: SOS arm expired after {now - self._armed_at:.2f}s (window was {self.SOS_ARM_CONFIRM_WINDOW_SEC}s)")
            self._armed_at = None

        if blink_type == "long" and not in_cooldown:

            if self._armed_at is not None:
                print(f"DEBUG: blink while armed - duration={blink_duration_frames}, time_since_armed={now - self._armed_at:.2f}s")

            if blink_duration_frames >= self.EXTRA_LONG_BLINK_MIN_FRAMES:
                if self._armed_at is not None:
                    # --- Second extra-long hold within the window: CONFIRMED ---
                    event = IntentEvent(
                        type=IntentType.SOS_TRIGGERED,
                        long_blink_count=2,
                        timestamp=now,
                    )
                    self._reset_sequence()
                    self._cooldown_until = now + self.COOLDOWN_SEC
                    return event
                else:
                    # --- First extra-long hold: ARM, cancel any pending selection ---
                    self._armed_at = now
                    self._pending_zone = None
                    self._last_long_blink_time = None
                    return IntentEvent(
                        type=IntentType.SOS_ARMED,
                        long_blink_count=1,
                        timestamp=now,
                    )

            # --- Normal selection long-blink - ignored entirely while armed,
            #     so it can't accidentally cancel or confirm anything ---
            if self._armed_at is None:
                self._last_long_blink_time = now

                if zone_stable_duration >= self.ZONE_STABLE_TIME_SEC and zone != "CENTER":
                    self._pending_zone = zone

                if self._pending_zone is not None:
                    return IntentEvent(
                        type=IntentType.PENDING,
                        zone=self._pending_zone,
                        long_blink_count=1,
                        timestamp=now,
                    )

        if self._armed_at is None and self._pending_zone is not None and not in_cooldown and self._last_long_blink_time is not None:
            time_since_last_blink = now - self._last_long_blink_time
            if time_since_last_blink >= self.LONG_BLINK_SEQUENCE_WINDOW_SEC:
                event = IntentEvent(
                    type=IntentType.ZONE_SELECTED,
                    zone=self._pending_zone,
                    long_blink_count=1,
                    timestamp=now,
                )
                self._reset_sequence()
                self._cooldown_until = now + self.COOLDOWN_SEC
                return event

        # --- Still within the confirm window: keep reporting ARMED every
        #     frame so the state machine/display stay in that state ---
        if self._armed_at is not None:
            return IntentEvent(type=IntentType.SOS_ARMED, long_blink_count=1, timestamp=now)

        return IntentEvent(type=IntentType.NONE, timestamp=now)

    def _reset_sequence(self):
        self._last_long_blink_time = None
        self._pending_zone = None
        self._armed_at = None
"""
src/core/intent_fusion_engine.py

Week 9/10: Multimodal Fusion Engine
------------------------------------
Combines the two independent outputs of GazeBlinkEngine (zone + blink)
into a single "intent" signal:

    - A stable gaze zone + a LONG blink -> ZONE_SELECTED immediately
    - Single-hold SOS with a confirm step: one extra-long hold
      (>= EXTRA_LONG_BLINK_MIN_FRAMES) ARMS SOS; a second extra-long
      hold within SOS_ARM_CONFIRM_WINDOW_SEC CONFIRMS it
      (SOS_TRIGGERED). No confirm in time -> silently cancels back to
      normal, no side effects.

SELECTION SIMPLIFIED: a selection long-blink's full duration is known
the instant the blink ends, so the old design's wait-then-resolve
delay (LONG_BLINK_SEQUENCE_WINDOW_SEC after the blink, left over from
an earlier multi-blink design) added pure latency with no benefit.
ZONE_SELECTED now fires in the same update() call as the qualifying
blink, provided the zone was already stable for ZONE_STABLE_TIME_SEC
and isn't CENTER. Live-confirmed: three selections fired instantly in
a row with no delay, immediately followed by a clean SOS arm+confirm.

EXTRA_LONG_BLINK_MIN_FRAMES lowered from 60 to 55 after live testing
showed a confirm-hold attempt (48 frames) falling just short of 60;
55 has since been live-confirmed working for both arm (66 frames) and
confirm (63 frames). Still well above the highest measured selection
blink (41-53 frames).
"""

from dataclasses import dataclass
from enum import Enum
import time


class IntentType(Enum):
    NONE = "none"
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
    COOLDOWN_SEC = 1.5

    def __init__(self):
        self._current_zone = None
        self._zone_stable_since = None
        self._armed_at = None
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

        if self._armed_at is not None and (now - self._armed_at) > self.SOS_ARM_CONFIRM_WINDOW_SEC:
            self._armed_at = None

        if blink_type == "long" and not in_cooldown:

            if blink_duration_frames >= self.EXTRA_LONG_BLINK_MIN_FRAMES:
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
                if zone_stable_duration >= self.ZONE_STABLE_TIME_SEC and zone != "CENTER":
                    event = IntentEvent(
                        type=IntentType.ZONE_SELECTED,
                        zone=zone,
                        long_blink_count=1,
                        timestamp=now,
                    )
                    self._cooldown_until = now + self.COOLDOWN_SEC
                    return event

        if self._armed_at is not None:
            return IntentEvent(type=IntentType.SOS_ARMED, long_blink_count=1, timestamp=now)

        return IntentEvent(type=IntentType.NONE, timestamp=now)

    def _reset_sequence(self):
        self._armed_at = None
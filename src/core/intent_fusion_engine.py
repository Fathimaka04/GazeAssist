"""
src/core/intent_fusion_engine.py

Week 9/10: Multimodal Fusion Engine
------------------------------------
Combines the two independent outputs of GazeBlinkEngine (zone + blink)
into a single "intent" signal:

    - A stable gaze zone + a LONG blink  -> pending zone selection
    - No further long blink within LONG_BLINK_SEQUENCE_WINDOW_SEC of
      the LAST long blink -> zone CONFIRMED
    - ONE extra-long blink hold (>= EXTRA_LONG_BLINK_MIN_FRAMES)
      -> SOS_TRIGGERED immediately (cancels any pending selection)

HISTORY: original design used 3 discrete long blinks for SOS. Live
testing (round 1, Week 9) showed this was hard to execute reliably
even for an able-bodied tester despite widening the timing window
1.0s -> 1.5s -> 2.5s - led to this single-hold, duration-based
redesign (validated: selection 16-39 frames, SOS holds 87-103 frames).

Later reverted back to 3-blink-count by request, with an added fix
gating SOS blinks on zone == CENTER to stop selection blinks and SOS
blinks from being confused with each other. That fix worked for the
mixing problem, but live testing (round 2) reproduced the exact same
timing/sequencing difficulty as round 1 - every attempt reset at
blink #1, individual blink hold durations were wildly inconsistent
(20-136 frames), and the 1.5s window was consistently missed. Same
root cause as before: timed multi-blink sequencing is unreliable
regardless of which zone gates it.

REVERTED AGAIN (by request) to this single-hold design, since it has
no sequencing/timing requirement at all - the difficulty class that
broke 3-blink twice doesn't apply here. EXTRA_LONG_BLINK_MIN_FRAMES
set to 60 (raised from the original 45) per earlier live-measured
data: selection blinks measured up to 41 frames, SOS holds measured
as low as 68 frames across two runs - 60 sits comfortably in that
gap, favoring a more deliberate hold over the original 45.
"""

from dataclasses import dataclass
from enum import Enum
import time


class IntentType(Enum):
    NONE = "none"
    PENDING = "pending"
    ZONE_SELECTED = "zone_selected"
    SOS_TRIGGERED = "sos_triggered"


@dataclass
class IntentEvent:
    type: IntentType
    zone: str = None
    long_blink_count: int = 0
    timestamp: float = 0.0


class IntentFusionEngine:
    # --- Tunable constants ---

    ZONE_STABLE_TIME_SEC = 0.5

    # Max gap after the last long (selection) blink before a pending
    # zone selection auto-confirms as ZONE_SELECTED.
    LONG_BLINK_SEQUENCE_WINDOW_SEC = 1.5

    # A single blink held at least this many frames is treated as an
    # SOS hold instead of a normal selection blink. Originally 45
    # (Week 9: selection 16-39, SOS 87-103). Raised to 60 after a
    # second live run showed selection blinks up to 41 frames and SOS
    # holds as low as 68 frames - 60 sits in the middle of that gap,
    # favoring a more deliberate SOS gesture.
    EXTRA_LONG_BLINK_MIN_FRAMES = 60

    COOLDOWN_SEC = 1.5

    def __init__(self):
        self._current_zone = None
        self._zone_stable_since = None

        self._last_long_blink_time = None
        self._pending_zone = None

        self._cooldown_until = 0.0

    def update(self, zone: str, blink_type: str, blink_duration_frames: int = 0,
               now: float = None) -> IntentEvent:
        if now is None:
            now = time.time()

        # --- Track zone stability ---
        if zone != self._current_zone:
            self._current_zone = zone
            self._zone_stable_since = now
        zone_stable_duration = now - self._zone_stable_since

        in_cooldown = now < self._cooldown_until

        if blink_type == "long" and not in_cooldown:

            # --- SOS: a single extra-long hold, checked first so it
            #     always takes priority over a pending selection ---
            if blink_duration_frames >= self.EXTRA_LONG_BLINK_MIN_FRAMES:
                event = IntentEvent(
                    type=IntentType.SOS_TRIGGERED,
                    long_blink_count=1,
                    timestamp=now,
                )
                self._reset_sequence()
                self._cooldown_until = now + self.COOLDOWN_SEC
                return event

            # --- Normal selection long-blink ---
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

        # --- Resolve a pending selection once the window since the
        #     last selection blink passes with no new one ---
        if self._pending_zone is not None and not in_cooldown and self._last_long_blink_time is not None:
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

        return IntentEvent(type=IntentType.NONE, timestamp=now)

    def _reset_sequence(self):
        self._last_long_blink_time = None
        self._pending_zone = None
"""
src/core/intent_fusion_engine.py

Week 9: Multimodal Fusion Engine
--------------------------------
Combines the two independent outputs of GazeBlinkEngine (zone + blink)
into a single "intent" signal:

    - A stable gaze zone + a LONG blink  -> pending zone selection
    - No further long blink within LONG_BLINK_SEQUENCE_WINDOW_SEC of
      the LAST long blink -> zone CONFIRMED
    - ONE extra-long blink hold (>= EXTRA_LONG_BLINK_MIN_FRAMES)
      -> SOS_TRIGGERED immediately (cancels any pending selection)

REDESIGNED (post-live-testing, round 3): the original SOS design
required 3 discrete long blinks landing within a tight consecutive-gap
window. Extensive live testing (including with the window widened
from 1.0s -> 1.5s -> 2.5s) showed this timed multi-blink sequence was
very difficult to execute reliably even for an able-bodied tester.
Since the target users are non-verbal/motor-impaired individuals with
LESS precise motor control, this was judged a fundamental usability
problem with the gesture design, not a tuning problem.

REPLACED WITH: a single sustained eye closure held past a second,
higher threshold (EXTRA_LONG_BLINK_MIN_FRAMES) triggers SOS directly.
This is one continuous motor action instead of a precisely-timed
sequence, and removes the entire class of bugs previously seen
(inter-blink gap timing, cooldown collisions with prior selections,
premature pending-resolution). It also better matches the physical
capabilities of the actual target population.

VALIDATED (live testing): selection long-blinks measured 16-39 frames,
SOS holds measured 87-103 frames - a clean ~48-frame gap around the
EXTRA_LONG_BLINK_MIN_FRAMES=45 threshold, confirming no retuning
needed on first measurement.

NOTE: this requires the raw blink_duration_frames value (not just the
short/long label) so SOS can be distinguished from a normal selection
long-blink. Exposed via GazeBlinkEngine.EngineResult.blink_duration_frames.

DEBUG FUSION prints removed after live confirmation that SOS_TRIGGERED
fires correctly and reliably (see docs/architecture_week9.md for the
measured data that validated this).
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
    # SOS hold instead of a normal selection blink. Validated via live
    # testing: selection blinks measured 16-39 frames, SOS holds
    # measured 87-103 frames - comfortable margin around this value.
    EXTRA_LONG_BLINK_MIN_FRAMES = 45

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
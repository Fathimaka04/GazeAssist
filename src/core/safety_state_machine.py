"""
src/core/safety_state_machine.py

Week 10: Safety State Machine
------------------------------
Wraps an overall app-level status around the existing outputs of
GazeBlinkEngine (EngineResult) and IntentFusionEngine (IntentEvent).
This is decision-logic only - no new sensing.

States:
    IDLE          - calibration in progress, system not yet usable
    ACTIVE        - calibration done, face tracked normally
    UNCERTAIN     - face lost / tracking unreliable right now
    SOS_ARMED     - reserved for a future confirm-before-SOS step,
                    NOT YET USED (design decision still pending)
    SOS_TRIGGERED - SOS was just triggered, shown briefly before
                    returning to ACTIVE

Design decisions so far:
- Face lost -> UNCERTAIN immediately, with an on-screen warning
  (decided; not silently ignored).
- SOS currently fires immediately on the single long-blink hold,
  going straight to SOS_TRIGGERED with no arming step - this may
  change later depending on the SOS_ARMED design decision, which
  is intentionally left open for now.
- SOS_TRIGGERED is transient: after SOS_DISPLAY_DURATION_SEC, the
  state machine automatically returns to ACTIVE on its own (it does
  NOT wait for any further blink/zone input to clear the alert).
"""

from dataclasses import dataclass
from enum import Enum
import time


class SafetyState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    UNCERTAIN = "uncertain"
    SOS_ARMED = "sos_armed"          # reserved, not yet used
    SOS_TRIGGERED = "sos_triggered"


@dataclass
class StateResult:
    state: SafetyState
    message: str = ""          # short human-readable status, for on-screen display
    state_changed: bool = False


# Calibration-related EngineResult statuses that map to IDLE
CALIBRATING_STATUSES = {
    "calibrating_blink",
    "calibrating_pose",
    "calibrating_directions",
    "calibration_restarted",
}

# How long SOS_TRIGGERED stays displayed before auto-returning to ACTIVE.
# Starting guess - not yet measured/tuned against real usage.
SOS_DISPLAY_DURATION_SEC = 3.0


class SafetyStateMachine:
    def __init__(self):
        self._state = SafetyState.IDLE
        self._sos_triggered_at = None

    def update(self, engine_status: str, intent_type=None, now: float = None) -> StateResult:
        """
        engine_status: EngineResult.status (e.g. "ready", "no_face", "calibrating_blink", ...)
        intent_type: IntentType from IntentFusionEngine, or None if fusion wasn't run this frame
                     (e.g. during calibration phases, before "ready")
        """
        if now is None:
            now = time.time()

        previous_state = self._state

        # --- SOS takes priority over everything else ---
        if intent_type is not None and intent_type.value == "sos_triggered":
            self._state = SafetyState.SOS_TRIGGERED
            self._sos_triggered_at = now

        # --- If currently showing SOS, check if display duration has passed ---
        elif self._state == SafetyState.SOS_TRIGGERED:
            if self._sos_triggered_at is not None and (now - self._sos_triggered_at) >= SOS_DISPLAY_DURATION_SEC:
                # Time's up - fall through to normal status-based logic below
                self._state = self._state_from_engine_status(engine_status)
            # else: stay in SOS_TRIGGERED, don't fall through

        else:
            self._state = self._state_from_engine_status(engine_status)

        message = self._message_for_state(self._state)
        state_changed = self._state != previous_state

        return StateResult(state=self._state, message=message, state_changed=state_changed)

    def _state_from_engine_status(self, engine_status: str) -> SafetyState:
        if engine_status in CALIBRATING_STATUSES:
            return SafetyState.IDLE
        if engine_status == "no_face":
            return SafetyState.UNCERTAIN
        if engine_status == "ready":
            return SafetyState.ACTIVE
        # Unknown status - fall back to UNCERTAIN rather than assuming ACTIVE
        return SafetyState.UNCERTAIN

    def _message_for_state(self, state: SafetyState) -> str:
        return {
            SafetyState.IDLE: "Calibrating...",
            SafetyState.ACTIVE: "Ready",
            SafetyState.UNCERTAIN: "WARNING: face not detected - please face the camera",
            SafetyState.SOS_ARMED: "SOS arming...",
            SafetyState.SOS_TRIGGERED: "SOS TRIGGERED",
        }[state]
        
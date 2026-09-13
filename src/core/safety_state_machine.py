"""
src/core/safety_state_machine.py
"""

from dataclasses import dataclass
from enum import Enum
import time


class SafetyState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    UNCERTAIN = "uncertain"
    SOS_ARMED = "sos_armed"
    SOS_TRIGGERED = "sos_triggered"


@dataclass
class StateResult:
    state: SafetyState
    message: str = ""
    state_changed: bool = False


CALIBRATING_STATUSES = {
    "calibrating_blink",
    "calibrating_pose",
    "calibrating_directions",
    "calibration_restarted",
}

SOS_DISPLAY_DURATION_SEC = 5.0


class SafetyStateMachine:
    def __init__(self):
        self._state = SafetyState.IDLE
        self._sos_triggered_at = None

    def update(self, engine_status: str, intent_type=None, now: float = None) -> StateResult:
        if now is None:
            now = time.time()

        previous_state = self._state

        if intent_type is not None and intent_type.value == "sos_triggered":
            self._state = SafetyState.SOS_TRIGGERED
            self._sos_triggered_at = now

        elif intent_type is not None and intent_type.value == "sos_armed":
            self._state = SafetyState.SOS_ARMED

        elif self._state == SafetyState.SOS_TRIGGERED:
            if self._sos_triggered_at is not None and (now - self._sos_triggered_at) >= SOS_DISPLAY_DURATION_SEC:
                self._state = self._state_from_engine_status(engine_status)

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
        return SafetyState.UNCERTAIN

    def _message_for_state(self, state: SafetyState) -> str:
        return {
            SafetyState.IDLE: "Calibrating...",
            SafetyState.ACTIVE: "Ready",
            SafetyState.UNCERTAIN: "WARNING: face not detected - please face the camera",
            SafetyState.SOS_ARMED: "SOS armed - hold blink again to confirm",
            SafetyState.SOS_TRIGGERED: "SOS TRIGGERED",
        }[state]
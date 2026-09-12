"""
verify_safety_state_machine.py

Week 10 unit tests for SafetyStateMachine, using fake engine statuses
and intent types (no camera needed) to verify state transitions in
isolation - same pattern as verify_intent_fusion.py.

Run with:
    python verify_safety_state_machine.py
"""

from src.core.safety_state_machine import SafetyStateMachine, SafetyState

PASS = "PASS"
FAIL = "FAIL"


def check(label, condition):
    result = PASS if condition else FAIL
    print(f"[{result}] {label}")
    return condition


class FakeIntentType:
    """Fake stand-in for IntentType.SOS_TRIGGERED, so this test file
    doesn't need to import the real IntentFusionEngine at all."""
    def __init__(self, value):
        self.value = value


def test_starts_idle():
    sm = SafetyStateMachine()
    result = sm.update("calibrating_blink", None, now=0.0)
    return check("starts_idle: calibrating status gives IDLE",
                 result.state == SafetyState.IDLE)


def test_ready_gives_active():
    sm = SafetyStateMachine()
    sm.update("calibrating_blink", None, now=0.0)
    result = sm.update("ready", None, now=1.0)
    return check("ready_gives_active: ready status gives ACTIVE",
                 result.state == SafetyState.ACTIVE)


def test_no_face_gives_uncertain():
    sm = SafetyStateMachine()
    sm.update("ready", None, now=0.0)
    result = sm.update("no_face", None, now=1.0)
    ok = check("no_face_gives_uncertain: no_face status gives UNCERTAIN",
               result.state == SafetyState.UNCERTAIN)
    ok &= check("no_face_gives_uncertain: message is a warning",
                "WARNING" in result.message.upper())
    return ok


def test_recovers_from_uncertain():
    sm = SafetyStateMachine()
    sm.update("ready", None, now=0.0)
    sm.update("no_face", None, now=1.0)
    result = sm.update("ready", None, now=2.0)
    return check("recovers_from_uncertain: back to ready gives ACTIVE again",
                 result.state == SafetyState.ACTIVE)


def test_sos_triggers_from_active():
    sm = SafetyStateMachine()
    sm.update("ready", None, now=0.0)
    result = sm.update("ready", FakeIntentType("sos_triggered"), now=1.0)
    return check("sos_triggers_from_active: SOS intent gives SOS_TRIGGERED",
                 result.state == SafetyState.SOS_TRIGGERED)


def test_sos_overrides_no_face():
    """Even if face is lost the SAME frame SOS fires, SOS should win -
    matches IntentFusionEngine's own override priority."""
    sm = SafetyStateMachine()
    sm.update("ready", None, now=0.0)
    result = sm.update("no_face", FakeIntentType("sos_triggered"), now=1.0)
    return check("sos_overrides_no_face: SOS intent wins even with no_face status",
                 result.state == SafetyState.SOS_TRIGGERED)


def test_sos_stays_until_display_duration():
    sm = SafetyStateMachine()
    sm.update("ready", None, now=0.0)
    sm.update("ready", FakeIntentType("sos_triggered"), now=1.0)

    # Check shortly after - should still be showing SOS
    result = sm.update("ready", None, now=1.5)
    return check("sos_stays_until_display_duration: still SOS_TRIGGERED before duration elapses",
                 result.state == SafetyState.SOS_TRIGGERED)


def test_sos_auto_returns_to_active():
    sm = SafetyStateMachine()
    sm.update("ready", None, now=0.0)
    sm.update("ready", FakeIntentType("sos_triggered"), now=1.0)

    # Check after SOS_DISPLAY_DURATION_SEC (3.0s default) has passed
    result = sm.update("ready", None, now=1.0 + 3.1)
    return check("sos_auto_returns_to_active: returns to ACTIVE after display duration",
                 result.state == SafetyState.ACTIVE)


def test_state_changed_flag():
    """SafetyStateMachine starts life already in IDLE (set in __init__),
    so the very first update() call to a calibrating_* status is NOT a
    real change - it's just confirming the state it already started in.
    Only a genuine transition (e.g. IDLE -> ACTIVE) should mark changed=True."""
    sm = SafetyStateMachine()
    result1 = sm.update("calibrating_blink", None, now=0.0)   # already IDLE from init, no real change
    result2 = sm.update("calibrating_blink", None, now=1.0)   # still same, no change
    result3 = sm.update("ready", None, now=2.0)                # real change: IDLE -> ACTIVE

    ok = check("state_changed_flag: False on first call (already IDLE from init)",
               result1.state_changed == False)
    ok &= check("state_changed_flag: False when status repeats",
                result2.state_changed == False)
    ok &= check("state_changed_flag: True when status changes",
                result3.state_changed == True)
    return ok


def main():
    tests = [
        test_starts_idle,
        test_ready_gives_active,
        test_no_face_gives_uncertain,
        test_recovers_from_uncertain,
        test_sos_triggers_from_active,
        test_sos_overrides_no_face,
        test_sos_stays_until_display_duration,
        test_sos_auto_returns_to_active,
        test_state_changed_flag,
    ]

    results = [t() for t in tests]
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} tests passed")


if __name__ == "__main__":
    main()
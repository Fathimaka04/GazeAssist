"""
verify_intent_fusion.py

Week 9 unit tests for IntentFusionEngine, using fake/controlled
timestamps (no camera needed) to verify state-machine logic in
isolation.

REWRITTEN for the SOS redesign: the old 5 tests assumed a 3-discrete-
long-blinks-in-sequence SOS design. That design was abandoned after
live testing showed it was unreliable even for an able-bodied tester
(see docs/architecture_week9.md). SOS is now a single extra-long
blink hold, so these tests are rewritten accordingly.

Run with:
    python verify_intent_fusion.py
"""

from src.core.intent_fusion_engine import IntentFusionEngine, IntentType

PASS = "PASS"
FAIL = "FAIL"


def check(label, condition):
    result = PASS if condition else FAIL
    print(f"[{result}] {label}")
    return condition


def test_basic_selection():
    """One long blink on a stable non-CENTER zone, then silence past
    the window, should confirm ZONE_SELECTED."""
    fusion = IntentFusionEngine()
    t = 1000.0

    # Zone becomes LEFT and holds stable
    fusion.update("LEFT", None, 0, now=t)
    t += fusion.ZONE_STABLE_TIME_SEC + 0.1

    # One normal-duration long blink (well below SOS threshold)
    event = fusion.update("LEFT", "long", 20, now=t)
    ok = check("basic_selection: first long blink gives PENDING",
               event.type == IntentType.PENDING and event.zone == "LEFT")

    # No further blink; wait past the confirm window
    t += fusion.LONG_BLINK_SEQUENCE_WINDOW_SEC + 0.1
    event = fusion.update("LEFT", None, 0, now=t)
    ok &= check("basic_selection: confirms ZONE_SELECTED after window",
                event.type == IntentType.ZONE_SELECTED and event.zone == "LEFT")
    return ok


def test_center_zone_ignored():
    """A long blink while zone is CENTER should never produce a
    pending selection (CENTER is explicitly excluded)."""
    fusion = IntentFusionEngine()
    t = 2000.0

    fusion.update("CENTER", None, 0, now=t)
    t += fusion.ZONE_STABLE_TIME_SEC + 0.1

    event = fusion.update("CENTER", "long", 20, now=t)
    return check("center_zone_ignored: no PENDING while zone is CENTER",
                 event.type == IntentType.NONE)


def test_sos_single_hold():
    """A single blink held past EXTRA_LONG_BLINK_MIN_FRAMES should
    trigger SOS_TRIGGERED immediately, with no PENDING step at all."""
    fusion = IntentFusionEngine()
    t = 3000.0

    fusion.update("RIGHT", None, 0, now=t)
    t += fusion.ZONE_STABLE_TIME_SEC + 0.1

    event = fusion.update("RIGHT", "long", fusion.EXTRA_LONG_BLINK_MIN_FRAMES, now=t)
    return check("sos_single_hold: extra-long hold gives SOS_TRIGGERED directly",
                 event.type == IntentType.SOS_TRIGGERED)


def test_sos_overrides_pending_selection():
    """If a normal selection is pending, a subsequent extra-long hold
    should still trigger SOS and cancel the pending selection."""
    fusion = IntentFusionEngine()
    t = 4000.0

    fusion.update("UP", None, 0, now=t)
    t += fusion.ZONE_STABLE_TIME_SEC + 0.1

    # Normal long blink starts a pending selection
    event = fusion.update("UP", "long", 20, now=t)
    ok = check("sos_overrides_pending: normal blink gives PENDING first",
               event.type == IntentType.PENDING)

    # Before the confirm window elapses, an extra-long hold occurs
    t += 0.3
    event = fusion.update("UP", "long", fusion.EXTRA_LONG_BLINK_MIN_FRAMES, now=t)
    ok &= check("sos_overrides_pending: SOS_TRIGGERED overrides pending selection",
                event.type == IntentType.SOS_TRIGGERED)

    # Pending state should be fully cleared afterward - waiting past
    # the old window should NOT also emit a stale ZONE_SELECTED
    t += fusion.LONG_BLINK_SEQUENCE_WINDOW_SEC + 0.1
    event = fusion.update("UP", None, 0, now=t)
    ok &= check("sos_overrides_pending: no stale ZONE_SELECTED after SOS",
                event.type == IntentType.NONE)
    return ok


def test_cooldown_blocks_retrigger():
    """After any confirmed event (selection or SOS), a new long blink
    within COOLDOWN_SEC should be ignored entirely."""
    fusion = IntentFusionEngine()
    t = 5000.0

    fusion.update("DOWN", None, 0, now=t)
    t += fusion.ZONE_STABLE_TIME_SEC + 0.1
    fusion.update("DOWN", "long", fusion.EXTRA_LONG_BLINK_MIN_FRAMES, now=t)  # SOS triggers, starts cooldown

    # Immediately try another long blink, well within cooldown
    t += 0.2
    event = fusion.update("DOWN", "long", 20, now=t)
    return check("cooldown_blocks_retrigger: blink ignored during cooldown",
                 event.type == IntentType.NONE)


def test_zone_instability_blocks_selection():
    """A long blink while the zone hasn't been stable for
    ZONE_STABLE_TIME_SEC yet should not start a pending selection."""
    fusion = IntentFusionEngine()
    t = 6000.0

    fusion.update("LEFT", None, 0, now=t)
    # Blink arrives immediately, zone not yet stable
    event = fusion.update("LEFT", "long", 20, now=t)
    return check("zone_instability_blocks_selection: no PENDING if zone just changed",
                 event.type == IntentType.NONE)


def main():
    tests = [
        test_basic_selection,
        test_center_zone_ignored,
        test_sos_single_hold,
        test_sos_overrides_pending_selection,
        test_cooldown_blocks_retrigger,
        test_zone_instability_blocks_selection,
    ]

    results = [t() for t in tests]
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} tests passed")


if __name__ == "__main__":
    main()
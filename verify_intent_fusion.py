"""
verify_intent_fusion.py
"""

from src.core.intent_fusion_engine import IntentFusionEngine, IntentType


PASS = []
FAIL = []


def check(name, condition):
    if condition:
        PASS.append(name)
        print(f"PASS: {name}")
    else:
        FAIL.append(name)
        print(f"FAIL: {name}")


def new_engine():
    return IntentFusionEngine()


def test_selection_resolves_immediately():
    fusion = new_engine()
    t = 0.0

    fusion.update("RIGHT", "short", 3, now=t)
    t += fusion.ZONE_STABLE_TIME_SEC + 0.1

    event = fusion.update("RIGHT", "long", 20, now=t)
    check("selection: fires ZONE_SELECTED in the same call, no wait",
          event.type == IntentType.ZONE_SELECTED and event.zone == "RIGHT")


def test_selection_requires_zone_stability_first():
    fusion = new_engine()
    t = 0.0

    fusion.update("RIGHT", "short", 3, now=t)
    t += 0.1  # not stable long enough yet

    event = fusion.update("RIGHT", "long", 20, now=t)
    check("selection: does not fire before ZONE_STABLE_TIME_SEC has passed",
          event.type == IntentType.NONE)


def test_center_zone_never_selects():
    fusion = new_engine()
    t = 0.0

    fusion.update("CENTER", "short", 3, now=t)
    t += fusion.ZONE_STABLE_TIME_SEC + 0.1

    event = fusion.update("CENTER", "long", 20, now=t)
    check("CENTER: normal long blink at CENTER never becomes ZONE_SELECTED",
          event.type == IntentType.NONE)


def test_zone_switch_resets_stability_timer():
    fusion = new_engine()
    t = 0.0

    fusion.update("UP", "short", 2, now=t)
    t += fusion.ZONE_STABLE_TIME_SEC - 0.1

    fusion.update("DOWN", "short", 2, now=t)
    t += 0.05

    event = fusion.update("DOWN", "long", 20, now=t)
    check("zone switch resets stability timer (no premature ZONE_SELECTED)",
          event.type == IntentType.NONE)


def test_extra_long_hold_arms_sos():
    fusion = new_engine()
    t = 0.0

    event = fusion.update("UP", "long", fusion.EXTRA_LONG_BLINK_MIN_FRAMES, now=t)
    check("SOS: single extra-long hold ARMS (does not trigger)",
          event.type == IntentType.SOS_ARMED)


def test_second_hold_within_window_confirms_sos():
    fusion = new_engine()
    t = 0.0

    fusion.update("UP", "long", fusion.EXTRA_LONG_BLINK_MIN_FRAMES, now=t)
    t += fusion.SOS_ARM_CONFIRM_WINDOW_SEC - 1.0

    event = fusion.update("UP", "long", fusion.EXTRA_LONG_BLINK_MIN_FRAMES, now=t)
    check("SOS: second extra-long hold within window CONFIRMS (SOS_TRIGGERED)",
          event.type == IntentType.SOS_TRIGGERED)


def test_second_hold_too_late_does_not_confirm():
    fusion = new_engine()
    t = 0.0

    fusion.update("UP", "long", fusion.EXTRA_LONG_BLINK_MIN_FRAMES, now=t)
    t += fusion.SOS_ARM_CONFIRM_WINDOW_SEC + 1.0

    event = fusion.update("UP", "long", fusion.EXTRA_LONG_BLINK_MIN_FRAMES, now=t)
    check("SOS: hold arriving after window expiry re-ARMS instead of confirming",
          event.type == IntentType.SOS_ARMED)


def test_arm_expires_silently_with_no_side_effects():
    fusion = new_engine()
    t = 0.0

    fusion.update("UP", "long", fusion.EXTRA_LONG_BLINK_MIN_FRAMES, now=t)
    t += fusion.SOS_ARM_CONFIRM_WINDOW_SEC + 1.0

    event = fusion.update("UP", "short", 2, now=t)
    check("SOS: expired arm leaves engine in a clean NONE state",
          event.type == IntentType.NONE)


def test_normal_selection_blink_ignored_while_armed():
    fusion = new_engine()
    t = 0.0

    fusion.update("UP", "long", fusion.EXTRA_LONG_BLINK_MIN_FRAMES, now=t)
    t += 0.5

    event = fusion.update("DOWN", "long", 20, now=t)
    check("SOS: normal selection blink while armed is ignored (not ZONE_SELECTED)",
          event.type == IntentType.SOS_ARMED)

    t += 1.0
    event = fusion.update("DOWN", "long", fusion.EXTRA_LONG_BLINK_MIN_FRAMES, now=t)
    check("SOS: confirm still works after an ignored normal blink while armed",
          event.type == IntentType.SOS_TRIGGERED)


def test_cooldown_blocks_immediate_reselection():
    fusion = new_engine()
    t = 0.0

    fusion.update("RIGHT", "short", 2, now=t)
    t += fusion.ZONE_STABLE_TIME_SEC + 0.1
    event = fusion.update("RIGHT", "long", 20, now=t)
    check("setup: ZONE_SELECTED fired", event.type == IntentType.ZONE_SELECTED)

    t += 0.1
    fusion.update("RIGHT", "short", 2, now=t)
    t += fusion.ZONE_STABLE_TIME_SEC + 0.1
    event = fusion.update("RIGHT", "long", 20, now=t)
    check("cooldown: blocks a new selection immediately after ZONE_SELECTED",
          event.type != IntentType.ZONE_SELECTED)


def test_cooldown_blocks_immediate_sos_rearm():
    fusion = new_engine()
    t = 0.0

    fusion.update("UP", "long", fusion.EXTRA_LONG_BLINK_MIN_FRAMES, now=t)
    t += 1.0
    event = fusion.update("UP", "long", fusion.EXTRA_LONG_BLINK_MIN_FRAMES, now=t)
    check("setup: SOS_TRIGGERED fired", event.type == IntentType.SOS_TRIGGERED)

    t += 0.1
    event = fusion.update("UP", "long", fusion.EXTRA_LONG_BLINK_MIN_FRAMES, now=t)
    check("cooldown: blocks immediate re-arm right after SOS_TRIGGERED",
          event.type != IntentType.SOS_ARMED)


def run_all():
    tests = [
        test_selection_resolves_immediately,
        test_selection_requires_zone_stability_first,
        test_center_zone_never_selects,
        test_zone_switch_resets_stability_timer,
        test_extra_long_hold_arms_sos,
        test_second_hold_within_window_confirms_sos,
        test_second_hold_too_late_does_not_confirm,
        test_arm_expires_silently_with_no_side_effects,
        test_normal_selection_blink_ignored_while_armed,
        test_cooldown_blocks_immediate_reselection,
        test_cooldown_blocks_immediate_sos_rearm,
    ]
    for test in tests:
        test()

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} tests passing")
    if FAIL:
        print(f"FAILED: {FAIL}")


if __name__ == "__main__":
    run_all()
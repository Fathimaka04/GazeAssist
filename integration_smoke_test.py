"""
integration_smoke_test.py

Week 8 — Integration smoke test for GazeBlinkEngine. Runs the full
pipeline (calibration + zone classification + blink detection) with a
live camera window (so you know when to turn/blink), and prints a
pass/fail summary at the end. Use this as a fast "did I break anything"
check before and after Week 9/10 changes.

WHAT IT CHECKS:
  1. Calibration completes successfully within a reasonable time/attempt
     budget (doesn't hang forever on repeated restarts).
  2. At least one blink is detected during the test window.
  3. At least 3 of the 5 zones are observed during the test window.
  4. No exceptions/crashes occur during the run.

Move your head through LEFT/RIGHT/UP/DOWN/CENTER and blink a few times
while the window is open, same as usual testing.

Run from the project root:
    python integration_smoke_test.py [duration_seconds]

Default duration is 45 seconds. Press 'q' or Ctrl+C to stop early - a
partial summary will still print.
"""

import sys
import time
import cv2

from src.sensing.camera_capture import CameraCapture
from src.core.gaze_blink_engine import GazeBlinkEngine

DEFAULT_DURATION_SECONDS = 45
CALIBRATION_TIMEOUT_SECONDS = 60

WINDOW_NAME = "Week 8 Smoke Test - move through zones + blink"


def run_smoke_test(duration_seconds=DEFAULT_DURATION_SECONDS):
    cam = CameraCapture(source=0)
    engine = GazeBlinkEngine(subject_label="smoke_test", enable_sqlite_logging=False)

    zones_seen = set()
    blink_count_final = 0
    calibration_completed = False
    calibration_restarts = 0
    crashed = False
    crash_message = None

    start_time = time.time()
    calibration_start_time = start_time

    print(f"Starting integration smoke test (duration: {duration_seconds}s)")
    print("Move through LEFT/RIGHT/UP/DOWN/CENTER and blink a few times once calibration finishes.\n")

    try:
        for frame in cam.frames():
            elapsed = time.time() - start_time
            if elapsed > duration_seconds:
                break

            result = engine.process_frame(frame)

            # --- On-screen feedback so you know what to do ---
            if result.status == "calibrating_blink":
                cv2.putText(frame, f"CALIBRATING BLINK ({result.progress_current}/{result.progress_total})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Look straight, hold still", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            elif result.status == "no_face":
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            elif result.status == "calibrating_pose":
                cv2.putText(frame, "CALIBRATING HEAD POSE - hold still", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            elif result.status == "calibrating_directions":
                color = (0, 165, 255) if result.direction_settling else (0, 255, 0)
                cv2.putText(frame, result.direction_prompt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                if result.direction_settling:
                    cv2.putText(frame, "Get ready...", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.putText(frame, f"Hold! ({result.direction_sample_progress}/"
                                        f"{result.direction_sample_total})",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            elif result.status == "calibration_restarted":
                calibration_restarts += 1
                print(f"[{elapsed:.1f}s] Calibration restarted: {result.restart_reason}")
                cv2.putText(frame, "RESTARTING - " + result.restart_reason[:40], (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if time.time() - calibration_start_time > CALIBRATION_TIMEOUT_SECONDS:
                    print("Calibration timeout exceeded - stopping test.")
                    cv2.imshow(WINDOW_NAME, frame)
                    cv2.waitKey(1)
                    break

            elif result.status == "ready":
                if not calibration_completed:
                    calibration_completed = True
                    print(f"[{elapsed:.1f}s] Calibration completed "
                          f"(restarts along the way: {calibration_restarts})")

                if result.zone is not None:
                    zones_seen.add(result.zone)
                if result.blink_count is not None:
                    blink_count_final = result.blink_count

                if result.zone_changed:
                    print(f"[{elapsed:.1f}s] Zone -> {result.zone}")
                if result.blink_just_occurred:
                    print(f"[{elapsed:.1f}s] Blink detected (total: {result.blink_count})")

                cv2.putText(frame, f"Blinks: {result.blink_count} | Zone: {result.zone}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if result.yaw is not None:
                    cv2.putText(frame, f"yaw={result.yaw:.1f} pitch={result.pitch:.1f}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                zones_str = f"zones seen: {sorted(zones_seen)}"
                cv2.putText(frame, zones_str, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Countdown so you know how much time is left
            remaining = max(0, duration_seconds - elapsed)
            cv2.putText(frame, f"Time left: {remaining:.0f}s", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Stopped early by user ('q').")
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user - printing partial summary.")
    except Exception as e:
        crashed = True
        crash_message = f"{type(e).__name__}: {e}"
        print(f"\nCRASH: {crash_message}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        engine.close()

    # --- Summary ---
    print("\n" + "=" * 50)
    print("INTEGRATION SMOKE TEST SUMMARY")
    print("=" * 50)

    checks = []
    checks.append(("No crashes", not crashed, crash_message if crashed else "OK"))
    checks.append(("Calibration completed", calibration_completed,
                    f"restarts: {calibration_restarts}" if calibration_completed
                    else "never completed within duration/timeout"))

    at_least_3_zones = len(zones_seen) >= 3
    checks.append(("At least 3 zones observed", at_least_3_zones,
                    f"zones seen: {sorted(zones_seen) if zones_seen else 'none'}"))

    blinked = blink_count_final >= 1
    checks.append(("At least 1 blink detected", blinked,
                    f"blink count: {blink_count_final}"))

    all_passed = all(passed for _, passed, _ in checks)

    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} ({detail})")

    print("=" * 50)
    print("OVERALL: " + ("PASS" if all_passed else "FAIL"))
    print("=" * 50)

    return all_passed


if __name__ == "__main__":
    duration = DEFAULT_DURATION_SECONDS
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            print(f"Invalid duration '{sys.argv[1]}', using default {DEFAULT_DURATION_SECONDS}s")

    passed = run_smoke_test(duration)
    sys.exit(0 if passed else 1)
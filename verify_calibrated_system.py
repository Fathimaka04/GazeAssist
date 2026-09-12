"""
verify_calibrated_system.py

Week 7 — Uses the calibration engine's personalized EAR threshold for
live blink detection (via BlinkClassifier), plus a directional
auto-calibration phase for gaze-zone classification.

CHANGE (this version): added SQLite logging of calibration events via
CalibrationLogger - every successful calibration is saved to
data/calibration_log.db with a timestamp and subject_label, providing
a queryable history for the personalized vs fixed-threshold comparison
still to be done with a volunteer.

Run from the project root:
    python verify_calibrated_system.py
Press 'q' to quit.
"""
"""
verify_calibrated_system.py

Week 8 — REWRITTEN to use src/core/gaze_blink_engine.py (GazeBlinkEngine)
instead of inline calibration/classification/blink logic.

WHAT CHANGED: All the verified Week 7 logic - calibration (blink baseline,
head-pose, directional LEFT/RIGHT/UP/DOWN), the symmetry check, raw pitch
separation check, spread checks, fused pitch+gaze_y vertical scoring,
yaw-priority hysteresis, CENTER-boundary hysteresis, blink detection via
BlinkClassifier, and SQLite logging via CalibrationLogger - has been moved
out of this script and into the GazeBlinkEngine class. This script now
only handles the camera loop and on-screen (cv2) display; it calls
engine.process_frame(frame) each iteration and reacts to the returned
EngineResult status.

WHY: Week 9 (multimodal fusion) and Week 10 (safety state machine) need to
reuse this same calibration + classification pipeline without depending on
a script full of cv2.imshow/cv2.waitKey calls. Separating the engine from
the display makes it reusable and testable headlessly (see the Week 8
integration smoke test).

VERIFICATION: This refactor should produce IDENTICAL behavior to the
pre-refactor version - same prompts, same thresholds, same zone/blink
results, same SQLite logging. Run the usual CENTER->LEFT->CENTER->RIGHT->
CENTER->UP->CENTER->DOWN->CENTER->blinks test sequence and confirm the
output matches prior verified runs before building anything new on top
of this.

Run from the project root:
    python verify_calibrated_system.py
Press 'q' to quit.
"""
import cv2

from src.sensing.camera_capture import CameraCapture
from src.core.gaze_blink_engine import GazeBlinkEngine

SUBJECT_LABEL = "self"  # change to a volunteer's name/id for that session


def main():
    cam = CameraCapture(source=0)
    engine = GazeBlinkEngine(subject_label=SUBJECT_LABEL)
    window_name = "Week 8 - GazeBlinkEngine Integration Check"

    for frame in cam.frames():
        result = engine.process_frame(frame)

        if result.status == "calibrating_blink":
            cv2.putText(frame, f"CALIBRATING BLINK BASELINE "
                                f"({result.progress_current}/{result.progress_total})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Look straight at the camera, hold still", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        elif result.status == "no_face":
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        elif result.status == "calibrating_pose":
            cv2.putText(frame, "CALIBRATING HEAD POSE - hold still, face camera", (10, 30),
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
            print(result.restart_reason)

        elif result.status == "ready":
            if result.calibration_summary is not None:
                s = result.calibration_summary
                print("DIRECTION CALIBRATION DONE (yaw + weighted fused pitch/gaze_y)")
                print(f"  LEFT:  yaw={s['left']['yaw']:.2f}, pitch={s['left']['pitch']:.2f}, "
                      f"gaze_y={s['left']['gaze_y']:.3f}")
                print(f"  RIGHT: yaw={s['right']['yaw']:.2f}, pitch={s['right']['pitch']:.2f}, "
                      f"gaze_y={s['right']['gaze_y']:.3f}")
                print(f"  UP:    yaw={s['up']['yaw']:.2f}, pitch={s['up']['pitch']:.2f}, "
                      f"gaze_y={s['up']['gaze_y']:.3f}")
                print(f"  DOWN:  yaw={s['down']['yaw']:.2f}, pitch={s['down']['pitch']:.2f}, "
                      f"gaze_y={s['down']['gaze_y']:.3f}")
                print(f"  center_yaw={s['center_yaw']:.2f} yaw_scale={s['yaw_scale']:.2f}")
                print(f"  pitch_center={s['pitch_center']:.2f} pitch_scale={s['pitch_scale']:.2f} "
                      f"raw_pitch_sep={s['raw_pitch_sep']:.2f}")
                print(f"  gaze_y_center={s['gaze_y_center']:.3f} gaze_y_scale={s['gaze_y_scale']:.3f}")
                print(f"  fused UP score={s['fused_up_score']:.2f}, "
                      f"fused DOWN score={s['fused_down_score']:.2f}")
                print(f"  [logged to SQLite as subject_label='{SUBJECT_LABEL}']")

            if result.zone_changed:
                gy_str = f"{result.raw_gaze[1]:.2f}" if result.raw_gaze else "None"
                print(f"Zone -> {result.zone}  (yaw={result.yaw:.2f}, pitch={result.pitch:.2f}, "
                      f"gaze_y={gy_str}, h_score={result.horizontal_score:.2f}, "
                      f"v_score={result.vertical_score:.2f})")

            cv2.putText(frame, f"Blinks: {result.blink_count} | Zone: {result.zone}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if result.yaw is not None:
                cv2.putText(frame, f"yaw={result.yaw:.1f} pitch={result.pitch:.1f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if result.ear is not None:
                cv2.putText(frame, f"EAR: {result.ear:.3f} (thr={result.ear_threshold:.3f})",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()
    engine.close()
    final_count = result.blink_count if result and result.blink_count is not None else 0
    print(f"\nSession ended. Total blinks: {final_count}")


if __name__ == "__main__":
    main()
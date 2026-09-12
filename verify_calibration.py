"""
verify_calibration.py

Week 7 verification. Runs the full pipeline (face mesh -> EAR -> head pose
-> gaze) and feeds live values into CalibrationEngine. Shows calibration
progress, then displays the personalized thresholds once calibration
finishes, alongside how they compare to the old fixed values from
Weeks 2 and 6.

Run from the project root:
    python verify_calibration.py
Press 'q' to quit.

WHAT TO CHECK: sit normally, look at the screen, during the calibration
countdown (don't do anything special — that's the point of zero-learning).
Once calibrated, the personalized EAR threshold and gaze baselines should
print, and should be reasonably close to (but not necessarily identical
to) the fixed values used in earlier weeks.
"""

import time
import cv2

from src.sensing.camera_capture import CameraCapture
from src.perception.face_mesh import FaceMeshDetector
from src.perception.head_pose import HeadPoseEstimator
from src.perception.gaze_vector import estimate_raw_gaze_x, estimate_raw_gaze_y
from src.signal_processing.ear_calculator import average_ear
from src.signal_processing.calibration_engine import CalibrationEngine

GAZE_Y_SIGN_FLIP = True  # matches the fix applied in verify_gaze_zone.py


def main():
    cam = CameraCapture(source=0)
    time.sleep(1)
    detector = FaceMeshDetector()
    pose_estimator = HeadPoseEstimator()
    calibration = CalibrationEngine(calibration_frames=45)

    window_name = "Week 7 - Zero-Learning Calibration"

    print("Starting calibration. Sit normally and look at the screen —")
    print("no special action needed. This measures your own natural")
    print("EAR and gaze baseline automatically.\n")

    for frame in cam.frames():
        results = detector.process(frame)
        right_eye_pts, left_eye_pts = detector.get_eye_points(results, frame.shape)
        ear = average_ear(right_eye_pts, left_eye_pts)

        pose = pose_estimator.estimate(results, frame.shape)
        raw_gaze_x = estimate_raw_gaze_x(results, frame.shape)
        raw_gaze_y = estimate_raw_gaze_y(results, frame.shape)
        if raw_gaze_y is not None and GAZE_Y_SIGN_FLIP:
            raw_gaze_y = -raw_gaze_y

        if not calibration.is_calibrated:
            calibration.add_sample(ear, raw_gaze_x, raw_gaze_y, time.time())
            current, total = calibration.get_progress()
            display_text = f"CALIBRATING ({current}/{total}) - sit normally"
            color = (0, 255, 255)

            if calibration.is_calibrated:
                summary = calibration.get_summary()
                print("\n" + "=" * 50)
                print("CALIBRATION COMPLETE")
                print("=" * 50)
                for k, v in summary.items():
                    print(f"  {k}: {v}")
                print(f"\n  (For comparison, Week 2's fixed EAR_THRESHOLD was 0.22)")
                print(f"  (For comparison, Week 6's fixed gaze thresholds were")
                print(f"   LEFT=-0.3, RIGHT=0.3, UP=-0.3, DOWN=0.2)")
                print("=" * 50 + "\n")
        else:
            summary = calibration.get_summary()
            display_text = (f"READY | EAR_thr={summary['ear_threshold']:.3f} "
                             f"gazeX_base={summary['gaze_x_baseline']:.3f}")
            color = (0, 255, 0)

        cv2.putText(frame, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        if ear is not None:
            cv2.putText(frame, f"EAR: {ear:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
"""
verify_gaze_zone.py

Week 6 verification + formal accuracy test. Shows the live gaze zone
(LEFT/RIGHT/UP/DOWN/CENTER) using combined head+eye movement (raw gaze,
not head-pose corrected — see Week 6 notes on why raw gaze was selected).

A vertical baseline offset is applied to gaze_y since, unlike gaze_x,
the resting/center vertical eye position was found NOT to sit at 0 —
it clustered around -0.25 in testing. This is a rough, single-value
correction (not full per-session auto-calibration like head_pose.py
uses) intended to validate the concept before building a more thorough
version.

Run from the project root:
    python verify_gaze_zone.py
Press 'q' to quit (with the video window focused).

ACCURACY TEST: after calibration, look at each of the 5 directions in
turn (LEFT, CENTER, RIGHT, CENTER, UP, CENTER, DOWN, CENTER), holding
each for a couple of seconds, so the printed zones can be checked
against what you were actually looking at.
"""

import time
import cv2

from src.sensing.camera_capture import CameraCapture
from src.perception.face_mesh import FaceMeshDetector
from src.perception.head_pose import HeadPoseEstimator
from src.perception.gaze_vector import estimate_raw_gaze_x, estimate_raw_gaze_y
from src.signal_processing.gaze_zone_classifier import GazeZoneClassifier

PRINT_EVERY_N_FRAMES = 3

# Measured from resting/center gaze_y values during earlier testing —
# the vertical offset formula doesn't naturally center at 0 the way
# the horizontal one does, so we correct for it here.
GAZE_Y_BASELINE_OFFSET = -0.25


def main():
    cam = CameraCapture(source=0)
    time.sleep(1)
    detector = FaceMeshDetector()
    pose_estimator = HeadPoseEstimator()
    zone_classifier = GazeZoneClassifier()

    window_name = "Week 6 - Gaze Zone Test (5 zones)"
    frame_number = 0

    print("Starting. Waiting for calibration...")
    print("Once calibrated, look LEFT, CENTER, RIGHT, CENTER, UP, CENTER,")
    print("DOWN, CENTER in turn, holding each for a couple seconds.\n")

    for frame in cam.frames():
        frame_number += 1
        results = detector.process(frame)
        pose = pose_estimator.estimate(results, frame.shape)
        raw_gaze_x = estimate_raw_gaze_x(results, frame.shape)
        raw_gaze_y = estimate_raw_gaze_y(results, frame.shape)

        if raw_gaze_y is not None:
            raw_gaze_y = -(raw_gaze_y - GAZE_Y_BASELINE_OFFSET)

        display_text = ""
        color = (255, 255, 255)

        if pose is None:
            display_text = "NO FACE DETECTED"
            color = (0, 0, 255)
            if frame_number % PRINT_EVERY_N_FRAMES == 0:
                print("No face detected.")

        elif pose[3]:
            progress = len(pose_estimator.calibration_samples)
            total = pose_estimator.calibration_frames_needed
            display_text = f"CALIBRATING ({progress}/{total})"
            color = (0, 255, 255)
            if frame_number % PRINT_EVERY_N_FRAMES == 0:
                print(f"Calibrating... ({progress}/{total})")

        else:
            zone = zone_classifier.classify(raw_gaze_x, raw_gaze_y)
            display_text = f"Zone: {zone}"
            color = (0, 255, 0)

            if frame_number % PRINT_EVERY_N_FRAMES == 0 and raw_gaze_x is not None and raw_gaze_y is not None:
                print(f"Zone: {zone:8s}  |  gaze_x: {raw_gaze_x:+.3f}  |  gaze_y: {raw_gaze_y:+.3f}")

        cv2.putText(frame, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()
    print("\nSession ended.")


if __name__ == "__main__":
    main()
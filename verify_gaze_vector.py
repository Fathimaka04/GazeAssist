"""
verify_gaze_vector.py

Week 5 verification + ablation test. Prints RAW gaze estimate (iris only)
alongside NORMALIZED gaze estimate (iris + head-pose correction) to the
terminal every few frames, so you can read and compare the numbers
without needing to interpret anything on the video overlay.

THE TEST: after calibration finishes, pick one fixed point to stare at
(e.g., your webcam lens itself) and DO NOT move your eyes off it. Then
slowly turn your head left and right while still staring at that same
point. Watch the terminal:

- "Raw" should change noticeably as you turn your head (even though
  your eyes never moved) — this shows the raw measurement being
  "fooled" by head rotation.
- "Normalized" should stay closer to steady during the same head
  movement — this shows the head-pose correction working.

If Normalized stays flatter than Raw during this test, the correction
is doing its job. If both jump around equally, the correction isn't
helping and needs adjustment.

Run from the project root:
    python verify_gaze_vector.py
Press 'q' to quit (with the video window focused).
"""

import time
import cv2

from src.sensing.camera_capture import CameraCapture
from src.perception.face_mesh import FaceMeshDetector
from src.perception.head_pose import HeadPoseEstimator
from src.perception.gaze_vector import estimate_raw_gaze_x, estimate_normalized_gaze_x

PRINT_EVERY_N_FRAMES = 10  # print roughly 2-3 times per second


def main():
    cam = CameraCapture(source=0)
    time.sleep(1)
    detector = FaceMeshDetector()
    pose_estimator = HeadPoseEstimator()

    window_name = "Week 5 - Gaze Vector Test"
    frame_number = 0

    print("Starting. Waiting for calibration...")
    print("Once calibration finishes, stare at one fixed point (e.g. your")
    print("webcam), then turn your head left/right WITHOUT moving your eyes.")
    print("Watch the Raw vs Normalized numbers below.\n")

    for frame in cam.frames():
        frame_number += 1
        results = detector.process(frame)
        pose = pose_estimator.estimate(results, frame.shape)
        raw_gaze = estimate_raw_gaze_x(results, frame.shape)

        if pose is None:
            if frame_number % PRINT_EVERY_N_FRAMES == 0:
                print("No face detected.")
            display_text = "NO FACE DETECTED"
            color = (0, 0, 255)

        elif pose[3]:  # still calibrating
            progress = len(pose_estimator.calibration_samples)
            total = pose_estimator.calibration_frames_needed
            if frame_number % PRINT_EVERY_N_FRAMES == 0:
                print(f"Calibrating... ({progress}/{total}) - hold still, face forward")
            display_text = f"CALIBRATING ({progress}/{total})"
            color = (0, 255, 255)

        else:
            yaw = pose[0]
            normalized_gaze = estimate_normalized_gaze_x(raw_gaze, yaw)

            if raw_gaze is not None and frame_number % PRINT_EVERY_N_FRAMES == 0:
                print(f"Raw: {raw_gaze:+.3f}  |  Normalized: {normalized_gaze:+.3f}  |  Yaw: {yaw:+.1f}")

            display_text = "Calibrated - see terminal for numbers"
            color = (0, 255, 0)

        cv2.putText(frame, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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
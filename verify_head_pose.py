"""
verify_head_pose.py

Verification script for head-pose estimation. Uses automatic per-session
calibration: on startup, hold still and face the camera for a couple of
seconds while it calibrates to your current position, then live yaw/pitch/
roll values are shown.

Run from the project root:
    python verify_head_pose.py
Press 'q' to quit.

WHAT TO CHECK (after calibration finishes):
- Turn your head RIGHT -> yaw should consistently move in one direction
- Turn your head LEFT -> yaw should move the opposite way
- Nod DOWN -> pitch should move one way, tip UP -> the other way
- Tilt your head sideways (ear toward shoulder) -> roll should respond
If any of these feel backwards, note it — sign conventions can be flipped
without affecting correctness, as long as it's CONSISTENT and known.
"""

import time
import cv2

from src.sensing.camera_capture import CameraCapture
from src.perception.face_mesh import FaceMeshDetector
from src.perception.head_pose import HeadPoseEstimator


def main():
    cam = CameraCapture(source=0)
    time.sleep(1)
    detector = FaceMeshDetector()
    pose_estimator = HeadPoseEstimator()

    window_name = "Head Pose Estimation"

    for frame in cam.frames():
        results = detector.process(frame)
        pose = pose_estimator.estimate(results, frame.shape)

        if pose is None:
            cv2.putText(frame, "NO FACE DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            yaw, pitch, roll, is_calibrating = pose

            if is_calibrating:
                progress = len(pose_estimator.calibration_samples)
                total = pose_estimator.calibration_frames_needed
                cv2.putText(frame, f"CALIBRATING... hold still ({progress}/{total})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(frame, f"Yaw:   {yaw:6.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Pitch: {pitch:6.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll:  {roll:6.1f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
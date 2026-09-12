"""
live webcam window with the MediaPipe Face Mesh
overlay, the 6-point eye landmarks, an FPS counter, and a face-detected
status indicator.

Run from the project root:
    python verify_face_tracking.py
Press 'q' to quit.
"""

import time
import cv2

from src.sensing.camera_capture import CameraCapture
from src.perception.face_mesh import FaceMeshDetector


def main():
    cam = CameraCapture(source=0)
    time.sleep(1)  # wait for 1 sec
    detector = FaceMeshDetector()

    prev_time = time.time()  # time.time() gives the current time
    fps_values = []

    print("Week 1 demo running. Press 'q' to quit.")
    window_name = "GazeAssist - Week 1 - Face Mesh Demo"

    for frame in cam.frames():
        results = detector.process(frame)

        frame = detector.draw_landmarks(frame, results)
        right_eye_pts, left_eye_pts = detector.get_eye_points(results, frame.shape)
        frame = detector.draw_eye_points(frame, right_eye_pts, left_eye_pts)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0  # FPS = 1 / time taken for one frame
        prev_time = curr_time
        fps_values.append(fps)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        face_detected = results.multi_face_landmarks is not None
        status_text = "FACE DETECTED" if face_detected else "NO FACE"
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # checks whether the user closed the webcam window using the X button.
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()

    if fps_values:
        avg_fps = sum(fps_values) / len(fps_values)
        print(f"\nSession ended. Average FPS: {avg_fps:.1f}")
        print("Write this number down — you'll need it later as a baseline.")

# If this file is being run directly, start the main() function.
if __name__ == "__main__":
    main()
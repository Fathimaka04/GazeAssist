"""
verify_blink_detection.py

Week 2 verification script. State-machine-based blink detection: requires
sustained closure AND sustained reopening before counting a blink, which
filters out EAR flicker near the threshold boundary (a common issue during
longer or slower closures).

Run from the project root:
    python verify_blink_detection.py
Press 'q' to quit.
"""

import time
import cv2

from src.sensing.camera_capture import CameraCapture
from src.perception.face_mesh import FaceMeshDetector
from src.signal_processing.ear_calculator import average_ear

EAR_THRESHOLD = 0.22
MIN_CLOSED_FRAMES = 2   # frames eye must stay below threshold to register as "closing"
MIN_OPEN_FRAMES = 4     # frames eye must stay above threshold to confirm "reopened"
                        # (higher than MIN_CLOSED_FRAMES on purpose — reopening
                        # needs to be more certain, since that's what ends a blink)
PRINT_EVERY_N_FRAMES = 15


def main():
    cam = CameraCapture(source=0)
    time.sleep(1)
    detector = FaceMeshDetector()

    window_name = "Week 2 - Blink Detection (state machine)"
    state = "OPEN"          # OPEN or CLOSED
    open_counter = 0
    closed_counter = 0
    blink_count = 0
    frame_number = 0
    ear_log = []

    print("Starting. Watch the terminal for live EAR readings and blink events.")
    print(f"Threshold = {EAR_THRESHOLD}, "
          f"min_closed={MIN_CLOSED_FRAMES}, min_open={MIN_OPEN_FRAMES}\n")

    for frame in cam.frames():
        frame_number += 1
        results = detector.process(frame)
        right_eye_pts, left_eye_pts = detector.get_eye_points(results, frame.shape)
        frame = detector.draw_eye_points(frame, right_eye_pts, left_eye_pts)

        ear = average_ear(right_eye_pts, left_eye_pts)

        if ear is not None:
            ear_log.append(ear)

            if ear < EAR_THRESHOLD:
                closed_counter += 1
                open_counter = 0
            else:
                open_counter += 1
                closed_counter = 0

            if state == "OPEN" and closed_counter >= MIN_CLOSED_FRAMES:
                state = "CLOSED"
            elif state == "CLOSED" and open_counter >= MIN_OPEN_FRAMES:
                state = "OPEN"
                blink_count += 1
                print(f"[BLINK #{blink_count}] confirmed at frame {frame_number}")

            if frame_number % PRINT_EVERY_N_FRAMES == 0:
                print(f"frame {frame_number:5d}  EAR = {ear:.3f}  state={state}")

            cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            state_color = (0, 0, 255) if state == "CLOSED" else (0, 255, 0)
            cv2.putText(frame, f"Eye: {state}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

        cv2.putText(frame, f"Blinks: {blink_count}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print("SESSION SUMMARY")
    print(f"{'='*50}")
    print(f"Total blinks detected: {blink_count}")
    if ear_log:
        print(f"EAR min:  {min(ear_log):.3f}")
        print(f"EAR max:  {max(ear_log):.3f}")
        print(f"EAR mean: {sum(ear_log)/len(ear_log):.3f}")
    print("Copy this whole summary block back for review.")


if __name__ == "__main__":
    main()
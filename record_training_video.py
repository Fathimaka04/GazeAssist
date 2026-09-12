"""
record_training_video.py

Week 3 — Dataset collection.

Records a labeled video clip for training data. You choose the label
before recording starts; the script saves the clip with a clean,
descriptive filename so labeling/extraction later is unambiguous.

Run from the project root:
    python record_training_video.py

You'll be prompted to choose a label and a duration, then recording
starts after a 3-second countdown (so you have time to get ready).
Press 'q' to stop early if needed.

Recordings save to: data/raw/<label>_<subject>_<timestamp>.avi
"""

import os
import time
import cv2

from src.sensing.camera_capture import CameraCapture

LABELS = {
    "1": "open",          # eyes open, no blinking — negative examples
    "2": "normal_blink",  # relaxed, natural blinking
    "3": "short_blink",   # deliberate quick blinks
    "4": "long_blink",    # deliberate held/long blinks (SOS pattern)
}

OUTPUT_DIR = os.path.join("data", "raw")    # Save the recorded videos inside data/raw.


def choose_label():
    print("\nWhat are you recording?")
    for key, val in LABELS.items():
        print(f"  {key}. {val}")
    choice = input("Enter number: ").strip()
    while choice not in LABELS:
        choice = input("Invalid. Enter 1-4: ").strip()
    return LABELS[choice]


def choose_subject_name():
    name = input("Subject name (e.g. 'self', 'volunteer1'): ").strip()
    return name if name else "self"


def choose_duration():
    raw = input("Duration in seconds (default 120): ").strip()
    if raw == "":
        return 120
    try:
        return int(raw)
    except ValueError:
        print("Invalid number, using default 120.")
        return 120


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    label = choose_label()
    subject = choose_subject_name()
    duration = choose_duration()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{subject}_{timestamp}.avi"
    filepath = os.path.join(OUTPUT_DIR, filename)

    cam = CameraCapture(source=0)
    time.sleep(1)

    # get one frame to determine frame size for the video writer
    frame_iter = cam.frames()
    first_frame = next(frame_iter)
    h, w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"XVID")            # encode the video
    writer = cv2.VideoWriter(filepath, fourcc, 20.0, (w, h))    # object that will save frames into a video file.

    print(f"\nRecording '{label}' for subject '{subject}'.")
    print(f"Saving to: {filepath}")
    print("Starting in 3 seconds — get ready...")
    for i in (3, 2, 1):
        print(i)
        time.sleep(1)
    print("GO! Press 'q' to stop early.\n")

    start_time = time.time()
    frame_count = 0
    window_name = f"Recording: {label} ({subject})"

    writer.write(first_frame)
    frame_count += 1
    cv2.imshow(window_name, first_frame)
    cv2.waitKey(1)

    for frame in frame_iter:
        elapsed = time.time() - start_time
        remaining = max(0, duration - elapsed)

        writer.write(frame)
        frame_count += 1

        display = frame.copy()
        cv2.putText(display, f"REC  {label}  {remaining:.1f}s left", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or elapsed >= duration:
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    writer.release()
    cam.release()
    cv2.destroyAllWindows()

    actual_duration = time.time() - start_time
    print(f"\nDone. Saved {frame_count} frames (~{actual_duration:.1f}s) to:")
    print(f"  {filepath}")
    print("Run this script again to record the next label/category.")


if __name__ == "__main__":
    main()
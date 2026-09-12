"""
extract_ear_dataset.py

Week 3 — Dataset preparation.

Reads every video in data/raw/, runs the existing Face Mesh + EAR pipeline
on each frame, and writes one row per frame to a CSV: the EAR value, the
frame's label (parsed from the filename), and the source file. Frames
labeled 'open' that dip below the Week 2 threshold are relabeled
'blink_artifact' (incidental natural blinks), so they don't pollute the
open-eye training class.

Run from the project root:
    python extract_ear_dataset.py

Output: data/ear_dataset_raw.csv
"""

import os
import glob
import csv

import cv2

from src.perception.face_mesh import FaceMeshDetector
from src.signal_processing.ear_calculator import average_ear

RAW_DIR = os.path.join("data", "raw")
OUTPUT_CSV = os.path.join("data", "ear_dataset_raw.csv")
EAR_THRESHOLD = 0.22  # same validated value from Week 2


def parse_label_from_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")

    two_word_labels = ["normal_blink", "short_blink", "long_blink"]
    for label in two_word_labels:
        if base.startswith(label):
            subject = base[len(label) + 1:].rsplit("_", 2)[0]
            return label, subject

    label = parts[0]
    subject = "_".join(parts[1:-2]) if len(parts) > 3 else parts[1]
    return label, subject


def process_video(filepath, detector, writer):
    label, subject = parse_label_from_filename(filepath)
    print(f"Processing: {os.path.basename(filepath)}  (label={label}, subject={subject})")

    cap = cv2.VideoCapture(filepath)
    frame_idx = 0
    rows_written = 0
    artifact_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = detector.process(frame)
        right_eye_pts, left_eye_pts = detector.get_eye_points(results, frame.shape)
        ear = average_ear(right_eye_pts, left_eye_pts)

        if ear is not None:
            effective_label = label
            if label == "open" and ear < EAR_THRESHOLD:
                effective_label = "blink_artifact"
                artifact_count += 1

            writer.writerow({
                "source_file": os.path.basename(filepath),
                "subject": subject,
                "label": effective_label,
                "frame_index": frame_idx,
                "ear": round(ear, 4),
            })
            rows_written += 1

        frame_idx += 1

    cap.release()
    print(f"  -> {rows_written} frames with valid EAR out of {frame_idx} total frames")
    if label == "open":
        print(f"  -> {artifact_count} incidental blink frames caught and relabeled")


def main():
    video_files = glob.glob(os.path.join(RAW_DIR, "*.avi"))
    video_files += glob.glob(os.path.join(RAW_DIR, "*.mp4"))

    if not video_files:
        print(f"No video files found in {RAW_DIR}. Run record_training_video.py first.")
        return

    detector = FaceMeshDetector()
    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = ["source_file", "subject", "label", "frame_index", "ear"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for filepath in video_files:
            process_video(filepath, detector, writer)

    print(f"\nDone. Combined dataset written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
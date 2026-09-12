"""
prepare_training_data.py

Week 3 — Turns the raw per-frame EAR CSV into labeled training windows.

For 'open' recordings: samples fixed-length windows of clean open-eye EAR
values, skipping any window that touches an incidental blink_artifact frame.

For blink recordings (normal_blink, short_blink, long_blink): runs the same
state-machine blink detector validated in Week 2 over each video's EAR
sequence to find actual blink events, measures each event's closed-frame
duration, and classifies it as 'short_blink' or 'long_blink' based on that
measured duration — not just the filename — since natural blinks inside a
'normal_blink' recording are always short regardless of the file's label.

Run from the project root:
    python prepare_training_data.py

Input:  data/ear_dataset_raw.csv
Output: data/ear_training_windows.csv
"""

import os
import csv
from collections import defaultdict

INPUT_CSV = os.path.join("data", "ear_dataset_raw.csv")
OUTPUT_CSV = os.path.join("data", "training_data.csv")

WINDOW_SIZE = 10

EAR_THRESHOLD = 0.22        # same validated Week 2 value
MIN_CLOSED_FRAMES = 2
MIN_OPEN_FRAMES = 3         # lowered from 4 — avoids merging rapid successive blinks
LONG_BLINK_MIN_CLOSED_FRAMES = 20  # raised from 12 — matches observed real durations
MAX_VALID_CLOSED_FRAMES_DEFAULT = 60   # for open/normal_blink/short_blink recordings —
                                        # anything longer is a merged rapid-blink artifact
MAX_VALID_CLOSED_FRAMES_LONG = 200     # for long_blink recordings — genuine deliberate
                                        # holds legitimately run this long

OPEN_WINDOW_STRIDE = 15     # sample an open-window every N frames
MAX_OPEN_WINDOWS_PER_FILE = 60  # cap so 'open' doesn't dominate the dataset

def load_rows_by_file(path):
    """Group CSV rows by source_file, sorted by frame_index."""
    files = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            files[row["source_file"]].append({
                "subject": row["subject"],
                "label": row["label"],
                "frame_index": int(row["frame_index"]),
                "ear": float(row["ear"]),
            })
    for fname in files:
        files[fname].sort(key=lambda r: r["frame_index"])
    return files


def file_original_label(rows):
    """The label a file 'is' overall, ignoring incidental blink_artifact rows."""
    for r in rows:
        if r["label"] != "blink_artifact":
            return r["label"]
    return rows[0]["label"]


def extract_open_windows(rows, subject, source_file, writer):
    """Sliding windows over clean open-eye frames only."""
    count = 0
    i = 0
    while i + WINDOW_SIZE <= len(rows) and count < MAX_OPEN_WINDOWS_PER_FILE:
        window = rows[i:i + WINDOW_SIZE]
        if all(r["label"] == "open" for r in window):
            ear_values = [r["ear"] for r in window]
            writer.writerow({
                **{f"ear_{j}": ear_values[j] for j in range(WINDOW_SIZE)},
                "duration": 0,
                "label": "open",
                "subject": subject,
                "source_file": source_file,
            })
            count += 1
        i += OPEN_WINDOW_STRIDE
    return count


def detect_blink_events(rows):
    """Runs the Week 2 state machine over one video's EAR sequence.
    Returns a list of (start_idx, end_idx, closed_duration) for each
    confirmed blink event. closed_duration is measured from the real
    last-closed frame, NOT the confirmation frame, to avoid inflating
    duration by the MIN_OPEN_FRAMES confirmation delay."""
    events = []
    state = "OPEN"
    open_counter = 0
    closed_counter = 0
    closing_start_idx = None
    last_closed_idx = None

    for idx, r in enumerate(rows):
        ear = r["ear"]
        if ear < EAR_THRESHOLD:
            closed_counter += 1
            open_counter = 0
            if closing_start_idx is None:
                closing_start_idx = idx
            last_closed_idx = idx
        else:
            open_counter += 1
            closed_counter = 0

        if state == "OPEN" and closed_counter >= MIN_CLOSED_FRAMES:
            state = "CLOSED"
        elif state == "CLOSED" and open_counter >= MIN_OPEN_FRAMES:
            state = "OPEN"
            closed_duration = last_closed_idx - closing_start_idx + 1
            events.append((closing_start_idx, last_closed_idx, closed_duration))
            closing_start_idx = None
            last_closed_idx = None

    return events

def extract_blink_windows(rows, subject, source_file, writer, original_label):
    """For each detected blink event, extract a fixed-size window centered
    on the event and classify it by measured duration."""
    events = detect_blink_events(rows)

    durations = [d for _, _, d in events]
    print(f"    durations for {source_file}: {durations}")

    max_valid = (MAX_VALID_CLOSED_FRAMES_LONG if original_label == "long_blink"
                 else MAX_VALID_CLOSED_FRAMES_DEFAULT)

    counts = {"short_blink": 0, "long_blink": 0}

    for start_idx, end_idx, closed_duration in events:
        if closed_duration > max_valid:
            print(f"    skipping merged/anomalous event "
                  f"(duration={closed_duration} frames) in {source_file}")
            continue
        center = (start_idx + end_idx) // 2
        half = WINDOW_SIZE // 2
        win_start = max(0, center - half)
        win_end = win_start + WINDOW_SIZE

        if win_end > len(rows):
            win_end = len(rows)
            win_start = max(0, win_end - WINDOW_SIZE)

        window = rows[win_start:win_end]
        if len(window) < WINDOW_SIZE:
            continue  # too close to the edge of the recording, skip

        if original_label == "normal_blink":
            event_label = ("long_blink" if closed_duration >= LONG_BLINK_MIN_CLOSED_FRAMES
                           else "short_blink")
        else:
            event_label = original_label  # "short_blink" or "long_blink" as recorded

        ear_values = [r["ear"] for r in window]
        writer.writerow({
            **{f"ear_{j}": ear_values[j] for j in range(WINDOW_SIZE)},
            "duration": closed_duration,
            "label": event_label,
            "subject": subject,
            "source_file": source_file,
        })
        counts[event_label] += 1

    return counts


def main():
    files = load_rows_by_file(INPUT_CSV)
    fieldnames = [f"ear_{j}" for j in range(WINDOW_SIZE)] + ["duration","label", "subject", "source_file"]

    totals = defaultdict(int)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for source_file, rows in files.items():
            subject = rows[0]["subject"]
            original_label = file_original_label(rows)

            if original_label == "open":
                n = extract_open_windows(rows, subject, source_file, writer)
                totals["open"] += n
                print(f"{source_file}: {n} open windows")
            else:
                counts = extract_blink_windows(rows, subject, source_file, writer, original_label)
                for k, v in counts.items():
                    totals[k] += v
                print(f"{source_file}: {counts['short_blink']} short_blink, "
                      f"{counts['long_blink']} long_blink events detected")

    print(f"\n{'='*50}")
    print("TOTAL WINDOWS PER CLASS")
    print(f"{'='*50}")
    for label, count in totals.items():
        print(f"  {label}: {count}")
    print(f"\nWritten to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
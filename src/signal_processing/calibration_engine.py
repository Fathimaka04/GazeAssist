"""
src/signal_processing/calibration_engine.py

Week 7 — Zero-learning calibration engine.
Updated — per-person EAR calibration now measures BOTH open-eye and
closed-eye baselines directly, instead of guessing the closed-eye value
from open-eye statistics alone.

WHY THIS CHANGED:
The original version computed ear_threshold = ear_mean - (1.5 * ear_std)
using only open-eye samples. This works only if the natural EAR wobble
captured during the short calibration window happens to be a good stand-in
for "how far EAR drops on a real blink." For some users (e.g. if their
eyes sit very steady during that window, or if landmark noise is higher
during live use than during the still calibration window — which can
happen with facial hair partially obscuring lower-eyelid landmarks), this
produced a threshold sitting too close to the natural open-eye value,
causing normal noise to be misread as blinks.

Fix: measure the closed-eye EAR directly (same "measure, don't guess"
principle already used for LEFT/RIGHT/UP/DOWN gaze calibration), then set
the threshold at a point between the two REAL measured values.
"""

import numpy as np

GAZE_SAMPLE_SPREAD_LIMIT = 0.5

# How far above the closed-eye EAR the threshold sits, as a fraction of
# the gap between open and closed. 0.5 = exact midpoint.
THRESHOLD_POSITION = 0.5

# Spread limit for detecting "a blink happened during the open-eye
# calibration window" - discard and retry instead of averaging a blink
# into the open-eye baseline.
OPEN_EAR_SPREAD_LIMIT = 0.08

# Closed-eye phase needs fewer samples - it's a deliberate, held action.
CLOSED_EAR_FRAMES_NEEDED = 20


class CalibrationEngine:
    def __init__(self, calibration_frames=45, closed_frames=CLOSED_EAR_FRAMES_NEEDED):
        self.calibration_frames_needed = calibration_frames
        self.closed_frames_needed = closed_frames
        self.is_calibrated = False

        # "open" -> collecting open-eye baseline
        # "closed" -> collecting closed-eye baseline (open phase done)
        self.phase = "open"

        self.ear_samples = []          # open-eye phase
        self.closed_ear_samples = []   # closed-eye phase
        self.gaze_x_samples = []
        self.gaze_y_samples = []

        self.ear_threshold = None
        self.ear_open_baseline = None
        self.ear_closed_baseline = None
        self.gaze_x_baseline = None
        self.gaze_y_baseline = None

        self._start_time = None
        self._end_time = None

    def add_sample(self, ear_value, gaze_x, gaze_y, timestamp):
        if self.is_calibrated:
            return

        if self._start_time is None:
            self._start_time = timestamp

        if self.phase == "open":
            if ear_value is not None:
                self.ear_samples.append(ear_value)
            if gaze_x is not None:
                self.gaze_x_samples.append(gaze_x)
            if gaze_y is not None:
                self.gaze_y_samples.append(gaze_y)

            if len(self.ear_samples) >= self.calibration_frames_needed:
                self._finish_open_phase(timestamp)

        elif self.phase == "closed":
            if ear_value is not None:
                self.closed_ear_samples.append(ear_value)

            if len(self.closed_ear_samples) >= self.closed_frames_needed:
                self._finalize_calibration(timestamp)

    def _finish_open_phase(self, timestamp):
        if self.gaze_x_samples:
            gx_spread = max(self.gaze_x_samples) - min(self.gaze_x_samples)
            if gx_spread > GAZE_SAMPLE_SPREAD_LIMIT:
                self._reset_open_phase()
                return

        if self.gaze_y_samples:
            gy_spread = max(self.gaze_y_samples) - min(self.gaze_y_samples)
            if gy_spread > GAZE_SAMPLE_SPREAD_LIMIT:
                self._reset_open_phase()
                return

        # NEW: if a blink happened during this window, EAR dips sharply -
        # spread would be much bigger than normal open-eye micro-variation.
        ear_spread = max(self.ear_samples) - min(self.ear_samples)
        if ear_spread > OPEN_EAR_SPREAD_LIMIT:
            self._reset_open_phase()
            return

        self.ear_open_baseline = float(np.mean(self.ear_samples))
        self.gaze_x_baseline = float(np.mean(self.gaze_x_samples)) if self.gaze_x_samples else 0.0
        self.gaze_y_baseline = float(np.mean(self.gaze_y_samples)) if self.gaze_y_samples else 0.0

        self.phase = "closed"

    def _reset_open_phase(self):
        self.ear_samples = []
        self.gaze_x_samples = []
        self.gaze_y_samples = []
        self._start_time = None

    def _finalize_calibration(self, timestamp):
        self.ear_closed_baseline = float(np.mean(self.closed_ear_samples))

        # Sanity check: closed-eye EAR should be meaningfully lower than
        # open-eye EAR. If not, the person probably didn't actually close
        # their eyes (or tracking failed) - retry the closed phase.
        gap = self.ear_open_baseline - self.ear_closed_baseline
        if gap < 0.03:
            self.closed_ear_samples = []
            return

        self.ear_threshold = self.ear_closed_baseline + (THRESHOLD_POSITION * gap)

        self._end_time = timestamp
        self.is_calibrated = True

    def get_progress(self):
        """Returns (current, total, phase) for on-screen display."""
        if self.phase == "open":
            return len(self.ear_samples), self.calibration_frames_needed, "open"
        return len(self.closed_ear_samples), self.closed_frames_needed, "closed"

    def get_calibration_duration(self):
        if self._start_time is None or self._end_time is None:
            return None
        return self._end_time - self._start_time

    def get_summary(self):
        if not self.is_calibrated:
            return None
        return {
            "ear_threshold": round(self.ear_threshold, 4),
            "ear_open_baseline": round(self.ear_open_baseline, 4),
            "ear_closed_baseline": round(self.ear_closed_baseline, 4),
            "gaze_x_baseline": round(self.gaze_x_baseline, 4),
            "gaze_y_baseline": round(self.gaze_y_baseline, 4),
            "calibration_duration_sec": round(self.get_calibration_duration(), 2),
            "ear_samples_count": len(self.ear_samples),
        }
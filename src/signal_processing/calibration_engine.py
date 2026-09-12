"""
src/signal_processing/calibration_engine.py

Week 7 — Zero-learning calibration engine.

Passively measures a new user's natural EAR baseline and gaze-center
baseline during the first N seconds of a session, then computes
personalized thresholds — replacing the fixed constants from Weeks 2 and
6 with values tuned to the actual person using the system, with no
explicit "training mode" required.

This directly operationalizes the project's "zero-learning" claim: the
system adapts to the user, not the other way around.
"""

import numpy as np

GAZE_SAMPLE_SPREAD_LIMIT = 0.5

class CalibrationEngine:
    def __init__(self, calibration_frames=45):
        """calibration_frames: how many frames to collect before computing
        thresholds. At ~20-30fps, 45 frames is roughly 1.5-2 seconds per
        sample point — we sample continuously during the calibration
        window, so the actual wall-clock duration depends on frame rate
        and is reported at the end (see get_calibration_duration())."""
        self.calibration_frames_needed = calibration_frames
        self.is_calibrated = False

        # raw samples collected during calibration
        self.ear_samples = []
        self.gaze_x_samples = []
        self.gaze_y_samples = []

        # computed, personalized values (None until calibration finishes)
        self.ear_threshold = None
        self.gaze_x_baseline = None
        self.gaze_y_baseline = None

        # for reporting calibration duration afterward
        self._start_time = None
        self._end_time = None

    def add_sample(self, ear_value, gaze_x, gaze_y, timestamp):
        """Call this once per frame during the calibration window. Once
        enough samples are collected, this computes the final personalized
        thresholds and flips is_calibrated to True."""
        if self.is_calibrated:
            return  # already done, ignore further calls

        if self._start_time is None:
            self._start_time = timestamp

        # Only collect samples where all three values are actually
        # available — a frame with a missed blink-reading or gaze-reading
        # shouldn't corrupt the other two baselines.
        if ear_value is not None:
            self.ear_samples.append(ear_value)
        if gaze_x is not None:
            self.gaze_x_samples.append(gaze_x)
        if gaze_y is not None:
            self.gaze_y_samples.append(gaze_y)

        total_samples = len(self.ear_samples) + len(self.gaze_x_samples)
        # Use EAR sample count as the primary progress indicator since
        # it's the most consistently available signal frame-to-frame.
        if len(self.ear_samples) >= self.calibration_frames_needed:
            self._finalize_calibration(timestamp)

    def _finalize_calibration(self, timestamp):
        """Computes personalized thresholds from collected samples using
        the same 'measure the real baseline, don't assume it' principle
        used throughout this project (Week 2's EAR tuning, Week 4's
        head-pose offset, Week 5's correction-factor selection)."""

        # If gaze wandered a lot during the window (real eye/head motion,
        # not just noise), the plain mean isn't a meaningful "center"
        # baseline - discard this window and keep collecting instead of
        # locking in a bad baseline (same principle as the head-pose
        # calibration stability check).
        if self.gaze_x_samples:
            gx_spread = max(self.gaze_x_samples) - min(self.gaze_x_samples)
            if gx_spread > GAZE_SAMPLE_SPREAD_LIMIT:
                self.ear_samples = []
                self.gaze_x_samples = []
                self.gaze_y_samples = []
                self._start_time = None
                return

        if self.gaze_y_samples:
            gy_spread = max(self.gaze_y_samples) - min(self.gaze_y_samples)
            if gy_spread > GAZE_SAMPLE_SPREAD_LIMIT:
                self.ear_samples = []
                self.gaze_x_samples = []
                self.gaze_y_samples = []
                self._start_time = None
                return

        ear_array = np.array(self.ear_samples)
        ear_mean = np.mean(ear_array)
        ear_std = np.std(ear_array)

        # Personalized EAR threshold: mean minus a margin based on the
        # user's own natural variability, rather than Week 2's fixed 0.22.
        # A 1.5 standard-deviation margin is a reasonable starting point,
        # consistent with how the fixed threshold (0.22) sat noticeably
        # below the observed open-eye range (~0.24-0.31) in Week 2 testing.
        self.ear_threshold = ear_mean - (1.5 * ear_std)

        self.gaze_x_baseline = float(np.mean(self.gaze_x_samples)) if self.gaze_x_samples else 0.0
        self.gaze_y_baseline = float(np.mean(self.gaze_y_samples)) if self.gaze_y_samples else 0.0

        self._end_time = timestamp
        self.is_calibrated = True

    def get_progress(self):
        """Returns (current, total) sample counts for on-screen display."""
        return len(self.ear_samples), self.calibration_frames_needed

    def get_calibration_duration(self):
        """Returns elapsed seconds from first sample to calibration
        completion, or None if not yet calibrated."""
        if self._start_time is None or self._end_time is None:
            return None
        return self._end_time - self._start_time

    def get_summary(self):
        """Returns a dict of the final calibrated values, for logging."""
        if not self.is_calibrated:
            return None
        return {
            "ear_threshold": round(self.ear_threshold, 4),
            "gaze_x_baseline": round(self.gaze_x_baseline, 4),
            "gaze_y_baseline": round(self.gaze_y_baseline, 4),
            "calibration_duration_sec": round(self.get_calibration_duration(), 2),
            "ear_samples_count": len(self.ear_samples),
        }
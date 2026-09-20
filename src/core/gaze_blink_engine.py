"""
src/core/gaze_blink_engine.py

Week 8 — Integration checkpoint. Extracts the core calibration + zone
classification + blink detection logic (verified working across many
sessions in Week 7) out of the test/demo script verify_calibrated_system.py
into a reusable, display-independent engine.

Week 9 addition: blink_type classification ("short"/"long") added to
EngineResult, based on BlinkClassifier.last_blink_duration_frames.

Week 9 SOS redesign addition: blink_duration_frames (the raw frame
count itself) is now also passed through onto EngineResult.

Week 11 REDESIGN (diagonal quadrant zones): the old zone system used a
priority/override mechanism (yaw override takes priority over
vertical, with hysteresis) to produce 5 zones: UP/DOWN/LEFT/RIGHT/
CENTER - only one axis "won" at a time. This has been replaced with
independent horizontal-side and vertical-side decisions that COMBINE
into one of 4 diagonal quadrant zones: LEFT_UP, LEFT_DOWN, RIGHT_UP,
RIGHT_DOWN. There is no more CENTER zone - every frame resolves to
exactly one quadrant. This matches the phrase board's 2x2 tile grid
directly (each tile IS a quadrant), instead of needing an arbitrary
mapping between 4 cross-zones and 4 grid tiles.

Each axis (horizontal, vertical) keeps its own small independent
hysteresis margin (QUADRANT_AXIS_MARGIN) so the side doesn't flicker
right at the zero-crossing - this replaces the old HORIZ_PRIORITY_*,
CENTER_ENTER/EXIT_THRESHOLD, and SWITCH_MARGIN mechanisms entirely,
which only made sense under the old single-axis-wins design.

QUADRANT_AXIS_MARGIN=0.15 is an UNTUNED starting guess, same caveat
as every other constant here - needs a live controlled test before
being trusted, per project methodology (never trust a number without
an isolated test).

Directional CALIBRATION itself (Phase 2, DIRECTIONS/DIRECTION_PROMPTS,
the LEFT/RIGHT/UP/DOWN sampling sequence, symmetry/spread checks) is
UNCHANGED - it still needs pure single-axis extremes to correctly
derive center_yaw/yaw_scale and pitch_center/pitch_scale. Only the
RUNTIME zone decision in _process_normal_operation changed.

TEMPORARY: a print() in _process_normal_operation measures real blink
duration frame counts. Remove once measurement is done.

Usage:
    engine = GazeBlinkEngine(subject_label="self")
    for frame in cam.frames():
        result = engine.process_frame(frame)
        if result.status == "ready":
            ...use result.zone (one of LEFT_UP/LEFT_DOWN/RIGHT_UP/
               RIGHT_DOWN), result.blink_type, result.blink_duration_frames...
"""

import time
import math
from dataclasses import dataclass, field
from typing import Optional

from src.sensing.camera_capture import CameraCapture  # noqa: F401 (re-exported for convenience)
from src.perception.face_mesh import FaceMeshDetector
from src.perception.head_pose import HeadPoseEstimator
from src.perception.gaze_vector import estimate_raw_gaze_x, estimate_raw_gaze_y
from src.signal_processing.ear_calculator import average_ear
from src.signal_processing.calibration_engine import CalibrationEngine
from src.signal_processing.blink_classifier import BlinkClassifier
from src.signal_processing.calibration_logger import CalibrationLogger

# --- Verified Week 7 calibration constants (unchanged) ---
DIRECTION_SAMPLE_FRAMES = 30
DIRECTION_SETTLE_FRAMES = 45

MIN_AXIS_SEPARATION_HORIZ = 5.0
MIN_RAW_PITCH_SEPARATION = 1.5
DIRECTION_SAMPLE_SPREAD_LIMIT_HORIZ = 30.0
DIRECTION_SAMPLE_SPREAD_LIMIT_VERT = 12.0

HORIZ_SYMMETRY_TOLERANCE = 0.25

PITCH_WEIGHT = 0.7
GAZE_Y_WEIGHT = 0.3
GAZE_Y_SCALE_FLOOR = 0.05

VERTICAL_SMOOTHING_FRAMES = 5

# --- Week 11: diagonal quadrant zone decision (replaces old
# priority/override + CENTER hysteresis system) ---
# Small per-axis hysteresis margin to avoid flicker right at the
# zero-crossing. UNTUNED - needs a live controlled test, same as
# LONG_BLINK_MIN_FRAMES did.
QUADRANT_AXIS_MARGIN = 0.15

# --- Blink duration classification (Week 9) ---
# MEASURED for this project's own tester (see handoff history):
# natural blinks clustered 6-11 frames, deliberate holds 13+.
LONG_BLINK_MIN_FRAMES = 13

DIRECTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]
DIRECTION_PROMPTS = {
    "LEFT": "Water",
    "RIGHT": "Bathroom",
    "UP": "Food",
    "DOWN": "Help",
}


@dataclass
class EngineResult:
    """Structured output of one process_frame() call."""
    status: str  # "no_face" | "calibrating_blink_open" | "calibrating_blink_closed" |
                 # "calibrating_pose" | "calibrating_directions" |
                 # "calibration_restarted" | "ready"

    progress_current: Optional[int] = None
    progress_total: Optional[int] = None

    direction_prompt: Optional[str] = None
    direction_settling: Optional[bool] = None
    direction_sample_progress: Optional[int] = None
    direction_sample_total: Optional[int] = None

    restart_reason: Optional[str] = None

    # ready (normal operation) - zone is now one of LEFT_UP/LEFT_DOWN/
    # RIGHT_UP/RIGHT_DOWN (no more CENTER)
    zone: Optional[str] = None
    zone_changed: Optional[bool] = None
    blink_count: Optional[int] = None
    blink_just_occurred: Optional[bool] = None
    blink_type: Optional[str] = None
    blink_duration_frames: Optional[int] = None
    ear: Optional[float] = None
    ear_threshold: Optional[float] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    horizontal_score: Optional[float] = None
    vertical_score: Optional[float] = None
    raw_gaze: Optional[tuple] = None

    calibration_summary: Optional[dict] = None


def _sample_spread_1d(values):
    if not values:
        return 0.0
    return max(values) - min(values)


class GazeBlinkEngine:
    """
    Display-independent core engine: calibration + 4-quadrant diagonal
    gaze classification (yaw for horizontal side, fused pitch+gaze_y
    for vertical side, combined into one of 4 quadrants) + blink
    detection.
    """

    def __init__(self, subject_label="self", enable_sqlite_logging=True):
        self.detector = FaceMeshDetector()
        self.pose_estimator = HeadPoseEstimator()
        self.calibration = CalibrationEngine(calibration_frames=45)
        self.blink_detector: Optional[BlinkClassifier] = None
        self.cal_logger = CalibrationLogger() if enable_sqlite_logging else None
        self.subject_label = subject_label

        self.zone = None
        self.last_zone = None
        self.horiz_side = None
        self.vert_side = None

        self.dir_phase_index = 0
        self.dir_samples = []
        self.dir_settle_counter = 0
        self.dir_deviations = {}
        self.directions_done = False

        self.center_yaw = 0.0
        self.yaw_scale = 1.0
        self.horiz_pos_label = self.horiz_neg_label = None

        self.pitch_center = self.pitch_scale = 0.0
        self.gaze_y_center = self.gaze_y_scale = 0.0
        self.pitch_up_positive = True
        self.gaze_y_up_positive = True
        self.vert_pos_label = self.vert_neg_label = None

        self.horizontal_score = 0.0
        self.vertical_score = 0.0
        self.vertical_score_history = []

    def _fused_vertical(self, p, gy):
        p_score = (p - self.pitch_center) / self.pitch_scale
        if not self.pitch_up_positive:
            p_score = -p_score
        gy_score = (gy - self.gaze_y_center) / self.gaze_y_scale
        if not self.gaze_y_up_positive:
            gy_score = -gy_score
        return (PITCH_WEIGHT * p_score) + (GAZE_Y_WEIGHT * gy_score)

    def process_frame(self, frame) -> EngineResult:
        results = self.detector.process(frame)
        right_eye_pts, left_eye_pts = self.detector.get_eye_points(results, frame.shape)
        ear = average_ear(right_eye_pts, left_eye_pts)

        pose_result = self.pose_estimator.estimate(results, frame.shape)
        raw_gaze_x = estimate_raw_gaze_x(results, frame.shape)
        raw_gaze_y = estimate_raw_gaze_y(results, frame.shape)

        if not self.calibration.is_calibrated:
            self.calibration.add_sample(ear, raw_gaze_x, raw_gaze_y, time.time())
            current, total, phase = self.calibration.get_progress()
            status = "calibrating_blink_open" if phase == "open" else "calibrating_blink_closed"
            return EngineResult(status=status,
                         progress_current=current, progress_total=total)
        if self.blink_detector is None:
            self.blink_detector = BlinkClassifier(ear_threshold=self.calibration.ear_threshold)

        if pose_result is None:
            return EngineResult(status="no_face")

        yaw, pitch, roll, pose_is_calibrating = pose_result

        if pose_is_calibrating:
            return EngineResult(status="calibrating_pose")

        gaze_y_rel = None
        if raw_gaze_y is not None:
            gaze_y_rel = raw_gaze_y - self.calibration.gaze_y_baseline

        if not self.directions_done:
            return self._process_direction_calibration(yaw, pitch, gaze_y_rel)

        return self._process_normal_operation(ear, yaw, pitch, gaze_y_rel, raw_gaze_x, raw_gaze_y)

    def _process_direction_calibration(self, yaw, pitch, gaze_y_rel):
        current_direction = DIRECTIONS[self.dir_phase_index]
        is_horizontal_direction = current_direction in ("LEFT", "RIGHT")

        if self.dir_settle_counter < DIRECTION_SETTLE_FRAMES:
            self.dir_settle_counter += 1
            return EngineResult(
                status="calibrating_directions",
                direction_prompt=DIRECTION_PROMPTS[current_direction],
                direction_settling=True,
                direction_sample_progress=0,
                direction_sample_total=DIRECTION_SAMPLE_FRAMES,
            )

        self.dir_samples.append((yaw, pitch, gaze_y_rel))

        if len(self.dir_samples) < DIRECTION_SAMPLE_FRAMES:
            return EngineResult(
                status="calibrating_directions",
                direction_prompt=DIRECTION_PROMPTS[current_direction],
                direction_settling=False,
                direction_sample_progress=len(self.dir_samples),
                direction_sample_total=DIRECTION_SAMPLE_FRAMES,
            )

        if is_horizontal_direction:
            spread = _sample_spread_1d([s[0] for s in self.dir_samples])
            spread_limit = DIRECTION_SAMPLE_SPREAD_LIMIT_HORIZ
        else:
            spread = _sample_spread_1d([s[1] for s in self.dir_samples])
            spread_limit = DIRECTION_SAMPLE_SPREAD_LIMIT_VERT

        if spread > spread_limit:
            reason = (f"{current_direction} CAPTURE TOO SHAKY (spread={spread:.2f}, "
                      f"limit={spread_limit:.1f}) - hold steadier")
            self.dir_samples = []
            self.dir_settle_counter = 0
            return EngineResult(status="calibration_restarted", restart_reason=reason)

        mean_yaw = sum(s[0] for s in self.dir_samples) / len(self.dir_samples)
        mean_pitch = sum(s[1] for s in self.dir_samples) / len(self.dir_samples)
        valid_gy = [s[2] for s in self.dir_samples if s[2] is not None]
        mean_gaze_y = sum(valid_gy) / len(valid_gy) if valid_gy else 0.0

        self.dir_deviations[current_direction] = (mean_yaw, mean_pitch, mean_gaze_y)
        self.dir_samples = []
        self.dir_settle_counter = 0

        if current_direction == "RIGHT" and "LEFT" in self.dir_deviations:
            left_yaw_check = abs(self.dir_deviations["LEFT"][0])
            right_yaw_check = abs(self.dir_deviations["RIGHT"][0])
            larger = max(left_yaw_check, right_yaw_check)
            smaller = min(left_yaw_check, right_yaw_check)

            if larger > 0 and (larger - smaller) / larger > HORIZ_SYMMETRY_TOLERANCE:
                reason = (f"LEFT/RIGHT ASYMMETRIC (left={left_yaw_check:.2f}, "
                          f"right={right_yaw_check:.2f}) - turn equally on both sides")
                self.dir_deviations.pop("LEFT", None)
                self.dir_deviations.pop("RIGHT", None)
                self.dir_phase_index = 0
                return EngineResult(status="calibration_restarted", restart_reason=reason)
            else:
                self.dir_phase_index += 1
        else:
            self.dir_phase_index += 1

        if self.dir_phase_index >= len(DIRECTIONS) and "DOWN" in self.dir_deviations:
            return self._finalize_calibration()

        return EngineResult(
            status="calibrating_directions",
            direction_prompt=DIRECTION_PROMPTS[DIRECTIONS[min(self.dir_phase_index, 3)]],
            direction_settling=True,
            direction_sample_progress=0,
            direction_sample_total=DIRECTION_SAMPLE_FRAMES,
        )

    def _finalize_calibration(self):
        left_yaw, left_pitch, left_gy = self.dir_deviations["LEFT"]
        right_yaw, right_pitch, right_gy = self.dir_deviations["RIGHT"]
        up_yaw, up_pitch, up_gy = self.dir_deviations["UP"]
        down_yaw, down_pitch, down_gy = self.dir_deviations["DOWN"]

        lr_sep = abs(left_yaw - right_yaw)
        raw_pitch_sep = abs(up_pitch - down_pitch)

        if lr_sep < MIN_AXIS_SEPARATION_HORIZ or raw_pitch_sep < MIN_RAW_PITCH_SEPARATION:
            reason = (f"CALIBRATION TOO CLOSE (lr_sep={lr_sep:.2f}, "
                      f"raw_pitch_sep={raw_pitch_sep:.2f}) - tilt your head more clearly for UP/DOWN")
            self.dir_phase_index = 0
            self.dir_deviations = {}
            return EngineResult(status="calibration_restarted", restart_reason=reason)

        self.center_yaw = (left_yaw + right_yaw) / 2
        self.yaw_scale = max(abs(right_yaw - left_yaw) / 2, 1e-6)
        self.horiz_pos_label = "RIGHT" if right_yaw > left_yaw else "LEFT"
        self.horiz_neg_label = "LEFT" if right_yaw > left_yaw else "RIGHT"

        self.pitch_center = (up_pitch + down_pitch) / 2
        self.pitch_scale = max(abs(down_pitch - up_pitch) / 2, 1e-6)
        self.gaze_y_center = (up_gy + down_gy) / 2
        self.gaze_y_scale = max(abs(down_gy - up_gy) / 2, GAZE_Y_SCALE_FLOOR)

        self.pitch_up_positive = up_pitch > down_pitch
        self.gaze_y_up_positive = up_gy > down_gy

        self.vert_pos_label = "UP"
        self.vert_neg_label = "DOWN"
        self.vertical_score_history = []

        # Sensible starting sides so the very first "ready" frame
        # doesn't need a None check downstream.
        self.horiz_side = self.horiz_pos_label
        self.vert_side = self.vert_pos_label
        self.zone = f"{self.horiz_side}_{self.vert_side}"
        self.last_zone = self.zone

        up_fused = self._fused_vertical(up_pitch, up_gy)
        down_fused = self._fused_vertical(down_pitch, down_gy)

        summary = {
            "left": {"yaw": left_yaw, "pitch": left_pitch, "gaze_y": left_gy},
            "right": {"yaw": right_yaw, "pitch": right_pitch, "gaze_y": right_gy},
            "up": {"yaw": up_yaw, "pitch": up_pitch, "gaze_y": up_gy},
            "down": {"yaw": down_yaw, "pitch": down_pitch, "gaze_y": down_gy},
            "center_yaw": self.center_yaw, "yaw_scale": self.yaw_scale,
            "pitch_center": self.pitch_center, "pitch_scale": self.pitch_scale,
            "raw_pitch_sep": raw_pitch_sep,
            "gaze_y_center": self.gaze_y_center, "gaze_y_scale": self.gaze_y_scale,
            "fused_up_score": up_fused, "fused_down_score": down_fused,
            "ear_threshold": self.calibration.ear_threshold,
        }

        if self.cal_logger is not None:
            self.cal_logger.log_calibration(
                left_yaw=left_yaw, left_pitch=left_pitch, left_gaze_y=left_gy,
                right_yaw=right_yaw, right_pitch=right_pitch, right_gaze_y=right_gy,
                up_yaw=up_yaw, up_pitch=up_pitch, up_gaze_y=up_gy,
                down_yaw=down_yaw, down_pitch=down_pitch, down_gaze_y=down_gy,
                center_yaw=self.center_yaw, yaw_scale=self.yaw_scale,
                pitch_center=self.pitch_center, pitch_scale=self.pitch_scale,
                raw_pitch_sep=raw_pitch_sep,
                gaze_y_center=self.gaze_y_center, gaze_y_scale=self.gaze_y_scale,
                ear_threshold=self.calibration.ear_threshold,
                subject_label=self.subject_label,
            )

        self.directions_done = True
        print(self.calibration.get_summary())
        return EngineResult(status="ready", zone=self.zone, zone_changed=False,
                             blink_count=0, blink_just_occurred=False, blink_type=None,
                             blink_duration_frames=None,
                             calibration_summary=summary)

    def _process_normal_operation(self, ear, yaw, pitch, gaze_y_rel, raw_gaze_x, raw_gaze_y):
        blink_just_occurred = self.blink_detector.update(ear)

        blink_type = None
        blink_duration_frames = None
        if blink_just_occurred:
            blink_duration_frames = self.blink_detector.last_blink_duration_frames
            print(f"DEBUG BLINK: zone={self.zone}, RAW duration_frames={blink_duration_frames}")
            if blink_duration_frames >= LONG_BLINK_MIN_FRAMES:
                blink_type = "long"
            else:
                blink_type = "short"

        self.horizontal_score = (yaw - self.center_yaw) / self.yaw_scale

        if gaze_y_rel is not None:
            raw_vertical = self._fused_vertical(pitch, gaze_y_rel)
        else:
            p_score = (pitch - self.pitch_center) / self.pitch_scale
            if not self.pitch_up_positive:
                p_score = -p_score
            raw_vertical = p_score

        self.vertical_score_history.append(raw_vertical)
        if len(self.vertical_score_history) > VERTICAL_SMOOTHING_FRAMES:
            self.vertical_score_history.pop(0)
        self.vertical_score = sum(self.vertical_score_history) / len(self.vertical_score_history)

        # --- Week 11: independent per-axis quadrant decision ---
        # Each axis flips side only once it clears QUADRANT_AXIS_MARGIN
        # past the zero-crossing; inside the margin, keep the previous
        # side (hysteresis against flicker). No CENTER dead zone -
        # every frame belongs to exactly one quadrant.
        if self.horizontal_score > QUADRANT_AXIS_MARGIN:
            self.horiz_side = self.horiz_pos_label
        elif self.horizontal_score < -QUADRANT_AXIS_MARGIN:
            self.horiz_side = self.horiz_neg_label
        # else: keep previous self.horiz_side

        if self.vertical_score > QUADRANT_AXIS_MARGIN:
            self.vert_side = self.vert_pos_label
        elif self.vertical_score < -QUADRANT_AXIS_MARGIN:
            self.vert_side = self.vert_neg_label
        # else: keep previous self.vert_side

        self.zone = f"{self.horiz_side}_{self.vert_side}"
        zone_changed = self.zone != self.last_zone
        self.last_zone = self.zone

        raw_gaze = None
        if raw_gaze_x is not None and raw_gaze_y is not None:
            raw_gaze = (raw_gaze_x, raw_gaze_y)

        return EngineResult(
            status="ready",
            zone=self.zone,
            zone_changed=zone_changed,
            blink_count=self.blink_detector.blink_count,
            blink_just_occurred=blink_just_occurred,
            blink_type=blink_type,
            blink_duration_frames=blink_duration_frames,
            ear=ear,
            ear_threshold=self.calibration.ear_threshold,
            yaw=yaw, pitch=pitch,
            horizontal_score=self.horizontal_score,
            vertical_score=self.vertical_score,
            raw_gaze=raw_gaze,
        )

    def close(self):
        if self.cal_logger is not None:
            self.cal_logger.close()
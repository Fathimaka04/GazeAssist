"""
src/core/gaze_blink_engine.py

Week 8 — Integration checkpoint. Extracts the core calibration + zone
classification + blink detection logic (verified working across many
sessions in Week 7) out of the test/demo script verify_calibrated_system.py
into a reusable, display-independent engine. This is what Week 9's
multimodal fusion and Week 10's safety state machine will call directly,
instead of importing from a script full of cv2.imshow/cv2.waitKey calls.

Design: feed frames in via process_frame(), get structured results out.
No OpenCV window handling here - that stays in the demo/verification
scripts. All tuned constants from Week 7 are preserved exactly, exposed
as constructor parameters with the same defaults so behavior is identical
to the verified system.

Week 9 addition: blink_type classification ("short"/"long") added to
EngineResult, based on BlinkClassifier.last_blink_duration_frames, so
IntentFusionEngine can distinguish deliberate long blinks from normal
ones without changing any Week 7 zone/blink logic.

Week 9 SOS redesign addition: blink_duration_frames (the raw frame
count itself, not just the short/long label) is now also passed
through onto EngineResult. This lets IntentFusionEngine distinguish a
normal selection long-blink from a much longer deliberate SOS hold,
without needing multiple timed blinks in sequence.

TEMPORARY: a print() has been added in _process_normal_operation to
measure real blink duration frame counts, to pick a real value for
LONG_BLINK_MIN_FRAMES instead of guessing. Remove it once measurement
is done.

Usage:
    engine = GazeBlinkEngine(subject_label="self")
    for frame in cam.frames():
        result = engine.process_frame(frame)
        if result.status == "calibrating_blink":
            ...show result.progress...
        elif result.status == "calibrating_pose":
            ...
        elif result.status == "calibrating_directions":
            ...show result.direction_prompt, result.direction_progress...
        elif result.status == "calibration_restarted":
            ...show result.restart_reason...
        elif result.status == "ready":
            ...use result.zone, result.blink_count, result.blink_just_occurred,
               result.blink_type, result.blink_duration_frames...
    engine.close()
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

# --- Verified Week 7 constants (unchanged) ---
DIRECTION_SAMPLE_FRAMES = 30
DIRECTION_SETTLE_FRAMES = 45

MIN_AXIS_SEPARATION_HORIZ = 5.0
MIN_RAW_PITCH_SEPARATION = 1.5
DIRECTION_SAMPLE_SPREAD_LIMIT_HORIZ = 30.0
DIRECTION_SAMPLE_SPREAD_LIMIT_VERT = 12.0

HORIZ_SYMMETRY_TOLERANCE = 0.25

SWITCH_MARGIN = 0.3

HORIZ_PRIORITY_ENTER = 12.0
HORIZ_PRIORITY_EXIT = 8.0

CENTER_EXIT_THRESHOLD = 0.7
CENTER_ENTER_THRESHOLD = 0.3

PITCH_WEIGHT = 0.7
GAZE_Y_WEIGHT = 0.3
GAZE_Y_SCALE_FLOOR = 0.05

VERTICAL_SMOOTHING_FRAMES = 5

# --- Blink duration classification (Week 9) ---
# Frame threshold for short-blink vs long-blink. This needs to be
# MEASURED against real long blinks like every other constant here -
# 10 frames (~0.33s at 30fps) is a starting guess, not a final value.
LONG_BLINK_MIN_FRAMES = 9

DIRECTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]
DIRECTION_PROMPTS = {
    "LEFT": "Look/Turn LEFT",
    "RIGHT": "Look/Turn RIGHT",
    "UP": "Tilt chin UP + eyes UP",
    "DOWN": "Tilt chin DOWN + eyes DOWN",
}


@dataclass
class EngineResult:
    """Structured output of one process_frame() call."""
    status: str  # "no_face" | "calibrating_blink" | "calibrating_pose" |
                 # "calibrating_directions" | "calibration_restarted" | "ready"

    # calibrating_blink
    progress_current: Optional[int] = None
    progress_total: Optional[int] = None

    # calibrating_directions
    direction_prompt: Optional[str] = None
    direction_settling: Optional[bool] = None
    direction_sample_progress: Optional[int] = None
    direction_sample_total: Optional[int] = None

    # calibration_restarted
    restart_reason: Optional[str] = None

    # ready (normal operation)
    zone: Optional[str] = None
    zone_changed: Optional[bool] = None
    blink_count: Optional[int] = None
    blink_just_occurred: Optional[bool] = None
    blink_type: Optional[str] = None  # "short" | "long" | None (no blink this frame)
    blink_duration_frames: Optional[int] = None  # raw closed-frame count for the blink that just occurred, else None
    ear: Optional[float] = None
    ear_threshold: Optional[float] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    horizontal_score: Optional[float] = None
    vertical_score: Optional[float] = None
    raw_gaze: Optional[tuple] = None  # (gaze_x, gaze_y) for reference/future fusion use

    # calibration summary, populated once when directions_done first becomes True
    calibration_summary: Optional[dict] = None


def _zone_score(zone_name, horizontal_score, vertical_score,
                 horiz_pos_label, horiz_neg_label, vert_pos_label, vert_neg_label):
    if zone_name == horiz_pos_label:
        return abs(horizontal_score) if horizontal_score > 0 else 0.0
    if zone_name == horiz_neg_label:
        return abs(horizontal_score) if horizontal_score < 0 else 0.0
    if zone_name == vert_pos_label:
        return abs(vertical_score) if vertical_score > 0 else 0.0
    if zone_name == vert_neg_label:
        return abs(vertical_score) if vertical_score < 0 else 0.0
    return 0.0


def _sample_spread_1d(values):
    if not values:
        return 0.0
    return max(values) - min(values)


class GazeBlinkEngine:
    """
    Display-independent core engine: calibration + 5-zone gaze
    classification (yaw for horizontal, fused pitch+gaze_y for vertical)
    + blink detection, exactly as verified in Week 7, plus Week 9's
    blink duration classification (short/long).
    """

    def __init__(self, subject_label="self", enable_sqlite_logging=True):
        self.detector = FaceMeshDetector()
        self.pose_estimator = HeadPoseEstimator()
        self.calibration = CalibrationEngine(calibration_frames=45)
        self.blink_detector: Optional[BlinkClassifier] = None
        self.cal_logger = CalibrationLogger() if enable_sqlite_logging else None
        self.subject_label = subject_label

        self.last_zone = "CENTER"
        self.zone = "CENTER"
        self.yaw_override_active = False

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

        # --- Phase 1: blink baseline calibration ---
        if not self.calibration.is_calibrated:
            self.calibration.add_sample(ear, raw_gaze_x, raw_gaze_y, time.time())
            current, total = self.calibration.get_progress()
            return EngineResult(status="calibrating_blink",
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

        # --- Phase 2: directional auto-calibration ---
        if not self.directions_done:
            return self._process_direction_calibration(yaw, pitch, gaze_y_rel)

        # --- Phase 3: normal operation ---
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

        # Sample window full - validate and process
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
        return EngineResult(status="ready", zone=self.zone, zone_changed=False,
                             blink_count=0, blink_just_occurred=False, blink_type=None,
                             blink_duration_frames=None,
                             calibration_summary=summary)

    def _process_normal_operation(self, ear, yaw, pitch, gaze_y_rel, raw_gaze_x, raw_gaze_y):
        blink_just_occurred = self.blink_detector.update(ear)

        # TEMPORARY - measuring real blink durations. Remove after
        # picking a real value for LONG_BLINK_MIN_FRAMES.
        

        blink_type = None
        blink_duration_frames = None
        if blink_just_occurred:
            blink_duration_frames = self.blink_detector.last_blink_duration_frames
            if blink_duration_frames >= LONG_BLINK_MIN_FRAMES:
                blink_type = "long"
            else:
                blink_type = "short"

        self.horizontal_score = (yaw - self.center_yaw) / self.yaw_scale
        raw_yaw_deviation = yaw - self.center_yaw
        abs_yaw_dev = abs(raw_yaw_deviation)

        if not self.yaw_override_active and abs_yaw_dev > HORIZ_PRIORITY_ENTER:
            self.yaw_override_active = True
        elif self.yaw_override_active and abs_yaw_dev < HORIZ_PRIORITY_EXIT:
            self.yaw_override_active = False

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

        if self.yaw_override_active:
            candidate_zone = self.horiz_pos_label if raw_yaw_deviation > 0 else self.horiz_neg_label
            candidate_strength = abs(self.horizontal_score)
        elif abs(self.vertical_score) < CENTER_EXIT_THRESHOLD:
            candidate_zone = "CENTER"
            candidate_strength = 0.0
        else:
            candidate_zone = self.vert_pos_label if self.vertical_score > 0 else self.vert_neg_label
            candidate_strength = abs(self.vertical_score)

        if candidate_zone == self.zone:
            pass
        elif self.zone == "CENTER" or candidate_zone == "CENTER":
            if candidate_zone == "CENTER" and self.zone != "CENTER":
                if abs(self.vertical_score) < CENTER_ENTER_THRESHOLD:
                    self.zone = "CENTER"
            else:
                self.zone = candidate_zone
        else:
            current_strength = _zone_score(
                self.zone, self.horizontal_score, self.vertical_score,
                self.horiz_pos_label, self.horiz_neg_label,
                self.vert_pos_label, self.vert_neg_label
            )
            if candidate_strength > current_strength + SWITCH_MARGIN:
                self.zone = candidate_zone

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
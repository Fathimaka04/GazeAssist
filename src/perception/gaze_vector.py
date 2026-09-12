"""
src/perception/gaze_vector.py

Week 5 — Gaze vector estimation, combining raw iris position with head-pose
to reduce (not fully eliminate) head-motion-induced drift in the raw
gaze measurement.
"""

import numpy as np

YAW_CORRECTION_FACTOR = 0.02 

LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_EYE_LEFT_CORNER = 362
LEFT_EYE_RIGHT_CORNER = 263
RIGHT_EYE_LEFT_CORNER = 33
RIGHT_EYE_RIGHT_CORNER = 133

def _iris_offset_in_eye(landmarks, iris_idx, corner_left_idx, corner_right_idx, frame_w, frame_h):
    iris = landmarks[iris_idx]
    left_corner = landmarks[corner_left_idx]
    right_corner = landmarks[corner_right_idx]

    eye_width = (right_corner.x - left_corner.x) * frame_w
    if abs(eye_width) < 1e-6:
        return 0.0

    eye_center_x = (left_corner.x + right_corner.x) / 2 * frame_w
    iris_x = iris.x * frame_w

    return (iris_x - eye_center_x) / (eye_width / 2)


def estimate_raw_gaze_x(results, frame_shape):
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = frame_shape[0], frame_shape[1]

    left_offset = _iris_offset_in_eye(landmarks, LEFT_IRIS_CENTER, LEFT_EYE_LEFT_CORNER, LEFT_EYE_RIGHT_CORNER, w, h)
    right_offset = _iris_offset_in_eye(landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER, w, h)

    return (left_offset + right_offset) / 2.0


def estimate_normalized_gaze_x(raw_gaze_x, yaw, yaw_correction_factor=YAW_CORRECTION_FACTOR):
    """Subtracts a small, conservative fraction of yaw from the raw
    estimate. Sign convention: verify empirically (see calibration
    routine) rather than assumed, since eye-tracking sign conventions
    are easy to get backwards."""
    if raw_gaze_x is None:
        return None
    return raw_gaze_x + (yaw * yaw_correction_factor)


# --- Vertical (up/down) gaze estimation — Week 6 addition ---

LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145


def _iris_offset_vertical(landmarks, iris_idx, eye_top_idx, eye_bottom_idx, frame_w, frame_h):
    """Same idea as horizontal offset, but for up/down: where the iris
    sits between the top and bottom of the eye socket."""
    iris = landmarks[iris_idx]
    top = landmarks[eye_top_idx]
    bottom = landmarks[eye_bottom_idx]

    eye_height = (bottom.y - top.y) * frame_h
    if abs(eye_height) < 3.0:  # eye too closed/small to measure reliably
        return None

    eye_center_y = (top.y + bottom.y) / 2 * frame_h
    iris_y = iris.y * frame_h

    return (iris_y - eye_center_y) / (eye_height / 2)


def estimate_raw_gaze_y(results, frame_shape):
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = frame_shape[0], frame_shape[1]

    left_offset = _iris_offset_vertical(landmarks, LEFT_IRIS_CENTER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, w, h)
    right_offset = _iris_offset_vertical(landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, w, h)

    if left_offset is None or right_offset is None:
        return None

    return (left_offset + right_offset) / 2.0


PITCH_CORRECTION_FACTOR = 0.02  # placeholder - verify empirically like YAW_CORRECTION_FACTOR


def estimate_normalized_gaze_y(raw_gaze_y, pitch, pitch_correction_factor=PITCH_CORRECTION_FACTOR):
    """Same idea as estimate_normalized_gaze_x, but for vertical head
    motion (nodding). Sign convention: verify empirically - don't trust
    this blindly, the same way YAW_CORRECTION_FACTOR's sign was verified
    with a fixation test in Week 5."""
    if raw_gaze_y is None:
        return None
    return raw_gaze_y + (pitch * pitch_correction_factor)
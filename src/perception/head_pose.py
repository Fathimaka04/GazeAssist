"""
src/perception/head_pose.py

Head-pose estimation using OpenCV's solvePnP, with automatic per-session
calibration instead of a fixed hardcoded offset — this handles differences
in sitting position, camera angle, or a different person using the system.
"""

import cv2
import numpy as np

POSE_LANDMARK_INDICES = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 263,
    "right_eye_corner": 33,
    "left_mouth_corner": 291,
    "right_mouth_corner": 61,
    "nose_bridge": 168,
}

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
    (0.0, 140.0, -50.0),
], dtype="double")

MAX_PLAUSIBLE_YAW = 60
MAX_PLAUSIBLE_PITCH = 60
MAX_PLAUSIBLE_ROLL = 45
RAW_PITCH_SANITY = 45
RAW_ROLL_SANITY = 45
MAX_CONSECUTIVE_BAD_FRAMES = 10
CALIBRATION_YAW_SPREAD_LIMIT = 20


def is_pose_plausible(yaw, pitch, roll):
    return (abs(yaw) < MAX_PLAUSIBLE_YAW and
            abs(pitch) < MAX_PLAUSIBLE_PITCH and
            abs(roll) < MAX_PLAUSIBLE_ROLL)


class HeadPoseEstimator:
    def __init__(self, calibration_frames=30):
        self.smoothing_window = []
        self.smoothing_size = 5

        self.calibration_frames_needed = calibration_frames
        self.calibration_samples = []
        self.yaw_offset = None
        self.pitch_offset = None
        self.roll_offset = None
        self.is_calibrated = False

        self.prev_rvec = None
        self.prev_tvec = None
        self.consecutive_bad_frames = 0

        # Fallback initial guess ("facing the camera, ~50cm away") used
        # only when there's no previous good frame to seed from. Without
        # this, the very first solvePnP call can converge to the
        # ambiguous "flipped" solution near +/-180 degrees yaw instead of
        # near 0 - which then poisons calibration, since averaging plain
        # degrees near the +/-180 wrap-around point produces meaningless
        # offsets (e.g. averaging +179 and -179 gives ~0, not ~180).
        self.initial_rvec = np.zeros((3, 1), dtype="double")
        self.initial_tvec = np.array([[0.0], [0.0], [500.0]], dtype="double")

    def _get_camera_matrix(self, frame_shape):
        h, w = frame_shape[0], frame_shape[1]
        focal_length = w
        center = (w / 2, h / 2)
        return np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

    def estimate(self, results, frame_shape):
        """Returns (yaw, pitch, roll, is_calibrating) — the 4th value
        tells the caller whether calibration is still in progress."""
        if not results.multi_face_landmarks:
            return None

        h, w = frame_shape[0], frame_shape[1]
        landmarks = results.multi_face_landmarks[0].landmark

        image_points = np.array([
            (landmarks[POSE_LANDMARK_INDICES[k]].x * w,
             landmarks[POSE_LANDMARK_INDICES[k]].y * h)
            for k in ["nose_tip", "chin", "left_eye_corner",
                      "right_eye_corner", "left_mouth_corner",
                      "right_mouth_corner", "nose_bridge"]
        ], dtype="double")

        camera_matrix = self._get_camera_matrix(frame_shape)
        dist_coeffs = np.zeros((4, 1))

        seed_rvec = self.prev_rvec if self.prev_rvec is not None else self.initial_rvec
        seed_tvec = self.prev_tvec if self.prev_tvec is not None else self.initial_tvec

        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
            rvec=seed_rvec.copy(), tvec=seed_tvec.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        raw_yaw, raw_pitch, raw_roll = self._rotation_matrix_to_angles(rotation_matrix)

        # Two independent sanity checks BEFORE this frame is trusted as a
        # seed or calibration sample:
        # (1) the ambiguous "flipped" solution (yaw near +/-180)
        # (2) a genuinely garbage solve (extreme raw pitch/roll) - a bad
        #     first/early frame (partial face, motion blur) can otherwise
        #     become the seed, and useExtrinsicGuess then keeps biasing
        #     every later frame back toward that same wrong solution.
        is_flip = abs(raw_yaw) > 150
        is_garbage = abs(raw_pitch) > RAW_PITCH_SANITY or abs(raw_roll) > RAW_ROLL_SANITY

        if is_flip or is_garbage:
            self.consecutive_bad_frames += 1
            # If stuck near a bad solution for too long, drop the seed
            # entirely so the next call re-solves from the initial
            # "facing camera" guess instead of the wrong basin.
            if self.consecutive_bad_frames >= MAX_CONSECUTIVE_BAD_FRAMES:
                self.prev_rvec = None
                self.prev_tvec = None
                self.consecutive_bad_frames = 0
            return None

        self.consecutive_bad_frames = 0
        self.prev_rvec, self.prev_tvec = rotation_vector, translation_vector

        # --- CALIBRATION PHASE: just collect samples, don't filter yet ---
        if not self.is_calibrated:
            self.calibration_samples.append((raw_yaw, raw_pitch, raw_roll))
            if len(self.calibration_samples) >= self.calibration_frames_needed:
                yaws = [s[0] for s in self.calibration_samples]
                pitches = [s[1] for s in self.calibration_samples]
                rolls = [s[2] for s in self.calibration_samples]

                # If yaw drifted a lot across the window (real head motion
                # during calibration, not just noise), the plain average
                # is meaningless - throw the window out and start over
                # instead of locking in a bad offset.
                yaw_spread = max(yaws) - min(yaws)
                if yaw_spread > CALIBRATION_YAW_SPREAD_LIMIT:
                    self.calibration_samples = []
                    return (0.0, 0.0, 0.0, True)

                self.yaw_offset = sum(yaws) / len(yaws)
                self.pitch_offset = sum(pitches) / len(pitches)
                self.roll_offset = sum(rolls) / len(rolls)
                self.is_calibrated = True
            return (0.0, 0.0, 0.0, True)  # still calibrating

        # --- NORMAL PHASE: apply this session's own measured offset ---
        corrected_yaw = raw_yaw - self.yaw_offset
        corrected_pitch = raw_pitch - self.pitch_offset
        corrected_roll = raw_roll - self.roll_offset

        self.smoothing_window.append((corrected_yaw, corrected_pitch, corrected_roll))
        if len(self.smoothing_window) > self.smoothing_size:
            self.smoothing_window.pop(0)

        avg_yaw = sum(v[0] for v in self.smoothing_window) / len(self.smoothing_window)
        avg_pitch = sum(v[1] for v in self.smoothing_window) / len(self.smoothing_window)
        avg_roll = sum(v[2] for v in self.smoothing_window) / len(self.smoothing_window)

        if not is_pose_plausible(avg_yaw, avg_pitch, avg_roll):
            return None

        return (avg_yaw, avg_pitch, avg_roll, False)

    def _rotation_matrix_to_angles(self, R):
        """Converts a 3x3 rotation matrix to yaw/pitch/roll in degrees.
        Uses the standard Tait-Bryan angle extraction, with axes verified
        empirically: yaw = head turning left/right, pitch = nodding
        up/down, roll = tilting sideways."""
        sy = np.sqrt(R[0, 0] ** 2 + R[2, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(-R[1, 0], sy)
            yaw = np.arctan2(R[2, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:
            pitch = np.arctan2(-R[1, 0], sy)
            yaw = 0
            roll = np.arctan2(-R[1, 2], R[1, 1])

        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)
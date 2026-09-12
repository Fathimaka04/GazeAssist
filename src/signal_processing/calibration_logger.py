"""
src/signal_processing/calibration_logger.py

SQLite logging for directional calibration events (Week 7 to-do).
Each time verify_calibrated_system.py successfully completes
calibration, one row is logged capturing the measured reference
points and derived parameters - useful evidence for the "zero-learning
adapts per-user/per-session" claim in the report, and for comparing
personalized vs fixed-threshold accuracy later.

Usage:
    from src.signal_processing.calibration_logger import CalibrationLogger

    logger = CalibrationLogger()  # creates/opens data/calibration_log.db
    logger.log_calibration(..., subject_label="self")
    logger.close()
"""

import sqlite3
import os
from datetime import datetime

DEFAULT_DB_PATH = os.path.join("data", "calibration_log.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS calibration_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    subject_label TEXT,

    left_yaw REAL, left_pitch REAL, left_gaze_y REAL,
    right_yaw REAL, right_pitch REAL, right_gaze_y REAL,
    up_yaw REAL, up_pitch REAL, up_gaze_y REAL,
    down_yaw REAL, down_pitch REAL, down_gaze_y REAL,

    center_yaw REAL, yaw_scale REAL,
    pitch_center REAL, pitch_scale REAL, raw_pitch_sep REAL,
    gaze_y_center REAL, gaze_y_scale REAL,

    ear_threshold REAL
);
"""


class CalibrationLogger:
    def __init__(self, db_path=DEFAULT_DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def log_calibration(self, *, left_yaw, left_pitch, left_gaze_y,
                         right_yaw, right_pitch, right_gaze_y,
                         up_yaw, up_pitch, up_gaze_y,
                         down_yaw, down_pitch, down_gaze_y,
                         center_yaw, yaw_scale,
                         pitch_center, pitch_scale, raw_pitch_sep,
                         gaze_y_center, gaze_y_scale,
                         ear_threshold,
                         subject_label="self"):
        self.conn.execute(
            """
            INSERT INTO calibration_events (
                timestamp, subject_label,
                left_yaw, left_pitch, left_gaze_y,
                right_yaw, right_pitch, right_gaze_y,
                up_yaw, up_pitch, up_gaze_y,
                down_yaw, down_pitch, down_gaze_y,
                center_yaw, yaw_scale,
                pitch_center, pitch_scale, raw_pitch_sep,
                gaze_y_center, gaze_y_scale,
                ear_threshold
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"), subject_label,
                left_yaw, left_pitch, left_gaze_y,
                right_yaw, right_pitch, right_gaze_y,
                up_yaw, up_pitch, up_gaze_y,
                down_yaw, down_pitch, down_gaze_y,
                center_yaw, yaw_scale,
                pitch_center, pitch_scale, raw_pitch_sep,
                gaze_y_center, gaze_y_scale,
                ear_threshold,
            )
        )
        self.conn.commit()

    def fetch_all(self):
        """Returns all logged rows - useful for the personalized vs
        fixed-threshold comparison later."""
        cursor = self.conn.execute("SELECT * FROM calibration_events ORDER BY id")
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self):
        self.conn.close()
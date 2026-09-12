"""
src/signal_processing/ear_calculator.py

Computes Eye Aspect Ratio (EAR) from the 6-point eye landmarks produced by
face_mesh.py. EAR drops sharply when the eye closes and recovers when it
opens — this is the core signal blink detection is built on.

Formula (Soukupová & Čech, 2016):
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

Where p1..p6 follow the exact order defined in face_mesh.py's RIGHT_EYE /
LEFT_EYE index lists: p1/p4 are the horizontal corners, p2/p3 and p5/p6 are
the vertical top/bottom pairs.
"""

import math


def _distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def calculate_ear(eye_points):
    """eye_points: list of 6 (x, y) tuples in [p1, p2, p3, p4, p5, p6] order.
    Returns a single float EAR value for that eye."""
    if eye_points is None or len(eye_points) != 6:
        return None

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = _distance(p2, p6)
    vertical_2 = _distance(p3, p5)
    horizontal = _distance(p1, p4)

    if horizontal == 0:
        return None  # avoid divide-by-zero on a bad frame

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def average_ear(right_eye_points, left_eye_points):
    """Average both eyes' EAR into one value — more robust than relying on
    a single eye, since camera angle can make one eye's landmarks noisier
    than the other."""
    right_ear = calculate_ear(right_eye_points)
    left_ear = calculate_ear(left_eye_points)

    if right_ear is None or left_ear is None:
        return None

    return (right_ear + left_ear) / 2.0
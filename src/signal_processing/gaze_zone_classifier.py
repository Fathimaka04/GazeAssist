"""
src/signal_processing/gaze_zone_classifier.py

Week 6 — Converts gaze estimates into discrete zones: LEFT, RIGHT, UP,
DOWN, CENTER. Horizontal (x) determines LEFT/RIGHT/CENTER; vertical (y)
determines UP/DOWN — horizontal takes priority if both axes cross their
thresholds simultaneously, since horizontal was validated first and more
thoroughly.

Using discrete zones instead of precise coordinates is a deliberate design
choice: Week 5 showed the underlying gaze estimate has real noise even
after head-pose correction. A handful of well-separated zones tolerate
that noise far better than treating the raw number as an exact position —
this is what the Phrase Board (Week 11) will actually read from.
"""

# Thresholds — tuned empirically (see verify_gaze_zone.py test) rather
# than assumed.
LEFT_THRESHOLD = -0.3
RIGHT_THRESHOLD = 0.3
UP_THRESHOLD = -0.3
DOWN_THRESHOLD = 0.2

# Hysteresis: how many consecutive frames must agree on a new zone before
# switching — prevents single noisy frames from causing rapid flickering.
MIN_FRAMES_TO_SWITCH = 3


class GazeZoneClassifier:
    def __init__(self):
        self.current_zone = "CENTER"
        self.candidate_zone = "CENTER"
        self.candidate_count = 0

    def _raw_zone(self, gaze_x, gaze_y):
        if gaze_x is None or gaze_y is None:
            return None

        if gaze_x < LEFT_THRESHOLD:
            return "LEFT"
        elif gaze_x > RIGHT_THRESHOLD:
            return "RIGHT"
        elif gaze_y < UP_THRESHOLD:
            return "UP"
        elif gaze_y > DOWN_THRESHOLD:
            return "DOWN"
        else:
            return "CENTER"

    def classify(self, gaze_x, gaze_y):
        """Returns the current stable zone: 'LEFT', 'RIGHT', 'UP', 'DOWN',
        'CENTER', or None if no valid gaze estimate. Uses hysteresis so a
        zone change only takes effect after MIN_FRAMES_TO_SWITCH
        consecutive frames agree, filtering out single-frame noise."""
        raw_zone = self._raw_zone(gaze_x, gaze_y)
        if raw_zone is None:
            return None

        if raw_zone == self.current_zone:
            self.candidate_zone = raw_zone
            self.candidate_count = 0
            return self.current_zone

        if raw_zone == self.candidate_zone:
            self.candidate_count += 1
        else:
            self.candidate_zone = raw_zone
            self.candidate_count = 1

        if self.candidate_count >= MIN_FRAMES_TO_SWITCH:
            self.current_zone = raw_zone
            self.candidate_count = 0

        return self.current_zone
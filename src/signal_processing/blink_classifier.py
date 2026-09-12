"""
src/signal_processing/blink_classifier.py
"""

MIN_CLOSED_FRAMES = 2
MIN_OPEN_FRAMES = 4


class BlinkClassifier:
    """
    State-machine blink detector.
    ...(docstring unchanged)...

    New: after update() returns True (a blink just completed), read
    self.last_blink_duration_frames to get how many consecutive frames
    the eye was closed for THAT blink. Use this to classify short vs
    long blinks (e.g. in GazeBlinkEngine) without changing update()'s
    return signature.
    """

    def __init__(self, ear_threshold, min_closed_frames=MIN_CLOSED_FRAMES,
                 min_open_frames=MIN_OPEN_FRAMES):
        self.ear_threshold = ear_threshold
        self.min_closed_frames = min_closed_frames
        self.min_open_frames = min_open_frames

        self.state = "OPEN"
        self.open_counter = 0
        self.closed_counter = 0
        self.blink_count = 0

        # NEW: tracks the peak consecutive-closed-frame count reached
        # during the current closure, since closed_counter resets to 0
        # the instant the eye reopens (before we get a chance to read it).
        self._peak_closed_counter = 0

        # NEW: duration (in frames) of the most recently completed blink.
        # Only meaningful on/after a frame where update() returned True.
        self.last_blink_duration_frames = 0

    def update(self, ear):
        if ear is None:
            return False

        if ear < self.ear_threshold:
            self.closed_counter += 1
            self.open_counter = 0
            if self.closed_counter > self._peak_closed_counter:
                self._peak_closed_counter = self.closed_counter
        else:
            self.open_counter += 1
            self.closed_counter = 0

        blinked = False

        if self.state == "OPEN" and self.closed_counter >= self.min_closed_frames:
            self.state = "CLOSED"
        elif self.state == "CLOSED" and self.open_counter >= self.min_open_frames:
            self.state = "OPEN"
            self.blink_count += 1
            blinked = True
            self.last_blink_duration_frames = self._peak_closed_counter
            self._peak_closed_counter = 0

        return blinked

    def reset(self):
        self.state = "OPEN"
        self.open_counter = 0
        self.closed_counter = 0
        self.blink_count = 0
        self._peak_closed_counter = 0
        self.last_blink_duration_frames = 0
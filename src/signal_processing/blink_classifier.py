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

    FIX: duration is now the TOTAL elapsed frames spent in the CLOSED
    state for this blink gesture, not the longest unbroken closed-EAR
    streak. The old approach (_peak_closed_counter) reset on any single
    frame where EAR blipped back above threshold, which fragmented long
    deliberate holds into several short segments and reported only the
    longest fragment - badly undercounting real hold duration whenever
    EAR noise crossed the threshold mid-hold.
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

        # Total frames elapsed since entering CLOSED state for the
        # current blink gesture. Increments every frame while CLOSED,
        # regardless of brief EAR flickers above threshold (those don't
        # exit CLOSED unless they last min_open_frames in a row).
        self._closed_state_frame_count = 0

        self.last_blink_duration_frames = 0

    def update(self, ear):
        if ear is None:
            return False

        if ear < self.ear_threshold:
            self.closed_counter += 1
            self.open_counter = 0
        else:
            self.open_counter += 1
            self.closed_counter = 0

        blinked = False

        if self.state == "OPEN" and self.closed_counter >= self.min_closed_frames:
            self.state = "CLOSED"
            self._closed_state_frame_count = self.closed_counter
        elif self.state == "CLOSED":
            self._closed_state_frame_count += 1
            if self.open_counter >= self.min_open_frames:
                self.state = "OPEN"
                self.blink_count += 1
                blinked = True
                self.last_blink_duration_frames = self._closed_state_frame_count
                self._closed_state_frame_count = 0

        return blinked

    def reset(self):
        self.state = "OPEN"
        self.open_counter = 0
        self.closed_counter = 0
        self.blink_count = 0
        self._closed_state_frame_count = 0
        self.last_blink_duration_frames = 0
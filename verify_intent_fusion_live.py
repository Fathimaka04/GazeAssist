"""
verify_intent_fusion_live.py

Week 9 live verification: wires IntentFusionEngine on top of
GazeBlinkEngine, using a real camera feed, to test ZONE_SELECTED and
SOS_TRIGGERED behavior with real blinks - not fake timestamps.

Does NOT modify verify_calibrated_system.py or GazeBlinkEngine's own
calibration/display flow - this is a new, separate script, following
the same "engine does logic, script does display" pattern as
verify_calibrated_system.py.

UPDATED (SOS redesign): now passes result.blink_duration_frames into
fusion.update() so IntentFusionEngine can distinguish a normal
selection long-blink from an extra-long SOS hold. Previously this
was omitted, silently defaulting to 0 inside update() - meaning SOS
could never trigger no matter how long the hold was.

IMPORTANT: if EngineResult does not actually expose a field called
blink_duration_frames, this line will throw an AttributeError - check
gaze_blink_engine.py's EngineResult definition and correct the
attribute name below if it's called something else.

Run with:
    python verify_intent_fusion_live.py
(terminal only - not the VS Code Run button)

Press 'q' to quit.
"""


import cv2
import time

from src.sensing.camera_capture import CameraCapture
from src.core.gaze_blink_engine import GazeBlinkEngine
from src.core.intent_fusion_engine import IntentFusionEngine, IntentType


def main():
    cam = CameraCapture()
    engine = GazeBlinkEngine(subject_label="self")
    fusion = IntentFusionEngine()

    print("Starting... calibration will run first, same as verify_calibrated_system.py")

    try:
        for frame in cam.frames():
            result = engine.process_frame(frame)

            display_frame = frame.copy()

            if result.status == "calibrating_blink":
                cv2.putText(display_frame, f"Calibrating blink: {result.progress_current}/{result.progress_total}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            elif result.status == "calibrating_pose":
                cv2.putText(display_frame, "Calibrating pose...",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            elif result.status == "calibrating_directions":
                cv2.putText(display_frame, result.direction_prompt,
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                if not result.direction_settling:
                    cv2.putText(display_frame, f"Sample {result.direction_sample_progress}/{result.direction_sample_total}",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            elif result.status == "calibration_restarted":
                print(f"CALIBRATION RESTARTED: {result.restart_reason}")
                cv2.putText(display_frame, "RESTARTING - see terminal",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            elif result.status == "ready":
                if result.blink_just_occurred:
                    print(f"DEBUG: zone={result.zone}, blink_type={result.blink_type}, duration_frames={result.blink_duration_frames}")
                intent_event = fusion.update(result.zone, result.blink_type, result.blink_duration_frames)

                cv2.putText(display_frame, f"Zone: {result.zone}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if intent_event.type == IntentType.PENDING:
                    msg = f"PENDING: {intent_event.zone} (blink #{intent_event.long_blink_count})"
                    print(msg)
                    cv2.putText(display_frame, msg,
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                elif intent_event.type == IntentType.ZONE_SELECTED:
                    msg = f"ZONE_SELECTED: {intent_event.zone}"
                    print(msg)
                    cv2.putText(display_frame, msg,
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

                elif intent_event.type == IntentType.SOS_TRIGGERED:
                    msg = "SOS_TRIGGERED"
                    print(msg)
                    cv2.putText(display_frame, msg,
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            cv2.imshow("Intent Fusion Live Test", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        engine.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
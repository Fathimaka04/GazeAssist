"""
verify_intent_fusion_live.py
"""

import cv2
import time

from src.sensing.camera_capture import CameraCapture
from src.core.gaze_blink_engine import GazeBlinkEngine
from src.core.intent_fusion_engine import IntentFusionEngine, IntentType
from src.core.safety_state_machine import SafetyStateMachine, SafetyState


STATE_COLORS = {
    SafetyState.IDLE: (0, 255, 255),
    SafetyState.ACTIVE: (0, 255, 0),
    SafetyState.UNCERTAIN: (0, 0, 255),
    SafetyState.SOS_ARMED: (0, 128, 255),
    SafetyState.SOS_TRIGGERED: (0, 0, 255),
}


def main():
    cam = CameraCapture()
    engine = GazeBlinkEngine(subject_label="self")
    fusion = IntentFusionEngine()
    safety = SafetyStateMachine()

    last_intent_type = IntentType.NONE

    print("Starting... calibration will run first, same as verify_calibrated_system.py")

    try:
        for frame in cam.frames():
            result = engine.process_frame(frame)

            display_frame = frame.copy()

            intent_event = None

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

            elif result.status == "no_face":
                cv2.putText(display_frame, "No face detected",
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

                elif intent_event.type == IntentType.SOS_ARMED:
                    if last_intent_type != IntentType.SOS_ARMED:
                        print("SOS_ARMED - hold another long blink to confirm")
                    cv2.putText(display_frame, "SOS ARMED - confirm with another long blink",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

                elif intent_event.type == IntentType.SOS_TRIGGERED:
                    msg = "SOS_TRIGGERED"
                    print(msg)
                    cv2.putText(display_frame, msg,
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                elif intent_event.type == IntentType.NONE and last_intent_type == IntentType.SOS_ARMED:
                    print("SOS ARM EXPIRED - cancelled (no confirm in time)")

                last_intent_type = intent_event.type

            state_result = safety.update(
                result.status,
                intent_event.type if intent_event is not None else None
            )

            if state_result.state_changed:
                print(f"STATE CHANGED -> {state_result.state.value}")

            color = STATE_COLORS.get(state_result.state, (255, 255, 255))
            frame_height = display_frame.shape[0]

            cv2.putText(display_frame, f"STATE: {state_result.state.value.upper()}",
                        (20, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display_frame, state_result.message,
                        (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Intent Fusion Live Test", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        engine.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    
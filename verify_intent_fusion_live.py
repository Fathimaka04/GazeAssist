"""
verify_intent_fusion_live.py

Week 9 live verification: wires IntentFusionEngine on top of
GazeBlinkEngine, using a real camera feed, to test ZONE_SELECTED and
SOS_TRIGGERED behavior with real blinks - not fake timestamps.

Does NOT modify verify_calibrated_system.py or GazeBlinkEngine's own
calibration/display flow - this is a new, separate script, following
the same "engine does logic, script does display" pattern as
verify_calibrated_system.py.

UPDATED (SOS redesign): passes result.blink_duration_frames into
fusion.update() so IntentFusionEngine can distinguish a normal
selection long-blink from an extra-long SOS hold.

UPDATED (Week 10): wires SafetyStateMachine on top of the engine
status + intent event, so the overall app-level state
(IDLE/ACTIVE/UNCERTAIN/SOS_ARMED/SOS_TRIGGERED) is computed and
displayed every frame.

UPDATED (Week 10, SOS confirm step): SOS now requires an arm + confirm
sequence (IntentType.SOS_ARMED then SOS_TRIGGERED). Displays "SOS
ARMED" on screen and prints when an arm attempt expires unconfirmed.

UPDATED (selection simplified): IntentType.PENDING no longer exists -
selection now resolves directly to ZONE_SELECTED in a single frame,
so the old PENDING display branch has been removed.

UPDATED (Week 11, repeat-to-confirm): IntentFusionEngine now returns
AWAITING_CONFIRM on the first qualifying long blink in a zone, and
only fires ZONE_SELECTED on a second qualifying long blink in the SAME
zone within CONFIRM_WINDOW_SEC. Added the missing AWAITING_CONFIRM
display branch below - previously this event printed/showed nothing,
making a correctly-armed first blink look like it had been ignored.

UPDATED (Week 11, tile-based calibration): calibration no longer shows
"Look/Turn LEFT" style text prompts. gaze_blink_engine.py's
DIRECTION_PROMPTS now holds the actual confirmed Page 1 phrases
(Food/Water/Bathroom/Help), and this script draws all four as tile
boxes at their real on-screen positions (matching the phrase board
cross layout) during calibrating_directions, highlighting whichever
one is currently being calibrated. Tiles only appear once
calibrating_directions starts, i.e. strictly after face detection and
blink-baseline calibration are already done ("face first, then
tile"). Blink-baseline and pose calibration phases are unchanged and
show no tiles.

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
from src.core.gaze_blink_engine import GazeBlinkEngine, DIRECTION_PROMPTS
from src.core.intent_fusion_engine import IntentFusionEngine, IntentType
from src.core.safety_state_machine import SafetyStateMachine, SafetyState


STATE_COLORS = {
    SafetyState.IDLE: (0, 255, 255),
    SafetyState.ACTIVE: (0, 255, 0),
    SafetyState.UNCERTAIN: (0, 0, 255),
    SafetyState.SOS_ARMED: (0, 128, 255),
    SafetyState.SOS_TRIGGERED: (0, 0, 255),
}

# Tile size in pixels - same box drawn for every direction.
TILE_W, TILE_H = 160, 70


def _tile_center(direction, frame_width, frame_height):
    """Screen position for each direction's tile, matching the
    approved phrase-board cross layout (UP top-center, DOWN
    bottom-center, LEFT left-center, RIGHT right-center)."""
    margin = 90
    if direction == "UP":
        return frame_width // 2, margin
    if direction == "DOWN":
        return frame_width // 2, frame_height - margin
    if direction == "LEFT":
        return margin + TILE_W // 2, frame_height // 2
    if direction == "RIGHT":
        return frame_width - margin - TILE_W // 2, frame_height // 2
    return frame_width // 2, frame_height // 2


def draw_calibration_tiles(frame, active_direction):
    """Draws all four phrase tiles at their real positions, highlighting
    whichever direction is currently being calibrated."""
    h, w = frame.shape[0], frame.shape[1]
    for direction, label in DIRECTION_PROMPTS.items():
        cx, cy = _tile_center(direction, w, h)
        x1, y1 = cx - TILE_W // 2, cy - TILE_H // 2
        x2, y2 = cx + TILE_W // 2, cy + TILE_H // 2

        is_active = (direction == active_direction)
        box_color = (0, 255, 0) if is_active else (120, 120, 120)
        thickness = 3 if is_active else 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = cx - text_size[0] // 2
        text_y = cy + text_size[1] // 2
        cv2.putText(frame, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)


def main():
    cam = CameraCapture()
    engine = GazeBlinkEngine(subject_label="self")
    fusion = IntentFusionEngine()
    safety = SafetyStateMachine()

    last_intent_type = IntentType.NONE
    # Reverse lookup: tile label ("Water") -> direction key ("LEFT"),
    # used to know which tile to highlight from result.direction_prompt.
    label_to_direction = {v: k for k, v in DIRECTION_PROMPTS.items()}

    print("Starting... calibration will run first, same as verify_calibrated_system.py")

    try:
        for frame in cam.frames():
            result = engine.process_frame(frame)

            display_frame = frame.copy()

            intent_event = None

            if result.status == "calibrating_blink_open":
                cv2.putText(display_frame, "Keep eyes OPEN, look at camera naturally...",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, f"{result.progress_current}/{result.progress_total}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            elif result.status == "calibrating_blink_closed":
                cv2.putText(display_frame, "Now CLOSE your eyes and HOLD...",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
                cv2.putText(display_frame, f"{result.progress_current}/{result.progress_total}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

            elif result.status == "calibrating_pose":
                cv2.putText(display_frame, "Calibrating pose...",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            elif result.status == "calibrating_directions":
                active_direction = label_to_direction.get(result.direction_prompt)
                draw_calibration_tiles(display_frame, active_direction)

                instruction = "Get ready..." if result.direction_settling else \
                    f"Look at it! ({result.direction_sample_progress}/{result.direction_sample_total})"
                cv2.putText(display_frame, instruction,
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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

                if intent_event.type == IntentType.AWAITING_CONFIRM:
                    if last_intent_type != IntentType.AWAITING_CONFIRM:
                        print(f"AWAITING_CONFIRM: {intent_event.zone} - blink again in the same zone to confirm")
                    cv2.putText(display_frame, f"Confirm {intent_event.zone}? Blink again",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 3)

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

                elif intent_event.type == IntentType.NONE and last_intent_type == IntentType.AWAITING_CONFIRM:
                    print("CONFIRM WINDOW EXPIRED - selection cancelled (no repeat blink in time)")

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
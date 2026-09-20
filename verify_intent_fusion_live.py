"""
verify_intent_fusion_live.py

Week 9 live verification: wires IntentFusionEngine on top of
GazeBlinkEngine, using a real camera feed, to test ZONE_SELECTED and
SOS_TRIGGERED behavior with real blinks - not fake timestamps.

Does NOT modify verify_calibrated_system.py or GazeBlinkEngine's own
calibration/display flow.

UPDATED (SOS redesign, Week 10, Week 11 repeat-to-confirm): see prior
handoffs - IntentFusionEngine does AWAITING_CONFIRM -> ZONE_SELECTED
for phrase selection, and a separate SOS_ARMED -> SOS_TRIGGERED
sequence off two extra-long blinks, gated to no zone in particular.

UPDATED (Week 11, phrase board module): BOARDS holds a navigable
board structure. Each zone maps to a tile: label + action ("speak" or
"navigate"). "main" board: Food/Bathroom/Water/Help(->help_menu).
"help_menu" board: Call Nurse/Reposition/Pain/Back(->main).
IntentFusionEngine never sees any of this - it only reports which
physical zone was confirmed; this script interprets that zone
according to whichever board is currently active.

UPDATED (Week 11, diagonal quadrant zones): GazeBlinkEngine's zone
values changed from 5 zones (UP/DOWN/LEFT/RIGHT/CENTER, single-axis-
wins) to 4 diagonal quadrants (LEFT_UP/LEFT_DOWN/RIGHT_UP/RIGHT_DOWN,
independent per-axis decision, no CENTER). BOARDS keys and
_quadrant_rects() below were updated to match exactly - tile position
now literally matches its zone name (LEFT_UP really is top-left).

UPDATED (Week 11, calibration-to-quadrant mapping): directional
calibration (Phase 2 in GazeBlinkEngine) is UNCHANGED - it still
prompts pure LEFT/RIGHT/UP/DOWN axis extremes, one at a time, because
that's what's needed to correctly derive yaw_scale/pitch_scale. A
single axis prompt doesn't correspond to one quadrant tile though - it
corresponds to the TWO quadrants sharing that side (e.g. "LEFT"
highlights both LEFT_UP and LEFT_DOWN together). AXIS_TO_QUADRANTS
below does this mapping, and calibration now highlights tiles drawn
from the CURRENT BOARD's own labels (not a separate fixed prompt set),
so the tester calibrates against the exact tiles/labels they'll
actually operate with afterward.

UPDATED (Week 11, equal 2x2 tile grid, full-bleed camera, color fix):
fill_screen() crops the 640x480 camera feed to fill the window with
no bars/distortion. All 4 tiles are exactly equal quadrants, zero
gaps. Tile colors avoid blue/cyan (gray/amber/magenta/green/orange-
red/red only). CENTER tile removed - SOS was never zone-gated in
IntentFusionEngine, so removing it needed zero engine changes; SOS
arm/trigger now flashes all four tile borders together.

IMPORTANT: QUADRANT_AXIS_MARGIN in gaze_blink_engine.py is an UNTUNED
starting guess (0.15) - this needs a real controlled test (deliberately
look at each of the 4 corners, check for flicker or hard-to-reach
quadrants) before being trusted, same as LONG_BLINK_MIN_FRAMES did.

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
    SafetyState.IDLE: (0, 220, 220),
    SafetyState.ACTIVE: (90, 200, 90),
    SafetyState.UNCERTAIN: (0, 0, 220),
    SafetyState.SOS_ARMED: (0, 140, 255),
    SafetyState.SOS_TRIGGERED: (0, 0, 220),
}

DISPLAY_W, DISPLAY_H = 1280, 720

SELECTED_FLASH_SEC = 1.0
TRIGGERED_FLASH_SEC = 2.0

TILE_STYLES = {
    "idle":              ((140, 140, 140), (30, 30, 30)),
    "active":            ((0, 175, 255), (35, 45, 60)),
    "awaiting_confirm":  ((190, 60, 200), (45, 30, 55)),
    "selected":          ((80, 200, 80), (20, 50, 20)),
    "danger_armed":      ((0, 90, 255), (35, 30, 60)),
    "danger_triggered":  ((0, 0, 220), (50, 15, 15)),
}

# --- Phrase board module ---
# Zone keys now match GazeBlinkEngine's diagonal quadrant names
# exactly. Each tile: label shown + action ("speak" or "navigate").
BOARDS = {
    "main": {
        "LEFT_UP":    {"label": "Food",     "action": "speak"},
        "RIGHT_UP":   {"label": "Bathroom", "action": "speak"},
        "LEFT_DOWN":  {"label": "Water",    "action": "speak"},
        "RIGHT_DOWN": {"label": "Help",     "action": "navigate", "target": "help_menu"},
    },
    "help_menu": {
        "LEFT_UP":    {"label": "Call Nurse", "action": "speak"},
        "RIGHT_UP":   {"label": "Reposition", "action": "speak"},
        "LEFT_DOWN":  {"label": "Pain",       "action": "speak"},
        "RIGHT_DOWN": {"label": "Back",       "action": "navigate", "target": "main"},
    },
}

# During calibration, a single axis prompt (LEFT/RIGHT/UP/DOWN) maps
# to the TWO quadrants that share that side, not one tile.
AXIS_TO_QUADRANTS = {
    "LEFT":  ["LEFT_UP", "LEFT_DOWN"],
    "RIGHT": ["RIGHT_UP", "RIGHT_DOWN"],
    "UP":    ["LEFT_UP", "RIGHT_UP"],
    "DOWN":  ["LEFT_DOWN", "RIGHT_DOWN"],
}


def fill_screen(frame, canvas_w, canvas_h):
    """Scales the frame up and center-crops it so it fills canvas_w x
    canvas_h completely, with no bars - preserves aspect ratio at the
    cost of cropping some outer edge instead of stretching."""
    h, w = frame.shape[:2]
    scale = max(canvas_w / w, canvas_h / h)
    new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
    resized = cv2.resize(frame, (new_w, new_h))
    x0 = (new_w - canvas_w) // 2
    y0 = (new_h - canvas_h) // 2
    return resized[y0:y0 + canvas_h, x0:x0 + canvas_w]


def _quadrant_rects(w, h):
    """Equal 2x2 quadrants, zero gaps, covering the entire window.
    Tile position now literally matches its zone name."""
    half_w, half_h = w // 2, h // 2
    return {
        "LEFT_UP":    (0, 0, half_w, half_h),
        "RIGHT_UP":   (half_w, 0, w, half_h),
        "LEFT_DOWN":  (0, half_h, half_w, h),
        "RIGHT_DOWN": (half_w, half_h, w, h),
    }


def draw_tile_rect(frame, rect, label, state):
    x1, y1, x2, y2 = rect
    border, fill = TILE_STYLES.get(state, TILE_STYLES["idle"])

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    thickness = 2 if state == "idle" else 4
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, thickness, cv2.LINE_AA)

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
    cv2.putText(frame, label, (cx - text_size[0] // 2, cy + text_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, border, 3, cv2.LINE_AA)


def draw_tile_board(frame, zone_labels, zone_states):
    """zone_labels: {"LEFT_UP": "Food", ...} - text per zone.
    zone_states: {"LEFT_UP": "active", ...} - style per zone."""
    h, w = frame.shape[0], frame.shape[1]
    rects = _quadrant_rects(w, h)
    for zone, rect in rects.items():
        draw_tile_rect(frame, rect, zone_labels.get(zone, ""),
                        zone_states.get(zone, "idle"))


def draw_status_line(frame, text, color=(255, 255, 255)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 45), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                color, 2, cv2.LINE_AA)


def main():
    cam = CameraCapture()
    engine = GazeBlinkEngine(subject_label="self")
    fusion = IntentFusionEngine()
    safety = SafetyStateMachine()

    last_intent_type = IntentType.NONE
    label_to_direction = {v: k for k, v in DIRECTION_PROMPTS.items()}

    current_board = "main"
    last_selected_zone = None
    last_selected_until = 0.0
    last_triggered_until = 0.0

    window_name = "GazeAssist"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H)

    print("Starting... calibration will run first, same as verify_calibrated_system.py")

    try:
        for frame in cam.frames():
            result = engine.process_frame(frame)

            display_frame = fill_screen(frame, DISPLAY_W, DISPLAY_H)

            intent_event = None
            now = time.time()

            if result.status == "calibrating_blink_open":
                draw_status_line(display_frame,
                    f"Keep eyes open, look at the camera... ({result.progress_current}/{result.progress_total})")

            elif result.status == "calibrating_blink_closed":
                draw_status_line(display_frame,
                    f"Now close your eyes and hold... ({result.progress_current}/{result.progress_total})",
                    color=(0, 170, 255))

            elif result.status == "calibrating_pose":
                draw_status_line(display_frame, "Calibrating pose - hold still")

            elif result.status == "calibrating_directions":
                active_direction = label_to_direction.get(result.direction_prompt)
                highlighted = AXIS_TO_QUADRANTS.get(active_direction, [])

                board = BOARDS[current_board]
                zone_labels = {z: t["label"] for z, t in board.items()}
                states = {z: ("active" if z in highlighted else "idle") for z in board}
                draw_tile_board(display_frame, zone_labels, states)

                instruction = f"Get ready... (look toward the {result.direction_prompt} side)" if result.direction_settling else \
                    f"Hold that side! ({result.direction_sample_progress}/{result.direction_sample_total})"
                draw_status_line(display_frame, instruction)

            elif result.status == "calibration_restarted":
                print(f"CALIBRATION RESTARTED: {result.restart_reason}")
                draw_status_line(display_frame, "Restarting calibration - see terminal", color=(0, 0, 220))

            elif result.status == "no_face":
                draw_status_line(display_frame, "No face detected", color=(0, 0, 220))

            elif result.status == "ready":
                if result.blink_just_occurred:
                    print(f"DEBUG: zone={result.zone}, blink_type={result.blink_type}, duration_frames={result.blink_duration_frames}")
                intent_event = fusion.update(result.zone, result.blink_type, result.blink_duration_frames)

                board = BOARDS[current_board]

                if intent_event.type == IntentType.AWAITING_CONFIRM:
                    if last_intent_type != IntentType.AWAITING_CONFIRM:
                        print(f"AWAITING_CONFIRM: {intent_event.zone} - blink again in the same zone to confirm")

                elif intent_event.type == IntentType.ZONE_SELECTED:
                    tile = board.get(intent_event.zone, {})
                    if tile.get("action") == "navigate":
                        print(f"NAVIGATE: {current_board} -> {tile['target']} (via {intent_event.zone}/{tile['label']})")
                        current_board = tile["target"]
                        board = BOARDS[current_board]
                    else:
                        print(f"ZONE_SELECTED: {intent_event.zone} -> speaking \"{tile.get('label', '?')}\"")
                    last_selected_zone = intent_event.zone
                    last_selected_until = now + SELECTED_FLASH_SEC

                elif intent_event.type == IntentType.SOS_ARMED:
                    if last_intent_type != IntentType.SOS_ARMED:
                        print("SOS_ARMED - hold another long blink to confirm")

                elif intent_event.type == IntentType.SOS_TRIGGERED:
                    print("SOS_TRIGGERED")
                    last_triggered_until = now + TRIGGERED_FLASH_SEC

                elif intent_event.type == IntentType.NONE and last_intent_type == IntentType.AWAITING_CONFIRM:
                    print("CONFIRM WINDOW EXPIRED - selection cancelled (no repeat blink in time)")

                elif intent_event.type == IntentType.NONE and last_intent_type == IntentType.SOS_ARMED:
                    print("SOS ARM EXPIRED - cancelled (no confirm in time)")

                last_intent_type = intent_event.type

                zone_labels = {z: t["label"] for z, t in board.items()}
                states = {}
                for zone in board:
                    if zone == last_selected_zone and now < last_selected_until:
                        states[zone] = "selected"
                    elif intent_event.type == IntentType.AWAITING_CONFIRM and intent_event.zone == zone:
                        states[zone] = "awaiting_confirm"
                    elif result.zone == zone:
                        states[zone] = "active"
                    else:
                        states[zone] = "idle"

                if now < last_triggered_until:
                    for z in states:
                        states[z] = "danger_triggered"
                elif intent_event.type == IntentType.SOS_ARMED:
                    for z in states:
                        if states[z] == "idle":
                            states[z] = "danger_armed"

                draw_tile_board(display_frame, zone_labels, states)

                state_result = safety.update(result.status, intent_event.type)
                if state_result.state_changed:
                    print(f"STATE CHANGED -> {state_result.state.value}")
                color = STATE_COLORS.get(state_result.state, (255, 255, 255))
                draw_status_line(display_frame,
                    f"{state_result.state.value.upper()} - {state_result.message}  |  SOS: two long blinks, any tile",
                    color=color)

            if result.status != "ready":
                state_result = safety.update(result.status, None)
                if state_result.state_changed:
                    print(f"STATE CHANGED -> {state_result.state.value}")

            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        engine.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
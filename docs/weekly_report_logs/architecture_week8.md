# GazeAssist — Architecture (Week 8 Integration Checkpoint)

**Project:** GazeAssist — A Zero-Learning Multimodal Eye-Blink Assistive
Communication System for Non-Verbal and Motor-Impaired Users
**Checkpoint:** Week 8, end of integration phase (Weeks 1–7 complete)

---

## 1. Overview

GazeAssist uses a standard laptop webcam to classify a user's attention
into one of five zones (LEFT, RIGHT, UP, DOWN, CENTER) and to detect
voluntary blinks, with no explicit training step required from the user
("zero-learning" — calibration is automatic and per-session).

Two design findings from Weeks 6–7 shaped the current architecture and
should be treated as established constraints, not open questions:

1. **Pure eye-only gaze is too weak/noisy for reliable vertical (UP/DOWN)
   classification** on a standard webcam. This was measured directly
   (Week 6–7 testing), not assumed.
2. **Head-pose pitch alone also has insufficient UP/DOWN separation** —
   a comfortable head nod produces only a small pitch range that sits
   close to resting noise. The current system instead **fuses head-pose
   pitch with gaze_y (iris vertical offset)**, weighted 70% pitch / 30%
   gaze_y, which is the actual working solution.

Horizontal (LEFT/RIGHT) classification uses **pure head-pose yaw**,
which has consistently shown strong, reliable separation on its own and
did not need fusion.

---

## 2. Module List

| Module | Path | Responsibility |
|---|---|---|
| `CameraCapture` | `src/sensing/camera_capture.py` | Webcam frame capture, mirrored (`cv2.flip`) for natural left/right orientation |
| `FaceMeshDetector` | `src/perception/face_mesh.py` | MediaPipe face mesh wrapper; extracts eye landmark points |
| `HeadPoseEstimator` | `src/perception/head_pose.py` | solvePnP-based yaw/pitch/roll estimation with per-session calibration, frame seeding (`useExtrinsicGuess`), flip/garbage-frame rejection, and a 5-frame smoothing window |
| `estimate_raw_gaze_x/y` | `src/perception/gaze_vector.py` | Raw iris-offset gaze vector (horizontal and vertical components) |
| `average_ear` | `src/signal_processing/ear_calculator.py` | Eye Aspect Ratio calculation from eye landmark points |
| `CalibrationEngine` | `src/signal_processing/calibration_engine.py` | Personalized EAR threshold + gaze baseline, collected during the initial "look straight ahead" phase |
| `BlinkClassifier` | `src/signal_processing/blink_classifier.py` | State-machine blink detection (debounced via `MIN_CLOSED_FRAMES`/`MIN_OPEN_FRAMES`) using the personalized EAR threshold |
| `CalibrationLogger` | `src/signal_processing/calibration_logger.py` | SQLite logging of each completed calibration session to `data/calibration_log.db` |
| `GazeBlinkEngine` | `src/core/gaze_blink_engine.py` | **Core integration point (new in Week 8).** Combines all of the above into a single frame-in / structured-result-out engine, independent of any display code |
| `verify_calibrated_system.py` | project root | Demo/verification script — camera loop + `cv2` display only; calls `GazeBlinkEngine.process_frame()` |

---

## 3. Data Flow

```
Webcam frame
     |
     v
CameraCapture (mirrored frame)
     |
     v
FaceMeshDetector.process() ---> eye landmark points ---> average_ear()  --> EAR value
     |
     |---> HeadPoseEstimator.estimate() ---> (yaw, pitch, roll, is_calibrating)
     |
     |---> estimate_raw_gaze_x/y() ---> raw iris offset (gaze_x, gaze_y)
     |
     v
GazeBlinkEngine.process_frame()
     |
     |-- Phase 1: CalibrationEngine collects EAR + gaze baseline (45 frames)
     |-- Phase 2: HeadPoseEstimator's own internal calibration (30 frames)
     |-- Phase 3: Directional auto-calibration (LOOK LEFT/RIGHT/UP/DOWN,
     |            30-frame hold each, with validation checks — see §4)
     |-- Phase 4 (normal operation):
     |     - horizontal_score = (yaw - center_yaw) / yaw_scale
     |     - vertical_score   = smoothed fused(pitch, gaze_y)   [70/30 weighted]
     |     - zone decision: yaw-priority override (hysteresis)
     |                       > CENTER boundary (hysteresis)
     |                       > fused vertical score
     |     - BlinkClassifier.update(ear) -> blink count
     |
     v
EngineResult (status, zone, blink_count, scores, calibration_summary, ...)
     |
     v
verify_calibrated_system.py — cv2 display + terminal logging
     |
     v
CalibrationLogger — one row per completed calibration -> data/calibration_log.db
```

---

## 4. Calibration Safeguards (all discovered empirically during Week 7 debugging)

These exist because each one was added after a **specific observed
failure**, not as precautionary defaults. Documenting the failure each
one fixes, since this is directly usable evidence for the report:

| Safeguard | Constant | Failure it fixes |
|---|---|---|
| Per-direction sample spread check | `DIRECTION_SAMPLE_SPREAD_LIMIT_HORIZ` (30°), `..._VERT` (12°) | A shaky/drifting hold during the 30-frame capture window produced a skewed mean that distorted both axes downstream |
| LEFT/RIGHT symmetry check | `HORIZ_SYMMETRY_TOLERANCE` (0.25) | An uneven turn (e.g. strong 40° LEFT but shallow 16° RIGHT) skewed `center_yaw` away from true center, causing near-center readings to misclassify as LEFT or RIGHT |
| Raw pitch separation check | `MIN_RAW_PITCH_SEPARATION` (1.5°) | The *fused-score* separation check could be fooled by its own math: a near-zero raw pitch difference collapses `pitch_scale` toward zero, which inflates the fused score artificially large enough to pass the old check — producing enormous, meaningless vertical scores in normal operation |
| CENTER entry/exit hysteresis (Schmitt trigger) | `CENTER_EXIT_THRESHOLD` (0.7), `CENTER_ENTER_THRESHOLD` (0.3) | A vertical score sitting near a single 0.5 threshold caused rapid CENTER↔UP/DOWN flicker from ordinary sensor noise, even while the head was nearly still |
| Yaw-priority hysteresis | `HORIZ_PRIORITY_ENTER` (12°), `HORIZ_PRIORITY_EXIT` (8°) | Yaw hovering near a single priority threshold caused the same class of flicker between "trust yaw outright" and "fall through to vertical score" |

---

## 5. Known Limitations (for report, not further debugging tonight)

- **Vertical axis requires more deliberate effort than horizontal.** A
  user must reproduce roughly the same head-tilt + eye-movement
  intensity during normal use as during calibration; a smaller nod may
  not cross the classification threshold. This is a genuine usability
  constraint for the target population (motor-impaired users) worth
  discussing in the report's limitations section.
- **Direct zone-to-zone transitions (e.g. LEFT → UP) are governed by a
  single qualification threshold** (`CENTER_EXIT_THRESHOLD`), not a
  fully independent hysteresis band per zone-pair. In practice this has
  performed acceptably in testing, but is a simplification versus a
  theoretically ideal per-transition hysteresis model.
- **Raw gaze (iris-only) is not used for the horizontal axis or as a
  standalone classifier** — it is only a fusion input for vertical
  scoring. This is a deliberate simplification based on measured
  cross-axis leakage in earlier raw-gaze-only attempts, not an
  oversight.
- **LEFT/RIGHT calibration is fairly wide-range tolerant** (accepts
  turns as small as ~16° up to ~50°+ if the symmetry check passes),
  which means `yaw_scale` — and therefore horizontal sensitivity —
  varies somewhat session to session. Not yet quantified how much this
  affects usability across sessions.

---

## 6. What Week 9 (Multimodal Fusion Engine) Should Build On

- Reuse `GazeBlinkEngine` as-is; do not duplicate its calibration or
  classification logic.
- The engine already returns `raw_gaze` (raw gaze_x, gaze_y) in its
  `EngineResult` even during normal operation, kept alive specifically
  for future fusion use beyond the current vertical-axis fusion.
- Consider whether blink state and zone state should be combined into a
  single fused "intent" signal (e.g. confidence-weighted agreement
  between zone and blink timing) as the actual "multimodal fusion"
  deliverable, since zone and blink are currently independent outputs.

## 7. What Week 10 (Safety State Machine) Should Build On

- `EngineResult.zone`, `.blink_count`, and `.blink_just_occurred` are
  the natural inputs to the `IDLE → ACTIVE → UNCERTAIN → SOS_ARMED →
  SOS_TRIGGERED` state machine.
- `GazeBlinkEngine` is stateful per instance — one instance should
  persist for the life of a session; do not recreate it per frame.
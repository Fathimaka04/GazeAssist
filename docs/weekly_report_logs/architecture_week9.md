## Week 9 — Multimodal Fusion Engine

### Overview
Combines the two independent outputs of GazeBlinkEngine (gaze zone +
blink) into a single "intent" signal via a new module,
`src/core/intent_fusion_engine.py` (`IntentFusionEngine`).

Two user intents are supported:
- **Zone selection**: user holds gaze on a stable non-CENTER zone and
  performs one deliberate long blink.
- **SOS**: user performs one deliberate *extra-long* blink hold,
  triggered regardless of current gaze zone, and overrides/cancels any
  pending selection.

### Design iteration 1: 3 discrete long blinks (abandoned)
The original design (per the initial project proposal) required 3
long blinks in quick succession to trigger SOS, mirroring common
assistive-tech conventions. This was implemented and passed 5/5
fake-timestamp unit tests (`verify_intent_fusion.py`).

Live testing surfaced two real implementation bugs, both fixed:
1. A separate `GRACE_WINDOW_SEC` (0.7s), shorter than
   `LONG_BLINK_SEQUENCE_WINDOW_SEC` (1.2s), meant a pending selection
   always auto-confirmed before a real 2nd/3rd blink could arrive —
   SOS could never trigger regardless of timing. Fixed by using one
   consistent window, measured from the *last* blink instead of the
   first.
2. Real measured blink-to-blink gaps during live SOS attempts landed
   at 1.0–1.3s, right at/above the original 1.0s window — widened to
   1.5s based on this measurement.

Even after both fixes, extended live testing (by the developer, who
has full motor control) showed the 3-blink sequence was very
difficult to execute reliably even with the window further widened
to 2.5s. Blinks either landed too close together (falsely resetting
the sequence) or too far apart (auto-confirming as a single
selection instead of continuing the SOS count). A secondary
`COOLDOWN_SEC` (1.5s) after any confirmed event also repeatedly ate
the first blink of a new attempt if it started too soon after a
prior selection, compounding the difficulty.

### Design decision: switch to single sustained-hold SOS
Given that the project's actual target users (ALS, locked-in
syndrome, stroke, non-verbal motor-impaired individuals) have
*less* precise motor control than the developer testing this, a
precisely-timed 3-blink sequence was judged a fundamental usability
mismatch, not a tuning problem. Per this project's own methodology
of measuring rather than assuming, the repeated real-world difficulty
was treated as evidence rather than user error.

**Redesign**: SOS is now triggered by a single blink held past a
second, higher duration threshold (`EXTRA_LONG_BLINK_MIN_FRAMES`),
distinct from the existing selection-blink threshold
(`LONG_BLINK_MIN_FRAMES=9`). This required exposing the raw blink
duration (`blink_duration_frames`), not just the short/long label,
on `GazeBlinkEngine.EngineResult` — a small, non-breaking addition
mirroring how `blink_type` itself was added earlier in Week 9.

This removes the entire class of timing-sensitive bugs from the
previous design (inter-blink gap timing, cooldown collisions,
premature pending-resolution) by making SOS a single continuous
motor action rather than a coordinated sequence.

### Measured data (live testing, initial threshold validation)
`EXTRA_LONG_BLINK_MIN_FRAMES` was set to an initial estimate of 45
frames (~1.5s at 30fps), pending real measurement — consistent with
this project's practice of never trusting a constant without
controlled data.

First live test round produced a clean, well-separated distribution:

| Blink purpose         | Measured durations (frames) |
|------------------------|------------------------------|
| Selection long-blink   | 16, 19, 22, 35, 39            |
| SOS hold               | 87, 98, 103                   |

The ~48-frame gap between the highest selection blink (39) and the
lowest SOS hold (87) gives comfortable margin around the 45-frame
threshold — no retuning was needed after this first measurement,
unlike most other constants in this project which required 2-3
rounds of adjustment.

### Open question for Week 10+
The original 3-blink SOS design's difficulty raises a broader
question, not yet resolved: whether *any* deliberate, timed gesture
(sequence or hold) is a realistic input for the most severely
motor-impaired end users, or whether SOS should eventually consider
alternative/redundant triggers (e.g. a longer hold combined with
zone, or an external switch input). Deferred as a Week 10+ design
conversation, consistent with the similar open question already
raised about calibration requiring large head tilts.
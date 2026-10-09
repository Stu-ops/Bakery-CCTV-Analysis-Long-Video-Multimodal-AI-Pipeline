# CHOICES.md — Engineering Decisions and Trade-offs

This document covers the architectural trade-offs behind the bakery CCTV
analysis pipeline (`NVR_ch13_main_20260902172508.mp4`: single fixed camera,
1280×720, ~15 FPS, 5 minutes).

## 1. Local CV First, VLM Only on Segments
**Decision**: Run YOLOv8n + tracking + zone logic locally over the whole
video; reserve any VLM/LLM for short event-anchored segments (seconds around
each entry/exit/dwell), which is currently a designed extension point, not a
wired-in call.
**Trade-off**:
Sending 4,500 frames (or even 1,500 sampled frames) to a VLM is
prohibitively expensive and slow, and per-frame VLM zone queries are noisy.
Local CV gives deterministic timestamps, bboxes, and confidences for the
full 5 minutes at zero token cost; the VLM then sees maybe a dozen short
clips instead of 5 minutes of video — ~100× less footage. We lose open-vocabulary
narration of uneventful stretches, which the assignment explicitly does not need.

## 2. 5 FPS Sampling
**Decision**: Analyse every 3rd frame of the ~15 FPS source via `FrameProcessor`.
**Trade-off**:
5 FPS (200 ms) captures 2–10 frames per door crossing while skipping 67% of
inference work. Faster motion (a run through the door) could fall between
samples, but bakery foot traffic is slow and the dwell/session filters
tolerate single missed detections.

## 3. YOLOv8n Over Larger Models
**Decision**: YOLOv8 Nano for detection (the runner loads `yolov8n.pt`),
keeping Medium as an offline high-precision option.
**Trade-off**:
Nano runs faster than real time on RTX 4060 and ~30–80 ms/frame on CPU,
but misses small/occluded figures at the back of the 720p frame that Medium
would catch. For entry/exit counting at the door this is the right cost/accuracy
point; Medium remains available for disputed segments.

## 4. Tracker: IoU Association + BotSort Default
**Decision**: Per-camera IoU/center-distance association in `detector.py`
(default tracker string `botsort`) rather than ByteTrack.
**Trade-off**:
ByteTrack needs the `lap` library, which frequently fails to compile on
Windows without MSVC Build Tools. The lightweight associator has no native
deps and is sufficient for one fixed camera, at the cost of more ID switches
during counter occlusions.

## 5. ReID: OSNet When Present, Histogram Fallback on Python 3.12
**Decision**: `CrossCameraTracker` imports `torchreid` lazily; without it
(the case on Python 3.12 — no wheels exist) it uses L2-normalized HSV
histogram embeddings with the same cosine-similarity matching and
`high`/`medium`/`none` confidence levels.
**Trade-off**:
Histograms are far weaker than OSNet across clothing/lighting changes, but
for a single bakery camera with short re-appearances they preserve session
continuity with zero extra dependencies. Unmatched tracks get a new ID
rather than a forced merge, so errors inflate visitor count instead of
fabricating identity links.

## 6. Zone Mapping: Dual Rule, Remap Required
**Decision**: Center-inside OR >50% bbox overlap via `shapely`.
**Trade-off**:
Robust to people leaning over the counter from outside the polygon, but each
overlap test costs geometry compute, and the shipped polygons in
`configs/store_layouts.json` describe the legacy 5-camera store — they must
be redrawn for the bakery's door/counter/seating areas or zone events will
be meaningless.

## 7. No Mock Data, No Forced Claims
**Decision**: All demo/mock inference paths removed; low-evidence outputs
(drop confidences, `reentry_confidence: "none"`, fresh IDs) are emitted as-is.
**Trade-off**:
Counts are conservative (possible over-counting of visitors) rather than
confident-sounding but unsupported. This directly satisfies the assignment's
demand to handle insufficient evidence explicitly.

## 8. Grace-Period Session Finalization
**Decision**: 10 s grace near entry/exit zones, 30 s in product zones before
finalizing a session.
**Trade-off**:
Prevents one visit from splitting when a person is briefly occluded by the
counter or another customer, at the cost of slightly delayed EXIT events and
short-term occupancy over-counting while the grace window is open.

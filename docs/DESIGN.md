# Architecture Design Document
## Bakery CCTV Analysis — Long-Video Multimodal AI Pipeline

---

## 1. System Overview

This system analyses one long bakery CCTV recording —
`NVR_ch13_main_20260902172508.mp4` (1280×720, ~15 FPS, 4,500 frames,
≈5 minutes) — and extracts customer movement as structured events.
It has two components:

### Detection Pipeline
Local computer vision over the full video; no VLM in the loop:
- **Frame sampling** to a constant 5 FPS (~1,500 of 4,500 frames analysed)
- **Person detection** with YOLOv8n (conf ≥ 0.45) + per-camera track association
- **Entry/exit tracking** via session state with grace-period finalization
- **Zone occupancy** with shapely polygon dual-rule logic (remap polygons to
  door/counter/seating for this bakery — see §2)
- **Dwell tracking** with state machines and oscillation filtering
- **Staff classification** with duration + zone-coverage heuristics
- **Re-identification** for re-entering visitors (OSNet when available,
  color-histogram fallback on Python 3.12)
- **Occupancy engine** for per-zone counts and time-spent estimates
- **Segment selection**: only frames with detections and only event-anchored
  segments are candidates for any downstream VLM review

### REST API
A FastAPI service that:
- Ingests detection events from the pipeline
- Stores events in SQLite (PostgreSQL-ready)
- Serves events, metrics (visitors, dwell, queue, funnel), and a live
  WebSocket dashboard

---

## 2. Pipeline Architecture

```
NVR mp4, 15 FPS (4500 frames)
    │
    ▼
┌─────────────────────────────────────┐
│ Frame Processor (5 FPS constant)    │
│ Analyse every 3rd frame             │
│ 200ms interval captures transitions │
│ Empty frames stop here (1 inference)│
└─────────────────────────────────────┘
    │ only frames with people continue
    ▼
┌─────────────────────────────────────┐
│ YOLOv8n + track association         │
│ Person detection (conf > 0.45)      │
│ Visitor validation (conf > 0.70)    │
│ Tiny crops (<20x40px) → no embedding│
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Zone Occupancy Detector             │
│ Rule: center inside OR >50% bbox    │
│ Shapely polygon geometry            │
│ NOTE: remap configs/store_layouts   │
│ .json to bakery door/counter zones  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Dwell Tracker (State Machine)       │
│ OUTSIDE → INSIDE → DWELLING → EXIT │
│ Oscillation filter (15 frames)      │
│ Dwell threshold: 30 seconds         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Staff Classifier                    │
│ 0.7 * duration + 0.3 * zones       │
│ Threshold: 0.6                      │
│ Auto-staff: > 20 minutes            │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Re-ID Matcher                       │
│ OSNet embedding if installed, else  │
│ color-histogram fallback            │
│ high / medium / none confidence;    │
│ unmatched tracks get a NEW id       │
│ (never a forced merge)              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Occupancy Engine                    │
│ Per-zone counts, time-spent, queue  │
│ depth = occupancy count             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Event Emitter → JSON + SQLite → API │
│ ENTRY, EXIT, ZONE_*, BILLING_*,     │
│ REENTRY, STAFF_CLASSIFIED           │
│ each with timestamp, observation,   │
│ evidence, confidence                │
└─────────────────────────────────────┘
```

Single-video mapping: copy the bakery file to `data/videos/CAM_1.mp4` and
run `python run_pipeline.py --video-dir data/videos`. Absent `CAM_2…CAM_5`
files only log errors; the run proceeds with the cameras present.

---

## 3. Key Design Decisions

### 3.1 Frame Rate: 5 FPS Constant
- Person crossings take 0.5–2 seconds; 5 FPS (200 ms) captures 2–10 frames each
- Analyses ~1,500 of 4,500 frames (67% of dense work skipped before it starts)
- Constant rate is predictable and matches the dwell/oscillation filters

### 3.2 Detection Gating Before Any Expensive Step
- Frames with no person detections terminate early
- Only tracked-person segments can emit events, so any VLM review set is
  bounded by event count, not video length

### 3.3 Zone Occupancy: Dual Rule
- Center point inside polygon → in zone (fast path)
- >50% bbox overlap → in zone (handles door/counter border cases)
- Uses Shapely for polygon intersection math

### 3.4 Dwell State Machine
- Prevents duplicate events and oscillation at boundaries
- 30-second threshold before emitting ZONE_DWELL
- 15-frame stability filter before confirming entry/exit

### 3.5 Staff Classification: Heuristic
- Duration is strongest signal (0.7 weight), zone coverage supports (0.3)
- Short bakery clips weaken the duration signal — expect staff/visitor
  confusion and read `staff_score`/`staff_explanation` in metadata

### 3.6 Re-ID Confidence Levels, No Forced Matches
- High: embedding similarity alone suffices
- Medium: embedding + same entry point + short time gap
- None: new visitor ID issued; insufficient evidence is reported, not hidden

### 3.7 Occupancy / Time Spent
- Occupancy = active sessions per zone/camera at each timestamp
- Dwell = zone-exit time − zone-entry time; visit length = exit − entry
- Queue depth = non-staff occupancy of the counter/billing polygon

---

## 4. API Design

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/events/ingest` | POST | Batch event ingestion |
| `/events/ingest/single` | POST | Single event ingestion |
| `/stores/{id}/metrics` | GET | Store metrics (visitors, dwell, queue) |
| `/stores/{id}/funnel` | GET | Conversion funnel |
| `/stores/{id}/events` | GET | Raw events with filtering |
| `/health` | GET | Health check |
| `/ws/live` | WS | Live event broadcast |

---

## 5. Data Model

### Events Schema
- `event_type`: ENTRY, EXIT, ZONE_ENTER, ZONE_EXIT, ZONE_DWELL, BILLING_QUEUE_JOIN, BILLING_QUEUE_EXIT, REENTRY, STAFF_CLASSIFIED
- `store_id`: premises identifier
- `visitor_id`: unique per session (`V_GL_xxxx` across re-entries when matched)
- `timestamp`: ISO-8601
- `track_id`, `zone`, `camera_id`: evidence pointers (which camera/track)
- `metadata`: observation + evidence detail — `confidence` (0–1),
  `dwell_seconds`, `queue_depth`, `is_staff`, `staff_score`,
  `staff_explanation`, `zones_visited`, `reentry_confidence`

### Database
- SQLite for single-premises deployment (`outputs/results/events.db`)
- Indexed on: store_id, visitor_id, timestamp, event_type
- PostgreSQL-ready for multi-site scale

---

## 6. Scale & Cost Considerations

- One 5-minute clip: ~1,500 YOLOv8n inferences, faster than real time on
  RTX 4060 CUDA, ~1–2 min on CPU — zero VLM spend
- VLM review (optional extension) touches only event-anchored segments of a
  few seconds each, not 4,500 frames — ~100× less footage in the worst case
- **First bottleneck** at multi-site scale: SQLite single-writer → migrate
  to PostgreSQL with a Redis ingest buffer
- **Second bottleneck**: query latency on large tables → composite indices,
  materialized metric views, time-based partitioning

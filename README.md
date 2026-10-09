# Bakery CCTV Analysis — Long-Video Multimodal AI Pipeline

Cost-efficient pipeline that analyses a single long bakery CCTV recording
(`NVR_ch13_main_20260902172508.mp4` — 1280×720, ~15 FPS, 4,500 frames,
~5 minutes, 37.5 MB) and produces structured customer-movement events:
entry/exit detection and tracking, timestamped entry/exit events, occupancy
and time-spent estimates, and relevant video segments — without sending the
entire video to a VLM/LLM.

Local computer vision (YOLOv8 + tracking + zone geometry) does the dense
work over the full video. Only short, event-anchored segments are ever
candidates for VLM review, which is how the pipeline avoids unnecessary
VLM/LLM calls. Every output carries `event_type`, `timestamp`,
`observation`, `evidence`, and `confidence`, and low-evidence cases are
reported as such instead of guessed.

## Input video

| Property | Value |
|---|---|
| File | `NVR_ch13_main_20260902172508.mp4` (repo root, git-ignored) |
| Download | [Download the video (SharePoint)](https://subko365-my.sharepoint.com/:v:/g/personal/nidhi_singh_subk_co_in/IQBGNzGepresSJgYFvUrN_3UAV_ZwIPhRzXbJ-ep-U_gJq8?e=iivAQH) — place it at the repo root |
| Scene | Single fixed bakery camera |
| Resolution | 1280×720 |
| Frame rate | ~15 FPS |
| Frames / duration | 4,500 frames ≈ 300 s (5 min) |
| Size | ~37.5 MB |

## Setup

Requires Python 3.12 (Windows + NVIDIA GPU tested on RTX 4060), with
`requirements.txt` already pinned for it (Torch CUDA 12.6 wheels,
`ultralytics`, `opencv`, `numpy`, `shapely`, `fastapi`, …).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` resolves torch/torchvision from the PyTorch CUDA index
(first line of the file), so the RTX 4060 is used when available.
`torchreid` is intentionally **not** a hard dependency (it has no Python
3.12 wheels); the tracker falls back to color-histogram embeddings when
OSNet is unavailable — see `docs/CHOICES.md`.

## Execution

The runner consumes `CAM_*.mp4` files from a video directory, so map the
single bakery recording to `CAM_1`:

```powershell
mkdir data\videos -Force
Copy-Item NVR_ch13_main_20260902172508.mp4 data\videos\CAM_1.mp4
```

Quick smoke test (~60 s of footage at 5 FPS processing):

```powershell
python run_pipeline.py --video-dir data/videos --max-frames 300
```

Full 5-minute video:

```powershell
python run_pipeline.py --video-dir data/videos
```

Missing `CAM_2…CAM_5` files only log `Missing video …` errors and the run
continues with the cameras present, so single-video mode works as-is.
(`--extract-videos` is a legacy flag for the old multi-camera ZIP flow and
is not needed here. `--live` streams events/frames to the dashboard API.)

Check the CV stack first if needed:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python train_model.py   # downloads/verifies yolov8 weights
```

Tests (no GPU/video needed — 94 passed, 14 skipped without the legacy POS CSV):

```powershell
python -m pytest tests/ -v
```

Optional live dashboard (Terminal 1: `python -m uvicorn app.main:app`,
browser: `http://127.0.0.1:8000/dashboard/index.html`, Terminal 2: add
`--live` to the pipeline command).

## Sample output

Each run writes timestamped events to `outputs/results/real_pos_events.json`
and the same events into SQLite at `outputs/results/events.db`
(`GET /stores/{id}/events`, `/metrics`, `/funnel` read from this DB).

Event schema (one row per event):

| Field | Meaning |
|---|---|
| `event_type` | `ENTRY`, `EXIT`, `ZONE_ENTER`, `ZONE_EXIT`, `ZONE_DWELL`, `BILLING_QUEUE_JOIN`, `BILLING_QUEUE_EXIT`, `REENTRY`, `STAFF_CLASSIFIED` |
| `timestamp` | ISO-8601 event time |
| `observation` | Human-readable description in `metadata` (zone, dwell seconds, queue depth, staff explanation) |
| `evidence` | `camera_id` + `track_id` + detection `confidence` + frame context in `metadata` |
| `confidence` | 0.0–1.0 detector/classifier confidence stored in `metadata.confidence` |

Format example (illustrative, field names exactly as emitted):

```json
{
  "event_type": "EXIT",
  "store_id": "STORE_BLR_002",
  "visitor_id": "V_GL_0007",
  "timestamp": "2026-09-02T17:27:10",
  "track_id": 3,
  "zone": "ENTRY",
  "camera_id": "CAM_1",
  "metadata": {
    "dwell_seconds": 94.5,
    "confidence": 0.82,
    "is_staff": false,
    "zones_visited": ["ENTRY", "COUNTER"]
  }
}
```

Low-evidence cases keep a low `confidence`, keep `reentry_confidence`
`"none"`, and create a new visitor ID rather than forcing a match — see
“Reliability” below.

## Architecture and approach

```
NVR mp4 (15 FPS, 4500 frames)
  │  FrameProcessor: sample every 3rd frame → 5 FPS (~1500 frames analysed)
  ▼
YOLOv8n person detection (conf ≥ 0.45) + per-camera IoU association tracker
  │  only frames with detections continue; empty frames cost one inference
  ▼
Zone mapping (shapely polygons, center-inside OR >50% overlap)
  ▼
Dwell state machine (OUTSIDE → INSIDE → DWELLING → EXIT, 15-frame filter)
  ▼
Staff heuristic (0.7 × duration + 0.3 × zone coverage) + cross-camera ReID
  ▼
EventEmitter → real_pos_events.json + SQLite → FastAPI (/metrics, /funnel, /ws/live)
```

- **Detect/track entry & exit:** `pipeline/detector.py` (YOLO + tracking) and
  `pipeline/session_manager.py` (session per visitor, grace-period
  finalization so brief occlusions don't split one visit into two).
- **Occupancy / time spent:** occupancy = active sessions per zone/camera;
  dwell = zone-entry → zone-exit deltas; visit length = entry → exit.
  See `pipeline/occupancy_engine.py`, `pipeline/dwell_tracker.py`.
- **Segment selection instead of full-video VLM:** sampling (5 FPS) plus
  detection-gating means only ~1/3 of frames reach inference and only
  segments containing tracked people produce events. Each event carries its
  timestamp/camera/track, so a VLM (optional, not wired in) can be pointed
  at seconds-long segments rather than the whole 5-minute video.
- **Insufficient evidence:** detections below 0.45 (0.7 for visitor
  validation) are dropped; person crops smaller than 20×40 px yield no
  embedding; re-entry matching returns `high`/`medium`/`none` and unmatched
  tracks get a fresh ID instead of a forced merge.

## Prompts used

No LLM/VLM prompts are part of the running pipeline (it is fully local CV).
The prompts documented in `docs/PROMPTS.md` cover the agentic build workflow
(dataset analysis, rebuild plan, audit). VLM review of flagged segments is a
designed extension point, not an implemented call — any segment-review
prompt used later should be logged alongside the segment timestamps.

## Accuracy / reliability and key limitations

- No ground-truth labels exist for this bakery video, so precision/recall
  are **not measured** — treat counts as estimates. The pytest suite (94
  tests) verifies pipeline logic (zones, sessions, correlation, API), not
  detection accuracy on this footage.
- Single fixed 720p view: counter occlusions, overlapping customers, and
  back-to-camera staff cause ID switches and split sessions.
- Python 3.12 runs the histogram ReID fallback (no OSNet wheels), which is
  weaker than deep embeddings across appearance changes.
- Staff heuristic was tuned for long-dwell retail staff; bakery staff behind
  the counter for short clips may classify as visitors and vice versa.
- Zone polygons in `configs/store_layouts.json` still describe the legacy
  5-camera store; remap them to this bakery's door/counter/seating areas for
  meaningful zone events (see `docs/DESIGN.md` §2).

## Cost / latency and VLM-call reduction

- Full video is 4,500 frames; 5 FPS sampling sends ~1,500 frames to YOLOv8n.
  On RTX 4060 CUDA this runs faster than real time; on CPU expect roughly
  30–80 ms/frame (≈1–2 min for the whole clip). No VLM tokens are spent.
- Detection-gating + event-anchored segments mean a VLM would review only
  the seconds around each entry/exit/dwell event (typically a handful of
  short clips) instead of 5 minutes of video — roughly two orders of
  magnitude less footage, and zero VLM cost when review is skipped.
- Local-first design also removes per-frame API latency and keeps the
  offline run (`--video-dir`, no `--live`) fully self-contained.
```

## Further reading

- `docs/DESIGN.md` — architecture, event schema, segment selection
- `docs/CHOICES.md` — engineering trade-offs (local CV vs VLM, sampling, ReID fallback)
- `docs/PROMPTS.md` — prompts used during the build

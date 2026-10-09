# PROMPTS.md — Prompts Used

The running pipeline makes **no LLM/VLM calls** — it is fully local CV
(YOLOv8n + tracking + zone geometry + heuristics), so there are no inference
prompts. The prompts below are the ones actually used to build and adapt the
work, plus the template reserved for the optional segment-review extension.

## 1. Build prompts (agentic workflow, prior rounds)

- *"Analyze all datasets inside the `datasets/` folder. Understand their
  schema, columns, labels, formats, missing values, and relationships.
  Compare the dataset structure with our current pipeline/codebase."*
  → Parsed the legacy POS CSV + store-layout workbook; identified mock
  `run_demo_pipeline` logic that never touched real video.
- *"Rebuild and clean the entire project using ONLY the datasets present
  inside the datasets/ folder. Remove all old/sample/temporary datasets…
  Rebuild the complete pipeline so the project works end-to-end…"*
  → Produced `extract_videos.py`, 5 FPS `FrameProcessor`, `SessionManager`
  re-entry logic, YOLOv8 + BotSort + Shapely stack (ByteTrack dropped over
  Windows `lap` build failures).
- *"Audit the project against the requirements… If anything is missing,
  implement/fix it before giving the final verdict."*
  → Docker packaging, `/metrics` endpoint, anomaly detection, `structlog`
  JSON logs, live WebSocket dashboard.

## 2. Adaptation prompts (this round — bakery single-video assignment)

- *"Install dependencies into venv compatible with my current Python version
  (torch etc.); change requirements.txt accordingly."*
  → Pinned `torch==2.8.0` / `torchvision==0.23.0` (CUDA 12.6 index),
  `ultralytics>=8.3.0`, `numpy>=1.26.0`, `pandas>=2.2.0`; removed `torchreid`
  hard dep with a lazy-import + histogram fallback in
  `pipeline/cross_cam_tracker.py`.
- *"Help me run the pipeline test."*
  → Full pytest run (94 passed); fixed stale `test_root` (app redirects `/`
  to the dashboard; test asserted old JSON body).
- *"Reword docs/README for the bakery long-video assignment; the new input
  video is in the codebase."*
  → Located `NVR_ch13_main_20260902172508.mp4` (1280×720, ~15 FPS, 4,500
  frames, ~5 min) and rewrote README + DESIGN + CHOICES + PROMPTS around
  entry/exit events, occupancy/dwell, segment selection, evidence/confidence
  outputs, and VLM cost reduction.

## 3. Reserved template — optional VLM segment review (not implemented)

If/when flagged segments are sent to a VLM, log the exact prompt with the
segment timestamps. Suggested template:

> *"You are reviewing a {duration_s}-second bakery CCTV segment
> ({start_ts} → {end_ts}, camera {camera_id}) flagged for event
> {event_type} involving track {track_id} (detector confidence {confidence}).
> Describe only what is visibly supported: entries, exits, people count,
> and counter interactions. For anything unclear, reply exactly
> INSUFFICIENT_EVIDENCE and state what is missing. Do not guess identities."*

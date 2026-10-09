"""
Purplle Retail Analytics — Configuration Settings
"""
import os
from pathlib import Path


# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = BASE_DIR / "datasets"
VIDEOS_DIR = DATA_DIR / "videos"


# ─── Store Identity ──────────────────────────────────────────────────────────
STORE_ID = "STORE_BLR_002"
STORE_NAME = "Brigade Road, Bangalore"

# Mapping from POS CSV store_id → pipeline store_id
# Real CSV uses ST1008 for Brigade_Bangalore
STORE_ID_MAPPING = {
    "ST1008": "STORE_BLR_002",
}

# Cameras (matching CCTV footage: 5 cameras)
CAMERAS = ["CAM_1", "CAM_2", "CAM_3", "CAM_4", "CAM_5"]

# POS data path
POS_CSV_FILENAME = "Brigade_Bangalore_10_April_26 (1)bc6219c.csv"
POS_CSV_PATH = DATASETS_DIR / POS_CSV_FILENAME

# CCTV footage ZIP
CCTV_ZIP_FILENAME = "CCTV Footage-20260529T160731Z-3-00144614ea (3).zip"
CCTV_ZIP_PATH = DATASETS_DIR / CCTV_ZIP_FILENAME


# ─── Frame Processing ───────────────────────────────────────────────────────
# CORRECTED: 5 FPS constant instead of 1 FPS + burst
SOURCE_FPS = 15           # Raw camera frame rate
TARGET_FPS = 5            # Processing frame rate (captures zone transitions)
FRAME_INTERVAL = SOURCE_FPS // TARGET_FPS  # = 3 (process every 3rd frame)


# ─── Detection ───────────────────────────────────────────────────────────────
YOLO_MODEL = "yolov8m.pt"                # Medium model: speed/accuracy balance
YOLO_CONFIDENCE_THRESHOLD = 0.45         # Person detection confidence
YOLO_PERSON_CLASS_ID = 0                 # COCO class 0 = person
TRACKER_TYPE = "bytetrack"               # Multi-object tracker


# ─── Zone Occupancy ─────────────────────────────────────────────────────────
ZONE_COVERAGE_THRESHOLD = 0.50           # >50% bbox overlap = in zone
ZONE_CENTER_FALLBACK = True              # Also check center point


# ─── Dwell Tracking ─────────────────────────────────────────────────────────
DWELL_THRESHOLD_SECONDS = 30             # 30s in zone → dwell event
OSCILLATION_STABILITY_FRAMES = 15        # >1s at 15 FPS before emitting


# ─── Staff Classification ───────────────────────────────────────────────────
# CORRECTED: 0.7*duration + 0.3*zones (not 0.4/0.4/0.2 composite)
STAFF_DURATION_WEIGHT = 0.70
STAFF_ZONE_WEIGHT = 0.30
STAFF_SCORE_THRESHOLD = 0.60
STAFF_DURATION_THRESHOLD_MIN = 15        # 15 minutes → full duration score
STAFF_ZONE_COUNT_THRESHOLD = 4           # 4 unique zones → full zone score
STAFF_AUTO_DURATION_MIN = 20             # 20+ min → auto-staff regardless


# ─── Re-ID ───────────────────────────────────────────────────────────────────
# CORRECTED: Simplified dual-factor (not 3-stage cascade)
REID_EMBEDDING_HIGH_THRESHOLD = 0.80     # High confidence re-entry
REID_EMBEDDING_MEDIUM_THRESHOLD = 0.70   # Medium (needs temporal support)
REID_TIME_WINDOW_SECONDS = 30 * 60       # 30 min max gap for re-entry
REID_MEDIUM_TIME_WINDOW_SECONDS = 5 * 60 # 5 min for medium confidence
REID_POSITION_PROXIMITY_PIXELS = 200     # Same entry point threshold


# ─── Queue Analysis ─────────────────────────────────────────────────────────
# CORRECTED: Simple occupancy count (not Y-axis clustering)
# No special config needed — just count people in billing zone


# ─── POS Correlation ────────────────────────────────────────────────────────
# CORRECTED: exit_time < txn_time <= exit_time + 5 min
POS_CORRELATION_WINDOW_SECONDS = 5 * 60  # 5-minute window after exit


# ─── Cross-Camera Dedup ─────────────────────────────────────────────────────
CROSS_CAMERA_TIME_DELTA_SECONDS = 10     # <10s gap = same person


# ─── API / Database ─────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'events.db'}")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))


# ─── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "json"  # Structured logging for production

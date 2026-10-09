"""
Object Detector — YOLOv8 + ByteTrack Wrapper

Wraps YOLOv8 for person detection and ByteTrack for multi-object tracking.
This module is the "eyes" of the pipeline — detecting and tracking people
across frames.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single person detection in a frame."""

    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0  # 0 = person in COCO

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        """Area of bounding box in pixels²."""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class TrackedPerson:
    """A tracked person across frames (from ByteTrack)."""

    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    frame_id: int
    is_staff: bool = False

    # Accumulated tracking data
    frames_seen: List[int] = field(default_factory=list)
    positions: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, bbox: Tuple[float, float, float, float], confidence: float, frame_id: int):
        """Update track with new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.frame_id = frame_id
        self.frames_seen.append(frame_id)
        self.positions.append(self.center)


class PersonDetector:
    """
    YOLOv8 + ByteTrack detector/tracker.

    In production, this wraps ultralytics. For testing without GPU,
    we provide a mock-compatible interface.
    """

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        confidence_threshold: float = 0.45,
        tracker: str = "botsort",
        device: str = "auto",
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.tracker = tracker
        self.device = device
        self._model = None
        # camera_id -> Dict[track_id, TrackedPerson]
        self._active_tracks: Dict[str, Dict[int, TrackedPerson]] = {}
        self._next_track_id: Dict[str, int] = {}
        self._last_frame_id: Dict[str, int] = {}

        logger.info(
            "PersonDetector: model=%s, conf=%.2f, tracker=%s",
            model_path,
            confidence_threshold,
            tracker,
        )

    def _load_model(self):
        """Lazy-load the YOLO model."""
        if self._model is None:
            try:
                from ultralytics import YOLO

                self._model = YOLO(self.model_path)
                logger.info("YOLO model loaded: %s", self.model_path)
            except ImportError:
                logger.warning(
                    "ultralytics not installed. Using mock detector for testing."
                )
                self._model = "mock"

    def detect_and_track(
        self, frame: np.ndarray, frame_id: int
    ) -> Dict[int, TrackedPerson]:
        """
        Run detection + tracking on a single frame (backward compatibility).
        """
        return self.detect_and_track_cam(frame, frame_id, "default")

    def detect_and_track_cam(
        self, frame: np.ndarray, frame_id: int, camera_id: str
    ) -> Dict[int, TrackedPerson]:
        """Run detection on a specific camera feed with isolated tracking state."""
        self._load_model()
        
        if camera_id not in self._active_tracks:
            self._active_tracks[camera_id] = {}
            self._next_track_id[camera_id] = 1
        self._last_frame_id[camera_id] = frame_id

        if self._model == "mock":
            return self._mock_detect(frame, frame_id)

        # Use YOLO for detection only and maintain a lightweight per-camera
        # tracker here. A single Ultralytics `persist=True` tracker can mix
        # state across sequential camera calls, which splits/merges people
        # incorrectly in this multi-camera pipeline.
        results = self._model.predict(
            frame,
            conf=self.confidence_threshold,
            classes=[0],  # Person class only
            verbose=False,
        )

        detections = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i])
                bbox_tuple = (bbox[0], bbox[1], bbox[2], bbox[3])
                detections.append((bbox_tuple, conf))

        return self._associate_detections(camera_id, detections, frame_id, frame.shape)

    def _associate_detections(
        self,
        camera_id: str,
        detections: List[Tuple[Tuple[float, float, float, float], float]],
        frame_id: int,
        frame_shape: tuple,
    ) -> Dict[int, TrackedPerson]:
        """Associate detections to existing per-camera tracks by IoU and distance."""
        active = self._active_tracks[camera_id]
        current_tracks: Dict[int, TrackedPerson] = {}
        used_tracks = set()

        frame_h, frame_w = frame_shape[:2]
        max_center_dist = max(frame_w, frame_h) * 0.12

        for bbox, conf in detections:
            best_track_id = None
            best_score = -1.0
            cx, cy = self._center(bbox)

            for track_id, person in active.items():
                if track_id in used_tracks:
                    continue

                frames_missing = frame_id - person.frame_id
                if frames_missing > 45:
                    continue

                iou = self._iou(bbox, person.bbox)
                px, py = person.center
                center_dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                distance_score = max(0.0, 1.0 - center_dist / max_center_dist)
                score = (0.70 * iou) + (0.30 * distance_score)

                if (iou >= 0.15 or center_dist <= max_center_dist) and score > best_score:
                    best_score = score
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = self._next_track_id[camera_id]
                self._next_track_id[camera_id] += 1
                x1, y1, x2, y2 = bbox
                active[best_track_id] = TrackedPerson(
                    track_id=best_track_id,
                    bbox=bbox,
                    confidence=conf,
                    frame_id=frame_id,
                    frames_seen=[frame_id],
                    positions=[((x1 + x2) / 2, (y1 + y2) / 2)],
                )
            else:
                active[best_track_id].update(bbox, conf, frame_id)

            used_tracks.add(best_track_id)
            current_tracks[best_track_id] = active[best_track_id]

        return current_tracks

    def _center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _iou(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _mock_detect(
        self, frame: np.ndarray, frame_id: int
    ) -> Dict[int, TrackedPerson]:
        """Mock detector for testing without YOLO weights."""
        return {}

    def get_lost_tracks(
        self, current_tracks: Dict[int, TrackedPerson], max_lost_frames: int = 30
    ) -> List[TrackedPerson]:
        """
        Find tracks that were active but are no longer detected.

        Args:
            current_tracks: Currently visible tracks.
            max_lost_frames: How many frames before a track is considered lost.

        Returns:
            List of TrackedPerson that have exited the scene.
        """
        return self.get_lost_tracks_cam(current_tracks, "default", max_lost_frames)

    def get_lost_tracks_cam(
        self, current_tracks: Dict[int, TrackedPerson], camera_id: str, max_lost_frames: int = 30
    ) -> List[TrackedPerson]:
        lost = []
        if camera_id not in self._active_tracks:
            return lost
            
        for track_id, person in list(self._active_tracks[camera_id].items()):
            if track_id not in current_tracks:
                if person.frames_seen:
                    last_frame = person.frames_seen[-1]
                    current_frame = (
                        max(t.frame_id for t in current_tracks.values())
                        if current_tracks
                        else self._last_frame_id.get(camera_id, person.frame_id)
                    )
                    if current_frame - last_frame >= max_lost_frames:
                        lost.append(person)
                        del self._active_tracks[camera_id][track_id]

        return lost

    @property
    def active_track_count(self) -> int:
        return sum(len(tracks) for tracks in self._active_tracks.values())

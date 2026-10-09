"""
Frame Processor — Constant 5 FPS Sampling

CORRECTION #1: Uses constant 5 FPS instead of ChatGPT's 1 FPS + 3 FPS burst.

Rationale:
- Zone transitions take 0.5–2 seconds
- At 1 FPS: only 0–2 frames captured during crossing → misses entry/exit
- At 5 FPS: 2–10 frames captured → reliable zone transition detection
- Still efficient: processes only 17% of raw frames (3,000 out of 18,000)
"""

from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FrameProcessor:
    """
    Manages frame sampling from video streams at a target FPS.

    Instead of ChatGPT's adaptive 1 FPS base + 3 FPS burst approach,
    we use a constant 5 FPS. This is simpler, more predictable, and
    captures zone transitions reliably.
    """

    def __init__(
        self,
        source_fps: int = 15,
        target_fps: int = 5,
        clip_start_time: Optional[datetime] = None,
    ):
        """
        Args:
            source_fps: Raw camera frame rate (typically 15 FPS).
            target_fps: Processing frame rate. Default 5 FPS.
                       - 5 FPS = 1 frame every 200ms
                       - Zone transition is 500–2000ms
                       - Captures 2–10 frames per transition ✓
            clip_start_time: Absolute timestamp of the first frame.
        """
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        if source_fps <= 0:
            raise ValueError(f"source_fps must be positive, got {source_fps}")
        if target_fps > source_fps:
            raise ValueError(
                f"target_fps ({target_fps}) cannot exceed source_fps ({source_fps})"
            )

        self.source_fps = source_fps
        self.target_fps = target_fps
        self.frame_interval = max(1, source_fps // target_fps)
        self.clip_start_time = clip_start_time or datetime(2026, 3, 3, 14, 0, 0)

        self._frames_processed = 0
        self._frames_total = 0

        logger.info(
            "FrameProcessor initialized: source=%d FPS, target=%d FPS, interval=%d",
            source_fps,
            target_fps,
            self.frame_interval,
        )

    def should_process_frame(self, frame_id: int) -> bool:
        """
        Determine if a frame should be processed based on constant-rate sampling.

        Simple rule: process every Nth frame where N = source_fps // target_fps.

        Args:
            frame_id: Zero-indexed frame number from video.

        Returns:
            True if this frame should be processed.
        """
        self._frames_total = max(self._frames_total, frame_id + 1)
        should = frame_id % self.frame_interval == 0
        if should:
            self._frames_processed += 1
        return should

    def get_timestamp_from_frame(self, frame_id: int) -> str:
        """
        Convert a frame index to an ISO-8601 timestamp.

        Args:
            frame_id: Zero-indexed frame number.

        Returns:
            ISO-8601 timestamp string (e.g., "2026-03-03T14:00:01.000Z").
        """
        seconds_from_start = frame_id / self.source_fps
        timestamp = self.clip_start_time + timedelta(seconds=seconds_from_start)
        return timestamp.isoformat(timespec="milliseconds") + "Z"

    def get_datetime_from_frame(self, frame_id: int) -> datetime:
        """
        Convert a frame index to a datetime object.

        Args:
            frame_id: Zero-indexed frame number.

        Returns:
            datetime object representing the frame's absolute time.
        """
        seconds_from_start = frame_id / self.source_fps
        return self.clip_start_time + timedelta(seconds=seconds_from_start)

    def get_effective_fps(self) -> float:
        """Return the actual processing FPS."""
        return self.source_fps / self.frame_interval

    @property
    def stats(self) -> dict:
        """Processing statistics."""
        return {
            "source_fps": self.source_fps,
            "target_fps": self.target_fps,
            "effective_fps": self.get_effective_fps(),
            "frame_interval": self.frame_interval,
            "frames_processed": self._frames_processed,
            "frames_total": self._frames_total,
            "processing_ratio": (
                self._frames_processed / self._frames_total
                if self._frames_total > 0
                else 0.0
            ),
        }

"""
Dwell Tracker — Zone Dwell State Machine

Implements the state machine for tracking how long a person stays in a zone.
This is one of the parts ChatGPT designed correctly — we keep the state
machine pattern but integrate it with the corrected pipeline.

State Machine:
  OUTSIDE → ZONE_ENTER → INSIDE → [30s elapsed] → ZONE_DWELL → ZONE_EXIT

Prevents:
  - Duplicate dwell events
  - Oscillation on zone boundary (stability filter)
  - Missing dwell during zone transitions
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DwellState(Enum):
    """Zone dwell states for the state machine."""

    OUTSIDE = "OUTSIDE"
    ENTERED = "ENTERED"         # Just entered, not yet stable
    INSIDE = "INSIDE"           # Stable in zone, dwell timer started
    DWELLING = "DWELLING"       # Dwell threshold exceeded, event emitted
    EXITING = "EXITING"         # Just left, checking if oscillation


class ZoneDwellRecord:
    """Tracks dwell state for one person in one zone."""

    def __init__(self, track_id: int, zone_name: str):
        self.track_id = track_id
        self.zone_name = zone_name
        self.state = DwellState.OUTSIDE

        # Timestamps
        self.enter_time: Optional[datetime] = None
        self.dwell_start_time: Optional[datetime] = None
        self.exit_time: Optional[datetime] = None

        # Oscillation filter
        self.consecutive_in_frames = 0
        self.consecutive_out_frames = 0
        self.stability_threshold = 15  # >1 second at 15 FPS

        # Accumulated dwell time
        self.total_dwell_seconds: float = 0.0

    @property
    def is_dwelling(self) -> bool:
        return self.state == DwellState.DWELLING

    @property
    def current_dwell_seconds(self) -> float:
        if self.enter_time and self.state in (DwellState.INSIDE, DwellState.DWELLING):
            return (datetime.utcnow() - self.enter_time).total_seconds()
        return self.total_dwell_seconds


class DwellTracker:
    """
    Tracks dwell time for all persons across all zones.

    Uses a per-person, per-zone state machine with oscillation filtering
    to prevent spurious enter/exit events from bbox jitter.

    Good parts from ChatGPT's design (kept as-is):
    - State machine for dwell emission
    - Oscillation filter for entry/exit stability
    """

    def __init__(
        self,
        dwell_threshold_seconds: float = 30.0,
        stability_frames: int = 15,
        source_fps: int = 15,
    ):
        """
        Args:
            dwell_threshold_seconds: Time in zone before emitting ZONE_DWELL event.
            stability_frames: Number of consecutive frames before confirming
                              entry/exit (oscillation filter).
            source_fps: Raw FPS for time calculations.
        """
        self.dwell_threshold = dwell_threshold_seconds
        self.stability_frames = stability_frames
        self.source_fps = source_fps

        # track_id -> zone_name -> ZoneDwellRecord
        self._records: Dict[int, Dict[str, ZoneDwellRecord]] = {}

    def _get_record(self, track_id: int, zone_name: str) -> ZoneDwellRecord:
        """Get or create a dwell record."""
        if track_id not in self._records:
            self._records[track_id] = {}
        if zone_name not in self._records[track_id]:
            self._records[track_id][zone_name] = ZoneDwellRecord(track_id, zone_name)
            self._records[track_id][zone_name].stability_threshold = self.stability_frames
        return self._records[track_id][zone_name]

    def update(
        self,
        track_id: int,
        zone_name: str,
        is_in_zone: bool,
        timestamp: datetime,
    ) -> List[dict]:
        """
        Update dwell state for a person in a zone.

        Implements the state machine:
          OUTSIDE ──[in_zone stable]──→ ENTERED ──[confirmed]──→ INSIDE
            ↑                                                      │
            │                                    [30s elapsed] ────┘
            │                                                      ↓
          EXITING ←──[out_zone stable]── INSIDE ←── DWELLING

        Args:
            track_id: Person track ID.
            zone_name: Zone being tracked.
            is_in_zone: Whether person is currently in this zone.
            timestamp: Current frame timestamp.

        Returns:
            List of events to emit (may be empty).
        """
        record = self._get_record(track_id, zone_name)
        events = []

        if is_in_zone:
            record.consecutive_in_frames += 1
            record.consecutive_out_frames = 0

            if record.state == DwellState.OUTSIDE:
                # Start tracking potential entry
                if record.consecutive_in_frames >= record.stability_threshold:
                    record.state = DwellState.INSIDE
                    record.enter_time = timestamp
                    events.append({
                        "type": "ZONE_ENTER",
                        "track_id": track_id,
                        "zone": zone_name,
                        "timestamp": timestamp.isoformat() + "Z",
                    })
                    logger.debug(
                        "Track %d ENTERED zone %s at %s",
                        track_id, zone_name, timestamp,
                    )

            elif record.state == DwellState.EXITING:
                # Was leaving but came back — oscillation, stay INSIDE
                record.state = DwellState.INSIDE if not record.is_dwelling else DwellState.DWELLING
                record.consecutive_out_frames = 0

            elif record.state in (DwellState.INSIDE, DwellState.DWELLING):
                # Check if dwell threshold exceeded
                if record.enter_time and record.state == DwellState.INSIDE:
                    elapsed = (timestamp - record.enter_time).total_seconds()
                    if elapsed >= self.dwell_threshold:
                        record.state = DwellState.DWELLING
                        record.dwell_start_time = timestamp
                        events.append({
                            "type": "ZONE_DWELL",
                            "track_id": track_id,
                            "zone": zone_name,
                            "timestamp": timestamp.isoformat() + "Z",
                            "dwell_seconds": elapsed,
                        })
                        logger.debug(
                            "Track %d DWELLING in zone %s (%.1fs)",
                            track_id, zone_name, elapsed,
                        )
        else:
            record.consecutive_out_frames += 1
            record.consecutive_in_frames = 0

            if record.state in (DwellState.INSIDE, DwellState.DWELLING):
                if record.consecutive_out_frames >= record.stability_threshold:
                    # Confirmed exit
                    record.exit_time = timestamp
                    if record.enter_time:
                        record.total_dwell_seconds = (
                            timestamp - record.enter_time
                        ).total_seconds()

                    events.append({
                        "type": "ZONE_EXIT",
                        "track_id": track_id,
                        "zone": zone_name,
                        "timestamp": timestamp.isoformat() + "Z",
                        "dwell_seconds": record.total_dwell_seconds,
                    })
                    logger.debug(
                        "Track %d EXITED zone %s (total %.1fs)",
                        track_id, zone_name, record.total_dwell_seconds,
                    )

                    # Reset for potential re-entry
                    record.state = DwellState.OUTSIDE
                    record.consecutive_in_frames = 0
                    record.consecutive_out_frames = 0
                elif record.consecutive_out_frames > 0:
                    # Start tracking potential exit
                    record.state = DwellState.EXITING

        return events

    def get_dwell_time(self, track_id: int, zone_name: str) -> float:
        """Get accumulated dwell time for a person in a zone."""
        if track_id in self._records and zone_name in self._records[track_id]:
            return self._records[track_id][zone_name].total_dwell_seconds
        return 0.0

    def get_all_zone_times(self, track_id: int) -> Dict[str, float]:
        """Get dwell times for all zones a person has visited."""
        if track_id not in self._records:
            return {}
        return {
            zone: record.total_dwell_seconds
            for zone, record in self._records[track_id].items()
            if record.total_dwell_seconds > 0
        }

    def get_zones_visited(self, track_id: int) -> List[str]:
        """Get list of zones a person has visited."""
        if track_id not in self._records:
            return []
        return [
            zone
            for zone, record in self._records[track_id].items()
            if record.enter_time is not None
        ]

    def finalize_track(self, track_id: int, timestamp: datetime) -> List[dict]:
        """
        Finalize all zones for a track that is leaving the scene.
        Emits ZONE_EXIT for any zones the person is still in.
        """
        events = []
        if track_id in self._records:
            for zone_name, record in self._records[track_id].items():
                if record.state in (DwellState.INSIDE, DwellState.DWELLING, DwellState.EXITING):
                    record.exit_time = timestamp
                    if record.enter_time:
                        record.total_dwell_seconds = (
                            timestamp - record.enter_time
                        ).total_seconds()
                    events.append({
                        "type": "ZONE_EXIT",
                        "track_id": track_id,
                        "zone": zone_name,
                        "timestamp": timestamp.isoformat() + "Z",
                        "dwell_seconds": record.total_dwell_seconds,
                    })
                    record.state = DwellState.OUTSIDE
        return events

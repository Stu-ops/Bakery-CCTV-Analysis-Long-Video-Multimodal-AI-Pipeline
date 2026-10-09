"""
Session Manager — Visitor Session Lifecycle

Manages the full lifecycle of a visitor session:
  ENTRY → zone events → billing → purchase → EXIT

Each session tracks:
- Entry/exit timestamps
- Zones visited and dwell times
- Staff classification result
- Billing zone interaction
- Re-entry detection
- Cross-camera deduplication

This is one of the parts ChatGPT designed correctly (session lifecycle).
We integrate it with the corrected pipeline components.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
import uuid
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Visitor session lifecycle states."""

    ACTIVE = "ACTIVE"
    EXITED = "EXITED"
    MERGED = "MERGED"  # Merged into another session (re-entry)


@dataclass
class VisitorSession:
    """
    Represents a single visitor's journey through the store.

    Lifecycle: ENTRY → zone visits → billing → exit → (POS correlation)
    """

    visitor_id: str
    track_id: int
    store_id: str

    # State
    state: SessionState = SessionState.ACTIVE

    # Timestamps
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None

    # Zone tracking
    zones_visited: Set[str] = field(default_factory=set)
    zone_dwell_times: Dict[str, float] = field(default_factory=dict)
    current_zone: Optional[str] = None

    # Billing
    billing_zone_enter_time: Optional[datetime] = None
    billing_zone_exit_time: Optional[datetime] = None

    # Staff classification (CORRECTED: 0.7/0.3 heuristic)
    is_staff: bool = False
    staff_score: float = 0.0
    staff_explanation: str = ""

    # Re-ID
    is_reentry: bool = False
    previous_session_id: Optional[str] = None
    reentry_confidence: str = "none"

    # Conversion (CORRECTED: exit_time ± 5 min)
    converted: bool = False
    basket_value: float = 0.0

    # Camera
    camera_id: Optional[str] = None
    last_seen_camera: Optional[str] = None  # Which camera last observed this person
    
    # Appearance (for Staff Uniform Detection)
    embedding: Optional[np.ndarray] = None

    # Session longevity
    last_seen: Optional[datetime] = None

    # Zone entry timestamps
    zone_entry_times: Dict[str, datetime] = field(default_factory=dict)

    # Role change audit trail
    role_change_history: List[dict] = field(default_factory=list)

    # Events emitted
    events: List[dict] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Total session duration in seconds."""
        if self.entry_time is None:
            return 0.0
        end = self.exit_time or self.last_seen or self.entry_time
        return (end - self.entry_time).total_seconds()

    @property
    def zone_count(self) -> int:
        """Number of unique zones visited."""
        return len(self.zones_visited)

    @property
    def visited_billing(self) -> bool:
        """Whether visitor entered the billing zone."""
        return self.billing_zone_enter_time is not None

    def record_role_change(self, old_role: str, new_role: str, timestamp: datetime, explanation: str = ""):
        """Record a role transition in the audit trail."""
        self.role_change_history.append({
            "old_role": old_role,
            "new_role": new_role,
            "timestamp": timestamp.isoformat() + "Z" if isinstance(timestamp, datetime) else timestamp,
            "explanation": explanation,
            "visitor_id": self.visitor_id,
        })

    def to_dict(self) -> dict:
        """Serialize session to dict for API/storage."""
        return {
            "visitor_id": self.visitor_id,
            "track_id": self.track_id,
            "store_id": self.store_id,
            "state": self.state.value,
            "entry_time": self.entry_time.isoformat() + "Z" if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() + "Z" if self.exit_time else None,
            "duration_seconds": round(self.duration_seconds, 1),
            "zones_visited": sorted(self.zones_visited),
            "zone_dwell_times": {
                k: round(v, 1) for k, v in self.zone_dwell_times.items()
            },
            "current_zone": self.current_zone,
            "is_staff": self.is_staff,
            "staff_score": round(self.staff_score, 3),
            "is_reentry": self.is_reentry,
            "converted": self.converted,
            "basket_value": self.basket_value,
            "billing_zone_enter_time": (
                self.billing_zone_enter_time.isoformat() + "Z"
                if self.billing_zone_enter_time
                else None
            ),
            "billing_zone_exit_time": (
                self.billing_zone_exit_time.isoformat() + "Z"
                if self.billing_zone_exit_time
                else None
            ),
            "camera_id": self.camera_id,
            "event_count": len(self.events),
        }


class SessionManager:
    """
    Manages all visitor sessions for a store.

    Handles:
    - Session creation on entry
    - Zone visit tracking
    - Session finalization on exit
    - Cross-camera deduplication
    - Re-entry session merging
    """

    def __init__(
        self,
        store_id: str,
        cross_camera_time_delta: float = 10.0,
    ):
        """
        Args:
            store_id: Store identifier.
            cross_camera_time_delta: Max seconds between detections
                to consider same person across cameras.
        """
        self.store_id = store_id
        self.cross_camera_time_delta = cross_camera_time_delta

        # Active sessions (track_id → VisitorSession)
        self._active_sessions: Dict[int, VisitorSession] = {}

        # Finalized sessions (visitor_id → VisitorSession)
        self._finalized_sessions: Dict[str, VisitorSession] = {}

        # Track ID to visitor ID mapping
        self._track_to_visitor: Dict[int, str] = {}

        logger.info("SessionManager initialized for store %s", store_id)

    def create_session(
        self,
        track_id: int,
        entry_time: datetime,
        camera_id: Optional[str] = None,
    ) -> VisitorSession:
        """
        Create a new visitor session on store entry.

        Args:
            track_id: ByteTrack track ID.
            entry_time: When the person entered.
            camera_id: Which camera detected them.

        Returns:
            New VisitorSession.
        """
        visitor_id = f"V_{uuid.uuid4().hex[:8]}_{track_id}"

        session = VisitorSession(
            visitor_id=visitor_id,
            track_id=track_id,
            store_id=self.store_id,
            entry_time=entry_time,
            camera_id=camera_id,
            last_seen_camera=camera_id,
            last_seen=entry_time,
        )

        self._active_sessions[track_id] = session
        self._track_to_visitor[track_id] = visitor_id

        logger.info(
            "Session created: %s (track=%s, camera=%s, time=%s)",
            visitor_id,
            track_id,
            camera_id,
            entry_time,
        )

        return session

    def update_zone(
        self,
        track_id: int,
        zone_name: str,
        timestamp: datetime,
        is_billing: bool = False,
    ) -> Optional[VisitorSession]:
        """
        Update zone for an active session.

        Args:
            track_id: Track ID.
            zone_name: Current zone name.
            timestamp: Current timestamp.
            is_billing: Whether this is the billing zone.

        Returns:
            Updated session, or None if track not found.
        """
        session = self._active_sessions.get(track_id)
        if session is None:
            return None

        # Auto-exit previous zone if transitioning directly to a different zone
        if session.current_zone and session.current_zone != zone_name:
            prev_zone = session.current_zone
            prev_entry = session.zone_entry_times.get(prev_zone, session.entry_time)
            dwell_sec = (timestamp - prev_entry).total_seconds() if prev_entry else 0.0
            self.update_zone_exit(track_id, prev_zone, dwell_sec, timestamp, is_billing=(prev_zone == "BILLING"))

        session.zones_visited.add(zone_name)
        session.current_zone = zone_name
        session.zone_entry_times[zone_name] = timestamp
        session.last_seen = timestamp

        if is_billing and session.billing_zone_enter_time is None:
            session.billing_zone_enter_time = timestamp
            logger.debug(
                "Session %s entered billing zone at %s",
                session.visitor_id,
                timestamp,
            )

        return session

    def update_zone_exit(
        self,
        track_id: int,
        zone_name: str,
        dwell_seconds: float,
        timestamp: datetime,
        is_billing: bool = False,
    ) -> Optional[VisitorSession]:
        """
        Record a zone exit with dwell time.

        Args:
            track_id: Track ID.
            zone_name: Zone that was exited.
            dwell_seconds: How long they were in the zone.
            timestamp: When they exited.
            is_billing: Whether this is the billing zone.
        """
        session = self._active_sessions.get(track_id)
        if session is None:
            return None

        # Add to accumulated dwell time for this zone
        session.zone_dwell_times[zone_name] = session.zone_dwell_times.get(zone_name, 0.0) + dwell_seconds
        session.last_seen = timestamp
        if session.current_zone == zone_name:
            session.current_zone = None

        if is_billing:
            session.billing_zone_exit_time = timestamp

        return session

    def finalize_session(
        self,
        track_id: int,
        exit_time: datetime,
    ) -> Optional[VisitorSession]:
        """
        Finalize a session when a visitor exits the store.

        Args:
            track_id: Track ID.
            exit_time: When the person exited.

        Returns:
            Finalized VisitorSession, or None if not found.
        """
        session = self._active_sessions.pop(track_id, None)
        if session is None:
            return None

        # Auto-exit current zone if finalized while inside a zone
        if session.current_zone:
            prev_zone = session.current_zone
            prev_entry = session.zone_entry_times.get(prev_zone, session.entry_time)
            dwell_sec = (exit_time - prev_entry).total_seconds() if prev_entry else 0.0
            session.zone_dwell_times[prev_zone] = session.zone_dwell_times.get(prev_zone, 0.0) + dwell_sec
            if prev_zone == "BILLING":
                session.billing_zone_exit_time = exit_time
            session.current_zone = None

        session.exit_time = exit_time
        session.state = SessionState.EXITED

        self._finalized_sessions[session.visitor_id] = session

        logger.info(
            "Session finalized: %s (duration=%.1fs, zones=%d, billing=%s)",
            session.visitor_id,
            session.duration_seconds,
            session.zone_count,
            session.visited_billing,
        )

        return session

    def merge_reentry(
        self,
        prev_visitor_id: str,
        new_track_id: int,
        confidence: str,
    ):
        """
        Merge a re-entry with a previous session.

        Args:
            prev_visitor_id: Previous session's visitor_id.
            new_track_id: New track ID for the re-entering person.
            confidence: Re-entry confidence ("high" or "medium").
        """
        new_session = self._active_sessions.get(new_track_id)
        if new_session is None:
            return

        new_session.is_reentry = True
        new_session.previous_session_id = prev_visitor_id
        new_session.reentry_confidence = confidence

        logger.info(
            "Re-entry merged: %s → %s (confidence=%s)",
            prev_visitor_id,
            new_session.visitor_id,
            confidence,
        )

    def check_cross_camera_dedup(
        self,
        track_id: int,
        entry_time: datetime,
        entry_position: tuple,
    ) -> Optional[str]:
        """
        Check if this entry is a duplicate from another camera.

        Cross-camera dedup rule: same person detected within
        10 seconds across cameras → same visitor.

        Args:
            track_id: New track ID.
            entry_time: Entry timestamp.
            entry_position: Entry position (x, y).

        Returns:
            Matched visitor_id if duplicate, else None.
        """
        for vid, session in self._finalized_sessions.items():
            if session.exit_time is None:
                continue

            time_delta = abs((entry_time - session.exit_time).total_seconds())
            if time_delta < self.cross_camera_time_delta:
                logger.info(
                    "Cross-camera dedup: track %d matches session %s "
                    "(delta=%.1fs)",
                    track_id,
                    vid,
                    time_delta,
                )
                return vid

        return None

    def get_active_sessions(self) -> Dict[str, VisitorSession]:
        """Get all currently active sessions."""
        return {
            s.visitor_id: s for s in self._active_sessions.values()
        }

    def get_finalized_sessions(
        self, exclude_staff: bool = True
    ) -> Dict[str, VisitorSession]:
        """
        Get all finalized sessions.

        Args:
            exclude_staff: If True, exclude sessions classified as staff.
        """
        if exclude_staff:
            return {
                vid: s
                for vid, s in self._finalized_sessions.items()
                if not s.is_staff
            }
        return dict(self._finalized_sessions)

    def get_session_by_track(self, track_id: int) -> Optional[VisitorSession]:
        """Get session for a track ID."""
        return self._active_sessions.get(track_id)

    @property
    def active_count(self) -> int:
        return len(self._active_sessions)

    @property
    def finalized_count(self) -> int:
        return len(self._finalized_sessions)

    @property
    def stats(self) -> dict:
        """Session manager statistics."""
        finalized = list(self._finalized_sessions.values())
        customers = [s for s in finalized if not s.is_staff]

        return {
            "active_sessions": self.active_count,
            "finalized_sessions": self.finalized_count,
            "total_customers": len(customers),
            "total_staff": len(finalized) - len(customers),
            "avg_duration_seconds": (
                sum(s.duration_seconds for s in customers) / max(len(customers), 1)
            ),
            "avg_zones_visited": (
                sum(s.zone_count for s in customers) / max(len(customers), 1)
            ),
            "billing_visitors": sum(1 for s in customers if s.visited_billing),
            "converted": sum(1 for s in customers if s.converted),
        }

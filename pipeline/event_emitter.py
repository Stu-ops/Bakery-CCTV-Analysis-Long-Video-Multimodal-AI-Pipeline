"""
Event Emitter — Structured Event Emission with Oscillation Filter

Emits events from the detection pipeline in a structured format.
Events are the bridge between the CV pipeline and the API/metrics layer.

Event schema complies with the Purple Tech Challenge specification:
  - event_id:    UUID-v4, globally unique
  - store_id:    from store_layout.json
  - camera_id:   which camera produced this event
  - visitor_id:  Re-ID token, unique per visit session
  - event_type:  see EventType catalogue
  - timestamp:   ISO-8601 UTC
  - zone_id:     zone name (null for ENTRY/EXIT)
  - dwell_ms:    duration in milliseconds
  - is_staff:    boolean
  - confidence:  detection confidence score
  - metadata:    {queue_depth, sku_zone, session_seq}
"""

from enum import Enum
from typing import Callable, Dict, List, Optional
from datetime import datetime
import json
import logging
import uuid

logger = logging.getLogger(__name__)


class EventType(Enum):
    """All event types in the pipeline."""

    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_EXIT = "BILLING_QUEUE_EXIT"
    REENTRY = "REENTRY"
    STAFF_CLASSIFIED = "STAFF_CLASSIFIED"
    ROLE_CHANGED = "ROLE_CHANGED"


class EventEmitter:
    """
    Emits structured events from the detection pipeline.

    Handles:
    - Event formatting to consistent schema
    - Oscillation filtering for entry/exit
    - Event buffering and batch emission
    - Structured logging of all events
    """

    def __init__(
        self,
        store_id: str,
        stability_frames: int = 15,
        source_fps: int = 15,
    ):
        """
        Args:
            store_id: Store identifier for all events.
            stability_frames: Frames to wait before confirming
                              entry/exit (oscillation filter).
            source_fps: Raw FPS for time calculations.
        """
        self.store_id = store_id
        self.stability_frames = stability_frames
        self.source_fps = source_fps

        # Event log
        self._events: List[dict] = []

        # Oscillation filter state per track
        self._entry_stability: Dict[int, int] = {}  # track_id → consecutive frames
        self._exit_stability: Dict[int, int] = {}

        # Session sequence counter per visitor
        self._session_seq: Dict[str, int] = {}

        # Callbacks for real-time event handling
        self._callbacks: List[Callable[[dict], None]] = []

        logger.info(
            "EventEmitter: store=%s, stability=%d frames",
            store_id,
            stability_frames,
        )

    def register_callback(self, callback: Callable[[dict], None]):
        """Register a callback for real-time event handling."""
        self._callbacks.append(callback)

    def emit(
        self,
        event_type: EventType,
        visitor_id: str,
        timestamp: datetime,
        track_id: Optional[int] = None,
        zone: Optional[str] = None,
        camera_id: Optional[str] = None,
        confidence: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Emit a structured event compliant with challenge schema.

        Args:
            event_type: Type of event.
            visitor_id: Unique visitor session identifier.
            timestamp: Event timestamp.
            track_id: ByteTrack track ID (optional).
            zone: Zone name (for zone events).
            camera_id: Camera identifier.
            confidence: Detection confidence score (0.0-1.0).
            metadata: Additional event-specific data.

        Returns:
            The emitted event dict.
        """
        # Increment session sequence for this visitor
        self._session_seq[visitor_id] = self._session_seq.get(visitor_id, 0) + 1
        seq = self._session_seq[visitor_id]

        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type.value,
            "store_id": self.store_id,
            "visitor_id": visitor_id,
            "timestamp": (
                timestamp.isoformat() + "Z"
                if isinstance(timestamp, datetime)
                else timestamp
            ),
            "track_id": track_id,
            "zone_id": zone,
            "camera_id": camera_id,
            "confidence": round(confidence, 3),
            "is_staff": False,
            "dwell_ms": 0,
            "converted": False,
            "basket_value": 0.0,
            "metadata": {
                "session_seq": seq,
            },
        }

        if metadata:
            # Merge metadata fields into event and into metadata sub-dict
            for key in ["is_staff", "dwell_ms", "dwell_seconds", "queue_depth", "converted", "basket_value"]:
                if key in metadata:
                    if key == "dwell_seconds":
                        event["dwell_ms"] = int(metadata[key] * 1000)
                    else:
                        event[key] = metadata[key]
            # Put extra fields into metadata sub-object
            for key, val in metadata.items():
                if key not in event:
                    event["metadata"][key] = val

        self._events.append(event)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error("Event callback error: %s", e)

        logger.info(
            "EVENT: %s | visitor=%s | zone=%s | conf=%.2f | time=%s",
            event_type.value,
            visitor_id,
            zone or "-",
            confidence,
            event["timestamp"],
        )

        return event

    def emit_entry(
        self,
        visitor_id: str,
        track_id: int,
        timestamp: datetime,
        camera_id: Optional[str] = None,
        confidence: float = 0.0,
        entry_position: Optional[tuple] = None,
    ) -> dict:
        """Emit ENTRY event."""
        metadata = {}
        if entry_position:
            metadata["entry_position"] = list(entry_position)

        return self.emit(
            EventType.ENTRY,
            visitor_id=visitor_id,
            track_id=track_id,
            timestamp=timestamp,
            camera_id=camera_id,
            confidence=confidence,
            metadata=metadata,
        )

    def emit_exit(
        self,
        visitor_id: str,
        track_id: int,
        timestamp: datetime,
        duration_seconds: float = 0.0,
        zones_visited: Optional[List[str]] = None,
        is_staff: bool = False,
        camera_id: Optional[str] = None,
        confidence: float = 0.0,
        converted: bool = False,
        basket_value: float = 0.0,
    ) -> dict:
        """Emit EXIT event with session summary."""
        return self.emit(
            EventType.EXIT,
            visitor_id=visitor_id,
            track_id=track_id,
            timestamp=timestamp,
            camera_id=camera_id,
            confidence=confidence,
            metadata={
                "dwell_seconds": round(duration_seconds, 1),
                "zones_visited": zones_visited or [],
                "is_staff": is_staff,
                "converted": converted,
                "basket_value": basket_value,
            },
        )

    def emit_zone_event(
        self,
        event_type: EventType,
        visitor_id: str,
        track_id: int,
        zone: str,
        timestamp: datetime,
        dwell_seconds: float = 0.0,
        confidence: float = 0.0,
    ) -> dict:
        """Emit ZONE_ENTER, ZONE_EXIT, or ZONE_DWELL event."""
        metadata = {}
        if dwell_seconds > 0:
            metadata["dwell_seconds"] = round(dwell_seconds, 1)

        return self.emit(
            event_type,
            visitor_id=visitor_id,
            track_id=track_id,
            zone=zone,
            timestamp=timestamp,
            confidence=confidence,
            metadata=metadata,
        )

    def emit_billing_event(
        self,
        event_type: EventType,
        visitor_id: str,
        track_id: int,
        timestamp: datetime,
        queue_depth: int = 0,
    ) -> dict:
        """Emit BILLING_QUEUE_JOIN or BILLING_QUEUE_EXIT event."""
        return self.emit(
            event_type,
            visitor_id=visitor_id,
            track_id=track_id,
            zone="BILLING",
            timestamp=timestamp,
            metadata={"queue_depth": queue_depth},
        )

    def emit_reentry(
        self,
        visitor_id: str,
        prev_visitor_id: str,
        track_id: int,
        timestamp: datetime,
        confidence: str = "none",
        similarity: float = 0.0,
    ) -> dict:
        """Emit REENTRY event."""
        return self.emit(
            EventType.REENTRY,
            visitor_id=visitor_id,
            track_id=track_id,
            timestamp=timestamp,
            metadata={
                "prev_visitor_id": prev_visitor_id,
                "confidence": confidence,
                "similarity": round(similarity, 3),
            },
        )

    def emit_staff_classification(
        self,
        visitor_id: str,
        track_id: int,
        timestamp: datetime,
        is_staff: bool,
        score: float,
        explanation: str,
        camera_id: str = "CAM_1",
    ):
        """Emit when someone is classified as staff."""
        return self.emit(
            event_type=EventType.STAFF_CLASSIFIED,
            visitor_id=visitor_id,
            timestamp=timestamp,
            track_id=track_id,
            camera_id=camera_id,
            confidence=score,
            metadata={
                "is_staff": is_staff,
                "explanation": explanation
            }
        )

    def emit_role_changed(
        self,
        visitor_id: str,
        track_id: int,
        timestamp: datetime,
        old_role: str,
        new_role: str,
        confidence: float,
        explanation: str,
        camera_id: str = "CAM_1",
    ):
        """Emit when an identity changes roles dynamically."""
        return self.emit(
            event_type=EventType.ROLE_CHANGED,
            visitor_id=visitor_id,
            timestamp=timestamp,
            track_id=track_id,
            camera_id=camera_id,
            confidence=confidence,
            metadata={
                "previous_role": old_role,
                "new_role": new_role,
                "explanation": explanation
            }
        )

    def check_entry_stability(self, track_id: int, is_inside: bool) -> bool:
        """
        Oscillation filter for entry detection.

        Returns True only when person has been consistently inside
        for stability_frames consecutive frames.
        """
        if is_inside:
            self._entry_stability[track_id] = (
                self._entry_stability.get(track_id, 0) + 1
            )
            if self._entry_stability[track_id] >= self.stability_frames:
                del self._entry_stability[track_id]
                return True
        else:
            self._entry_stability.pop(track_id, None)

        return False

    def check_exit_stability(self, track_id: int, is_outside: bool) -> bool:
        """
        Oscillation filter for exit detection.

        Returns True only when person has been consistently outside
        for stability_frames consecutive frames.
        """
        if is_outside:
            self._exit_stability[track_id] = (
                self._exit_stability.get(track_id, 0) + 1
            )
            if self._exit_stability[track_id] >= self.stability_frames:
                del self._exit_stability[track_id]
                return True
        else:
            self._exit_stability.pop(track_id, None)

        return False

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        visitor_id: Optional[str] = None,
    ) -> List[dict]:
        """
        Get emitted events, optionally filtered.

        Args:
            event_type: Filter by event type.
            visitor_id: Filter by visitor ID.

        Returns:
            List of matching events.
        """
        events = self._events

        if event_type:
            events = [e for e in events if e["event_type"] == event_type.value]

        if visitor_id:
            events = [e for e in events if e["visitor_id"] == visitor_id]

        return events

    def export_events_json(self) -> str:
        """Export all events as JSON string."""
        return json.dumps(self._events, indent=2, default=str)

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def stats(self) -> dict:
        """Event emission statistics."""
        type_counts = {}
        for event in self._events:
            t = event["event_type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_events": len(self._events),
            "by_type": type_counts,
            "unique_visitors": len(set(e["visitor_id"] for e in self._events)),
        }

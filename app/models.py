"""
Pydantic Models — API Request/Response Schemas

Defines the data models for the REST API endpoints.
All event and metric models follow the challenge specification.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


# ─── Event Models ────────────────────────────────────────────────────────────


class EventTypeEnum(str, Enum):
    """Valid event types."""

    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_EXIT = "BILLING_QUEUE_EXIT"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"
    STAFF_CLASSIFIED = "STAFF_CLASSIFIED"
    ROLE_CHANGED = "ROLE_CHANGED"


class EventIngest(BaseModel):
    """Single event for ingestion — challenge-compliant schema."""

    # Required fields per challenge spec
    event_id: Optional[str] = None
    event_type: EventTypeEnum
    store_id: str
    visitor_id: str
    timestamp: str
    camera_id: Optional[str] = None
    track_id: Optional[Union[str, int]] = None

    # Zone and dwell
    zone_id: Optional[str] = None
    dwell_ms: Optional[int] = 0
    confidence: Optional[float] = 0.0
    is_staff: Optional[bool] = False

    # Metadata sub-object
    metadata: Optional[Dict[str, Any]] = None

    # Legacy compatibility (zone → zone_id)
    zone: Optional[str] = None

    # Optional metadata fields (flat, for backward compatibility)
    duration_seconds: Optional[float] = None
    dwell_seconds: Optional[float] = None
    zones_visited: Optional[List[str]] = None
    staff_score: Optional[float] = None
    queue_depth: Optional[int] = None
    prev_visitor_id: Optional[str] = None
    similarity: Optional[float] = None
    entry_position: Optional[List[float]] = None
    explanation: Optional[str] = None
    basket_value: Optional[float] = None


class EventBatchIngest(BaseModel):
    """Batch of events for ingestion."""

    events: List[EventIngest]


class EventResponse(BaseModel):
    """Response after ingesting events."""

    status: str = "ok"
    events_ingested: int
    message: str = ""


# ─── Metrics Models ──────────────────────────────────────────────────────────


class ZoneDwellMetric(BaseModel):
    """Average dwell time per zone."""

    zone: str
    avg_dwell_seconds: float
    total_visitors: int
    total_dwell_seconds: float


class QueueMetric(BaseModel):
    """Queue depth statistics."""

    current_depth: int = 0
    average_depth: float = 0.0
    max_depth: int = 0
    min_depth: int = 0
    samples: int = 0


class FunnelStage(BaseModel):
    """Single stage in the conversion funnel."""

    stage: str
    count: int
    percentage: float = Field(
        ..., description="Percentage of total entries reaching this stage"
    )


class AnomalyItem(BaseModel):
    """Structured anomaly with severity and suggested action."""

    type: str
    severity: str = "INFO"  # INFO / WARN / CRITICAL
    message: str = ""
    suggested_action: str = ""
    timestamp: Optional[str] = None


class HeatmapZone(BaseModel):
    """Zone data for heatmap rendering."""

    zone_id: str
    visit_count: int = 0
    avg_dwell_seconds: float = 0.0
    normalized_score: float = 0.0  # 0-100


class StoreMetrics(BaseModel):
    """Complete metrics for a store."""

    store_id: str
    time_range: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Start/end timestamps of data",
    )

    # Visitor counts
    unique_visitors: int = 0
    total_entries: int = 0
    total_exits: int = 0
    staff_count: int = 0

    # Conversion
    conversion_rate: float = 0.0
    total_conversions: int = 0

    # Dwell
    avg_session_duration_seconds: float = 0.0
    zone_dwell_times: List[ZoneDwellMetric] = Field(default_factory=list)

    # Queue
    queue_stats: QueueMetric = Field(default_factory=QueueMetric)

    # Re-entry
    reentry_count: int = 0

    # Funnel
    funnel: List[FunnelStage] = Field(default_factory=list)

    # Anomalies (structured)
    anomalies: List[Any] = Field(default_factory=list, description="List of detected anomalies")


class HealthResponse(BaseModel):
    """Health check response — challenge compliant."""

    status: str = "healthy"
    version: str = "1.0.0"
    store_id: Optional[str] = None
    events_stored: int = 0
    uptime_seconds: float = 0.0
    last_event_timestamp: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

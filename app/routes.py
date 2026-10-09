"""
API Routes — FastAPI Endpoints

Challenge-compliant endpoint set:
- POST /events/ingest                — Ingest detection events (batch, idempotent by event_id)
- POST /events/ingest/single         — Ingest a single event
- GET  /stores/{store_id}/metrics    — Store metrics (unique visitors, conversion, dwell, queue)
- GET  /stores/{store_id}/funnel     — Conversion funnel (session-based)
- GET  /stores/{store_id}/heatmap    — Zone visit frequency + avg dwell, normalized 0-100
- GET  /stores/{store_id}/anomalies  — Structured anomalies with severity + suggested_action
- GET  /stores/{store_id}/events     — Raw events with filters
- GET  /metrics                      — Default store metrics shortcut
- GET  /health                       — Health check with STALE_FEED detection
- WS   /ws/live                      — WebSocket real-time event stream
"""

from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
import asyncio
import logging

from app.models import (
    EventIngest,
    EventBatchIngest,
    EventResponse,
    StoreMetrics,
    HealthResponse,
)
from app.database import EventDatabase
from app.metrics import MetricsComputer
from app.event_bus import event_bus

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared instances (initialized in main.py)
db: Optional[EventDatabase] = None
metrics_computer = MetricsComputer()
_start_time = datetime.now(timezone.utc)


def init_routes(database: EventDatabase):
    """Initialize routes with database instance."""
    global db
    db = database


# ─── WebSocket — Real-Time Live Stream ───────────────────────────────────────


@router.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """
    WebSocket endpoint for real-time event streaming.

    Clients connect here and receive:
    - A snapshot of recent events on connect
    - Each new event as it's ingested in real time
    - Periodic metrics updates
    """
    await event_bus.subscribe(ws)
    try:
        # Keep connection alive — listen for pings/close
        while True:
            # Wait for client messages (ping/pong or close)
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30)
                # Client can request a metrics refresh
                if data == "refresh_metrics":
                    if db:
                        events = db.get_events(store_id="STORE_BLR_002")
                        if events:
                            metrics = metrics_computer.compute_all_metrics(events, "STORE_BLR_002")
                            await event_bus.broadcast_metrics_update(metrics)
            except asyncio.TimeoutError:
                # Send a heartbeat to keep the connection alive
                try:
                    await ws.send_json({
                        "type": "heartbeat",
                        "server_time": datetime.now(timezone.utc).isoformat(),
                        "event_count": event_bus.total_events_broadcast,
                        "subscribers": event_bus.subscriber_count,
                    })
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        event_bus.unsubscribe(ws)


# ─── Event Ingestion ─────────────────────────────────────────────────────────


@router.post("/events/ingest", response_model=EventResponse)
async def ingest_events(payload: EventBatchIngest):
    """
    Ingest a batch of detection events.

    Accepts up to 500 events. Validates, deduplicates by event_id, and stores.
    Each event is broadcast via WebSocket to connected dashboard clients.
    """
    if db is None:
        raise HTTPException(status_code=503, detail={"error": "Database not initialized", "code": "DB_UNAVAILABLE"})

    events = []
    for event in payload.events:
        event_dict = event.model_dump(exclude_none=True)
        # Convert enum to string
        event_dict["event_type"] = event_dict["event_type"].value if hasattr(event_dict["event_type"], "value") else event_dict["event_type"]
        events.append(event_dict)

    count = db.insert_events_batch(events)

    # ── REAL-TIME: Broadcast each event to WebSocket clients ──
    for event_dict in events:
        await event_bus.broadcast(event_dict)

    # ── REAL-TIME: After batch, broadcast updated metrics ──
    if events:
        try:
            store_id = events[0].get("store_id", "STORE_BLR_002")
            all_events = db.get_events(store_id=store_id)
            if all_events:
                metrics = metrics_computer.compute_all_metrics(all_events, store_id)
                await event_bus.broadcast_metrics_update(metrics)
        except Exception as e:
            logger.error("Failed to broadcast metrics update: %s", e)

    logger.info("Ingested %d events (broadcast to %d clients)", count, event_bus.subscriber_count)

    return EventResponse(
        status="ok",
        events_ingested=count,
        message=f"Successfully ingested {count} events",
    )


@router.post("/events/ingest/single", response_model=EventResponse)
async def ingest_single_event(event: EventIngest):
    """Ingest a single detection event."""
    if db is None:
        raise HTTPException(status_code=503, detail={"error": "Database not initialized", "code": "DB_UNAVAILABLE"})

    event_dict = event.model_dump(exclude_none=True)
    event_dict["event_type"] = event_dict["event_type"].value if hasattr(event_dict["event_type"], "value") else event_dict["event_type"]
    db.insert_event(event_dict)

    # ── REAL-TIME: Broadcast single event ──
    await event_bus.broadcast(event_dict)

    return EventResponse(
        status="ok",
        events_ingested=1,
        message="Event ingested",
    )


from pydantic import BaseModel

class FrameIngest(BaseModel):
    camera_id: str
    frame: str  # base64 encoded jpeg

@router.post("/stream/frame")
async def ingest_frame(payload: FrameIngest):
    """Receive a video frame from pipeline and broadcast to dashboard."""
    await event_bus.broadcast_frame(payload.camera_id, payload.frame)
    return {"status": "ok"}

# ─── Metrics ─────────────────────────────────────────────────────────────────


@router.get("/stores/{store_id}/metrics", response_model=StoreMetrics)
async def get_store_metrics(
    store_id: str,
    start_time: Optional[str] = Query(None, description="ISO-8601 start time"),
    end_time: Optional[str] = Query(None, description="ISO-8601 end time"),
):
    """
    Get comprehensive metrics for a store.

    Returns unique visitors, conversion rate, average dwell times,
    queue stats, and conversion funnel.

    All metrics exclude staff members.
    """
    if db is None:
        raise HTTPException(status_code=503, detail={"error": "Database not initialized", "code": "DB_UNAVAILABLE"})

    events = db.get_events(
        store_id=store_id,
        start_time=start_time,
        end_time=end_time,
    )

    if not events:
        return StoreMetrics(store_id=store_id)

    metrics = metrics_computer.compute_all_metrics(events, store_id)

    return StoreMetrics(**metrics)


@router.get("/metrics", response_model=StoreMetrics)
@router.get("/Metrics", response_model=StoreMetrics)
async def get_general_metrics(
    store_id: str = Query("STORE_BLR_002", description="Store ID"),
    start_time: Optional[str] = Query(None, description="ISO-8601 start time"),
    end_time: Optional[str] = Query(None, description="ISO-8601 end time"),
):
    """
    Get comprehensive metrics for a default store.
    This fulfills the '/metrics' and '/Metrics' endpoints requirements.
    """
    return await get_store_metrics(store_id, start_time, end_time)


@router.get("/stores/{store_id}/funnel")
async def get_store_funnel(store_id: str):
    """
    Get conversion funnel for a store.

    Session-based funnel. Re-entries do NOT double-count a visitor.
    Stages: Entry → Zone Browse → Zone Dwell (>30s) → Billing Queue → Conversion
    """
    if db is None:
        raise HTTPException(status_code=503, detail={"error": "Database not initialized", "code": "DB_UNAVAILABLE"})

    events = db.get_events(store_id=store_id)
    metrics = metrics_computer.compute_all_metrics(events, store_id)

    return {
        "store_id": store_id,
        "funnel": metrics.get("funnel", []),
    }


@router.get("/stores/{store_id}/heatmap")
async def get_store_heatmap(store_id: str):
    """
    Get zone heatmap data for a store.

    Returns zone visit frequency + average dwell time, normalized 0-100.
    Includes data_confidence flag if fewer than 20 sessions in window.
    """
    if db is None:
        raise HTTPException(status_code=503, detail={"error": "Database not initialized", "code": "DB_UNAVAILABLE"})

    events = db.get_events(store_id=store_id)
    heatmap = metrics_computer.compute_heatmap(events, store_id)

    return heatmap


@router.get("/stores/{store_id}/anomalies")
async def get_store_anomalies(store_id: str):
    """
    Get active anomalies for a store.

    Returns structured anomaly objects with:
    - type: BILLING_QUEUE_SPIKE, CONVERSION_DROP, DEAD_ZONE, LONG_DWELL
    - severity: INFO / WARN / CRITICAL
    - message: Human-readable description
    - suggested_action: Actionable recommendation
    """
    if db is None:
        raise HTTPException(status_code=503, detail={"error": "Database not initialized", "code": "DB_UNAVAILABLE"})

    events = db.get_events(store_id=store_id)
    metrics = metrics_computer.compute_all_metrics(events, store_id)

    return {
        "store_id": store_id,
        "anomalies": metrics.get("anomalies", []),
    }


@router.get("/stores/{store_id}/events")
async def get_store_events(
    store_id: str,
    event_type: Optional[str] = Query(None),
    visitor_id: Optional[str] = Query(None),
    limit: int = Query(1000, le=10000),
):
    """Get raw events for a store with optional filtering."""
    if db is None:
        raise HTTPException(status_code=503, detail={"error": "Database not initialized", "code": "DB_UNAVAILABLE"})

    events = db.get_events(
        store_id=store_id,
        event_type=event_type,
        visitor_id=visitor_id,
        limit=limit,
    )

    return {
        "store_id": store_id,
        "total_events": len(events),
        "events": events,
    }


# ─── Health ──────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Challenge requirement: Must show last event timestamp per store
    and emit STALE_FEED warning if >10 min lag.
    """
    event_count = db.get_event_count() if db else 0
    uptime = (datetime.now(timezone.utc) - _start_time).total_seconds()

    # Get last event timestamp
    last_ts = None
    warnings = []
    if db:
        try:
            events = db.get_events(limit=1)
            if events:
                # Get the most recent event
                all_events = db.get_events(limit=10000)
                if all_events:
                    last_ts = max(e["timestamp"] for e in all_events if "timestamp" in e)
                    # Check for STALE_FEED
                    try:
                        last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                        lag = (datetime.now(timezone.utc) - last_dt).total_seconds()
                        if lag > 600:  # 10 minutes
                            warnings.append(f"STALE_FEED: Last event was {int(lag)}s ago (>{int(lag/60)} min)")
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            logger.error("Health check error: %s", e)

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        store_id="STORE_BLR_002",
        events_stored=event_count,
        uptime_seconds=round(uptime, 1),
        last_event_timestamp=last_ts,
        warnings=warnings,
    )

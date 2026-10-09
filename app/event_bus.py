"""
Event Bus — In-Memory Real-Time Event Broadcast System

Provides a publish/subscribe mechanism for WebSocket clients.
When events are ingested via the API, they are broadcast to all
connected WebSocket clients in real time.

This is the bridge that proves Pipeline → API → Dashboard are
genuinely connected, not batch-processed.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class EventBus:
    """
    Central event bus for real-time broadcasting.

    - Pipeline POSTs events to /events/ingest
    - Ingest handler calls event_bus.broadcast(event)
    - All connected WebSocket clients receive the event instantly
    """

    def __init__(self):
        self._subscribers: Set[WebSocket] = set()
        self._recent_events: List[dict] = []
        self._max_recent = 200
        self._event_counter = 0
        self._start_time = datetime.now(timezone.utc)

    async def subscribe(self, ws: WebSocket):
        """Add a WebSocket client to the broadcast list."""
        await ws.accept()
        self._subscribers.add(ws)
        logger.info("WebSocket client connected. Total: %d", len(self._subscribers))

        # Send current state snapshot to the new client
        now = datetime.now(timezone.utc)
        snapshot = {
            "type": "snapshot",
            "event_count": self._event_counter,
            "recent_events": self._recent_events[-50:],
            "connected_at": now.isoformat().replace("+00:00", "Z"),
            "uptime_seconds": (now - self._start_time).total_seconds(),
        }
        try:
            await ws.send_json(snapshot)
        except Exception:
            pass

    def unsubscribe(self, ws: WebSocket):
        """Remove a WebSocket client."""
        self._subscribers.discard(ws)
        logger.info("WebSocket client disconnected. Total: %d", len(self._subscribers))

    async def broadcast(self, event: dict):
        """
        Broadcast a single event to ALL connected WebSocket clients.
        Called from the ingest endpoint whenever new events arrive.
        """
        self._event_counter += 1
        event_with_meta = {
            "type": "event",
            "seq": self._event_counter,
            "server_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "data": event,
        }

        self._recent_events.append(event_with_meta)
        if len(self._recent_events) > self._max_recent:
            self._recent_events = self._recent_events[-self._max_recent:]

        # Broadcast to all subscribers concurrently
        if self._subscribers:
            async def send_safe(ws):
                try:
                    await asyncio.wait_for(ws.send_json(event_with_meta), timeout=0.2)
                except Exception:
                    return ws
                return None

            results = await asyncio.gather(*[send_safe(ws) for ws in self._subscribers])
            dead = [ws for ws in results if ws is not None]
            for ws in dead:
                self._subscribers.discard(ws)

    async def broadcast_metrics_update(self, metrics: dict):
        """
        Broadcast a full metrics snapshot.
        Called periodically or after significant events.
        """
        msg = {
            "type": "metrics",
            "server_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "data": metrics,
        }
        if self._subscribers:
            async def send_safe(ws):
                try:
                    await asyncio.wait_for(ws.send_json(msg), timeout=0.2)
                except Exception:
                    return ws
                return None

            results = await asyncio.gather(*[send_safe(ws) for ws in self._subscribers])
            dead = [ws for ws in results if ws is not None]
            for ws in dead:
                self._subscribers.discard(ws)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    async def broadcast_frame(self, camera_id: str, frame_b64: str):
        """
        Broadcast a live video frame to WebSocket clients.
        """
        msg = {
            "type": "video_frame",
            "camera_id": camera_id,
            "frame": frame_b64,
            "server_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if self._subscribers:
            async def send_safe(ws):
                try:
                    await asyncio.wait_for(ws.send_json(msg), timeout=0.2)
                except Exception:
                    return ws
                return None

            results = await asyncio.gather(*[send_safe(ws) for ws in self._subscribers])
            dead = [ws for ws in results if ws is not None]
            for ws in dead:
                self._subscribers.discard(ws)

    @property
    def total_events_broadcast(self) -> int:
        return self._event_counter


# Global singleton
event_bus = EventBus()

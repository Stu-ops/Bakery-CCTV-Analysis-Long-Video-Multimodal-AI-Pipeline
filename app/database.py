"""
Database — SQLite Event Storage

Stores events from the detection pipeline for metrics computation.
Uses SQLite for single-store deployment; PostgreSQL-ready for scale.

Indices on (store_id, visitor_id, timestamp) for efficient queries.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EventDatabase:
    """
    SQLite-backed event storage.

    Stores detection events with indices for fast querying by
    store_id, visitor_id, and timestamp.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: Path to SQLite database file.
                     Default: data/events.db
        """
        if db_path is None:
            db_dir = Path(__file__).resolve().parent.parent / "outputs" / "results"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "events.db")

        self.db_path = db_path
        self._init_db()
        logger.info("EventDatabase initialized: %s", db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Create tables and indices."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                store_id TEXT NOT NULL,
                visitor_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                track_id INTEGER,
                zone TEXT,
                camera_id TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indices for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_store
            ON events (store_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_visitor
            ON events (visitor_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_timestamp
            ON events (timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type
            ON events (event_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_store_type
            ON events (store_id, event_type)
        """)

        conn.commit()
        conn.close()

    def insert_event(self, event: dict) -> int:
        """
        Insert a single event.

        Args:
            event: Event dict with required keys:
                   event_type, store_id, visitor_id, timestamp.

        Returns:
            Row ID of inserted event.
        """
        conn = self._connect()
        cursor = conn.cursor()

        # Extract metadata (everything beyond core fields)
        core_keys = {
            "event_type", "store_id", "visitor_id", "timestamp",
            "track_id", "zone", "zone_id", "camera_id",
        }
        metadata = {k: v for k, v in event.items() if k not in core_keys}
        # Use zone_id if zone not present
        zone_val = event.get("zone") or event.get("zone_id")

        cursor.execute(
            """
            INSERT INTO events
            (event_type, store_id, visitor_id, timestamp, track_id, zone, camera_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["event_type"],
                event["store_id"],
                event["visitor_id"],
                event["timestamp"],
                event.get("track_id"),
                zone_val,
                event.get("camera_id"),
                json.dumps(metadata) if metadata else None,
            ),
        )

        row_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return row_id

    def insert_events_batch(self, events: List[dict]) -> int:
        """
        Insert a batch of events efficiently.

        Args:
            events: List of event dicts.

        Returns:
            Number of events inserted.
        """
        if not events:
            return 0

        conn = self._connect()
        cursor = conn.cursor()

        core_keys = {
            "event_type", "store_id", "visitor_id", "timestamp",
            "track_id", "zone", "zone_id", "camera_id",
        }

        rows = []
        for event in events:
            metadata = {k: v for k, v in event.items() if k not in core_keys}
            rows.append((
                event["event_type"],
                event["store_id"],
                event["visitor_id"],
                event["timestamp"],
                event.get("track_id"),
                event.get("zone") or event.get("zone_id"),
                event.get("camera_id"),
                json.dumps(metadata) if metadata else None,
            ))

        cursor.executemany(
            """
            INSERT INTO events
            (event_type, store_id, visitor_id, timestamp, track_id, zone, camera_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        count = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info("Batch inserted %d events", count)
        return count

    def get_events(
        self,
        store_id: Optional[str] = None,
        event_type: Optional[str] = None,
        visitor_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 10000,
    ) -> List[dict]:
        """
        Query events with optional filters.

        Args:
            store_id: Filter by store.
            event_type: Filter by event type.
            visitor_id: Filter by visitor.
            start_time: Filter events after this time.
            end_time: Filter events before this time.
            limit: Maximum results.

        Returns:
            List of event dicts.
        """
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if store_id:
            query += " AND store_id = ?"
            params.append(store_id)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if visitor_id:
            query += " AND visitor_id = ?"
            params.append(visitor_id)
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        events = []
        for row in rows:
            event = {
                "id": row["id"],
                "event_type": row["event_type"],
                "store_id": row["store_id"],
                "visitor_id": row["visitor_id"],
                "timestamp": row["timestamp"],
                "track_id": row["track_id"],
                "zone_id": row["zone"],
                "zone": row["zone"],  # backward compat
                "camera_id": row["camera_id"],
            }

            if row["metadata"]:
                try:
                    metadata = json.loads(row["metadata"])
                    event.update(metadata)
                except json.JSONDecodeError:
                    pass

            events.append(event)

        return events

    def get_event_count(self, store_id: Optional[str] = None) -> int:
        """Get total event count."""
        conn = self._connect()
        cursor = conn.cursor()

        if store_id:
            cursor.execute(
                "SELECT COUNT(*) FROM events WHERE store_id = ?",
                (store_id,),
            )
        else:
            cursor.execute("SELECT COUNT(*) FROM events")

        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_unique_visitors(self, store_id: str) -> int:
        """Get count of unique visitors (DISTINCT visitor_id with ENTRY events)."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(DISTINCT visitor_id) FROM events
            WHERE store_id = ? AND event_type = 'ENTRY'
            AND (metadata IS NULL OR json_extract(metadata, '$.is_staff') IS NOT 1)
            """,
            (store_id,),
        )

        count = cursor.fetchone()[0]
        conn.close()
        return count

    def clear_store(self, store_id: str):
        """Delete all events for a store."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM events WHERE store_id = ?", (store_id,))
        conn.commit()
        conn.close()
        logger.info("Cleared all events for store %s", store_id)

    def clear_all(self):
        """Delete all events."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM events")
        conn.commit()
        conn.close()
        logger.info("Cleared all events")

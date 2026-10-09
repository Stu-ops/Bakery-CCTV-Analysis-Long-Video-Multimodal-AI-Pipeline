# PROMPT: Generate mock client endpoints for testing the FastAPI application including ingestion, metrics, funnel, heatmaps, and zero-occupancy scenarios.
# CHANGES MADE: Added explicit JSON schema assertions and boundary conditions for nonexistent store IDs.
"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import EventDatabase
from app.routes import init_routes
import tempfile
import os


@pytest.fixture
def client(tmp_path):
    """Create test client with temporary database."""
    db_path = str(tmp_path / "test_events.db")
    db = EventDatabase(db_path=db_path)
    init_routes(db)

    with TestClient(app) as client:
        yield client


class TestAPIEndpoints:
    """Test suite for API endpoints."""

    def test_root(self, client):
        """Root endpoint should redirect to the live dashboard."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (200, 307, 308)
        # App redirects "/" -> "/dashboard/"; TestClient may follow it.
        if response.status_code in (307, 308):
            assert "/dashboard" in response.headers["location"]
        else:
            # Followed redirect lands on dashboard HTML, not JSON.
            assert "text/html" in response.headers.get("content-type", "")

    def test_health(self, client):
        """Health check should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_ingest_single_event(self, client):
        """Single event ingestion should work."""
        event = {
            "event_type": "ENTRY",
            "store_id": "STORE_BLR_002",
            "visitor_id": "V_test_001",
            "timestamp": "2026-03-03T14:00:00Z",
        }
        response = client.post(
            "/events/ingest/single",
            json=event,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["events_ingested"] == 1

    def test_ingest_batch(self, client):
        """Batch event ingestion should work."""
        payload = {
            "events": [
                {
                    "event_type": "ENTRY",
                    "store_id": "STORE_BLR_002",
                    "visitor_id": "V_001",
                    "timestamp": "2026-03-03T14:00:00Z",
                },
                {
                    "event_type": "ZONE_ENTER",
                    "store_id": "STORE_BLR_002",
                    "visitor_id": "V_001",
                    "timestamp": "2026-03-03T14:01:00Z",
                    "zone": "MAKEUP",
                },
                {
                    "event_type": "EXIT",
                    "store_id": "STORE_BLR_002",
                    "visitor_id": "V_001",
                    "timestamp": "2026-03-03T14:10:00Z",
                    "duration_seconds": 600,
                    "is_staff": False,
                },
            ]
        }
        response = client.post("/events/ingest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["events_ingested"] == 3

    def test_get_metrics(self, client):
        """
        Metrics endpoint should return valid JSON.
        Acceptance gate: GET /stores/{id}/metrics returns JSON.
        """
        # Ingest some events first
        payload = {
            "events": [
                {
                    "event_type": "ENTRY",
                    "store_id": "STORE_BLR_002",
                    "visitor_id": "V_001",
                    "timestamp": "2026-03-03T14:00:00Z",
                },
                {
                    "event_type": "EXIT",
                    "store_id": "STORE_BLR_002",
                    "visitor_id": "V_001",
                    "timestamp": "2026-03-03T14:10:00Z",
                    "duration_seconds": 600,
                },
            ]
        }
        client.post("/events/ingest", json=payload)

        response = client.get("/stores/STORE_BLR_002/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["store_id"] == "STORE_BLR_002"
        assert "unique_visitors" in data
        assert "conversion_rate" in data

    def test_get_metrics_empty_store(self, client):
        """Empty store should return zero metrics (not error)."""
        response = client.get("/stores/NONEXISTENT/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["unique_visitors"] == 0

    def test_get_funnel(self, client):
        """Funnel endpoint should return funnel stages."""
        # Ingest events
        payload = {
            "events": [
                {
                    "event_type": "ENTRY",
                    "store_id": "STORE_BLR_002",
                    "visitor_id": "V_001",
                    "timestamp": "2026-03-03T14:00:00Z",
                },
                {
                    "event_type": "ZONE_ENTER",
                    "store_id": "STORE_BLR_002",
                    "visitor_id": "V_001",
                    "timestamp": "2026-03-03T14:01:00Z",
                    "zone": "MAKEUP",
                },
            ]
        }
        client.post("/events/ingest", json=payload)

        response = client.get("/stores/STORE_BLR_002/funnel")
        assert response.status_code == 200
        data = response.json()
        assert "funnel" in data

    def test_get_events(self, client):
        """Events endpoint should return stored events."""
        # Ingest
        payload = {
            "events": [
                {
                    "event_type": "ENTRY",
                    "store_id": "STORE_BLR_002",
                    "visitor_id": "V_001",
                    "timestamp": "2026-03-03T14:00:00Z",
                },
            ]
        }
        client.post("/events/ingest", json=payload)

        response = client.get("/stores/STORE_BLR_002/events")
        assert response.status_code == 200
        data = response.json()
        assert data["total_events"] >= 1

    def test_ingest_with_all_fields(self, client):
        """Event with all optional fields should work."""
        event = {
            "event_type": "EXIT",
            "store_id": "STORE_BLR_002",
            "visitor_id": "V_full",
            "timestamp": "2026-03-03T14:10:00Z",
            "track_id": 42,
            "zone": "ENTRY",
            "camera_id": "CAM_01",
            "duration_seconds": 600.0,
            "zones_visited": ["MAKEUP", "SKINCARE"],
            "is_staff": False,
        }
        response = client.post("/events/ingest/single", json=event)
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

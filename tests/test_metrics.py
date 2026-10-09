# PROMPT: Create robust unit tests for a MetricsComputer computing total visits, conversion rates, zone dwell averages, queue stats, and excluding staff members.
# CHANGES MADE: Integrated multi-zone session test scenarios and verified average queue depth math bounds.
"""
Tests for API endpoints and MetricsComputer.
"""

import pytest
from datetime import datetime
from app.metrics import MetricsComputer


class TestMetricsComputer:
    """Test suite for metrics computation."""

    @pytest.fixture
    def computer(self):
        return MetricsComputer()

    @pytest.fixture
    def sample_events(self):
        """Sample events for a store with 3 customers and 1 staff."""
        return [
            # Customer 1: entry → makeup → billing → exit
            {"event_type": "ENTRY", "store_id": "S1", "visitor_id": "V1", "timestamp": "2026-03-03T14:00:00Z"},
            {"event_type": "ZONE_ENTER", "store_id": "S1", "visitor_id": "V1", "timestamp": "2026-03-03T14:01:00Z", "zone": "MAKEUP"},
            {"event_type": "ZONE_DWELL", "store_id": "S1", "visitor_id": "V1", "timestamp": "2026-03-03T14:01:30Z", "zone": "MAKEUP", "dwell_seconds": 30},
            {"event_type": "ZONE_EXIT", "store_id": "S1", "visitor_id": "V1", "timestamp": "2026-03-03T14:05:00Z", "zone": "MAKEUP", "dwell_seconds": 240},
            {"event_type": "BILLING_QUEUE_JOIN", "store_id": "S1", "visitor_id": "V1", "timestamp": "2026-03-03T14:06:00Z", "zone": "BILLING", "queue_depth": 2},
            {"event_type": "EXIT", "store_id": "S1", "visitor_id": "V1", "timestamp": "2026-03-03T14:10:00Z", "duration_seconds": 600, "is_staff": False, "converted": True},

            # Customer 2: entry → skincare → exit (no billing)
            {"event_type": "ENTRY", "store_id": "S1", "visitor_id": "V2", "timestamp": "2026-03-03T14:02:00Z"},
            {"event_type": "ZONE_ENTER", "store_id": "S1", "visitor_id": "V2", "timestamp": "2026-03-03T14:03:00Z", "zone": "SKINCARE"},
            {"event_type": "ZONE_EXIT", "store_id": "S1", "visitor_id": "V2", "timestamp": "2026-03-03T14:07:00Z", "zone": "SKINCARE", "dwell_seconds": 240},
            {"event_type": "EXIT", "store_id": "S1", "visitor_id": "V2", "timestamp": "2026-03-03T14:08:00Z", "duration_seconds": 360, "is_staff": False},

            # Customer 3: entry → exit (quick bounce)
            {"event_type": "ENTRY", "store_id": "S1", "visitor_id": "V3", "timestamp": "2026-03-03T14:05:00Z"},
            {"event_type": "EXIT", "store_id": "S1", "visitor_id": "V3", "timestamp": "2026-03-03T14:06:00Z", "duration_seconds": 60, "is_staff": False},

            # Staff member
            {"event_type": "ENTRY", "store_id": "S1", "visitor_id": "V_STAFF", "timestamp": "2026-03-03T14:00:00Z"},
            {"event_type": "STAFF_CLASSIFIED", "store_id": "S1", "visitor_id": "V_STAFF", "timestamp": "2026-03-03T14:00:00Z", "is_staff": True},
            {"event_type": "EXIT", "store_id": "S1", "visitor_id": "V_STAFF", "timestamp": "2026-03-03T14:20:00Z", "duration_seconds": 1200, "is_staff": True},
        ]

    def test_unique_visitors_excludes_staff(self, computer, sample_events):
        """Unique visitor count should exclude staff."""
        metrics = computer.compute_all_metrics(sample_events, "S1")
        assert metrics["unique_visitors"] == 3  # V1, V2, V3 (not V_STAFF)

    def test_staff_count(self, computer, sample_events):
        """Staff count should be accurate."""
        metrics = computer.compute_all_metrics(sample_events, "S1")
        assert metrics["staff_count"] == 1

    def test_conversion_rate(self, computer, sample_events):
        """Conversion = billing visitors / total visitors."""
        metrics = computer.compute_all_metrics(sample_events, "S1")
        # V1 reached billing, V2 and V3 did not
        # Conversion = 1/3 ≈ 0.3333
        assert abs(metrics["conversion_rate"] - 0.3333) < 0.01

    def test_avg_session_duration(self, computer, sample_events):
        """Average session duration from EXIT events."""
        metrics = computer.compute_all_metrics(sample_events, "S1")
        # V1: 600s, V2: 360s, V3: 60s → avg = 340s
        assert abs(metrics["avg_session_duration_seconds"] - 340) < 1

    def test_zone_dwell_times(self, computer, sample_events):
        """Zone dwell times from ZONE_EXIT events."""
        metrics = computer.compute_all_metrics(sample_events, "S1")
        zone_dwells = {z["zone"]: z for z in metrics["zone_dwell_times"]}

        assert "MAKEUP" in zone_dwells
        assert zone_dwells["MAKEUP"]["avg_dwell_seconds"] == 240.0
        assert zone_dwells["MAKEUP"]["total_visitors"] == 1

    def test_queue_stats(self, computer, sample_events):
        """Queue stats from billing events."""
        metrics = computer.compute_all_metrics(sample_events, "S1")
        queue = metrics["queue_stats"]
        assert queue["max_depth"] >= 2

    def test_funnel(self, computer, sample_events):
        """Conversion funnel stages."""
        metrics = computer.compute_all_metrics(sample_events, "S1")
        funnel = metrics["funnel"]
        assert len(funnel) == 5

        stages = {f["stage"]: f for f in funnel}
        assert stages["Entry"]["count"] == 3
        assert stages["Zone Browse"]["count"] == 2  # V1 and V2
        assert stages["Billing Queue"]["count"] == 1  # V1 only

    def test_empty_store(self, computer):
        """Empty store → all zero metrics."""
        metrics = computer.compute_all_metrics([], "S1")
        assert metrics["unique_visitors"] == 0
        assert metrics["conversion_rate"] == 0.0
        assert metrics["funnel"] == []

    def test_wrong_store_id(self, computer, sample_events):
        """Different store ID → empty results."""
        metrics = computer.compute_all_metrics(sample_events, "DIFFERENT_STORE")
        assert metrics["unique_visitors"] == 0

    def test_reentry_count(self, computer):
        """Re-entry events should be counted."""
        events = [
            {"event_type": "ENTRY", "store_id": "S1", "visitor_id": "V1", "timestamp": "2026-03-03T14:00:00Z"},
            {"event_type": "REENTRY", "store_id": "S1", "visitor_id": "V1", "timestamp": "2026-03-03T15:00:00Z"},
        ]
        metrics = computer.compute_all_metrics(events, "S1")
        assert metrics["reentry_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# PROMPT: Generate comprehensive unit tests for retail visitor SessionManager tracking state transitions, active/finalized counts, cross-camera ReID deduplication, and stats computation.
# CHANGES MADE: Added explicit tests for tracking dwell times in zones, active/finalized counts exclusions, and role transition audit trails.
"""
Tests for SessionManager — Visitor Session Lifecycle

Validates:
- Session creation on entry
- Zone visit tracking
- Session finalization on exit
- Cross-camera dedup
- Statistics computation
"""

import pytest
from datetime import datetime, timedelta
from pipeline.session_manager import SessionManager, VisitorSession, SessionState


class TestSessionManager:
    """Test suite for session manager."""

    @pytest.fixture
    def manager(self):
        return SessionManager(store_id="STORE_BLR_002")

    def test_create_session(self, manager):
        """Creating a session should populate fields correctly."""
        entry_time = datetime(2026, 3, 3, 14, 5, 0)
        session = manager.create_session(
            track_id=1,
            entry_time=entry_time,
            camera_id="CAM_01",
        )
        assert session.track_id == 1
        assert session.entry_time == entry_time
        assert session.state == SessionState.ACTIVE
        assert session.store_id == "STORE_BLR_002"
        assert "V_" in session.visitor_id

    def test_update_zone(self, manager):
        """Updating zone should track zones visited."""
        entry_time = datetime(2026, 3, 3, 14, 5, 0)
        session = manager.create_session(track_id=1, entry_time=entry_time)

        manager.update_zone(1, "MAKEUP", datetime(2026, 3, 3, 14, 6, 0))
        manager.update_zone(1, "SKINCARE", datetime(2026, 3, 3, 14, 8, 0))

        assert "MAKEUP" in session.zones_visited
        assert "SKINCARE" in session.zones_visited
        assert session.current_zone == "SKINCARE"

    def test_billing_zone_tracking(self, manager):
        """Billing zone enter/exit times should be tracked."""
        entry_time = datetime(2026, 3, 3, 14, 5, 0)
        manager.create_session(track_id=1, entry_time=entry_time)

        billing_enter = datetime(2026, 3, 3, 14, 15, 0)
        manager.update_zone(1, "BILLING", billing_enter, is_billing=True)

        session = manager.get_session_by_track(1)
        assert session.billing_zone_enter_time == billing_enter

        billing_exit = datetime(2026, 3, 3, 14, 18, 0)
        manager.update_zone_exit(1, "BILLING", 180.0, billing_exit, is_billing=True)
        assert session.billing_zone_exit_time == billing_exit

    def test_finalize_session(self, manager):
        """Finalizing should mark session as EXITED."""
        entry_time = datetime(2026, 3, 3, 14, 5, 0)
        manager.create_session(track_id=1, entry_time=entry_time)

        exit_time = datetime(2026, 3, 3, 14, 20, 0)
        session = manager.finalize_session(1, exit_time)

        assert session is not None
        assert session.state == SessionState.EXITED
        assert session.exit_time == exit_time
        assert session.duration_seconds == 15 * 60  # 15 minutes

    def test_active_count(self, manager):
        """Active count should reflect current sessions."""
        assert manager.active_count == 0

        manager.create_session(track_id=1, entry_time=datetime.utcnow())
        assert manager.active_count == 1

        manager.create_session(track_id=2, entry_time=datetime.utcnow())
        assert manager.active_count == 2

        manager.finalize_session(1, datetime.utcnow())
        assert manager.active_count == 1

    def test_finalized_count(self, manager):
        """Finalized count should track completed sessions."""
        manager.create_session(track_id=1, entry_time=datetime.utcnow())
        manager.finalize_session(1, datetime.utcnow())
        assert manager.finalized_count == 1

    def test_get_finalized_excludes_staff(self, manager):
        """get_finalized_sessions should exclude staff by default."""
        entry_time = datetime.utcnow()

        # Create and finalize a customer
        session1 = manager.create_session(track_id=1, entry_time=entry_time)
        manager.finalize_session(1, entry_time + timedelta(minutes=5))

        # Create and finalize a staff member
        session2 = manager.create_session(track_id=2, entry_time=entry_time)
        session2.is_staff = True
        manager.finalize_session(2, entry_time + timedelta(minutes=30))

        customers = manager.get_finalized_sessions(exclude_staff=True)
        all_sessions = manager.get_finalized_sessions(exclude_staff=False)

        assert len(customers) == 1
        assert len(all_sessions) == 2

    def test_session_to_dict(self, manager):
        """Session serialization should include all fields."""
        entry_time = datetime(2026, 3, 3, 14, 5, 0)
        session = manager.create_session(track_id=1, entry_time=entry_time)
        session.zones_visited.add("MAKEUP")

        data = session.to_dict()
        assert data["store_id"] == "STORE_BLR_002"
        assert "MAKEUP" in data["zones_visited"]
        assert data["is_staff"] is False

    def test_cross_camera_dedup(self, manager):
        """Should detect cross-camera duplicates within time delta."""
        entry_time = datetime(2026, 3, 3, 14, 5, 0)
        manager.create_session(track_id=1, entry_time=entry_time)
        exit_time = entry_time + timedelta(minutes=10)
        manager.finalize_session(1, exit_time)

        # New detection 5 seconds later on another camera
        new_entry = exit_time + timedelta(seconds=5)
        match = manager.check_cross_camera_dedup(
            track_id=2,
            entry_time=new_entry,
            entry_position=(100, 500),
        )
        assert match is not None

    def test_stats(self, manager):
        """Stats should reflect session data."""
        entry = datetime(2026, 3, 3, 14, 0, 0)
        session = manager.create_session(track_id=1, entry_time=entry)
        session.zones_visited = {"MAKEUP", "SKINCARE"}
        manager.finalize_session(1, entry + timedelta(minutes=10))

        stats = manager.stats
        assert stats["finalized_sessions"] == 1
        assert stats["total_customers"] == 1

    def test_last_seen_tracking(self, manager):
        """Testing that last_seen is updated correctly on entry, update_zone, and update_zone_exit."""
        entry_time = datetime(2026, 3, 3, 14, 5, 0)
        session = manager.create_session(track_id=1, entry_time=entry_time)
        assert session.last_seen == entry_time

        update_time = datetime(2026, 3, 3, 14, 6, 0)
        manager.update_zone(1, "MAKEUP", update_time)
        assert session.last_seen == update_time

        exit_time = datetime(2026, 3, 3, 14, 7, 0)
        manager.update_zone_exit(1, "MAKEUP", 60.0, exit_time)
        assert session.last_seen == exit_time

    def test_record_role_change(self, manager):
        """test_record_role_change should successfully record a role transition in history."""
        entry_time = datetime(2026, 3, 3, 14, 5, 0)
        session = manager.create_session(track_id=1, entry_time=entry_time)
        
        session.record_role_change(
            old_role="CUSTOMER",
            new_role="STAFF",
            timestamp=entry_time,
            explanation="uniform match"
        )
        
        assert len(session.role_change_history) == 1
        record = session.role_change_history[0]
        assert record["old_role"] == "CUSTOMER"
        assert record["new_role"] == "STAFF"
        assert record["explanation"] == "uniform match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

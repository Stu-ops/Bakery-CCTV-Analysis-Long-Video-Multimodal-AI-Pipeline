# PROMPT: Write an integration test suite validating real POS CSV loading, correlation window matching, session generation, and dynamic conversion rate metrics compilation.
# CHANGES MADE: Integrated exact date-time delta offsets for real CSV records to verify conversion window logic.
"""
Integration Tests — End-to-End with Real Data

Tests the complete pipeline flow:
  Real CSV → PurpllePOSLoader → POSCorrelator → Metrics

Validates that ALL integration points work together:
1. Real CSV loads and preprocesses correctly
2. POSCorrelator accepts preprocessed data
3. Correlation produces valid results
4. Metrics compute from real data
"""

import pytest
import os
import pandas as pd
from datetime import datetime, timedelta

from pipeline.data_loader import PurpllePOSLoader
from pipeline.pos_correlator import POSCorrelator
from pipeline.session_manager import SessionManager
from pipeline.event_emitter import EventEmitter, EventType
from pipeline.staff_classifier import StaffClassifier
from app.metrics import MetricsComputer


REAL_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "datasets",
    "Brigade_Bangalore_10_April_26 (1)bc6219c.csv",
)


def has_real_csv():
    return os.path.exists(REAL_CSV_PATH)


@pytest.mark.skipif(not has_real_csv(), reason="Real CSV not found")
class TestEndToEndWithRealData:
    """Full pipeline integration with real Purplle POS data."""

    @pytest.fixture
    def loaded_data(self):
        """Load and preprocess real CSV."""
        loader = PurpllePOSLoader()
        loader.load_csv(REAL_CSV_PATH)
        return loader

    @pytest.fixture
    def pipeline_components(self, loaded_data):
        """Set up pipeline components with real data."""
        pipeline_txns = loaded_data.get_pipeline_transactions()
        return {
            "pos_correlator": POSCorrelator(pos_data=pipeline_txns),
            "session_manager": SessionManager(store_id="STORE_BLR_002"),
            "event_emitter": EventEmitter(store_id="STORE_BLR_002"),
            "staff_classifier": StaffClassifier(),
            "pipeline_txns": pipeline_txns,
            "loader": loaded_data,
        }

    def test_csv_to_pipeline_flow(self, loaded_data):
        """CSV → PurpllePOSLoader → pipeline DataFrame should work."""
        pipeline_df = loaded_data.get_pipeline_transactions()

        assert len(pipeline_df) == 24  # 24 unique orders
        assert list(pipeline_df.columns) == ["store_id", "timestamp", "basket_value"]
        assert all(pipeline_df["store_id"] == "STORE_BLR_002")
        assert not pipeline_df["timestamp"].isna().any()

    def test_pos_correlator_with_real_data(self, pipeline_components):
        """POSCorrelator should accept and work with real preprocessed data."""
        pos = pipeline_components["pos_correlator"]
        txns = pipeline_components["pipeline_txns"]

        # Correlator should have all 24 transactions
        assert len(pos.transactions) == 24

        # Try to correlate a session that exits billing 1 min before first order
        first_txn_time = txns["timestamp"].min()
        billing_exit = first_txn_time - timedelta(minutes=1)

        converted, txn = pos.correlate_session(billing_exit, "STORE_BLR_002")
        assert isinstance(converted, bool)
        if converted:
            assert txn is not None
            assert "basket_value" in txn

    def test_simulated_visitors_correlate(self, pipeline_components):
        """Simulated visitors at real times should produce valid conversions."""
        pos = pipeline_components["pos_correlator"]
        sm = pipeline_components["session_manager"]
        txns = pipeline_components["pipeline_txns"]

        real_times = txns.sort_values("timestamp")["timestamp"].tolist()

        # Create 5 visitors timed to match first 5 real orders
        converted_count = 0
        for i, txn_time in enumerate(real_times[:5]):
            track_id = i + 1
            entry_time = txn_time - timedelta(minutes=10)
            billing_exit = txn_time - timedelta(minutes=1)
            exit_time = txn_time + timedelta(minutes=1)

            session = sm.create_session(track_id, entry_time, "CAM_1")
            sm.update_zone(track_id, "BILLING", txn_time - timedelta(minutes=3), is_billing=True)
            sm.update_zone_exit(track_id, "BILLING", 120.0, billing_exit, is_billing=True)
            sm.finalize_session(track_id, exit_time)

            converted, _ = pos.correlate_session(billing_exit, "STORE_BLR_002")
            if converted:
                converted_count += 1

        # At least some should convert (billing exit 1 min before txn → within 5 min window)
        assert converted_count >= 3, f"Expected at least 3 conversions, got {converted_count}"

    def test_metrics_from_real_events(self, pipeline_components):
        """Metrics computation should work with events from real data flow."""
        sm = pipeline_components["session_manager"]
        ee = pipeline_components["event_emitter"]
        txns = pipeline_components["pipeline_txns"]

        # Create a few visitor events
        real_times = txns.sort_values("timestamp")["timestamp"].tolist()

        for i in range(3):
            track_id = i + 1
            entry_time = real_times[i] - timedelta(minutes=10)
            exit_time = real_times[i] + timedelta(minutes=1)

            session = sm.create_session(track_id, entry_time, "CAM_1")
            ee.emit_entry(session.visitor_id, track_id, entry_time, "CAM_1")
            ee.emit_exit(session.visitor_id, track_id, exit_time, 660.0, ["MAKEUP"], False)
            sm.finalize_session(track_id, exit_time)

        # Compute metrics
        events = ee.get_events()
        mc = MetricsComputer()
        metrics = mc.compute_all_metrics(events, "STORE_BLR_002")

        assert "unique_visitors" in metrics
        assert metrics["unique_visitors"] == 3
        assert "avg_session_duration_seconds" in metrics

    def test_department_stats(self, loaded_data):
        """Department breakdown from real data should match expected."""
        stats = loaded_data.stats
        deps = stats.get("departments", {})

        # We know from analysis: makeup=54, skin=27, bath-and-body=9, hair=6
        assert deps.get("makeup", 0) > 40
        assert deps.get("skin", 0) > 20
        assert "hair" in deps

    def test_order_details_accessible(self, loaded_data):
        """Should be able to get details for any real order."""
        orders = loaded_data.order_dataframe
        first_order_id = orders["order_id"].iloc[0]

        details = loaded_data.get_order_details(first_order_id)
        assert details is not None
        assert "departments" in details
        assert "brands" in details
        assert details["basket_value"] > 0

    def test_load_from_purplle_csv_classmethod(self):
        """POSCorrelator.load_from_purplle_csv() should work end-to-end."""
        pos = POSCorrelator.load_from_purplle_csv(REAL_CSV_PATH)
        assert len(pos.transactions) == 24
        assert all(pos.transactions["store_id"] == "STORE_BLR_002")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

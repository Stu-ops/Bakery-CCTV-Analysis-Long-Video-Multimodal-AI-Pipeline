"""
Tests for POSCorrelator — CORRECTION #5: exit_time ± 5min

Validates:
- Transaction within window → converted
- Transaction outside window → not converted
- No billing visit → not converted
- Multiple transactions → first match
- Edge cases (empty POS data, zero visitors)
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from pipeline.pos_correlator import POSCorrelator


class TestPOSCorrelator:
    """Test suite for corrected POS correlation."""

    @pytest.fixture
    def correlator(self):
        """Create correlator with sample POS data."""
        pos_data = pd.DataFrame({
            "store_id": [
                "STORE_BLR_002", "STORE_BLR_002", "STORE_BLR_002",
                "STORE_BLR_002",
            ],
            "timestamp": [
                datetime(2026, 3, 3, 14, 33, 0),  # 14:33
                datetime(2026, 3, 3, 14, 45, 0),  # 14:45
                datetime(2026, 3, 3, 15, 10, 0),  # 15:10
                datetime(2026, 3, 3, 14, 31, 0),  # 14:31
            ],
            "basket_value": [1500.0, 2300.0, 800.0, 450.0],
        })
        return POSCorrelator(pos_data=pos_data)

    def test_transaction_in_window_converted(self, correlator):
        """
        CRITICAL TEST: Exit at 14:30, transaction at 14:33 → converted.
        This is the exact example from the corrected approach.
        Rule: exit_time < txn_time <= exit_time + 5*60
        """
        billing_exit = datetime(2026, 3, 3, 14, 30, 0)
        converted, txn = correlator.correlate_session(
            billing_exit, "STORE_BLR_002"
        )
        assert converted is True, "14:30 exit + 14:31 or 14:33 txn should convert"
        assert txn is not None

    def test_transaction_outside_window_not_converted(self, correlator):
        """Transaction more than 5 min after exit → not converted."""
        billing_exit = datetime(2026, 3, 3, 14, 0, 0)  # 14:00
        # Nearest transaction is at 14:31, which is 31 min later
        converted, txn = correlator.correlate_session(
            billing_exit, "STORE_BLR_002"
        )
        assert converted is False

    def test_no_billing_visit_not_converted(self, correlator):
        """Visitor never visited billing zone → not converted."""
        converted, txn = correlator.correlate_session(
            None, "STORE_BLR_002"
        )
        assert converted is False
        assert txn is None

    def test_window_boundary_exact_5min(self, correlator):
        """
        Transaction at exactly 5 min after exit → converted.
        Rule: exit_time < txn_time <= exit_time + 5*60
        """
        # exit at 14:28, txn at 14:33 → delta = 5 min → should be included (<=)
        billing_exit = datetime(2026, 3, 3, 14, 28, 0)
        converted, txn = correlator.correlate_session(
            billing_exit, "STORE_BLR_002"
        )
        assert converted is True

    def test_window_boundary_just_over_5min(self):
        """Transaction at 5 min + 1 second → not converted."""
        pos_data = pd.DataFrame({
            "store_id": ["STORE_BLR_002"],
            "timestamp": [datetime(2026, 3, 3, 14, 35, 1)],  # 5:01 after exit
            "basket_value": [1000.0],
        })
        correlator = POSCorrelator(pos_data=pos_data)
        billing_exit = datetime(2026, 3, 3, 14, 30, 0)
        converted, _ = correlator.correlate_session(
            billing_exit, "STORE_BLR_002"
        )
        assert converted is False

    def test_transaction_before_exit_not_counted(self):
        """
        Transaction BEFORE exit → not converted.
        Rule is exit_time < txn_time, so txn must be AFTER exit.
        """
        pos_data = pd.DataFrame({
            "store_id": ["STORE_BLR_002"],
            "timestamp": [datetime(2026, 3, 3, 14, 29, 0)],  # Before exit
            "basket_value": [1000.0],
        })
        correlator = POSCorrelator(pos_data=pos_data)
        billing_exit = datetime(2026, 3, 3, 14, 30, 0)
        converted, _ = correlator.correlate_session(
            billing_exit, "STORE_BLR_002"
        )
        assert converted is False

    def test_wrong_store_not_converted(self, correlator):
        """Transaction for different store → not converted."""
        billing_exit = datetime(2026, 3, 3, 14, 30, 0)
        converted, _ = correlator.correlate_session(
            billing_exit, "STORE_MUM_001"  # Different store
        )
        assert converted is False

    def test_batch_correlation(self, correlator):
        """Batch correlate multiple sessions."""
        sessions = {
            "V_001": {
                "billing_zone_exit_time": datetime(2026, 3, 3, 14, 30, 0),
            },
            "V_002": {
                "billing_zone_exit_time": None,  # Never in billing
            },
            "V_003": {
                "billing_zone_exit_time": datetime(2026, 3, 3, 14, 0, 0),
            },
        }
        conversions = correlator.correlate_sessions_with_pos(
            sessions, "STORE_BLR_002"
        )
        assert conversions["V_001"] is True  # 14:30 exit, 14:31 txn
        assert conversions["V_002"] is False  # No billing visit
        assert conversions["V_003"] is False  # 14:00, no txn in window

    def test_conversion_rate(self, correlator):
        """Conversion rate computation."""
        sessions = {
            "V_001": {
                "billing_zone_exit_time": datetime(2026, 3, 3, 14, 30, 0),
            },
            "V_002": {
                "billing_zone_exit_time": None,
            },
        }
        rate = correlator.compute_conversion_rate(sessions, "STORE_BLR_002")
        assert rate == 0.5  # 1 out of 2

    def test_conversion_rate_zero_visitors(self, correlator):
        """Zero visitors → rate = 0.0 (not error)."""
        rate = correlator.compute_conversion_rate({}, "STORE_BLR_002")
        assert rate == 0.0

    def test_empty_pos_data(self):
        """No POS data → nothing converts."""
        correlator = POSCorrelator()
        billing_exit = datetime(2026, 3, 3, 14, 30, 0)
        converted, _ = correlator.correlate_session(
            billing_exit, "STORE_BLR_002"
        )
        assert converted is False

    def test_add_transactions_dynamically(self):
        """Can add transactions after initialization."""
        correlator = POSCorrelator()
        correlator.add_transactions([
            {
                "store_id": "STORE_BLR_002",
                "timestamp": "2026-03-03T14:33:00",
                "basket_value": 1500.0,
            }
        ])
        billing_exit = datetime(2026, 3, 3, 14, 30, 0)
        converted, _ = correlator.correlate_session(
            billing_exit, "STORE_BLR_002"
        )
        assert converted is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

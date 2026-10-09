"""
POS Correlator — Explicit Exit-Time Window

CORRECTION #5: Uses explicit rule `exit_time < txn_time <= exit_time + 5min`
instead of ChatGPT's ambiguous "within 5 minutes before transaction".

Why ChatGPT's approach is ambiguous:
- "Before transaction" could mean:
  a) txn_time - 5min < billing_presence < txn_time (presence-based)
  b) exit_time < txn_time < exit_time + 5min (exit-based)
- These produce DIFFERENT results and DIFFERENT conversion rates.

Corrected interpretation (from problem statement):
- "A visitor who was in billing zone in the 5-minute window
   before transaction timestamp counts as converted."
- This means: the visitor EXITS billing zone, then a transaction
  occurs within 5 minutes → converted.
- Rule: exit_time < txn_time <= exit_time + 5*60

Example:
  Visitor exits billing at 14:30
  Transaction at 14:33
  14:30 < 14:33 <= 14:35 ✓ → Converted
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class POSCorrelator:
    """
    Correlates visitor sessions with POS transactions to compute
    conversion rates.

    Uses the corrected explicit window:
      exit_time < txn_time <= exit_time + 5 minutes

    This replaces ChatGPT's ambiguous "within 5 minutes before transaction".
    """

    def __init__(
        self,
        pos_data: Optional[pd.DataFrame] = None,
        pos_csv_path: Optional[str] = None,
        correlation_window_seconds: int = 5 * 60,
    ):
        """
        Args:
            pos_data: DataFrame with columns [store_id, timestamp, basket_value].
            pos_csv_path: Path to CSV file with POS transactions.
                          If this is a Purplle CSV (38 columns), use
                          load_from_purplle_csv() instead.
            correlation_window_seconds: Window after exit for transaction match
                                        (default: 5 minutes = 300 seconds).
        """
        self.correlation_window = timedelta(seconds=correlation_window_seconds)

        if pos_data is not None:
            self.transactions = pos_data.copy()
        elif pos_csv_path:
            self.transactions = pd.read_csv(pos_csv_path)
        else:
            # Create empty DataFrame with expected schema
            self.transactions = pd.DataFrame(
                columns=["store_id", "timestamp", "basket_value"]
            )

        # Ensure timestamp column is datetime
        if not self.transactions.empty and "timestamp" in self.transactions.columns:
            self.transactions["timestamp"] = pd.to_datetime(
                self.transactions["timestamp"]
            )

        logger.info(
            "POSCorrelator: %d transactions loaded, window=%ds",
            len(self.transactions),
            correlation_window_seconds,
        )

    @classmethod
    def load_from_purplle_csv(
        cls,
        csv_path: str,
        correlation_window_seconds: int = 5 * 60,
    ) -> "POSCorrelator":
        """
        Create a POSCorrelator from a real Purplle POS CSV file.

        This handles all the preprocessing:
        - Combining order_date + order_time → timestamp
        - Mapping store IDs (ST1008 → STORE_BLR_002)
        - Aggregating multi-item orders
        - Handling missing values

        Args:
            csv_path: Path to the Purplle POS CSV.
            correlation_window_seconds: Correlation window.

        Returns:
            Configured POSCorrelator with preprocessed data.
        """
        from pipeline.data_loader import PurpllePOSLoader

        loader = PurpllePOSLoader()
        loader.load_csv(csv_path)

        # Get the pipeline-ready transactions
        pipeline_df = loader.get_pipeline_transactions()

        logger.info(
            "Loaded Purplle CSV: %d raw rows → %d orders",
            loader.stats.get("raw_rows", 0),
            len(pipeline_df),
        )

        return cls(
            pos_data=pipeline_df,
            correlation_window_seconds=correlation_window_seconds,
        )

    def add_transactions(self, transactions: List[dict]):
        """
        Add transactions dynamically.

        Args:
            transactions: List of dicts with keys [store_id, timestamp, basket_value].
        """
        new_df = pd.DataFrame(transactions)
        if "timestamp" in new_df.columns:
            new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
        self.transactions = pd.concat(
            [self.transactions, new_df], ignore_index=True
        )
        logger.info("Added %d transactions (total: %d)", len(transactions), len(self.transactions))

    def correlate_session(
        self,
        billing_exit_time: Optional[datetime],
        store_id: str,
    ) -> Tuple[bool, Optional[dict]]:
        """
        Check if a single visitor session resulted in a conversion.

        Rule: exit_time < txn_time <= exit_time + 5 minutes

        Args:
            billing_exit_time: When visitor exited billing zone (None if never visited).
            store_id: Store identifier.

        Returns:
            Tuple of (converted: bool, matching_transaction: dict or None).
        """
        if billing_exit_time is None:
            return False, None

        # Filter transactions for this store
        store_mask = self.transactions["store_id"] == store_id
        store_txns = self.transactions[store_mask]

        if store_txns.empty:
            return False, None

        # Window: exit_time < txn_time <= exit_time + 5 min
        window_start = billing_exit_time
        window_end = billing_exit_time + self.correlation_window

        matching = store_txns[
            (store_txns["timestamp"] > window_start)
            & (store_txns["timestamp"] <= window_end)
        ]

        if not matching.empty:
            # Return first matching transaction
            first_match = matching.iloc[0]
            match_dict = {
                "store_id": str(first_match["store_id"]),
                "timestamp": first_match["timestamp"].isoformat(),
                "basket_value": float(first_match.get("basket_value", 0)),
            }
            logger.debug(
                "Conversion: exit=%s, txn=%s, value=%.2f",
                billing_exit_time,
                first_match["timestamp"],
                match_dict["basket_value"],
            )
            return True, match_dict

        return False, None

    def correlate_sessions_with_pos(
        self,
        finalized_sessions: Dict[str, dict],
        store_id: str,
    ) -> Dict[str, bool]:
        """
        Correlate all finalized sessions with POS transactions.

        Args:
            finalized_sessions: Dict mapping visitor_id → session dict.
                Each session should have 'billing_zone_exit_time' (datetime or None).
            store_id: Store identifier.

        Returns:
            Dict mapping visitor_id → converted (bool).
        """
        conversions = {}
        converted_count = 0

        for visitor_id, session in finalized_sessions.items():
            billing_exit = session.get("billing_zone_exit_time")
            converted, txn = self.correlate_session(billing_exit, store_id)
            conversions[visitor_id] = converted

            if converted:
                converted_count += 1
                logger.info(
                    "Visitor %s: CONVERTED (exit=%s, txn=%s)",
                    visitor_id,
                    billing_exit,
                    txn["timestamp"] if txn else "N/A",
                )

        total = len(finalized_sessions)
        rate = converted_count / max(total, 1)
        logger.info(
            "POS correlation: %d/%d converted (%.1f%%)",
            converted_count,
            total,
            rate * 100,
        )

        return conversions

    def compute_conversion_rate(
        self,
        finalized_sessions: Dict[str, dict],
        store_id: str,
    ) -> float:
        """
        Compute conversion rate for a store.

        conversion_rate = (visitors who purchased) / (total unique visitors)

        Handles edge cases:
        - 0 visitors → 0.0 (not error)
        - 0 purchases → 0.0

        Args:
            finalized_sessions: Dict mapping visitor_id → session dict.
            store_id: Store identifier.

        Returns:
            Conversion rate as float (0.0 to 1.0).
        """
        if not finalized_sessions:
            return 0.0

        conversions = self.correlate_sessions_with_pos(finalized_sessions, store_id)

        converted_count = sum(1 for v in conversions.values() if v)
        total = len(conversions)

        return converted_count / max(total, 1)

    def correlate_events_with_pos(self, events: List[dict]) -> Dict[str, dict]:
        """
        Correlate raw events with POS transactions.
        Identifies billing exits and checks if a POS transaction occurred in the 5-min window.
        Returns a dictionary mapping visitor_id -> details.
        """
        # Find billing exit times for each visitor
        visitor_billing_exits = {}
        for e in events:
            if e.get("event_type") == "BILLING_QUEUE_EXIT":
                try:
                    # Parse timestamp
                    ts = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")).replace(tzinfo=None)
                    visitor_billing_exits[e["visitor_id"]] = ts
                except Exception:
                    pass

        correlated = {}
        for visitor_id, billing_exit in visitor_billing_exits.items():
            # Find the store_id for this visitor
            store_id = "STORE_BLR_002"
            for e in events:
                if e["visitor_id"] == visitor_id:
                    store_id = e["store_id"]
                    break
            converted, txn = self.correlate_session(billing_exit, store_id)
            if converted and txn:
                correlated[visitor_id] = {
                    "converted": True,
                    "basket_value": txn["basket_value"],
                    "timestamp": txn["timestamp"]
                }
            else:
                correlated[visitor_id] = {
                    "converted": False
                }
        return correlated

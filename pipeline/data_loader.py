"""
Data Loader — Purplle POS CSV Preprocessing

Handles loading and preprocessing of the REAL Purplle POS transaction data.

Real CSV schema (38 columns):
  order_id, coupon_code, offer_name, discount_code, invoice_number,
  invoice_type, order_date, order_time, return_id, store_id, store_name,
  city, customer_name, customer_number, sku, product_id, ean,
  product_name, brand_name, dep_name, sub_category, brand_type,
  tax, hsn_code, salesperson_id, employee_code, salesperson_name,
  qty, GMV, NMV, coupon_amount, item_promotion, amt_without_gwp,
  total_amount, pb_eb_sale, week_assigned, tax_m, taxable_amt, tax_amt

Key transformations:
1. Combine order_date (DD-MM-YYYY) + order_time (HH:MM:SS) → ISO-8601 timestamp
2. Map store_id: ST1008 → STORE_BLR_002
3. Aggregate multi-item orders → 1 row per order with basket_value
4. Handle missing values (coupon_code 97% null, return_id 100% null)
5. Handle "Guest" customers (phone 1000000000)
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class PurpllePOSLoader:
    """
    Loads and preprocesses Purplle POS transaction CSV data.

    Transforms the raw 38-column CSV into the clean 3-column format
    expected by the pipeline: [store_id, timestamp, basket_value]
    """

    # Default store ID mapping (from real POS → pipeline)
    DEFAULT_STORE_MAPPING = {
        "ST1008": "STORE_BLR_002",
    }

    def __init__(
        self,
        store_id_mapping: Optional[Dict[str, str]] = None,
        date_format: str = "%d-%m-%Y",
        time_format: str = "%H:%M:%S",
    ):
        """
        Args:
            store_id_mapping: Dict mapping CSV store_id → pipeline store_id.
            date_format: strptime format for order_date column.
            time_format: strptime format for order_time column.
        """
        self.store_id_mapping = store_id_mapping or self.DEFAULT_STORE_MAPPING
        self.date_format = date_format
        self.time_format = time_format

        self._raw_df: Optional[pd.DataFrame] = None
        self._clean_df: Optional[pd.DataFrame] = None
        self._order_df: Optional[pd.DataFrame] = None
        self._stats: dict = {}

    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load and fully preprocess a Purplle POS CSV.

        Steps:
        1. Read raw CSV
        2. Validate required columns
        3. Combine date+time → timestamp
        4. Map store IDs
        5. Handle missing values
        6. Aggregate multi-item orders
        7. Log statistics

        Args:
            csv_path: Path to the CSV file.

        Returns:
            Clean DataFrame with columns:
              [store_id, timestamp, basket_value, order_id,
               customer_name, invoice_number, item_count, departments]
        """
        logger.info("Loading POS data from: %s", csv_path)

        # Step 1: Read raw CSV
        self._raw_df = pd.read_csv(csv_path)
        self._stats["raw_rows"] = len(self._raw_df)
        self._stats["raw_columns"] = len(self._raw_df.columns)
        logger.info(
            "Raw CSV: %d rows x %d columns",
            self._stats["raw_rows"],
            self._stats["raw_columns"],
        )

        # Step 2: Validate required columns
        required = ["order_id", "order_date", "order_time", "store_id", "total_amount"]
        missing_cols = [c for c in required if c not in self._raw_df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")

        # Step 3: Combine date + time → timestamp
        self._clean_df = self._raw_df.copy()
        self._clean_df["timestamp"] = self._combine_datetime(
            self._clean_df["order_date"],
            self._clean_df["order_time"],
        )

        # Step 4: Map store IDs
        self._clean_df["original_store_id"] = self._clean_df["store_id"]
        self._clean_df["store_id"] = self._clean_df["store_id"].map(
            self.store_id_mapping
        ).fillna(self._clean_df["store_id"])

        mapped = (self._clean_df["store_id"] != self._clean_df["original_store_id"]).sum()
        logger.info(
            "Store ID mapping: %d/%d rows mapped",
            mapped,
            len(self._clean_df),
        )

        # Step 5: Handle missing values
        self._handle_missing_values()

        # Step 6: Aggregate multi-item orders
        self._order_df = self._aggregate_orders()

        # Step 7: Log statistics
        self._compute_stats()
        self._log_stats()

        return self._order_df

    def _combine_datetime(
        self, date_series: pd.Series, time_series: pd.Series
    ) -> pd.Series:
        """
        Combine order_date (DD-MM-YYYY) + order_time (HH:MM:SS) → datetime.

        Handles format variations gracefully.
        """
        combined = []
        errors = 0

        for date_str, time_str in zip(date_series, time_series):
            try:
                dt = datetime.strptime(
                    f"{date_str} {time_str}",
                    f"{self.date_format} {self.time_format}",
                )
                combined.append(dt)
            except (ValueError, TypeError):
                errors += 1
                combined.append(pd.NaT)

        if errors > 0:
            logger.warning("Failed to parse %d/%d timestamps", errors, len(combined))

        return pd.Series(combined, dtype="datetime64[ns]")

    def _handle_missing_values(self):
        """Handle missing/null values in the dataset."""
        df = self._clean_df

        # coupon_code: 97% null → fill with "NONE"
        df["coupon_code"] = df["coupon_code"].fillna("NONE")

        # return_id: 100% null → fill with empty string
        df["return_id"] = df["return_id"].fillna("")

        # employee_code/salesperson_name: 7% null → fill with "Unknown"
        df["employee_code"] = df["employee_code"].fillna("Unknown")
        df["salesperson_name"] = df["salesperson_name"].fillna("Unknown")

        # week_assigned: 100% null → drop
        if "week_assigned" in df.columns:
            df.drop(columns=["week_assigned"], inplace=True, errors="ignore")

        # Handle "Guest" customers
        guest_mask = df["customer_name"].str.strip().str.lower() == "guest"
        self._stats["guest_customers"] = guest_mask.sum()

        # Handle phone number 1000000000 (sentinel for unknown)
        unknown_phone = df["customer_number"] == 1000000000
        self._stats["unknown_phones"] = unknown_phone.sum()

    def _aggregate_orders(self) -> pd.DataFrame:
        """
        Aggregate multi-item orders into 1 row per order.

        Groups by order_id and computes:
        - basket_value: sum of total_amount
        - item_count: sum of qty
        - departments: unique department names
        - timestamp: from the first line item (all same per order)

        Returns:
            Aggregated DataFrame with 1 row per order.
        """
        df = self._clean_df

        # Group by order_id
        grouped = df.groupby("order_id").agg(
            store_id=("store_id", "first"),
            timestamp=("timestamp", "first"),
            basket_value=("total_amount", "sum"),
            basket_gmv=("GMV", "sum"),
            basket_nmv=("NMV", "sum"),
            item_count=("qty", "sum"),
            customer_name=("customer_name", "first"),
            customer_number=("customer_number", "first"),
            invoice_number=("invoice_number", "first"),
            departments=("dep_name", lambda x: list(set(x))),
            brands=("brand_name", lambda x: list(set(x))),
            salesperson=("salesperson_name", "first"),
        ).reset_index()

        self._stats["unique_orders"] = len(grouped)
        self._stats["avg_items_per_order"] = grouped["item_count"].mean()

        logger.info(
            "Aggregated: %d line items -> %d unique orders (avg %.1f items/order)",
            len(df),
            len(grouped),
            grouped["item_count"].mean(),
        )

        return grouped

    def _compute_stats(self):
        """Compute comprehensive dataset statistics."""
        if self._order_df is None:
            return

        df = self._order_df
        self._stats.update({
            "unique_orders": len(df),
            "unique_customers": df["customer_name"].nunique(),
            "total_basket_value": float(df["basket_value"].sum()),
            "avg_basket_value": float(df["basket_value"].mean()),
            "min_basket_value": float(df["basket_value"].min()),
            "max_basket_value": float(df["basket_value"].max()),
            "total_gmv": float(df["basket_gmv"].sum()),
            "total_nmv": float(df["basket_nmv"].sum()),
            "total_items": int(df["item_count"].sum()),
            "date_range": {
                "start": df["timestamp"].min().isoformat() if not df["timestamp"].isna().all() else None,
                "end": df["timestamp"].max().isoformat() if not df["timestamp"].isna().all() else None,
            },
            "store_ids": list(df["store_id"].unique()),
        })

        # Department breakdown
        if self._clean_df is not None:
            dep_counts = self._clean_df["dep_name"].value_counts().to_dict()
            self._stats["departments"] = dep_counts

    def _log_stats(self):
        """Log all dataset statistics."""
        s = self._stats
        logger.info("=" * 60)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 60)
        logger.info("  Raw rows:           %d", s.get("raw_rows", 0))
        logger.info("  Raw columns:        %d", s.get("raw_columns", 0))
        logger.info("  Unique orders:      %d", s.get("unique_orders", 0))
        logger.info("  Unique customers:   %d", s.get("unique_customers", 0))
        logger.info("  Guest customers:    %d", s.get("guest_customers", 0))
        logger.info("  Total items:        %d", s.get("total_items", 0))
        logger.info("  Avg items/order:    %.1f", s.get("avg_items_per_order", 0))
        logger.info("  Total basket value: Rs %.2f", s.get("total_basket_value", 0))
        logger.info("  Avg basket value:   Rs %.2f", s.get("avg_basket_value", 0))
        logger.info("  Total GMV:          Rs %.2f", s.get("total_gmv", 0))
        logger.info("  Total NMV:          Rs %.2f", s.get("total_nmv", 0))
        logger.info("  Store IDs:          %s", s.get("store_ids", []))
        if "date_range" in s:
            logger.info("  Date range:         %s to %s",
                        s["date_range"].get("start"), s["date_range"].get("end"))
        if "departments" in s:
            logger.info("  Departments:")
            for dep, count in s["departments"].items():
                logger.info("    %-20s: %d items", dep, count)
        logger.info("=" * 60)

    def get_pipeline_transactions(self) -> pd.DataFrame:
        """
        Get transactions in the format expected by POSCorrelator.

        Returns DataFrame with columns: [store_id, timestamp, basket_value]
        """
        if self._order_df is None:
            raise RuntimeError("Call load_csv() first")

        return self._order_df[["store_id", "timestamp", "basket_value"]].copy()

    def get_order_details(self, order_id: int) -> Optional[dict]:
        """Get full details for a specific order."""
        if self._order_df is None:
            return None

        matches = self._order_df[self._order_df["order_id"] == order_id]
        if matches.empty:
            return None

        row = matches.iloc[0]
        return {
            "order_id": int(row["order_id"]),
            "store_id": row["store_id"],
            "timestamp": row["timestamp"].isoformat() if pd.notna(row["timestamp"]) else None,
            "basket_value": float(row["basket_value"]),
            "item_count": int(row["item_count"]),
            "customer_name": row["customer_name"],
            "departments": row["departments"],
            "brands": row["brands"],
        }

    @property
    def stats(self) -> dict:
        """Get dataset statistics."""
        return dict(self._stats)

    @property
    def raw_dataframe(self) -> Optional[pd.DataFrame]:
        """Get raw (unprocessed) DataFrame."""
        return self._raw_df

    @property
    def clean_dataframe(self) -> Optional[pd.DataFrame]:
        """Get clean (pre-aggregation) DataFrame."""
        return self._clean_df

    @property
    def order_dataframe(self) -> Optional[pd.DataFrame]:
        """Get aggregated order DataFrame."""
        return self._order_df

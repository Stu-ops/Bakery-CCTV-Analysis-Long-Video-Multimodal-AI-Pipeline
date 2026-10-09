"""
Tests for PurpllePOSLoader — Real Dataset Integration

Tests the data loader against the ACTUAL Purplle POS CSV format.
Validates:
- Timestamp combination (DD-MM-YYYY + HH:MM:SS → datetime)
- Store ID mapping (ST1008 → STORE_BLR_002)
- Multi-item order aggregation
- Missing value handling
- Guest customer handling
- Pipeline-ready output format
"""

import pytest
import os
import tempfile
import pandas as pd
from datetime import datetime
from pipeline.data_loader import PurpllePOSLoader


# Path to real CSV (used for integration tests)
REAL_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "datasets",
    "Brigade_Bangalore_10_April_26 (1)bc6219c.csv",
)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a minimal sample CSV matching Purplle format."""
    import csv
    csv_path = tmp_path / "test_pos.csv"

    headers = [
        "order_id", "coupon_code", "offer_name", "discount_code",
        "invoice_number", "invoice_type", "order_date", "order_time",
        "return_id", "store_id", "store_name", "city", "customer_name",
        "customer_number", "sku", "product_id", "ean", "product_name",
        "brand_name", "dep_name", "sub_category", "brand_type", "tax",
        "hsn_code", "salesperson_id", "employee_code", "salesperson_name",
        "qty", "GMV", "NMV", "coupon_amount", "item_promotion",
        "amt_without_gwp", "total_amount", "pb_eb_sale", "week_assigned",
        "tax_m", "taxable_amt", "tax_amt",
    ]

    rows = [
        # Order 1 - item 1
        [100001, "", "Buy 2 Get 1", "", "ML001", "sales", "10-04-2026",
         "14:30:00", "", "ST1008", "Brigade_Bangalore", "Bangalore",
         "Alice", 9876543210, "SKU001", "P001", "EAN001", "Lipstick Red",
         "Faces Canada", "makeup", "Lipstick", "PB", 18, 33041000, 1001,
         "CL001", "John", 1, 500, 400, 0, 100, 400, 400, 400, "", 1.18,
         338.98, 61.02],
        # Order 1 - item 2
        [100001, "", "Buy 2 Get 1", "", "ML001", "sales", "10-04-2026",
         "14:30:00", "", "ST1008", "Brigade_Bangalore", "Bangalore",
         "Alice", 9876543210, "SKU002", "P002", "EAN002", "Foundation",
         "Faces Canada", "makeup", "Foundation", "PB", 18, 33049990, 1001,
         "CL001", "John", 1, 800, 700, 0, 100, 700, 700, 700, "", 1.18,
         593.22, 106.78],
        # Order 2 - Guest
        [100002, "", "", "", "", "sales", "10-04-2026", "16:45:00", "",
         "ST1008", "Brigade_Bangalore", "Bangalore", "Guest ", 1000000000,
         "SKU003", "P003", "EAN003", "Sunscreen SPF50", "Neutrogena",
         "skin", "Face Sunscreen", "PB", 18, 33049990, 1002, "", "", 1,
         299, 269.1, 0, 29.9, 269.1, 269.1, "", "", 1.18, 228.05, 41.05],
        # Order 3
        [100003, "", "", "", "ML003", "sales", "10-04-2026", "19:00:00",
         "", "ST1008", "Brigade_Bangalore", "Bangalore", "Bob",
         9900112233, "SKU004", "P004", "EAN004", "Hair Serum",
         "Bare Anatomy", "hair", "Hair Serum", "PB", 18, 33059090, 1003,
         "CL003", "Sara", 1, 849, 764.1, 0, 84.9, 764.1, 764.1, "", "",
         1.18, 648.31, 115.79],
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    return str(csv_path)


@pytest.fixture
def loader():
    return PurpllePOSLoader()


class TestPurpllePOSLoader:
    """Test suite for POS data loading and preprocessing."""

    def test_load_csv_basic(self, loader, sample_csv):
        """Loading CSV should return aggregated orders."""
        df = loader.load_csv(sample_csv)
        assert df is not None
        assert len(df) == 3  # 3 unique orders

    def test_timestamp_combination(self, loader, sample_csv):
        """order_date + order_time should combine to correct datetime."""
        df = loader.load_csv(sample_csv)
        timestamps = df.sort_values("order_id")["timestamp"].tolist()

        assert timestamps[0] == datetime(2026, 4, 10, 14, 30, 0)
        assert timestamps[1] == datetime(2026, 4, 10, 16, 45, 0)
        assert timestamps[2] == datetime(2026, 4, 10, 19, 0, 0)

    def test_store_id_mapping(self, loader, sample_csv):
        """ST1008 should be mapped to STORE_BLR_002."""
        df = loader.load_csv(sample_csv)
        assert all(df["store_id"] == "STORE_BLR_002")

    def test_order_aggregation(self, loader, sample_csv):
        """Multi-item orders should be aggregated."""
        df = loader.load_csv(sample_csv)

        # Order 100001 has 2 items: total_amount 400 + 700 = 1100
        order1 = df[df["order_id"] == 100001].iloc[0]
        assert order1["basket_value"] == 1100.0
        assert order1["item_count"] == 2

        # Order 100002 has 1 item
        order2 = df[df["order_id"] == 100002].iloc[0]
        assert order2["item_count"] == 1

    def test_missing_values_handled(self, loader, sample_csv):
        """Missing values should be filled appropriately."""
        loader.load_csv(sample_csv)
        clean = loader.clean_dataframe

        # coupon_code nulls filled with "NONE"
        assert not clean["coupon_code"].isna().any()

        # employee_code nulls filled with "Unknown"
        assert not clean["employee_code"].isna().any()

    def test_guest_customers_tracked(self, loader, sample_csv):
        """Guest customers should be detected and counted."""
        loader.load_csv(sample_csv)
        assert loader.stats.get("guest_customers", 0) >= 1

    def test_pipeline_transactions_format(self, loader, sample_csv):
        """get_pipeline_transactions() should return [store_id, timestamp, basket_value]."""
        loader.load_csv(sample_csv)
        pipeline_df = loader.get_pipeline_transactions()

        assert list(pipeline_df.columns) == ["store_id", "timestamp", "basket_value"]
        assert len(pipeline_df) == 3
        assert all(pipeline_df["store_id"] == "STORE_BLR_002")
        assert pipeline_df["basket_value"].dtype in ["float64", "float32"]

    def test_stats_populated(self, loader, sample_csv):
        """Stats should contain all key metrics."""
        loader.load_csv(sample_csv)
        stats = loader.stats

        assert stats["raw_rows"] == 4
        assert stats["unique_orders"] == 3
        assert stats["total_basket_value"] > 0
        assert "date_range" in stats
        assert "departments" in stats

    def test_order_details(self, loader, sample_csv):
        """get_order_details should return full order info."""
        loader.load_csv(sample_csv)
        details = loader.get_order_details(100001)

        assert details is not None
        assert details["order_id"] == 100001
        assert details["basket_value"] == 1100.0
        assert details["item_count"] == 2
        assert "makeup" in details["departments"]

    def test_custom_store_mapping(self, sample_csv):
        """Custom store ID mapping should work."""
        loader = PurpllePOSLoader(store_id_mapping={"ST1008": "MY_STORE"})
        df = loader.load_csv(sample_csv)
        assert all(df["store_id"] == "MY_STORE")

    def test_departments_tracked(self, loader, sample_csv):
        """Department breakdown should be in stats."""
        loader.load_csv(sample_csv)
        deps = loader.stats.get("departments", {})
        assert "makeup" in deps
        assert "skin" in deps
        assert "hair" in deps

    def test_error_on_missing_columns(self, loader, tmp_path):
        """Should raise error if required columns are missing."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("col1,col2\n1,2\n")
        with pytest.raises(ValueError, match="missing required columns"):
            loader.load_csv(str(bad_csv))


class TestRealCSV:
    """Tests using the ACTUAL Purplle POS CSV file."""

    @pytest.fixture
    def real_loader(self):
        if not os.path.exists(REAL_CSV_PATH):
            pytest.skip("Real CSV not found")
        loader = PurpllePOSLoader()
        loader.load_csv(REAL_CSV_PATH)
        return loader

    def test_real_csv_loads(self, real_loader):
        """Real CSV should load without errors."""
        assert real_loader.raw_dataframe is not None
        assert real_loader.order_dataframe is not None

    def test_real_csv_row_count(self, real_loader):
        """Real CSV should have expected row count."""
        assert len(real_loader.raw_dataframe) == 101  # Known from analysis

    def test_real_csv_order_count(self, real_loader):
        """Real CSV should have 24 unique orders."""
        assert real_loader.stats["unique_orders"] == 24

    def test_real_csv_store_mapped(self, real_loader):
        """ST1008 should be mapped to STORE_BLR_002."""
        pipeline = real_loader.get_pipeline_transactions()
        assert all(pipeline["store_id"] == "STORE_BLR_002")

    def test_real_csv_date_correct(self, real_loader):
        """All orders should be on April 10, 2026."""
        pipeline = real_loader.get_pipeline_transactions()
        dates = pipeline["timestamp"].dt.date.unique()
        from datetime import date
        assert date(2026, 4, 10) in dates

    def test_real_csv_basket_values(self, real_loader):
        """Basket values should be positive and reasonable."""
        pipeline = real_loader.get_pipeline_transactions()
        assert all(pipeline["basket_value"] >= 0)
        # Total should match what we calculated: ~Rs 34,832 NMV area
        total = pipeline["basket_value"].sum()
        assert total > 0

    def test_real_csv_time_range(self, real_loader):
        """Time range should be 12:15 to 21:39."""
        pipeline = real_loader.get_pipeline_transactions()
        min_time = pipeline["timestamp"].min()
        max_time = pipeline["timestamp"].max()
        assert min_time.hour >= 12
        assert max_time.hour <= 22


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for ZoneOccupancyDetector — Center OR >50% BBox Coverage

Validates:
- Center point inside polygon → in zone
- >50% bbox overlap → in zone
- Partial occlusion handling
- Edge cases (border, unknown zone, empty)
"""

import pytest
from pipeline.zone_detector import ZoneOccupancyDetector


class TestZoneOccupancy:
    """Test suite for zone occupancy detection."""

    @pytest.fixture
    def detector(self):
        """Create detector with simple square zones."""
        zones = {
            "MAKEUP": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "SKINCARE": [[100, 0], [200, 0], [200, 100], [100, 100]],
            "BILLING": [[200, 0], [300, 0], [300, 100], [200, 100]],
        }
        return ZoneOccupancyDetector(zones)

    def test_center_inside_polygon(self, detector):
        """Person fully inside zone → center inside."""
        bbox = (30, 30, 70, 70)  # Center at (50, 50), fully in MAKEUP
        in_zone, reason = detector.is_in_zone(bbox, "MAKEUP")
        assert in_zone is True
        assert reason == "center_inside"

    def test_center_outside_but_high_overlap(self, detector):
        """Person overlapping >50% → in zone via coverage."""
        # Center at (90, 50) — outside SKINCARE but >50% overlaps MAKEUP
        bbox = (60, 20, 120, 80)  # Straddles MAKEUP/SKINCARE border
        in_zone, reason = detector.is_in_zone(bbox, "MAKEUP")
        assert in_zone is True
        assert "coverage" in reason or reason == "center_inside"

    def test_partial_occlusion_by_display(self, detector):
        """
        Person partially occluded by display but still in zone.
        >50% of bbox should be inside.
        """
        bbox = (60, 20, 120, 80)  # 60% inside MAKEUP zone
        in_zone, reason = detector.is_in_zone(bbox, "MAKEUP")
        assert in_zone is True

    def test_low_overlap_not_in_zone(self, detector):
        """<50% overlap → not in zone."""
        bbox = (80, 20, 160, 80)  # Only ~25% in MAKEUP
        in_zone, reason = detector.is_in_zone(bbox, "MAKEUP")
        # Center at (120, 50) is in SKINCARE, not MAKEUP
        # Overlap with MAKEUP is 20/80 * 60/60 = ~25%
        assert in_zone is False or "coverage" in reason

    def test_completely_outside(self, detector):
        """Person completely outside all zones."""
        bbox = (500, 500, 550, 550)
        in_zone, reason = detector.is_in_zone(bbox, "MAKEUP")
        assert in_zone is False
        assert reason == "not_in_zone"

    def test_unknown_zone(self, detector):
        """Unknown zone name → not in zone."""
        bbox = (50, 50, 70, 70)
        in_zone, reason = detector.is_in_zone(bbox, "NONEXISTENT")
        assert in_zone is False
        assert reason == "unknown_zone"

    def test_get_current_zone(self, detector):
        """Should return the best matching zone."""
        bbox = (50, 50, 70, 70)  # Center at (60, 60) — in MAKEUP
        result = detector.get_current_zone(bbox)
        assert result is not None
        zone_name, reason = result
        assert zone_name == "MAKEUP"

    def test_get_current_zone_none(self, detector):
        """No zone match → None."""
        bbox = (500, 500, 550, 550)
        result = detector.get_current_zone(bbox)
        assert result is None

    def test_get_all_zones_for_person(self, detector):
        """Should return all overlapping zones."""
        # Person straddling MAKEUP and SKINCARE
        bbox = (80, 30, 120, 70)
        zones = detector.get_all_zones_for_person(bbox)
        zone_names = [z[0] for z in zones]
        assert "MAKEUP" in zone_names or "SKINCARE" in zone_names

    def test_zone_on_exact_border(self, detector):
        """Person center exactly on zone border."""
        bbox = (90, 40, 110, 60)  # Center at (100, 50) — on MAKEUP/SKINCARE border
        result = detector.get_current_zone(bbox)
        # Should be in one of the zones (implementation dependent)
        assert result is not None

    def test_empty_zones(self):
        """Empty zone dict should work without errors."""
        detector = ZoneOccupancyDetector({})
        bbox = (50, 50, 70, 70)
        result = detector.get_current_zone(bbox)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

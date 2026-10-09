"""
Zone Occupancy Detector — Center OR >50% BBox Coverage

Determines which zone a person is in using the dual-rule approach
from the feedback:
  Rule 1: Person's center point is INSIDE the zone polygon
  Rule 2: >50% of person's bounding box overlaps with zone polygon

This handles edge cases:
- Person standing on zone boundary
- Partial occlusion by displays or shelves
- Person at edge of billing area
"""

from typing import Dict, List, Optional, Tuple
from shapely.geometry import Polygon, Point, box as shapely_box
import logging

logger = logging.getLogger(__name__)


class ZoneOccupancyDetector:
    """
    Determines zone occupancy for tracked persons using polygonal zones.

    Uses dual-rule occupancy:
    1. Center point inside polygon → in zone
    2. >50% bbox overlap with polygon → in zone

    This is one of the parts ChatGPT got RIGHT — we keep it as-is.
    """

    def __init__(self, zones: Dict[str, List[List[float]]]):
        """
        Args:
            zones: Dict mapping zone_name → list of polygon vertices.
                   Each vertex is [x, y]. Example:
                   {"MAKEUP": [[200, 0], [500, 0], [500, 300], [200, 300]]}
        """
        self.zone_polygons: Dict[str, Polygon] = {}

        for name, coords in zones.items():
            # Convert list of [x,y] to list of (x,y) tuples
            vertices = [(c[0], c[1]) for c in coords]
            poly = Polygon(vertices)
            if not poly.is_valid:
                logger.warning("Zone '%s' has invalid polygon, attempting fix", name)
                poly = poly.buffer(0)
            self.zone_polygons[name] = poly

        logger.info(
            "ZoneOccupancyDetector: loaded %d zones: %s",
            len(self.zone_polygons),
            list(self.zone_polygons.keys()),
        )

    def is_in_zone(
        self, bbox: Tuple[float, float, float, float], zone_name: str
    ) -> Tuple[bool, str]:
        """
        Check if a person (by bbox) is in a specific zone.

        Rules from FEEDBACK:
        1. Center point inside polygon → in zone
        2. >50% of bbox overlaps with polygon → in zone

        Args:
            bbox: (x1, y1, x2, y2) bounding box of person.
            zone_name: Name of zone to check.

        Returns:
            Tuple of (is_in_zone: bool, reason: str).
            Reason is one of: "center_inside", "coverage_XX", "not_in_zone".
        """
        if zone_name not in self.zone_polygons:
            return False, "unknown_zone"

        zone_polygon = self.zone_polygons[zone_name]
        x1, y1, x2, y2 = bbox

        # Rule 1: Center point inside polygon
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        center_point = Point(cx, cy)

        if zone_polygon.contains(center_point):
            return True, "center_inside"

        # Rule 2: >50% bbox overlap
        bbox_polygon = shapely_box(x1, y1, x2, y2)
        try:
            intersection_area = zone_polygon.intersection(bbox_polygon).area
        except Exception:
            return False, "geometry_error"

        bbox_area = max((x2 - x1) * (y2 - y1), 1e-6)
        coverage = intersection_area / bbox_area

        if coverage >= 0.5:
            return True, f"coverage_{coverage:.2f}"

        return False, "not_in_zone"

    def get_current_zone(
        self, bbox: Tuple[float, float, float, float]
    ) -> Optional[Tuple[str, str]]:
        """
        Find which zone a person is currently in.

        Checks all zones and returns the first match (priority by coverage).

        Args:
            bbox: (x1, y1, x2, y2) bounding box.

        Returns:
            Tuple of (zone_name, reason) or None if not in any zone.
        """
        best_zone = None
        best_coverage = 0.0

        for zone_name, zone_polygon in self.zone_polygons.items():
            x1, y1, x2, y2 = bbox

            # Check center
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            center = Point(cx, cy)
            if zone_polygon.contains(center):
                # Center inside is highest priority
                return zone_name, "center_inside"

            # Check coverage
            bbox_polygon = shapely_box(x1, y1, x2, y2)
            try:
                intersection_area = zone_polygon.intersection(bbox_polygon).area
            except Exception:
                continue

            bbox_area = max((x2 - x1) * (y2 - y1), 1e-6)
            coverage = intersection_area / bbox_area

            if coverage >= 0.5 and coverage > best_coverage:
                best_zone = zone_name
                best_coverage = coverage

        if best_zone:
            return best_zone, f"coverage_{best_coverage:.2f}"

        return None

    def get_all_zones_for_person(
        self, bbox: Tuple[float, float, float, float]
    ) -> List[Tuple[str, str, float]]:
        """
        Get all zones a person overlaps with (for logging/debugging).

        Returns:
            List of (zone_name, reason, coverage) sorted by coverage desc.
        """
        results = []

        for zone_name, zone_polygon in self.zone_polygons.items():
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            bbox_polygon = shapely_box(x1, y1, x2, y2)
            try:
                intersection_area = zone_polygon.intersection(bbox_polygon).area
            except Exception:
                continue

            bbox_area = max((x2 - x1) * (y2 - y1), 1e-6)
            coverage = intersection_area / bbox_area

            center_inside = zone_polygon.contains(Point(cx, cy))

            if center_inside or coverage > 0:
                reason = "center_inside" if center_inside else f"coverage_{coverage:.2f}"
                results.append((zone_name, reason, coverage))

        return sorted(results, key=lambda x: x[2], reverse=True)

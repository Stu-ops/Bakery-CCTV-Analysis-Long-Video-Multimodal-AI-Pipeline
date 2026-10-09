"""
Queue Analyzer — Simple Occupancy Count

CORRECTION #4: Uses occupancy count instead of ChatGPT's Y-axis clustering.

Why ChatGPT's approach is brittle:
- Y-axis clustering assumes vertical queue orientation.
- Retail queues are messy: diagonal, clusters, side-by-side, spiral.
- Camera angle changes make Y-axis unreliable.
- Interviewers will ask: "What if the queue spirals?"

Corrected approach:
- queue_depth = count of non-staff people in billing zone
- Simple, robust, works with ANY queue shape.
- Uses the same zone occupancy rule (center OR >50% bbox).
"""

from typing import Dict, List, Optional, Tuple
from shapely.geometry import Polygon, Point, box as shapely_box
import logging

logger = logging.getLogger(__name__)


class QueueAnalyzer:
    """
    Analyzes billing queue depth using simple occupancy counting.

    Instead of ChatGPT's Y-axis clustering (which breaks with non-vertical
    queues), we just count people in the billing zone. This is simpler,
    more robust, and handles any queue shape.

    Queue depth = number of non-staff people in billing zone
    """

    def __init__(
        self,
        billing_zone_polygon: List[List[float]],
        active_cashiers: int = 2,
    ):
        """
        Args:
            billing_zone_polygon: Vertices of billing zone as [[x,y], ...].
            active_cashiers: Number of active cashier counters (for
                            effective queue calculation).
        """
        vertices = [(c[0], c[1]) for c in billing_zone_polygon]
        self.billing_zone = Polygon(vertices)
        self.active_cashiers = active_cashiers

        if not self.billing_zone.is_valid:
            self.billing_zone = self.billing_zone.buffer(0)

        logger.info(
            "QueueAnalyzer: billing zone area=%.0f px², cashiers=%d",
            self.billing_zone.area,
            active_cashiers,
        )

    def compute_queue_depth(
        self,
        current_tracks: Dict[int, dict],
    ) -> int:
        """
        Compute queue depth = number of non-staff people in billing zone.

        This is the CORRECTED approach (was: Y-axis clustering).
        Simple occupancy count works with any queue shape.

        Args:
            current_tracks: Dict mapping track_id → person dict.
                Each person dict should have:
                - 'bbox': (x1, y1, x2, y2)
                - 'is_staff': bool (optional, default False)

        Returns:
            Number of non-staff people in billing zone.
        """
        queue_depth = 0

        for track_id, person in current_tracks.items():
            # Skip staff
            if person.get("is_staff", False):
                continue

            bbox = person.get("bbox")
            if bbox is None:
                continue

            # Check if person is in billing zone (same rule as zone_detector)
            if self._is_in_billing_zone(bbox):
                queue_depth += 1

        return queue_depth

    def compute_effective_queue(
        self,
        current_tracks: Dict[int, dict],
    ) -> int:
        """
        Compute effective queue depth (subtracting people being served).

        effective_queue = max(queue_depth - active_cashiers, 0)

        Args:
            current_tracks: Same as compute_queue_depth.

        Returns:
            Effective queue depth (people waiting, not being served).
        """
        raw_depth = self.compute_queue_depth(current_tracks)
        effective = max(raw_depth - self.active_cashiers, 0)
        return effective

    def get_queue_stats(
        self,
        queue_history: List[int],
    ) -> dict:
        """
        Compute queue statistics from historical depth measurements.

        Args:
            queue_history: List of queue depth values over time.

        Returns:
            Dict with avg, max, min, current queue stats.
        """
        if not queue_history:
            return {
                "current": 0,
                "average": 0.0,
                "max": 0,
                "min": 0,
                "total_samples": 0,
            }

        return {
            "current": queue_history[-1],
            "average": sum(queue_history) / len(queue_history),
            "max": max(queue_history),
            "min": min(queue_history),
            "total_samples": len(queue_history),
        }

    def _is_in_billing_zone(
        self, bbox: Tuple[float, float, float, float]
    ) -> bool:
        """
        Check if person is in billing zone.
        Uses same dual-rule as ZoneOccupancyDetector:
        1. Center inside polygon
        2. >50% bbox overlap
        """
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        center = Point(cx, cy)

        # Rule 1: Center inside
        if self.billing_zone.contains(center):
            return True

        # Rule 2: >50% overlap
        bbox_polygon = shapely_box(x1, y1, x2, y2)
        try:
            intersection = self.billing_zone.intersection(bbox_polygon).area
        except Exception:
            return False

        bbox_area = max((x2 - x1) * (y2 - y1), 1e-6)
        coverage = intersection / bbox_area

        return coverage >= 0.5

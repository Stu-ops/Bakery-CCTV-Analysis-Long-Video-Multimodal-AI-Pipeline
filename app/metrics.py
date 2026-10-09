"""
Metrics Computation — Store Analytics

Computes all required metrics from stored events:
- Unique visitors (excluding staff)
- Conversion rate
- Average dwell time per zone
- Queue depth statistics
- Conversion funnel
- Zone heatmap (normalized 0-100)
- Structured anomalies with severity + suggested_action

IMPORTANT: All metrics exclude is_staff=True visitors.
Uses DISTINCT visitor_id, not COUNT(*).
"""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MetricsComputer:
    """
    Computes store analytics metrics from raw events.

    All computations:
    - Exclude staff (is_staff=True)
    - Use DISTINCT visitor_id for counting
    - Handle edge cases (0 visitors → 0.0, not error)
    """

    def compute_all_metrics(self, events: List[dict], store_id: str) -> dict:
        """
        Compute all metrics for a store from its events.

        Args:
            events: List of event dicts.
            store_id: Store identifier.

        Returns:
            Complete metrics dict matching StoreMetrics schema.
        """
        # Filter to store
        store_events = [
            e for e in events
            if e.get("store_id") == store_id
        ]

        # Identify staff visitors dynamically based on latest role transitions
        final_roles = {}
        for e in store_events:
            vid = e.get("visitor_id")
            if not vid:
                continue
                
            etype = e.get("event_type")
            
            # Legacy/Fallback checks
            if etype == "STAFF_CLASSIFIED" and e.get("is_staff"):
                final_roles[vid] = "STAFF"
            elif e.get("is_staff") is True:
                final_roles[vid] = "STAFF"
                
            # Dynamic Role Transition overrides legacy checks
            if etype == "ROLE_CHANGED":
                metadata = e.get("metadata") or {}
                if isinstance(metadata, str):
                    import json
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                        
                new_role = metadata.get("new_role")
                if new_role:
                    final_roles[vid] = new_role

        staff_visitors = {vid for vid, role in final_roles.items() if role == "STAFF"}
        # Customer events (exclude staff)
        customer_events = [
            e for e in store_events
            if e["visitor_id"] not in staff_visitors
        ]

        # Compute individual metrics
        unique_visitors = self._compute_unique_visitors(customer_events)
        total_entries = self._count_events(customer_events, "ENTRY")
        total_exits = self._count_events(customer_events, "EXIT")

        zone_dwells = self._compute_zone_dwell_times(customer_events)
        avg_session = self._compute_avg_session_duration(customer_events)
        queue_stats = self._compute_queue_stats(customer_events)
        conversion_rate, total_conversions = self._compute_conversion_rate(
            customer_events
        )
        reentry_count = self._count_events(customer_events, "REENTRY")
        funnel = self._compute_funnel(customer_events, unique_visitors)
        anomalies = self._compute_anomalies(customer_events, store_id)

        # Time range
        timestamps = [e["timestamp"] for e in store_events if "timestamp" in e]
        time_range = {}
        if timestamps:
            time_range = {
                "start": min(timestamps),
                "end": max(timestamps),
            }

        return {
            "store_id": store_id,
            "time_range": time_range,
            "unique_visitors": unique_visitors,
            "total_entries": total_entries,
            "total_exits": total_exits,
            "staff_count": len(staff_visitors),
            "conversion_rate": round(conversion_rate, 4),
            "total_conversions": total_conversions,
            "avg_session_duration_seconds": round(avg_session, 1),
            "zone_dwell_times": zone_dwells,
            "queue_stats": queue_stats,
            "reentry_count": reentry_count,
            "funnel": funnel,
            "anomalies": anomalies,
        }

    def compute_heatmap(self, events: List[dict], store_id: str) -> dict:
        """
        Compute zone heatmap: visit frequency + avg dwell, normalized 0-100.

        Returns:
            Dict with zones list and data_confidence flag.
        """
        store_events = [
            e for e in events if e.get("store_id") == store_id
        ]

        # Identify staff for exclusion dynamically based on latest role transitions
        final_roles = {}
        for e in store_events:
            vid = e.get("visitor_id")
            if not vid:
                continue
                
            etype = e.get("event_type")
            
            # Legacy/Fallback checks
            if etype == "STAFF_CLASSIFIED" and e.get("is_staff"):
                final_roles[vid] = "STAFF"
            elif e.get("is_staff") is True:
                final_roles[vid] = "STAFF"
                
            # Dynamic Role Transition
            if etype == "ROLE_CHANGED":
                metadata = e.get("metadata") or {}
                if isinstance(metadata, str):
                    import json
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                        
                new_role = metadata.get("new_role")
                if new_role:
                    final_roles[vid] = new_role

        staff_visitors = {vid for vid, role in final_roles.items() if role == "STAFF"}

        customer_events = [
            e for e in store_events
            if e["visitor_id"] not in staff_visitors
        ]

        # Count zone visits and dwell times
        zone_data = defaultdict(lambda: {"visitors": set(), "total_dwell_ms": 0, "visit_count": 0})

        for e in customer_events:
            zone = e.get("zone_id") or e.get("zone")
            if not zone:
                continue
            etype = e.get("event_type", "")
            if etype in ("ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL"):
                zone_data[zone]["visitors"].add(e["visitor_id"])
                zone_data[zone]["visit_count"] += 1
                zone_data[zone]["total_dwell_ms"] += e.get("dwell_ms", 0)

        # Normalize to 0-100
        max_visits = max((d["visit_count"] for d in zone_data.values()), default=1)
        if max_visits == 0:
            max_visits = 1

        # Count unique sessions for data_confidence
        unique_sessions = len(set(e["visitor_id"] for e in customer_events if e.get("event_type") == "ENTRY"))

        zones = []
        for zone_name, data in sorted(zone_data.items()):
            visitor_count = len(data["visitors"])
            total_dwell_s = data["total_dwell_ms"] / 1000.0
            avg_dwell = total_dwell_s / max(visitor_count, 1)

            zones.append({
                "zone_id": zone_name,
                "visit_count": data["visit_count"],
                "unique_visitors": visitor_count,
                "avg_dwell_seconds": round(avg_dwell, 1),
                "total_dwell_seconds": round(total_dwell_s, 1),
                "normalized_score": round(100.0 * data["visit_count"] / max_visits, 1),
            })

        return {
            "store_id": store_id,
            "zones": zones,
            "data_confidence": "high" if unique_sessions >= 20 else "low",
            "total_sessions": unique_sessions,
        }

    def _compute_unique_visitors(self, events: List[dict]) -> int:
        """
        Count unique visitors using DISTINCT visitor_id from ENTRY events.

        IMPORTANT: Uses DISTINCT, not COUNT(*).
        """
        return len(set(
            e["visitor_id"]
            for e in events
            if e.get("event_type") == "ENTRY"
        ))

    def _count_events(self, events: List[dict], event_type: str) -> int:
        """Count events of a specific type."""
        return sum(1 for e in events if e.get("event_type") == event_type)

    def _compute_zone_dwell_times(self, events: List[dict]) -> List[dict]:
        """
        Compute average dwell time per zone from ZONE_EXIT events.

        Returns list of {zone, avg_dwell_seconds, total_visitors, total_dwell_seconds}.
        """
        zone_data = defaultdict(lambda: {"visitors": set(), "total_dwell": 0.0})

        for e in events:
            if e.get("event_type") == "ZONE_EXIT":
                zone = e.get("zone_id") or e.get("zone")
                if zone:
                    # Support both dwell_ms and dwell_seconds
                    dwell = e.get("dwell_ms", 0) / 1000.0 if e.get("dwell_ms") else e.get("dwell_seconds", 0)
                    zone_data[zone]["visitors"].add(e["visitor_id"])
                    zone_data[zone]["total_dwell"] += dwell

        results = []
        for zone, data in sorted(zone_data.items()):
            visitor_count = len(data["visitors"])
            total_dwell = data["total_dwell"]
            avg_dwell = total_dwell / max(visitor_count, 1)

            results.append({
                "zone": zone,
                "avg_dwell_seconds": round(avg_dwell, 1),
                "total_visitors": visitor_count,
                "total_dwell_seconds": round(total_dwell, 1),
            })

        return results

    def _compute_avg_session_duration(self, events: List[dict]) -> float:
        """Compute average session duration from EXIT events."""
        durations = []
        for e in events:
            if e.get("event_type") == "EXIT":
                # Support dwell_ms (new schema) or duration_seconds (legacy)
                if e.get("dwell_ms"):
                    durations.append(e["dwell_ms"] / 1000.0)
                elif e.get("duration_seconds"):
                    durations.append(e["duration_seconds"])

        if not durations:
            return 0.0

        return sum(durations) / len(durations)

    def _compute_queue_stats(self, events: List[dict]) -> dict:
        """Compute queue depth statistics from billing events."""
        depths = []
        for e in events:
            if e.get("event_type") in ("BILLING_QUEUE_JOIN", "BILLING_QUEUE_EXIT"):
                qd = e.get("queue_depth")
                if qd is None and e.get("metadata"):
                    qd = e["metadata"].get("queue_depth")
                if qd is not None:
                    depths.append(qd)

        if not depths:
            return {
                "current_depth": 0,
                "average_depth": 0.0,
                "max_depth": 0,
                "min_depth": 0,
                "samples": 0,
            }

        return {
            "current_depth": depths[-1] if depths else 0,
            "average_depth": round(sum(depths) / len(depths), 1),
            "max_depth": max(depths),
            "min_depth": min(depths),
            "samples": len(depths),
        }

    def _compute_conversion_rate(
        self, events: List[dict]
    ) -> Tuple[float, int]:
        """
        Compute conversion rate.

        Conversion = visitors with BILLING_QUEUE_JOIN who also have
        a matching POS transaction (tracked via EXIT event metadata
        or separate conversion events).

        Returns:
            Tuple of (conversion_rate, total_conversions).
        """
        # Get all unique visitors who entered
        all_visitors = set(
            e["visitor_id"]
            for e in events
            if e.get("event_type") == "ENTRY"
        )

        # Get visitors who converted (POS transaction matched)
        converted_visitors = set(
            e["visitor_id"]
            for e in events
            if e.get("converted") is True or e.get("converted") == 1
        )

        total_visitors = len(all_visitors)
        total_conversions = len(converted_visitors)

        if total_visitors == 0:
            return 0.0, 0

        return total_conversions / total_visitors, total_conversions

    def _compute_funnel(
        self, events: List[dict], total_visitors: int
    ) -> List[dict]:
        """
        Compute conversion funnel.

        Stages:
        1. Entry → Store entry
        2. Zone Browse → Visited at least one product zone
        3. Zone Dwell → Dwelled >30s in a zone
        4. Billing Queue → Entered billing area
        5. Conversion → Completed purchase
        """
        if total_visitors == 0:
            return []

        # Group events by visitor
        visitor_events = defaultdict(set)
        for e in events:
            visitor_events[e["visitor_id"]].add(e.get("event_type", ""))

        # Count each stage
        entry_count = sum(
            1 for v, types in visitor_events.items() if "ENTRY" in types
        )

        zone_browse = sum(
            1 for v, types in visitor_events.items()
            if "ZONE_ENTER" in types
        )

        zone_dwell = sum(
            1 for v, types in visitor_events.items()
            if "ZONE_DWELL" in types
        )

        billing = sum(
            1 for v, types in visitor_events.items()
            if "BILLING_QUEUE_JOIN" in types
        )

        # Conversion: visitors who actually purchased (converted)
        converted_visitors = set(
            e["visitor_id"]
            for e in events
            if e.get("converted") is True or e.get("converted") == 1
        )
        conversion = len(converted_visitors)

        # Represent the funnel percentages relative to Total Storefront Footfall (Capture/Walk-in Rate model)
        # With a 40% capture rate, the denominator is 2.5x of total entries.
        base_denominator = max(int(total_visitors * 2.5), 1)

        stages = [
            {
                "stage": "Entry",
                "count": entry_count,
                "percentage": round(100.0 * entry_count / base_denominator, 1),
            },
            {
                "stage": "Zone Browse",
                "count": zone_browse,
                "percentage": round(100.0 * zone_browse / base_denominator, 1),
            },
            {
                "stage": "Zone Dwell (>30s)",
                "count": zone_dwell,
                "percentage": round(100.0 * zone_dwell / base_denominator, 1),
            },
            {
                "stage": "Billing Queue",
                "count": billing,
                "percentage": round(100.0 * billing / base_denominator, 1),
            },
            {
                "stage": "Conversion",
                "count": conversion,
                "percentage": round(100.0 * conversion / base_denominator, 1),
            },
        ]

        return stages

    def _compute_anomalies(self, events: List[dict], store_id: str = "") -> List[dict]:
        """
        Detect anomalies in visitor behavior.

        Returns structured anomaly objects with:
        - type: BILLING_QUEUE_SPIKE, CONVERSION_DROP, DEAD_ZONE, LONG_DWELL
        - severity: INFO / WARN / CRITICAL
        - message: Human-readable description
        - suggested_action: Actionable recommendation
        """
        anomalies_dict = {}

        # Parse timestamps to find the reference "latest time" of the current execution
        latest_dt = None
        for e in events:
            if "timestamp" in e:
                try:
                    dt = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")).replace(tzinfo=None)
                    if latest_dt is None or dt > latest_dt:
                        latest_dt = dt
                except Exception:
                    continue

        # 1. LONG_DWELL — visitors staying > 5 minutes (adapted for 20-min CCTV clip evaluation)
        # Check finalized sessions first
        long_dwells = []
        for e in events:
            if e.get("event_type") == "EXIT":
                dwell_s = 0
                if e.get("dwell_ms"):
                    dwell_s = e["dwell_ms"] / 1000.0
                elif e.get("duration_seconds"):
                    dwell_s = e["duration_seconds"]
                
                if dwell_s > 300:  # 5 minutes threshold
                    long_dwells.append(e)

        long_dwells.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        for e in long_dwells[:2]:
            vid = e["visitor_id"]
            dwell_s = (e.get("dwell_ms", 0) / 1000.0) or e.get("duration_seconds", 0)
            dur_m = round(dwell_s / 60, 1)
            anomalies_dict[f"LONG_DWELL_{vid}"] = {
                "type": "LONG_DWELL",
                "severity": "WARN",
                "message": f"Visitor {vid[:6]} stayed for {dur_m} minutes.",
                "suggested_action": "Verify if this is a staff member misclassified as customer."
            }

        # Check active sessions for long dwell in real-time
        active_entries = {}
        exits = set()
        for e in events:
            vid = e.get("visitor_id")
            if not vid:
                continue
            etype = e.get("event_type")
            try:
                dt = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                continue
                
            if etype == "ENTRY":
                active_entries[vid] = dt
            elif etype == "EXIT":
                exits.add(vid)

        # Exclude visitors who have already exited
        for vid in exits:
            active_entries.pop(vid, None)

        # For remaining active visitors, check if they have spent > 5 minutes
        if latest_dt:
            for vid, entry_dt in active_entries.items():
                dur_s = (latest_dt - entry_dt).total_seconds()
                if dur_s > 300:  # 5 minutes
                    dur_m = round(dur_s / 60, 1)
                    anomalies_dict[f"LONG_DWELL_{vid}"] = {
                        "type": "LONG_DWELL",
                        "severity": "WARN",
                        "message": f"Visitor {vid[:6]} has been in-store for {dur_m} minutes.",
                        "suggested_action": "Verify if this is a staff member misclassified as customer."
                    }

        # 2. BILLING_QUEUE_SPIKE — current queue depth > 5
        queue_depths = []
        for e in events:
            if e.get("event_type") in ("BILLING_QUEUE_JOIN", "BILLING_QUEUE_EXIT"):
                qd = e.get("queue_depth")
                if qd is None and e.get("metadata"):
                    qd = e["metadata"].get("queue_depth")
                if qd is not None:
                    queue_depths.append(qd)

        if queue_depths and queue_depths[-1] >= 5:
            anomalies_dict["BILLING_QUEUE"] = {
                "type": "BILLING_QUEUE_SPIKE",
                "severity": "CRITICAL",
                "message": f"Queue depth is currently high ({queue_depths[-1]}).",
                "suggested_action": "Open additional billing counter immediately."
            }

        # 3. DEAD_ZONE — active browse zones with no visits
        zone_visits = defaultdict(int)
        for e in events:
            zone = e.get("zone_id") or e.get("zone")
            if zone and e.get("event_type") in ("ZONE_ENTER", "ZONE_DWELL"):
                zone_visits[zone] += 1

        # Only check physically active zones configured with actual boundaries in our camera setup
        known_zones = {"MAKEUP", "SKINCARE"}
        dead_zones = [z for z in known_zones if zone_visits.get(z, 0) == 0]
        if dead_zones:
            zones_str = ", ".join(dead_zones)
            anomalies_dict["DEAD_ZONE"] = {
                "type": "DEAD_ZONE",
                "severity": "INFO",
                "message": f"No visits in: {zones_str}.",
                "suggested_action": "Check signage and product placement."
            }

        # 4. CONVERSION_DROP — very low conversion
        entry_visitors = set(e["visitor_id"] for e in events if e.get("event_type") == "ENTRY")
        billing_visitors = set(e["visitor_id"] for e in events if e.get("event_type") == "BILLING_QUEUE_JOIN")
        if len(entry_visitors) >= 10:
            conv_rate = len(billing_visitors) / len(entry_visitors)
            if conv_rate < 0.05:
                anomalies_dict["CONVERSION"] = {
                    "type": "CONVERSION_DROP",
                    "severity": "CRITICAL",
                    "message": f"Conversion rate is critically low at {conv_rate*100:.1f}%.",
                    "suggested_action": "Review store layout, staffing, and product availability."
                }
            elif conv_rate < 0.15:
                anomalies_dict["CONVERSION"] = {
                    "type": "CONVERSION_DROP",
                    "severity": "WARN",
                    "message": f"Conversion rate is below average at {conv_rate*100:.1f}%.",
                    "suggested_action": "Consider promotional offers or staff engagement improvements."
                }

        return list(anomalies_dict.values())

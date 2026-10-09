import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class StaffClassifier:
    """
    Hybrid Staff Classifier using 5 signals:
    1. Appearance (Uniform clustering)
    2. Visibility Ratio (Duration inside)
    3. Zone Roaming (Number of zones)
    4. Behavior (Billing affinity)
    5. Storage access (Auto-staff)
    
    This operates in two phases:
    1. Online: Only immediate auto-staff detection (Storage access or very long duration).
    2. Offline (finalize): Clusters embeddings to find staff uniform, applies hybrid rules.
    """

    def __init__(
        self,
        # Thresholds
        score_threshold: float = 0.65,
        auto_staff_ratio: float = 0.80,      # Visible >80% of clip → auto-staff online
        min_visibility_ratio: float = 0.40,  # Must be visible >40% for offline staff rules
        billing_dwell_ratio: float = 0.50,    # >50% of time at BILLING → cashier signal
    ):
        self.score_threshold = score_threshold
        self.auto_staff_ratio = auto_staff_ratio
        self.min_visibility_ratio = min_visibility_ratio
        self.billing_dwell_ratio = billing_dwell_ratio
        
        # Clip duration
        self.clip_duration_seconds: Optional[float] = None
        
        # Staff cluster embeddings (found during offline phase)
        self.staff_cluster_embeddings: List[np.ndarray] = []
        
        # Live Staff Uniform Embeddings (found dynamically online)
        self.live_staff_uniform_embeddings: List[np.ndarray] = []

    def set_clip_duration(self, clip_duration_seconds: float):
        self.clip_duration_seconds = clip_duration_seconds
        logger.info(f"StaffClassifier ADAPTED to clip duration {clip_duration_seconds:.1f}s")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.flatten()
        b_flat = b.flatten()
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))

    def _effective_zone_dwells(self, visitor_session) -> Dict[str, float]:
        """Return dwell times including the currently active zone."""
        dwells = dict(getattr(visitor_session, "zone_dwell_times", {}) or {})
        current_zone = getattr(visitor_session, "current_zone", None)
        zone_entry_times = getattr(visitor_session, "zone_entry_times", {}) or {}
        last_seen = getattr(visitor_session, "last_seen", None)

        if current_zone and current_zone in zone_entry_times and last_seen:
            try:
                active_dwell = (last_seen - zone_entry_times[current_zone]).total_seconds()
                if active_dwell > 0:
                    dwells[current_zone] = dwells.get(current_zone, 0.0) + active_dwell
            except Exception:
                pass

        return dwells

    def classify_session(self, visitor_session) -> bool:
        """
        ONLINE PHASE: Multi-factor confidence scoring with hysteresis.
        Returns whether the session is currently classified as STAFF.
        """
        duration = getattr(visitor_session, "duration_seconds", 0)
        zones = set(getattr(visitor_session, "zones_visited", []))
        zone_dwells = self._effective_zone_dwells(visitor_session)
        unique_zones = len(zones)
        billing_dwell = float(zone_dwells.get("BILLING", 0.0))
        emb = getattr(visitor_session, "embedding", None)
        
        confidence_score = 0.0
        explanations = []
        
        # 1. Cashier Role (Definitive Staff)
        if billing_dwell >= 45.0:
            confidence_score = max(confidence_score, 0.95)
            explanations.append(f"Live Cashier (BILLING: {billing_dwell:.0f}s)")
            
            # Learn uniform
            if emb is not None:
                already_known = any(self._cosine_similarity(emb, k_emb) > 0.85 for k_emb in self.live_staff_uniform_embeddings)
                if not already_known:
                    self.live_staff_uniform_embeddings.append(emb)
                    logger.info(f"LEARNED LIVE DRESS CODE: Captured uniform from Cashier {visitor_session.visitor_id}")

        # 2. Dress Code Match
        appearance_match = False
        if emb is not None and self.live_staff_uniform_embeddings:
            for staff_emb in self.live_staff_uniform_embeddings:
                if self._cosine_similarity(emb, staff_emb) > 0.85:
                    appearance_match = True
                    confidence_score += 0.60
                    explanations.append("Uniform Match")
                    break
                    
        # 3. Behavioral Consistency (Active Roaming)
        if unique_zones >= 2:
            confidence_score += 0.30
            explanations.append(f"Active Roaming ({unique_zones} zones)")
        
        # Cap confidence at 1.0
        confidence_score = min(1.0, confidence_score)
        
        # Store for dashboard
        visitor_session.staff_score = confidence_score
        visitor_session.staff_explanation = " + ".join(explanations) if explanations else "Customer"
        
        # 4. Hysteresis State Transition
        was_staff = getattr(visitor_session, "is_staff", False)
        
        if not was_staff and confidence_score > 0.85:
            visitor_session.is_staff = True
        elif was_staff and confidence_score < 0.30:
            visitor_session.is_staff = False
        else:
            # Maintain current state
            visitor_session.is_staff = was_staff
            
        return visitor_session.is_staff

    def finalize_staff_classification(self, sessions_dict: Dict[str, any]):
        """
        OFFLINE PHASE:
        1. Find "staff uniform" cluster
        2. Apply 5-factor rules
        """
        if not self.clip_duration_seconds:
            logger.warning("clip_duration_seconds not set for offline classification!")
            return

        min_vis_sec = self.clip_duration_seconds * self.min_visibility_ratio

        # Extract embeddings for clustering (only for sessions with sufficient duration)
        valid_sessions = []
        embeddings = []
        for vid, session in sessions_dict.items():
            duration = getattr(session, "duration_seconds", 0)
            emb = getattr(session, "embedding", None)
            
            if duration >= min_vis_sec and emb is not None:
                valid_sessions.append(session)
                embeddings.append(emb)

        if not embeddings:
            logger.info("No candidates found for staff uniform clustering.")
            return

        # Simple clustering O(N^2)
        clusters = [] # list of lists of indices
        for i, emb_i in enumerate(embeddings):
            placed = False
            for cluster in clusters:
                # Compare with first element of cluster
                if self._cosine_similarity(emb_i, embeddings[cluster[0]]) > 0.80:
                    cluster.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])
                
        # Find the cluster with highest average duration
        best_cluster = None
        best_avg_duration = -1
        for cluster in clusters:
            avg_dur = sum(getattr(valid_sessions[idx], "duration_seconds", 0) for idx in cluster) / len(cluster)
            # A valid staff cluster should have at least 2 people usually, but could be 1 in testing
            if avg_dur > best_avg_duration:
                best_avg_duration = avg_dur
                best_cluster = cluster
                
        if best_cluster:
            self.staff_cluster_embeddings = [embeddings[idx] for idx in best_cluster]
            logger.info(f"Staff uniform cluster found with {len(best_cluster)} members, avg duration {best_avg_duration:.1f}s")
            
        # Apply rules to all sessions
        for vid, session in sessions_dict.items():
            # Skip if already auto-staffed
            if getattr(session, "is_staff", False):
                continue
                
            duration = getattr(session, "duration_seconds", 0)
            zones = getattr(session, "zones_visited", [])
            unique_zones = len(set(zones))
            zone_dwells = self._effective_zone_dwells(session)
            emb = getattr(session, "embedding", None)
            
            # Rule 1: Must meet minimum visibility
            if duration < min_vis_sec:
                continue
                
            # Rule 2: Appearance match
            appearance_match = False
            if emb is not None and self.staff_cluster_embeddings:
                for staff_emb in self.staff_cluster_embeddings:
                    if self._cosine_similarity(emb, staff_emb) > 0.85:
                        appearance_match = True
                        break
                        
            # Rule 3 & 4: Roaming or Cashier
            billing_dwell = zone_dwells.get("BILLING", 0.0)
            is_cashier = (duration > 0) and (billing_dwell / duration > self.billing_dwell_ratio)
            is_roamer = unique_zones >= 3
            
            duration_score = min(duration / max(self.clip_duration_seconds * self.auto_staff_ratio, 1.0), 1.0)
            zone_score = min(unique_zones / 4.0, 1.0)
            behavior_score = max(1.0 if is_cashier else 0.0, zone_score if is_roamer else 0.0)
            score = 0.55 * duration_score + 0.25 * behavior_score + 0.20 * (1.0 if appearance_match else 0.0)

            # Final staff decision
            if appearance_match and (is_cashier or is_roamer) and score >= self.score_threshold:
                session.is_staff = True
                session.staff_score = round(score, 3)
                session.staff_explanation = (
                    f"STAFF (Offline): score={score:.2f}, appearance match + "
                    f"{'Cashier' if is_cashier else 'Roamer'} ({duration:.0f}s)"
                )
                logger.info(session.staff_explanation)

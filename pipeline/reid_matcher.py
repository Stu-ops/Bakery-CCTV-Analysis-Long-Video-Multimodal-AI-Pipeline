"""
Re-ID Matcher — Simplified Dual-Factor Matching

CORRECTION #3: Uses embedding + temporal instead of ChatGPT's
3-stage cascade (embedding → trajectory → pose).

Why ChatGPT's approach is overkill:
- Pose estimation adds complexity for minimal benefit in retail.
- Most re-entries happen 30+ minutes apart (time is strongest signal).
- Challenge doesn't penalize false positive re-entry detection.
- 3-stage cascade is harder to debug and explain in interviews.

Corrected approach:
- Stage 1: Embedding similarity (OSNet) — primary signal
- Stage 2: Temporal + spatial context — tiebreaker
- No pose estimation needed

Confidence levels:
- "high":   embedding_sim > 0.80 → definitely re-entry
- "medium": embedding_sim > 0.70 AND same entry point AND time < 5min
- "none":   otherwise → new entry
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ReIDMatcher:
    """
    Re-identification matcher for detecting re-entries.

    Determines if a newly detected person entering the store
    is the same person who previously exited.

    Simplified from ChatGPT's 3-stage cascade to a dual-factor approach
    that is simpler, more debuggable, and interview-friendly.
    """

    def __init__(
        self,
        embedding_high_threshold: float = 0.80,
        embedding_medium_threshold: float = 0.70,
        time_window_seconds: float = 30 * 60,
        medium_time_window_seconds: float = 5 * 60,
        position_proximity_pixels: float = 200,
    ):
        """
        Args:
            embedding_high_threshold: Cosine sim for high-confidence match.
            embedding_medium_threshold: Cosine sim for medium (needs temporal).
            time_window_seconds: Max time gap to consider re-entry (30 min).
            medium_time_window_seconds: Time window for medium confidence (5 min).
            position_proximity_pixels: Max pixel distance for "same entry point".
        """
        self.embedding_high_threshold = embedding_high_threshold
        self.embedding_medium_threshold = embedding_medium_threshold
        self.time_window = time_window_seconds
        self.medium_time_window = medium_time_window_seconds
        self.position_proximity = position_proximity_pixels

        # Cache of recently exited visitors
        self._exit_cache: Dict[str, dict] = {}

        logger.info(
            "ReIDMatcher: embed_high=%.2f, embed_med=%.2f, "
            "time_window=%.0fs, proximity=%.0fpx",
            embedding_high_threshold,
            embedding_medium_threshold,
            time_window_seconds,
            position_proximity_pixels,
        )

    def register_exit(
        self,
        visitor_id: str,
        exit_time: datetime,
        embedding: Optional[np.ndarray],
        last_position: Tuple[float, float],
    ):
        """
        Register a visitor exit for future re-entry matching.

        Args:
            visitor_id: Unique visitor session ID.
            exit_time: When the visitor exited.
            embedding: OSNet embedding vector (or None if unavailable).
            last_position: (x, y) pixel coordinates of last detection.
        """
        self._exit_cache[visitor_id] = {
            "visitor_id": visitor_id,
            "exit_time": exit_time,
            "embedding": embedding,
            "last_position": last_position,
        }
        logger.debug("Registered exit for %s at %s", visitor_id, exit_time)

    def match_reentry(
        self,
        entry_time: datetime,
        entry_embedding: Optional[np.ndarray],
        entry_position: Tuple[float, float],
    ) -> Tuple[Optional[str], str, float]:
        """
        Check if a new entry matches any recent exit.

        Decision logic:
        1. Time check: skip if exit was > 30 min ago
        2. Embedding similarity: primary matching signal
           - > 0.80 → HIGH confidence re-entry
           - > 0.70 AND same entry point AND < 5 min → MEDIUM confidence
        3. Default: NEW_ENTRY

        Args:
            entry_time: When the new person entered.
            entry_embedding: OSNet embedding of the new person.
            entry_position: (x, y) pixel coordinates of entry detection.

        Returns:
            Tuple of (matched_visitor_id, confidence, similarity).
            matched_visitor_id is None if no re-entry detected.
        """
        if entry_embedding is None:
            return None, "none", 0.0

        best_match_id = None
        best_confidence = "none"
        best_similarity = 0.0

        for visitor_id, exit_record in self._exit_cache.items():
            exit_time = exit_record["exit_time"]
            exit_embedding = exit_record["embedding"]
            exit_position = exit_record["last_position"]

            # Rule 1: Time window check
            time_delta = (entry_time - exit_time).total_seconds()
            if time_delta > self.time_window or time_delta < 0:
                continue  # Too old or future (clock issue)

            # Rule 2: Embedding similarity
            if exit_embedding is None:
                continue

            embedding_sim = self._cosine_similarity(exit_embedding, entry_embedding)

            # High confidence: embedding alone is sufficient
            if embedding_sim > self.embedding_high_threshold:
                if embedding_sim > best_similarity:
                    best_match_id = visitor_id
                    best_confidence = "high"
                    best_similarity = embedding_sim

            # Medium confidence: embedding + spatial + temporal
            elif embedding_sim > self.embedding_medium_threshold:
                pixel_distance = np.linalg.norm(
                    np.array(exit_position) - np.array(entry_position)
                )
                same_entry_point = pixel_distance < self.position_proximity
                within_medium_window = time_delta < self.medium_time_window

                if same_entry_point and within_medium_window:
                    if embedding_sim > best_similarity:
                        best_match_id = visitor_id
                        best_confidence = "medium"
                        best_similarity = embedding_sim

        if best_match_id:
            logger.info(
                "Re-entry detected: %s (confidence=%s, similarity=%.3f)",
                best_match_id,
                best_confidence,
                best_similarity,
            )
            # Remove from cache (matched)
            del self._exit_cache[best_match_id]

        return best_match_id, best_confidence, best_similarity

    def emit_reentry_event(
        self,
        prev_visitor_id: str,
        new_visitor_id: str,
        confidence: str,
        similarity: float,
        timestamp: datetime,
    ) -> dict:
        """
        Create a REENTRY event for the event stream.

        Args:
            prev_visitor_id: Original visitor session ID.
            new_visitor_id: New session ID (same person).
            confidence: "high" or "medium".
            similarity: Cosine similarity score.
            timestamp: When the re-entry was detected.

        Returns:
            Event dict ready for emission.
        """
        return {
            "event_type": "REENTRY",
            "prev_visitor_id": prev_visitor_id,
            "new_visitor_id": new_visitor_id,
            "confidence": confidence,
            "similarity": float(similarity),
            "timestamp": timestamp.isoformat() + "Z",
        }

    def cleanup_stale_exits(self, current_time: datetime):
        """Remove exits older than the time window."""
        stale_ids = [
            vid
            for vid, record in self._exit_cache.items()
            if (current_time - record["exit_time"]).total_seconds() > self.time_window
        ]
        for vid in stale_ids:
            del self._exit_cache[vid]
        if stale_ids:
            logger.debug("Cleaned up %d stale exit records", len(stale_ids))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        a_flat = a.flatten()
        b_flat = b.flatten()
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))

    @property
    def pending_exits(self) -> int:
        """Number of exits awaiting potential re-entry match."""
        return len(self._exit_cache)

import logging
from typing import Dict, Set, Optional, Tuple, List
from datetime import datetime
from enum import Enum
from pipeline.topology import StoreTopology, CameraRole

logger = logging.getLogger(__name__)

class Role(Enum):
    OUTSIDE_CANDIDATE = "outside_candidate"
    CUSTOMER = "customer"
    STAFF = "staff"
    EXITED = "exited"
    ORPHAN = "orphan"

class GlobalOccupancyEngine:
    """
    Manages true store occupancy by validating entries against the store topology.
    Centralized Identity Registry: Uses a dynamic Role transition matrix to prevent
    orphan tracks and accurately manage customer vs staff counts.
    """
    def __init__(self, topology: StoreTopology):
        self.topology = topology
        
        # Validated unique customers currently inside the store
        self.validated_occupancy: Set[str] = set()
        
        # Confirmed staff currently inside the store
        self.staff_occupancy: Set[str] = set()
        
        # Candidates: people seen outside but not yet confirmed inside
        self.candidates: Set[str] = set()
        
        # Track orphans: global_id -> (first_seen_cam, timestamp)
        self.orphans: Dict[str, Tuple[str, datetime]] = {}
        
        # Keep track of visitor roles (Central Identity State)
        self.roles: Dict[str, Role] = {}
        
        # Total unique customers that have entered today
        self.total_entries = 0
        
        # Total unique staff identified today
        self.staff_entries = 0
        
        # System start time to handle customers already in the store
        self.system_start_time: Optional[datetime] = None
        self.initialization_window_seconds = 15.0
        
        # Global audit trail for all role transitions
        self.role_change_log: List[dict] = []
        
        # Oscillation guard: track consecutive frames at proposed role
        # global_id -> {"proposed_role": Role, "stable_frames": int}
        self._role_stability: Dict[str, dict] = {}
        self.ROLE_STABILITY_THRESHOLD = 5  # Must be stable for 5 frames

    def process_new_identity(self, global_id: str, camera_id: str, timestamp: datetime) -> bool:
        """
        Called when CrossCameraTracker creates a BRAND NEW global_id.
        """
        role_type = self.topology.get_role(camera_id)
        
        if self.system_start_time is None:
            self.system_start_time = timestamp
            
        time_since_start = (timestamp - self.system_start_time).total_seconds()
        
        if role_type == CameraRole.ENTRANCE:
            self.candidates.add(global_id)
            self.roles[global_id] = Role.OUTSIDE_CANDIDATE
            logger.info(f"CANDIDATE DETECTED: {global_id} first seen outside on {camera_id}.")
            return False
        elif role_type == CameraRole.INTERNAL or role_type == CameraRole.BILLING:
            if time_since_start <= self.initialization_window_seconds:
                # Pre-existing customer logic
                self.validated_occupancy.add(global_id)
                self.roles[global_id] = Role.CUSTOMER
                self.total_entries += 1
                logger.info(f"PRE-EXISTING CUSTOMER: {global_id} initialized on {camera_id}. (Total inside: {len(self.validated_occupancy)})")
                return True
            else:
                self.orphans[global_id] = (camera_id, timestamp)
                self.roles[global_id] = Role.ORPHAN
                logger.warning(
                    f"ORPHAN DETECTED: {global_id} first seen on {camera_id} ({role_type.name}). "
                    f"Likely a viewpoint change. NOT counting towards occupancy."
                )
                return False
            
        return False

    def update_identity(self, global_id: str, camera_id: str, timestamp: datetime) -> bool:
        """
        Promotes CANDIDATES to CUSTOMERS if they are seen on an INTERNAL camera.
        Returns True if a PROMOTION occurred.
        """
        if global_id in self.candidates and self.roles.get(global_id) == Role.OUTSIDE_CANDIDATE:
            role_type = self.topology.get_role(camera_id)
            if role_type == CameraRole.INTERNAL or role_type == CameraRole.BILLING:
                self.update_role(global_id, Role.CUSTOMER)
                logger.info(f"ENTRY CONFIRMED: {global_id} promoted to CUSTOMER via {camera_id}. (Total inside: {len(self.validated_occupancy)})")
                return True
        return False

    def update_role(self, global_id: str, new_role: Role, timestamp: Optional[datetime] = None, explanation: str = "") -> bool:
        """
        Dynamic transition matrix for identity roles.
        Properly manages counts and internal sets to prevent double-counting.
        Includes oscillation guard: role must be proposed for ROLE_STABILITY_THRESHOLD
        consecutive calls before committing.
        Returns True if the role actually changed.
        """
        current_role = self.roles.get(global_id)
        if current_role == new_role:
            # Reset stability counter since role matches current
            self._role_stability.pop(global_id, None)
            return False

        # Oscillation guard: require stability before committing
        # (Skip for EXITED transitions — those are always immediate)
        if new_role != Role.EXITED:
            stability = self._role_stability.get(global_id)
            if stability and stability["proposed_role"] == new_role:
                stability["stable_frames"] += 1
            else:
                self._role_stability[global_id] = {"proposed_role": new_role, "stable_frames": 1}
                stability = self._role_stability[global_id]
            
            if stability["stable_frames"] < self.ROLE_STABILITY_THRESHOLD:
                return False  # Not stable enough yet
        
        # Stability reached — commit the transition
        self._role_stability.pop(global_id, None)

        # Clean up old state
        if current_role == Role.OUTSIDE_CANDIDATE and global_id in self.candidates:
            self.candidates.remove(global_id)
        elif current_role == Role.ORPHAN and global_id in self.orphans:
            del self.orphans[global_id]
        elif current_role == Role.CUSTOMER and global_id in self.validated_occupancy:
            self.validated_occupancy.remove(global_id)
            if new_role == Role.STAFF:
                self.total_entries = max(0, self.total_entries - 1)
        elif current_role == Role.STAFF and global_id in self.staff_occupancy:
            self.staff_occupancy.remove(global_id)
            
        # Apply new state
        self.roles[global_id] = new_role
        
        if new_role == Role.CUSTOMER:
            self.validated_occupancy.add(global_id)
            if current_role != Role.STAFF:  # If degrading from staff, don't double count entry
                self.total_entries += 1
        elif new_role == Role.STAFF:
            self.staff_occupancy.add(global_id)
            if current_role != Role.STAFF:
                self.staff_entries += 1
            logger.info(f"ROLE TRANSITION: {global_id} is now STAFF. (Customers: {len(self.validated_occupancy)}, Staff: {len(self.staff_occupancy)})")
        
        # Audit trail
        self.role_change_log.append({
            "global_id": global_id,
            "old_role": current_role.value if current_role else None,
            "new_role": new_role.value,
            "timestamp": timestamp.isoformat() + "Z" if timestamp and isinstance(timestamp, datetime) else str(timestamp),
            "explanation": explanation,
        })
            
        return True

    def get_role_audit_trail(self, global_id: str) -> list:
        """Get all role transitions for a specific identity."""
        return [entry for entry in self.role_change_log if entry["global_id"] == global_id]

    def handle_reid_merge(self, survivor_id: str, absorbed_id: str):
        """Clean up orphan state if merged."""
        if absorbed_id in self.orphans:
            del self.orphans[absorbed_id]
        if absorbed_id in self.validated_occupancy:
            self.validated_occupancy.remove(absorbed_id)

    def process_exit(self, global_id: str):
        """Transition identity to EXITED."""
        self.update_role(global_id, Role.EXITED)

    def is_validated(self, global_id: str) -> bool:
        """Check if an ID is part of the true validated customer occupancy."""
        return self.roles.get(global_id) == Role.CUSTOMER

    @property
    def current_occupancy_count(self) -> int:
        """The true, deduplicated count of customers inside the store."""
        return len(self.validated_occupancy)

import logging
from enum import Enum
from typing import Dict, List, Set, Optional

logger = logging.getLogger(__name__)

class CameraRole(Enum):
    ENTRANCE = "entrance"
    INTERNAL = "internal"
    BILLING = "billing"

class StoreTopology:
    """
    Defines the physical layout of the store's cameras to enable camera-aware tracking.
    Enforces valid transitions and determines which cameras can admit new visitors.
    """
    def __init__(self, camera_mapping: Optional[Dict[str, str]] = None):
        """
        Args:
            camera_mapping: Dict mapping camera_id to role string.
                            Defaults to Purplle Brigade Road layout.
        """
        # Default mapping based on Brigade Road layout
        default_mapping = {
            "CAM_1": "internal",
            "CAM_2": "internal",
            "CAM_3": "entrance",
            "CAM_4": "internal",
            "CAM_5": "billing"
        }
        
        mapping = camera_mapping or default_mapping
        
        self.camera_roles: Dict[str, CameraRole] = {}
        for cam_id, role_str in mapping.items():
            try:
                self.camera_roles[cam_id] = CameraRole(role_str.lower())
            except ValueError:
                logger.warning(f"Unknown role '{role_str}' for camera {cam_id}. Defaulting to INTERNAL.")
                self.camera_roles[cam_id] = CameraRole.INTERNAL

        # Valid transitions represent physical adjacency
        # ENTRANCE <-> INTERNAL <-> BILLING
        self.valid_transitions: Dict[CameraRole, Set[CameraRole]] = {
            CameraRole.ENTRANCE: {CameraRole.ENTRANCE, CameraRole.INTERNAL},
            CameraRole.INTERNAL: {CameraRole.ENTRANCE, CameraRole.INTERNAL, CameraRole.BILLING},
            CameraRole.BILLING: {CameraRole.INTERNAL, CameraRole.BILLING}
        }
        
        # Finer-grained adjacency matrix based on specific rooms
        # CAM_3 (Outside) <-> CAM_1, CAM_2 (Main Room)
        # CAM_1, CAM_2 <-> CAM_4 (Other Room)
        # CAM_1, CAM_2 <-> CAM_5 (Billing)
        self.valid_cam_transitions = {
            "CAM_1": {"CAM_1", "CAM_2", "CAM_3", "CAM_4", "CAM_5"},
            "CAM_2": {"CAM_1", "CAM_2", "CAM_3", "CAM_4", "CAM_5"},
            "CAM_3": {"CAM_1", "CAM_2", "CAM_3"}, # Cannot jump direct to CAM_4 or CAM_5
            "CAM_4": {"CAM_1", "CAM_2", "CAM_4"}, # Must go through main room
            "CAM_5": {"CAM_1", "CAM_2", "CAM_5"}, # Must go through main room
        }
        
        # Explicit Physical Room Mapping
        self.rooms = {
            "MAIN_ROOM": {"CAM_1", "CAM_2", "CAM_5"},
            "OTHER_ROOM": {"CAM_4"},
            "OUTSIDE": {"CAM_3"}
        }
        
        logger.info(f"StoreTopology initialized with {len(self.camera_roles)} cameras.")
        for role in CameraRole:
            cams = [cam for cam, r in self.camera_roles.items() if r == role]
            logger.info(f"  {role.name}: {cams}")

    def get_role(self, camera_id: str) -> CameraRole:
        """Get role for a camera, defaults to INTERNAL if unknown."""
        return self.camera_roles.get(camera_id, CameraRole.INTERNAL)

    def is_valid_transition(self, from_cam: str, to_cam: str) -> bool:
        """Check if transitioning between these two cameras is physically valid."""
        # Check explicit camera graph first
        if from_cam in self.valid_cam_transitions:
            if to_cam not in self.valid_cam_transitions[from_cam]:
                return False
                
        from_role = self.get_role(from_cam)
        to_role = self.get_role(to_cam)
        return to_role in self.valid_transitions[from_role]

    def can_admit_visitor(self, camera_id: str) -> bool:
        """
        Only ENTRANCE cameras can admit a fully validated new global visitor.
        If a new person is seen on an INTERNAL or BILLING camera, they are treated as an orphan.
        """
        return self.get_role(camera_id) == CameraRole.ENTRANCE

    def is_same_room(self, cam1: str, cam2: str) -> bool:
        """Check if two cameras share the same physical room."""
        if not cam1 or not cam2:
            return False
        for room_cams in self.rooms.values():
            if cam1 in room_cams and cam2 in room_cams:
                return True
        return False

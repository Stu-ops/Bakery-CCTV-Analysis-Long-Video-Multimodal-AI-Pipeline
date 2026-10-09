import cv2
import numpy as np
import time
from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import sys
from typing import Dict, List, Optional
import base64

from pipeline.detector import PersonDetector
from pipeline.zone_detector import ZoneOccupancyDetector
from pipeline.dwell_tracker import DwellTracker
from pipeline.staff_classifier import StaffClassifier
from pipeline.session_manager import SessionManager
from pipeline.event_emitter import EventEmitter, EventType
from pipeline.queue_analyzer import QueueAnalyzer
from pipeline.pos_correlator import POSCorrelator
from pipeline.data_loader import PurpllePOSLoader
from pipeline.cross_cam_tracker import CrossCameraTracker
from pipeline.live_pusher import LiveEventPusher, LiveFramePusher
from pipeline.topology import StoreTopology
from pipeline.occupancy_engine import GlobalOccupancyEngine
import requests

logger = logging.getLogger(__name__)

# Config
TARGET_FPS = 5

class MultiCamRunner:
    """
    Runs inference on all 5 cameras simultaneously, uses ReID for global tracking,
    and streams frames + metadata to the backend.
    """
    def __init__(self, config_path: Path, store_id: str, video_dir: Path, api_url: str, live: bool = False):
        self.store_id = store_id
        self.api_url = api_url
        self.video_dir = video_dir
        self.live = live
        
        # Load layout
        with open(config_path) as f:
            layouts = json.load(f)
        self.layout = layouts[store_id]
        
        self.clip_start = datetime.fromisoformat(
            self.layout["clip_start_time"].replace("Z", "+00:00")
        ).replace(tzinfo=None)
        
        # Use YOLOv8 Nano for significantly faster inference times
        self.detector = PersonDetector(model_path="models/yolov8n.pt")
        
        # Zones
        zones = {name: data["polygon"] for name, data in self.layout["zones"].items()}
        self.zone_detector = ZoneOccupancyDetector(zones)
        
        # Camera-Aware Topology & Occupancy
        self.topology = StoreTopology()
        self.occupancy_engine = GlobalOccupancyEngine(self.topology)
        self.cross_tracker = CrossCameraTracker(topology=self.topology, similarity_threshold=0.7)
        self.staff_classifier = StaffClassifier()
        self.session_manager = SessionManager(store_id=store_id)
        self.event_emitter = EventEmitter(store_id=store_id)
        self.pos_correlator = POSCorrelator()
        
        # Live pushers are enabled only for --live runs. Offline scoring runs
        # should not spend time retrying a backend that was never requested.
        self.live_pusher = None
        self.live_frame_pusher = None
        if self.live:
            self.live_pusher = LiveEventPusher(api_url=api_url)
            self.live_frame_pusher = LiveFramePusher(api_url=api_url)
            self.event_emitter.register_callback(self.live_pusher.push)
        
        self.last_processed_time = None
        
        # Component 1: Zone Hysteresis Filter state
        self.pending_zone_state = {}  # global_id -> {"zone": str, "count": int}

    def _current_billing_depth(self, exclude_track_id: Optional[str] = None) -> int:
        """Count active non-staff VALIDATED sessions currently occupying billing."""
        depth = 0
        for session in self.session_manager.get_active_sessions().values():
            if session.track_id == exclude_track_id:
                continue
            if not self.occupancy_engine.is_validated(session.track_id):
                continue
            if session.current_zone == "BILLING" and not session.is_staff:
                depth += 1
        return depth

    def _push_video_frame(self, camera_id: str, frame: np.ndarray, tracks: list):
        """Pushes annotated frame to API for live dashboard viewing."""
        if not self.live or self.live_frame_pusher is None:
            return

        # Annotate
        annotated = frame.copy()
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t['bbox']]
            label = f"{t['global_id']} ({'STAFF' if t['is_staff'] else 'VISITOR'})"
            color = (200, 200, 200) if t['is_staff'] else (255, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Resize for lightweight transmission
        annotated = cv2.resize(annotated, (320, 180))
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
        b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Async background pushing
        self.live_frame_pusher.push(camera_id, b64)

    def run(self, max_frames: int = 0):
        logger.info("Initializing Multi-Camera Streams...")
        
        cameras = ["CAM_1", "CAM_2", "CAM_3", "CAM_4", "CAM_5"]
        caps = {}
        cam_fps = {}
        cam_frame_counts = {}
        cam_frame_idx = {}  # Per-camera frame counter
        
        for cam in cameras:
            path = self.video_dir / f"{cam}.mp4"
            if path.exists():
                cap = cv2.VideoCapture(str(path))
                caps[cam] = cap
                fps = cap.get(cv2.CAP_PROP_FPS)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cam_fps[cam] = fps if fps > 0 else 30.0
                cam_frame_counts[cam] = total
                cam_frame_idx[cam] = 0
                duration_sec = total / cam_fps[cam]
                logger.info(f"  {cam}: {total} frames @ {cam_fps[cam]:.0f}fps = {duration_sec:.1f}s")
            else:
                logger.error(f"Missing video {path}")
                
        if not caps:
            return

        # Determine actual clip duration from longest video
        max_clip_duration_sec = max(
            (cam_frame_counts[c] / cam_fps[c]) for c in caps
        )
        logger.info(f"Actual clip duration: {max_clip_duration_sec:.1f}s ({max_clip_duration_sec/60:.1f} min)")
        
        # Configure staff classifier with adaptive thresholds based on actual clip length
        self.staff_classifier.set_clip_duration(max_clip_duration_sec)
        
        # Track active/completed cameras
        active_cameras = set(caps.keys())
        completed_cameras = set()
        
        global_frame_idx = 0
        
        while active_cameras:
            # Read one frame from each active camera
            frames = {}
            newly_completed = []
            
            for cam in list(active_cameras):
                ret, frame = caps[cam].read()
                if ret:
                    cam_frame_idx[cam] += 1
                    frames[cam] = frame
                else:
                    # This camera has finished
                    newly_completed.append(cam)
            
            # Process newly completed cameras
            for cam in newly_completed:
                active_cameras.discard(cam)
                completed_cameras.add(cam)
                caps[cam].release()
                logger.info(
                    f"  ✓ {cam} COMPLETED — processed {cam_frame_idx[cam]}/{cam_frame_counts[cam]} frames "
                    f"({len(active_cameras)} cameras still active: {sorted(active_cameras)})"
                )
            
            if not frames:
                # All cameras exhausted this tick (shouldn't happen if active_cameras is checked)
                break
                
            global_frame_idx += 1
            
            # Frame skipping: use the fastest camera's FPS for the skip interval
            max_fps = max(cam_fps[c] for c in caps)
            skip_interval = max(1, int(max_fps // TARGET_FPS))
            
            if global_frame_idx % skip_interval != 0:
                continue
                
            # Use per-camera timestamp based on each camera's actual FPS
            # For session/event timing, use the median timestamp across active cameras
            cam_timestamps = {}
            for cam in frames:
                cam_time_offset = cam_frame_idx[cam] / cam_fps[cam]
                cam_timestamps[cam] = self.clip_start + __import__("datetime").timedelta(seconds=cam_time_offset)
            
            # Global current_time = latest camera timestamp (for session timeout checks)
            current_time = max(cam_timestamps.values())
            self.last_processed_time = current_time
            
            # Process each camera sequentially in this tick
            for cam, frame in frames.items():
                cam_time = cam_timestamps[cam]
                
                # 1. Detect & Track locally
                local_tracks = self.detector.detect_and_track_cam(frame, cam_frame_idx[cam], cam)
                
                display_tracks = []
                
                # 2. Global Association & Store Logic
                for original_track_id, person in list(local_tracks.items()):
                    # Explicit Visitor Validation: Discard low-confidence detections
                    if person.confidence < 0.7:
                        continue
                        
                    # Associate global ID
                    global_id = self.cross_tracker.associate_track(cam, original_track_id, frame, person.bbox, cam_time.timestamp())
                    
                    # Phase 3: Continuous State Machine Check
                    # Check if this person was a CANDIDATE (from outside) and is now CONFIRMED (inside)
                    is_promoted = self.occupancy_engine.update_identity(global_id, cam, cam_time)
                    
                    # Store logic
                    zone_result = self.zone_detector.get_current_zone(person.bbox)
                    detected_zone = zone_result[0] if zone_result else None
                    
                    # Strict Passersby logic: Only create session if they enter a valid internal zone or ENTRY
                    session = self.session_manager.get_session_by_track(global_id)
                    
                    if not session and (detected_zone or is_promoted):
                        # Process new identity through the occupancy engine
                        # Check if it's a valid new entry, candidate, or an orphan track
                        is_valid_entry = self.occupancy_engine.process_new_identity(global_id, cam, cam_time)
                        
                        # Create session to track state locally
                        session = self.session_manager.create_session(global_id, cam_time, cam)
                        
                        if is_valid_entry:
                            self.event_emitter.emit_entry(session.visitor_id, global_id, cam_time, cam, confidence=person.confidence)
                    
                    if session and is_promoted:
                        # They were just promoted from CANDIDATE to CONFIRMED
                        self.event_emitter.emit_entry(session.visitor_id, global_id, cam_time, cam, confidence=person.confidence)
                        
                    if session:
                        # Update last_seen
                        session.last_seen = cam_time
                        session.last_seen_camera = cam

                        # Attach latest appearance embedding for uniform detection
                        global_track = self.cross_tracker.global_tracks.get(global_id)
                        if global_track is not None and global_track.embedding is not None:
                            session.embedding = global_track.embedding
                            
                        # Check Staff
                        was_staff = session.is_staff
                        is_staff = self.staff_classifier.classify_session(session)
                        
                        if is_staff != was_staff:
                            from pipeline.occupancy_engine import Role
                            new_role = Role.STAFF if is_staff else Role.CUSTOMER
                            changed = self.occupancy_engine.update_role(
                                global_id,
                                new_role,
                                timestamp=cam_time,
                                explanation=session.staff_explanation
                            )
                            if changed:
                                session.record_role_change(
                                    old_role="CUSTOMER" if is_staff else "STAFF",
                                    new_role="STAFF" if is_staff else "CUSTOMER",
                                    timestamp=cam_time,
                                    explanation=session.staff_explanation
                                )
                                self.event_emitter.emit_role_changed(
                                    visitor_id=session.visitor_id,
                                    track_id=global_id,
                                    timestamp=cam_time,
                                    old_role="CUSTOMER" if is_staff else "STAFF",
                                    new_role="STAFF" if is_staff else "CUSTOMER",
                                    confidence=session.staff_score,
                                    explanation=session.staff_explanation,
                                    camera_id=cam
                                )
                                # Also emit STAFF_CLASSIFIED for metrics backward compatibility
                                if is_staff:
                                    self.event_emitter.emit_staff_classification(
                                        visitor_id=session.visitor_id,
                                        track_id=global_id,
                                        timestamp=cam_time,
                                        is_staff=True,
                                        score=session.staff_score,
                                        explanation=session.staff_explanation,
                                        camera_id=cam
                                    )
                        
                        old_zone = session.current_zone
                        
                        # Component 1: Zone Hysteresis Filter
                        if global_id not in self.pending_zone_state:
                            self.pending_zone_state[global_id] = {"zone": old_zone, "count": 0}
                            
                        # If detector says we are in a new zone, increment pending count
                        if detected_zone != self.pending_zone_state[global_id]["zone"]:
                            self.pending_zone_state[global_id] = {"zone": detected_zone, "count": 1}
                        else:
                            self.pending_zone_state[global_id]["count"] += 1
                            
                        # Only commit transition if we've seen this zone for 3 consecutive frames
                        if self.pending_zone_state[global_id]["count"] >= 3:
                            new_zone = self.pending_zone_state[global_id]["zone"]
                        else:
                            new_zone = old_zone

                        if old_zone != new_zone:
                            # 1. Exit the old zone if there was one
                            if old_zone is not None:
                                prev_entry = session.zone_entry_times.get(old_zone, session.entry_time)
                                dwell_sec = (cam_time - prev_entry).total_seconds() if prev_entry else 0.0
                                self.session_manager.update_zone_exit(global_id, old_zone, dwell_sec, cam_time, is_billing=(old_zone == "BILLING"))
                                
                                if old_zone == "BILLING":
                                    queue_depth = self._current_billing_depth(exclude_track_id=global_id)
                                    self.event_emitter.emit_billing_event(
                                        EventType.BILLING_QUEUE_EXIT,
                                        visitor_id=session.visitor_id,
                                        track_id=global_id,
                                        timestamp=cam_time,
                                        queue_depth=queue_depth
                                    )
                                else:
                                    self.event_emitter.emit_zone_event(
                                        EventType.ZONE_EXIT,
                                        visitor_id=session.visitor_id,
                                        track_id=global_id,
                                        zone=old_zone,
                                        timestamp=cam_time,
                                        dwell_seconds=dwell_sec,
                                        confidence=person.confidence
                                    )

                            # 2. Enter the new zone if there is one
                            if new_zone is not None:
                                self.session_manager.update_zone(global_id, new_zone, cam_time, is_billing=(new_zone == "BILLING"))
                                
                                if new_zone == "BILLING":
                                    # Component 4: True global queue depth
                                    queue_depth = self._current_billing_depth()
                                    self.event_emitter.emit_billing_event(
                                        EventType.BILLING_QUEUE_JOIN,
                                        visitor_id=session.visitor_id,
                                        track_id=global_id,
                                        timestamp=cam_time,
                                        queue_depth=queue_depth
                                    )
                                else:
                                    self.event_emitter.emit_zone_event(
                                        EventType.ZONE_ENTER,
                                        visitor_id=session.visitor_id,
                                        track_id=global_id,
                                        zone=new_zone,
                                        timestamp=cam_time,
                                        dwell_seconds=0.0,
                                        confidence=person.confidence
                                    )

                    display_tracks.append({
                        "global_id": global_id,
                        "bbox": person.bbox,
                        "is_staff": is_staff
                    })
                    
                # Process lost tracks for this camera (clears local detector tracks)
                lost_tracks = self.detector.get_lost_tracks_cam(local_tracks, cam)
                # Cleanup pending state for lost tracks to prevent memory leak
                for trk in lost_tracks:
                    global_id = self.cross_tracker.remove_local_track(cam, trk.track_id)
                    if global_id and global_id in self.pending_zone_state:
                        del self.pending_zone_state[global_id]

                # 3. Stream to Dashboard
                self._push_video_frame(cam, frame, display_tracks)
                
            # 4. Check for timed-out sessions across all cameras (grace-period finalization)
            active_sessions = list(self.session_manager.get_active_sessions().values())
            for session in active_sessions:
                # CAM_3 (entrance): 15s grace -> emit EXIT
                # Internal cameras: 120s grace -> finalize session silently (no EXIT event)
                if session.last_seen_camera == "CAM_3":
                    grace_period = 15.0
                else:
                    grace_period = 120.0
                
                if session.last_seen and (current_time - session.last_seen).total_seconds() > grace_period:
                    # Emit zone/billing exit first if in a zone
                    if session.current_zone:
                        prev_entry = session.zone_entry_times.get(session.current_zone, session.entry_time)
                        dwell_sec = (current_time - prev_entry).total_seconds() if prev_entry else 0.0
                        if session.current_zone == "BILLING":
                            queue_depth = self._current_billing_depth(exclude_track_id=session.track_id)
                            self.event_emitter.emit_billing_event(
                                EventType.BILLING_QUEUE_EXIT,
                                visitor_id=session.visitor_id,
                                track_id=session.track_id,
                                timestamp=current_time,
                                queue_depth=queue_depth
                            )
                        else:
                            self.event_emitter.emit_zone_event(
                                EventType.ZONE_EXIT,
                                visitor_id=session.visitor_id,
                                track_id=session.track_id,
                                zone=session.current_zone,
                                timestamp=current_time,
                                dwell_seconds=dwell_sec,
                                confidence=0.8
                            )
                    
                    finalized = self.session_manager.finalize_session(session.track_id, current_time)
                    if finalized:
                        self.occupancy_engine.process_exit(finalized.visitor_id)
                        self.occupancy_engine.process_exit(finalized.track_id)
                        if finalized.track_id in self.pending_zone_state:
                            del self.pending_zone_state[finalized.track_id]
                            
                        is_staff = self.staff_classifier.classify_session(finalized)
                        
                        # POS Correlation
                        converted = False
                        basket_value = 0.0
                        if hasattr(self, "pos_correlator") and self.pos_correlator:
                            billing_exit = finalized.billing_zone_exit_time
                            is_conv, txn = self.pos_correlator.correlate_session(billing_exit, self.store_id)
                            if is_conv and txn:
                                converted = True
                                basket_value = txn["basket_value"]

                        # Only emit EXIT event if last_seen_camera is CAM_3
                        if finalized.last_seen_camera == "CAM_3":
                            self.event_emitter.emit_exit(
                                finalized.visitor_id,
                                finalized.track_id,
                                current_time,
                                finalized.duration_seconds,
                                list(finalized.zones_visited),
                                is_staff,
                                camera_id=finalized.camera_id or "CAM_3",
                                confidence=0.8,
                                converted=converted,
                                basket_value=basket_value,
                            )
                
            if max_frames > 0 and global_frame_idx >= max_frames:
                break
        
        # Release any remaining captures
        for cam, cap in caps.items():
            if cam in active_cameras:
                cap.release()
            
        logger.info("============================================================")
        logger.info("Multi-Camera Pipeline Finished")
        logger.info(f"  Cameras completed: {sorted(completed_cameras)}")
        logger.info(f"  Per-camera frames: {dict(cam_frame_idx)}")
        logger.info("============================================================")

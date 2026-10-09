"""
Purplle Retail Analytics — Detection Pipeline Package

Corrected pipeline implementing all 5 fixes from analysis:
1. Frame processing at 5 FPS constant
2. Staff detection: 0.7*duration + 0.3*zones
3. Re-ID: simplified dual-factor
4. Queue depth: occupancy count
5. POS correlation: exit_time ± 5 min
"""

from pipeline.frame_processor import FrameProcessor
from pipeline.zone_detector import ZoneOccupancyDetector
from pipeline.dwell_tracker import DwellTracker, DwellState
from pipeline.staff_classifier import StaffClassifier
from pipeline.reid_matcher import ReIDMatcher
from pipeline.queue_analyzer import QueueAnalyzer
from pipeline.pos_correlator import POSCorrelator
from pipeline.session_manager import SessionManager, VisitorSession
from pipeline.event_emitter import EventEmitter, EventType

__all__ = [
    "FrameProcessor",
    "ZoneOccupancyDetector",
    "DwellTracker",
    "DwellState",
    "StaffClassifier",
    "ReIDMatcher",
    "QueueAnalyzer",
    "POSCorrelator",
    "SessionManager",
    "VisitorSession",
    "EventEmitter",
    "EventType",
]

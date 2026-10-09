"""
Multi-Camera Retail Analytics Pipeline
"""

import argparse
import logging
from pathlib import Path
import cv2
import json
import sys

from pipeline.multi_cam_runner import MultiCamRunner
from pipeline.event_emitter import EventType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)

def main(
    extract: bool,
    pos_csv_path: str,
    max_frames: int,
    video_dir: str,
    live: bool,
    api_url: str
):
    video_dir = Path(video_dir)
    store_id = "STORE_BLR_002"
    
    logger.info("============================================================")
    logger.info("STARTING TRUE MULTI-CAMERA RETAIL ANALYTICS PLATFORM")
    logger.info("============================================================")

    db_path = Path("outputs/results/events.db")
    if db_path.exists():
        try:
            db_path.unlink()
            logger.info("Deleted previous events.db to start a fresh run.")
        except Exception as e:
            logger.warning(f"Could not delete events.db: {e}")
            
    # Re-initialize tables so the backend doesn't crash with 'no such table'
    from app.database import EventDatabase
    EventDatabase(str(db_path))

    # ─── 1. Initialize Runner ────────────────────────────────────────────────
    logger.info("\n--- 1. Initializing MultiCamRunner ---")
    
    runner = MultiCamRunner(
        config_path=Path("configs/store_layouts.json"),
        store_id=store_id,
        video_dir=video_dir,
        api_url=api_url if live else "http://localhost:8000",
        live=live,
    )
    
    if pos_csv_path:
        from pipeline.data_loader import PurpllePOSLoader
        loader = PurpllePOSLoader()
        loader.load_csv(pos_csv_path)
        pipeline_txns = loader.get_pipeline_transactions()
        runner.pos_correlator.transactions = pipeline_txns
        logger.info(f"Loaded {len(pipeline_txns)} transactions from POS.")

    # ─── 2. Run Pipeline ─────────────────────────────────────────────────────
    logger.info("\n--- 2. Processing 5 Cameras Concurrently ---")
    runner.run(max_frames=max_frames)

    # Finalize remaining sessions
    active_sessions = list(runner.session_manager.get_active_sessions().values())
    end_time = runner.last_processed_time or (runner.clip_start + __import__("datetime").timedelta(minutes=runner.layout["clip_duration_minutes"]))
    for s in active_sessions:
        duration = (end_time - s.entry_time).total_seconds() if s.entry_time else 0.0
        
        # Emit zone/billing exit first if currently inside a zone
        if s.current_zone:
            prev_entry = s.zone_entry_times.get(s.current_zone, s.entry_time)
            dwell_sec = (end_time - prev_entry).total_seconds() if prev_entry else 0.0
            if s.current_zone == "BILLING":
                queue_depth = runner._current_billing_depth(exclude_track_id=s.track_id)
                runner.event_emitter.emit_billing_event(
                    EventType.BILLING_QUEUE_EXIT,
                    visitor_id=s.visitor_id,
                    track_id=s.track_id,
                    timestamp=end_time,
                    queue_depth=queue_depth
                )
            else:
                runner.event_emitter.emit_zone_event(
                    EventType.ZONE_EXIT,
                    visitor_id=s.visitor_id,
                    track_id=s.track_id,
                    zone=s.current_zone,
                    timestamp=end_time,
                    dwell_seconds=dwell_sec,
                    confidence=0.8
                )

        # Finalize the session first
        was_staff = s.is_staff
        finalized = runner.session_manager.finalize_session(s.track_id, end_time)
        if finalized:
            is_staff = runner.staff_classifier.classify_session(finalized)
            if is_staff and not was_staff:
                runner.event_emitter.emit_staff_classification(
                    visitor_id=finalized.visitor_id,
                    track_id=finalized.track_id,
                    timestamp=end_time,
                    is_staff=True,
                    score=finalized.staff_score,
                    explanation=finalized.staff_explanation
                )
            
            # POS Correlation
            converted = False
            basket_value = 0.0
            if hasattr(runner, "pos_correlator") and runner.pos_correlator:
                billing_exit = finalized.billing_zone_exit_time
                is_conv, txn = runner.pos_correlator.correlate_session(billing_exit, runner.store_id)
                if is_conv and txn:
                    converted = True
                    basket_value = txn["basket_value"]
            # Only emit EXIT event if last_seen_camera is CAM_3
            if finalized.last_seen_camera == "CAM_3":
                runner.event_emitter.emit_exit(
                    finalized.visitor_id,
                    finalized.track_id,
                    end_time,
                    duration,
                    list(finalized.zones_visited),
                    is_staff,
                    camera_id=finalized.camera_id or "CAM_3",
                    converted=converted,
                    basket_value=basket_value
                )

    logger.info("\n--- 3. Finalizing Staff Classification (Offline) ---")
    finalized_sessions = runner.session_manager._finalized_sessions
    runner.staff_classifier.finalize_staff_classification(finalized_sessions)
    
    events = runner.event_emitter.get_events()
    
    # Update events with offline staff classification results
    for event in events:
        if event["visitor_id"] in finalized_sessions:
            session = finalized_sessions[event["visitor_id"]]
            event["is_staff"] = session.is_staff
            if event["event_type"] == "EXIT" and "metadata" in event:
                event["metadata"]["is_staff"] = session.is_staff
    
    # Emit missing STAFF_CLASSIFIED events for newly offline-classified staff
    for vid, session in finalized_sessions.items():
        if session.is_staff:
            has_staff_event = any(e["event_type"] == "STAFF_CLASSIFIED" and e["visitor_id"] == vid for e in events)
            if not has_staff_event:
                # Add to the end of events list
                runner.event_emitter.emit_staff_classification(
                    visitor_id=vid,
                    track_id=session.track_id,
                    timestamp=session.exit_time or end_time,
                    is_staff=True,
                    score=getattr(session, "staff_score", 1.0),
                    explanation=getattr(session, "staff_explanation", "STAFF (Offline)")
                )
    
    # Re-fetch events to include newly emitted STAFF_CLASSIFIED
    events = runner.event_emitter.get_events()

    if not live and events:
        # Keep the REST dashboard useful after a normal offline run. The API
        # reads this same SQLite database, while the JSON file is only an export.
        from app.database import EventDatabase
        EventDatabase(str(db_path)).insert_events_batch(events)
        logger.info("Persisted %d offline events to %s for API/dashboard reads.", len(events), db_path)

    logger.info("\n--- 4. Exporting Results ---")

    # Create output directory
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "real_pos_events.json", "w") as f:
        json.dump(events, f, indent=2)
        
    logger.info(f"Pipeline complete! Generated {len(events)} events.")
    
    if pos_csv_path:
        correlated = runner.pos_correlator.correlate_events_with_pos(events)
        with open(output_dir / "final_correlated_sessions.json", "w") as f:
            json.dump(correlated, f, indent=2)
        logger.info(f"POS Correlation complete. Matched {len(correlated)} sessions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run True Multi-Camera Analytics Pipeline")
    parser.add_argument("--extract-videos", action="store_true", help="Extract video datasets first")
    parser.add_argument("--pos-csv", type=str, help="Path to POS transaction CSV")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    parser.add_argument("--video-dir", type=str, default="data/videos", help="Directory containing CAM_X.mp4")
    parser.add_argument("--live", action="store_true", help="Push events in real-time to dashboard API")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000", help="Live API URL")
    
    args = parser.parse_args()
    
    main(
        extract=args.extract_videos,
        pos_csv_path=args.pos_csv,
        max_frames=args.max_frames,
        video_dir=args.video_dir,
        live=args.live,
        api_url=args.api_url
    )

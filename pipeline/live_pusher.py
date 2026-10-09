import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
import queue
import time

logger = logging.getLogger(__name__)

class LiveEventPusher:
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        
        self.session = requests.Session()
        # Only retry on actual server errors, NOT connection refused
        retries = Retry(total=1, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        
        self.q = queue.Queue()
        self._disabled = False
        self._consecutive_failures = 0
        self._max_failures = 3  # Circuit breaker: disable after 3 consecutive failures
        self.worker = threading.Thread(target=self._push_worker, daemon=True)
        self.worker.start()

    def push(self, event: dict):
        if not self._disabled:
            self.q.put(event)

    def _push_worker(self):
        while True:
            batch = []
            try:
                # get at least one
                batch.append(self.q.get(timeout=0.5))
                # drain the rest
                while not self.q.empty():
                    batch.append(self.q.get_nowait())
            except queue.Empty:
                continue
                
            if not batch or self._disabled:
                continue
                
            try:
                if len(batch) == 1:
                    res = self.session.post(f"{self.api_url}/events/ingest/single", json=batch[0], timeout=1.0)
                else:
                    res = self.session.post(f"{self.api_url}/events/ingest", json={"events": batch}, timeout=2.0)
                res.raise_for_status()
                self._consecutive_failures = 0  # Reset on success
            except Exception as e:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_failures:
                    self._disabled = True
                    logger.warning("LiveEventPusher DISABLED — backend unreachable after %d failures. Events will be saved locally only.", self._max_failures)
                else:
                    logger.warning(f"LiveEventPusher failed ({self._consecutive_failures}/{self._max_failures}): {e}")


class LiveFramePusher:
    """
    Asynchronously pushes compressed video frames to the dashboard API.
    Uses a thread-safe Queue and worker thread. If the queue fills up,
    old frames are discarded to prevent latency and memory bloat.
    """
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        self.session = requests.Session()
        # Non-blocking connections
        retries = Retry(total=1, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        
        # Lossy frame queue (maxsize=25 matches 5 cams * 5 frames)
        self.q = queue.Queue(maxsize=25)
        self._disabled = False
        self._consecutive_failures = 0
        self._max_failures = 3
        
        self.worker = threading.Thread(target=self._push_worker, daemon=True)
        self.worker.start()

    def push(self, camera_id: str, frame_b64: str):
        if self._disabled:
            return
            
        # If queue is full, discard the oldest items to maintain a real-time feed
        while self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
                
        self.q.put((camera_id, frame_b64))

    def _push_worker(self):
        while True:
            try:
                camera_id, frame_b64 = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
                
            if self._disabled:
                continue
                
            try:
                res = self.session.post(
                    f"{self.api_url}/stream/frame",
                    json={"camera_id": camera_id, "frame": frame_b64},
                    timeout=0.5
                )
                res.raise_for_status()
                self._consecutive_failures = 0
            except Exception as e:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_failures:
                    self._disabled = True
                    logger.warning(
                        "LiveFramePusher DISABLED — backend unreachable after %d failures. Frame streaming disabled.",
                        self._max_failures
                    )
                else:
                    logger.debug(f"LiveFramePusher failed ({self._consecutive_failures}/{self._max_failures}): {e}")

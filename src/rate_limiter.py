import json
import os
import tempfile
import time
from contextlib import contextmanager
from typing import Dict, List


@contextmanager
def _exclusive_file_lock(lock_path: str):
    """
    Cross-platform advisory file lock used to protect shared limiter state.
    """
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "a+b") as fh:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class CrossProcessRateLimiter:
    """
    Shared file-backed rate limiter so multiple Python processes cooperate.
    """

    def __init__(self, namespace: str = "gemini"):
        safe_namespace = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in namespace
        )
        base_dir = os.path.join(tempfile.gettempdir(), "hitl_gemini_rate_limit")
        self.state_path = os.path.join(base_dir, f"{safe_namespace}.json")
        self.lock_path = os.path.join(base_dir, f"{safe_namespace}.lock")

    def _load_state(self) -> Dict[str, object]:
        if not os.path.exists(self.state_path):
            return {
                "timestamps": [],
                "last_call_ts": 0.0,
                "backoff_until": 0.0,
            }

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {
                "timestamps": [],
                "last_call_ts": 0.0,
                "backoff_until": 0.0,
            }

        timestamps = data.get("timestamps", [])
        if not isinstance(timestamps, list):
            timestamps = []

        cleaned_timestamps: List[float] = []
        for ts in timestamps:
            if isinstance(ts, (int, float)):
                cleaned_timestamps.append(float(ts))

        last_call_ts = data.get("last_call_ts", 0.0)
        if not isinstance(last_call_ts, (int, float)):
            last_call_ts = 0.0

        backoff_until = data.get("backoff_until", 0.0)
        if not isinstance(backoff_until, (int, float)):
            backoff_until = 0.0

        return {
            "timestamps": cleaned_timestamps,
            "last_call_ts": float(last_call_ts),
            "backoff_until": float(backoff_until),
        }

    def _save_state(self, state: Dict[str, object]) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        tmp_path = f"{self.state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp_path, self.state_path)

    def wait_for_slot(
        self,
        requests_per_minute: int,
        window_seconds: float,
        min_interval_seconds: float,
        local_backoff_until: float,
    ) -> None:
        """
        Blocks until a request slot is available, then records the reservation.
        """
        while True:
            wait_time = 0.0
            now = time.time()

            with _exclusive_file_lock(self.lock_path):
                state = self._load_state()

                timestamps = [
                    ts for ts in state["timestamps"] if (now - ts) < window_seconds
                ]
                state["timestamps"] = timestamps

                global_backoff_until = float(state.get("backoff_until", 0.0))
                effective_backoff_until = max(global_backoff_until, local_backoff_until)
                if effective_backoff_until > now:
                    wait_time = effective_backoff_until - now

                if wait_time <= 0:
                    last_call_ts = float(state.get("last_call_ts", 0.0))
                    if min_interval_seconds > 0 and last_call_ts > 0:
                        since_last = now - last_call_ts
                        if since_last < min_interval_seconds:
                            wait_time = min_interval_seconds - since_last

                if wait_time <= 0 and len(timestamps) >= requests_per_minute:
                    oldest = timestamps[0]
                    wait_time = window_seconds - (now - oldest) + 0.05

                if wait_time <= 0:
                    state["timestamps"] = timestamps + [now]
                    state["last_call_ts"] = now
                    self._save_state(state)
                    return

                self._save_state(state)

            if wait_time > 0:
                time.sleep(wait_time)

    def note_server_backoff(self, delay_seconds: float) -> None:
        if delay_seconds <= 0:
            return

        until = time.time() + delay_seconds
        with _exclusive_file_lock(self.lock_path):
            state = self._load_state()
            state["backoff_until"] = max(float(state.get("backoff_until", 0.0)), until)
            self._save_state(state)

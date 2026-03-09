import os
import time
import threading
from queue import Queue


class GeminiRequestQueue:
    """
    Global request queue that serializes all Gemini API calls.
    Prevents bursts and eliminates 429 errors.
    """

    def __init__(self, min_interval: float = 35.0):
        self.queue = Queue()
        self.min_interval = min_interval
        self.last_call_time = 0.0

        worker = threading.Thread(target=self._worker_loop, daemon=True)
        worker.start()

    def _worker_loop(self):
        while True:
            fn, args, kwargs, result_queue = self.queue.get()

            # enforce spacing
            now = time.time()
            since_last = now - self.last_call_time
            if since_last < self.min_interval:
                time.sleep(self.min_interval - since_last)

            try:
                result = fn(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                result_queue.put(e)

            self.last_call_time = time.time()
            self.queue.task_done()

    def run(self, fn, *args, **kwargs):
        """
        Submit a request and block until result is ready.
        """
        result_queue = Queue(maxsize=1)
        self.queue.put((fn, args, kwargs, result_queue))

        result = result_queue.get()

        if isinstance(result, Exception):
            raise result

        return result


# GLOBAL QUEUE INSTANCE
_QUEUE_MIN_INTERVAL = float(
    os.getenv(
        "GEMINI_QUEUE_MIN_INTERVAL_SECONDS",
        os.getenv("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "0"),
    )
)
GLOBAL_GEMINI_QUEUE = GeminiRequestQueue(min_interval=max(0.0, _QUEUE_MIN_INTERVAL))

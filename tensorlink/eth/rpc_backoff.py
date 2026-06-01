import threading
import random
import time


class RPCBackoff:
    """
    Tracks consecutive RPC failures and enforces an exponential back-off so
    that a 429 or transient network error never floods Infura.

    Usage::

        backoff = RPCBackoff()
        while True:
            backoff.wait()          # sleeps if we are in a cool-down period
            try:
                result = rpc_call()
                backoff.success()   # resets the failure counter
                break
            except Exception as e:
                if backoff.is_rate_limit(e):
                    backoff.failure()   # increases cool-down
                else:
                    raise
    """

    # seconds: [0, 5, 10, 20, 40, 80, 160, 300]
    _DELAYS = [0, 5, 10, 20, 40, 80, 160, 300]

    def __init__(self):
        self._failures = 0
        self._next_allowed: float = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until it is safe to make another RPC call."""
        with self._lock:
            wait_secs = self._next_allowed - time.monotonic()
        if wait_secs > 0:
            time.sleep(wait_secs)

    def success(self) -> None:
        """Reset back-off after a successful call."""
        with self._lock:
            self._failures = 0
            self._next_allowed = 0.0

    def failure(self) -> float:
        """Record a failure and return the new delay in seconds."""
        with self._lock:
            self._failures = min(self._failures + 1, len(self._DELAYS) - 1)
            delay = self._DELAYS[self._failures]
            # Add a small jitter to avoid thundering-herd from multiple threads
            jitter = random.uniform(0, delay * 0.1)
            self._next_allowed = time.monotonic() + delay + jitter
        return delay

    @staticmethod
    def is_rate_limit(exc: Exception) -> bool:
        return "429" in str(exc) or "too many requests" in str(exc).lower()

    @staticmethod
    def is_nonce_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "nonce too low" in msg or "replacement transaction underpriced" in msg

    @property
    def consecutive_failures(self) -> int:
        return self._failures

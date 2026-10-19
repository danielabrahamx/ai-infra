"""Request queues. Default: in-memory or file. Optional: Redis."""
import json, os, threading


class MemoryQueue:
    """Thread-safe counters: pending (waiting) and in_flight (being served)."""

    def __init__(self, pending: int = 0, in_flight: int = 0):
        self._l = threading.Lock()
        self.pending, self.in_flight = pending, in_flight

    def depth(self) -> tuple[int, int]:
        with self._l:
            return self.pending, self.in_flight

    def push(self, n: int = 1):
        with self._l:
            self.pending += n


class FileQueue:
    """JSON file {"pending": int, "in_flight": int}. Another process may write it."""

    def __init__(self, path: str):
        self.path = path

    def depth(self) -> tuple[int, int]:
        if not os.path.exists(self.path):
            return 0, 0
        with open(self.path) as f:
            d = json.load(f)
        return int(d.get("pending", 0)), int(d.get("in_flight", 0))


class RedisQueue:
    """Pending = LLEN of a list. In-flight = SCARD of a set. Needs `redis` package.

    UNTESTED against a live Redis server.
    """

    def __init__(self, url="redis://localhost:6379/0", pending_key="requests:pending", inflight_key="requests:inflight"):
        import redis  # optional dependency
        self.r = redis.Redis.from_url(url)
        self.pk, self.ik = pending_key, inflight_key

    def depth(self) -> tuple[int, int]:
        return int(self.r.llen(self.pk)), int(self.r.scard(self.ik))

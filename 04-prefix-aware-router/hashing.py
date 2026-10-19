"""Prefix hashing and replica selection. Pure python, no I/O."""
import bisect, hashlib, math

def h64(s: str) -> int:
    """Stable 64-bit hash (Python's hash() is salted per process)."""
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")

def prefix_key(messages: list[dict], max_chars: int = 4096) -> str:
    """Key = system prompt if present, else the first message. Truncated."""
    for m in messages:
        if m.get("role") == "system":
            return m.get("content", "")[:max_chars]
    return (messages[0].get("content", "") if messages else "")[:max_chars]

def rendezvous_rank(key: str, replicas: list[str]) -> list[str]:
    """Highest-random-weight order. Removing a replica only moves its own keys."""
    kh = h64(key)
    return sorted(replicas, key=lambda r: h64(f"{r}|{kh}"), reverse=True)

class HashRing:
    """Consistent-hash ring with virtual nodes."""
    def __init__(self, replicas: list[str], vnodes: int = 128):
        self.points = sorted((h64(f"{r}#{i}"), r) for r in replicas for i in range(vnodes))
        self.keys = [p[0] for p in self.points]
        self.n = len(set(replicas))

    def rank(self, key: str) -> list[str]:
        """Walk the ring clockwise; first-seen order of distinct replicas."""
        i = bisect.bisect(self.keys, h64(key)) % len(self.points)
        out = []
        for j in range(len(self.points)):
            r = self.points[(i + j) % len(self.points)][1]
            if r not in out:
                out.append(r)
                if len(out) == self.n: break
        return out

def pick_bounded(ranking: list[str], load: dict[str, int], c: float = 1.25) -> str:
    """Bounded loads: take the first replica whose load stays <= ceil(c * avg).
    avg counts the incoming request. Falls back to least-loaded if none fit."""
    cap = math.ceil(c * (sum(load.values()) + 1) / len(ranking))
    for r in ranking:
        if load.get(r, 0) + 1 <= cap:
            return r
    return min(ranking, key=lambda r: load.get(r, 0))

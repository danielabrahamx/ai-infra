"""Pure scaling policy: queue depth in, desired replicas out. No I/O."""
import math
from dataclasses import dataclass


@dataclass
class Policy:
    target_pending_per_replica: float = 4.0
    min_replicas: int = 0
    max_replicas: int = 8
    scale_up_cooldown_s: float = 30.0
    scale_down_cooldown_s: float = 120.0
    scale_to_zero: bool = True        # allow 0 replicas when queue is empty
    idle_before_zero_s: float = 300.0  # queue must stay empty this long


@dataclass
class State:
    last_scale_up: float = -1e18
    last_scale_down: float = -1e18
    empty_since: float | None = None


def raw_desired(pending: int, in_flight: int, p: Policy) -> int:
    """Replicas needed for the load, ignoring cooldown."""
    if pending <= 0 and in_flight <= 0:
        return 0
    want = math.ceil(pending / p.target_pending_per_replica)
    return max(want, 1)  # work in flight keeps at least one replica (drain)


def decide(now: float, pending: int, in_flight: int, current: int, p: Policy, s: State) -> int:
    """Return the replica count to run now. Mutates State."""
    want = raw_desired(pending, in_flight, p)

    # Track how long the system has been fully empty.
    if want == 0:
        s.empty_since = now if s.empty_since is None else s.empty_since
    else:
        s.empty_since = None

    if want == 0:
        zero_ok = p.scale_to_zero and now - s.empty_since >= p.idle_before_zero_s
        if zero_ok:
            want = p.min_replicas
        elif current > 0:
            want = 1                       # drain to one warm replica
        else:
            want = p.min_replicas
    if not p.scale_to_zero:
        want = max(want, 1)
    want = min(max(want, p.min_replicas), p.max_replicas)

    if want > current:
        if now - s.last_scale_up < p.scale_up_cooldown_s:
            return current
        s.last_scale_up = now
        return want
    if want < current:
        if now - s.last_scale_down < p.scale_down_cooldown_s:
            return current
        s.last_scale_down = now
        return want
    return current

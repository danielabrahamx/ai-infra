"""SIMULATED load demo. Fake clock, fake provider, synthetic arrivals. No GPUs.

Model: each replica serves RATE requests/s. Replicas start after STARTUP_S.
Run: python -m autoscaler.demo
"""
from .controller import Controller
from .policy import Policy
from .providers import FakeProvider
from .queue import MemoryQueue

RATE, STARTUP_S, TICK = 0.5, 20, 5


def arrivals(t):
    """Synthetic traffic: quiet, burst at 60-180s, quiet again."""
    return 3.0 if 60 <= t < 180 else 0.0


def simulate(duration=900):
    clock = [0.0]
    q, prov = MemoryQueue(), FakeProvider(0)
    pol = Policy(target_pending_per_replica=10, max_replicas=6, scale_up_cooldown_s=15,
                 scale_down_cooldown_s=60, idle_before_zero_s=120)
    ctl = Controller(q, prov, pol, clock=lambda: clock[0])
    ready_at, ready = {}, 0  # simple warmup: replicas count as ready STARTUP_S after request
    log, prev, carry = [], 0, 0.0
    for t in range(0, duration, TICK):
        clock[0] = float(t)
        carry += arrivals(t) * TICK
        q.push(int(carry)); carry -= int(carry)
        want = ctl.step()
        if want > prev:
            ready_at[t + STARTUP_S] = want
        prev = want
        ready = min(want, max([v for k, v in ready_at.items() if k <= t], default=0)) if want else 0
        served = min(q.pending, int(ready * RATE * TICK))
        q.pending -= served
        q.in_flight = int(ready * RATE * TICK) if q.pending else 0
        log.append((t, q.pending, want, ready))
    return log, prov


if __name__ == "__main__":
    log, prov = simulate()
    print("SIMULATED run (synthetic traffic, fake provider)")
    print(" t(s) pending desired ready")
    for t, p, w, r in log[::6]:
        print(f"{t:5d} {p:7d} {w:7d} {r:5d}")
    peak = max(l[2] for l in log)
    print(f"peak replicas={peak}, final replicas={log[-1][2]}, scale calls={prov.calls}")

"""Discrete-event simulator: continuous batching, chunked vs unchunked prefill.
ALL RESULTS ARE SIMULATED from a linear step-cost model. Not measured on GPU.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field

@dataclass
class Cost:
    """step_ms = base + per_decode*n_decode + per_prefill_tok*n_prefill_tokens."""
    base_ms: float = 8.0
    per_decode_ms: float = 0.05
    per_prefill_tok_ms: float = 0.12

@dataclass(eq=False)
class Req:
    rid: int
    arrival: float          # ms
    prompt: int
    out: int
    done_prefill: int = 0
    emitted: int = 0
    token_times: list = field(default_factory=list)

def make_workload(n=200, rate_per_s=1.0, long_frac=0.1, short=(100, 500),
                  long=(16000, 32000), out=(64, 256), seed=0):
    """Poisson arrivals; mix of short chat prompts and long-context prompts."""
    rng = random.Random(seed)
    t, reqs = 0.0, []
    for i in range(n):
        t += rng.expovariate(rate_per_s) * 1000
        lo, hi = long if rng.random() < long_frac else short
        reqs.append(Req(i, t, rng.randint(lo, hi), rng.randint(*out)))
    return reqs

def simulate(reqs, budget, chunked, cost=Cost(), max_seqs=64):
    """Iteration-level scheduler. budget = max_num_batched_tokens.
    chunked: decodes first, leftover budget goes to (partial) prefills.
    unchunked: waiting prompts are prefilled whole (prefill-priority step,
    decodes stall); a prompt longer than budget still runs alone, whole."""
    pending = sorted(reqs, key=lambda r: r.arrival)
    waiting, running, finished = [], [], []
    now, pi = 0.0, 0
    while len(finished) < len(reqs):
        while pi < len(pending) and pending[pi].arrival <= now:
            waiting.append(pending[pi]); pi += 1
        if not waiting and not running:
            now = pending[pi].arrival; continue
        decoding = [r for r in running if r.done_prefill >= r.prompt]
        prefilling = [r for r in running if r.done_prefill < r.prompt]
        while waiting and len(running) < max_seqs:   # admit
            r = waiting.pop(0); running.append(r); prefilling.append(r)
        plan = {}  # req -> prefill tokens this step
        if chunked:
            left = budget - len(decoding)
            for r in prefilling:
                take = min(r.prompt - r.done_prefill, max(left, 0))
                if take > 0: plan[r] = take; left -= take
            dec = decoding
        elif prefilling:
            left = budget
            for r in prefilling:   # whole prompts only
                need = r.prompt - r.done_prefill
                if need <= left or not plan: plan[r] = need; left -= need
            dec = []               # decode stalls during prefill step
        else:
            dec = decoding
        ptoks = sum(plan.values())
        now += cost.base_ms + cost.per_decode_ms * len(dec) + cost.per_prefill_tok_ms * ptoks
        for r, k in plan.items():
            r.done_prefill += k
            if r.done_prefill >= r.prompt:   # prefill completes -> first token
                r.emitted = 1; r.token_times.append(now)
        for r in dec:
            r.emitted += 1; r.token_times.append(now)
        for r in list(running):
            if r.emitted >= r.out:
                running.remove(r); finished.append(r)
    return summarize(reqs, now)

def pct(xs, p):
    if not xs: return float("nan")
    xs = sorted(xs)   # nearest-rank percentile, p may be fractional
    return xs[min(len(xs) - 1, max(0, int(-(-p * len(xs) // 100)) - 1))]

def summarize(reqs, end):
    ttft = [r.token_times[0] - r.arrival for r in reqs]
    itl = [b - a for r in reqs for a, b in zip(r.token_times, r.token_times[1:])]
    toks = sum(len(r.token_times) for r in reqs)
    return {"ttft_p50_ms": pct(ttft, 50), "ttft_p99_ms": pct(ttft, 99),
            "itl_p50_ms": pct(itl, 50), "itl_p99_ms": pct(itl, 99),
            "itl_p999_ms": pct(itl, 99.9), "itl_max_ms": max(itl),
            "throughput_tok_s": toks / (end - reqs[0].arrival) * 1000}

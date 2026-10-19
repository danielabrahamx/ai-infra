"""Load-test harness for any OpenAI-compatible /chat/completions endpoint (streaming).

Measures per request: TTFT, inter-token latencies (ITL), end-to-end time, output tokens.
Runs a concurrency sweep. Writes one JSON file that follows results/schema.json.
"""
import argparse
import asyncio
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import httpx

HARNESS_VERSION = "0.1.0"
WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau".split()


def make_prompt(n_words: int, seed: int) -> str:
    """Deterministic synthetic prompt of about n_words words. Same seed -> same prompt."""
    rng = random.Random(seed)
    return "Continue this text: " + " ".join(rng.choice(WORDS) for _ in range(n_words))


def pct(values: list, p: float):
    """Nearest-rank percentile. Returns None for empty input."""
    if not values:
        return None
    v = sorted(values)
    return v[max(0, min(len(v) - 1, math.ceil(p / 100 * len(v)) - 1))]


def summarize(values: list) -> dict:
    return {"mean": statistics.fmean(values) if values else None,
            "p50": pct(values, 50), "p95": pct(values, 95), "p99": pct(values, 99)}


async def one_request(client: httpx.AsyncClient, url: str, headers: dict, payload: dict) -> dict:
    """Send one streaming request. Token = one SSE chunk with non-empty content."""
    t0 = time.perf_counter()
    stamps, usage_tokens, err = [], None, None
    try:
        async with client.stream("POST", url, json=payload, headers=headers) as r:
            if r.status_code != 200:
                await r.aread()
                return {"ok": False, "error": f"HTTP {r.status_code}", "e2e_ms": (time.perf_counter() - t0) * 1000}
            async for line in r.aiter_lines():
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                try:
                    obj = json.loads(line[5:])
                except ValueError:
                    continue
                if obj.get("usage"):
                    usage_tokens = obj["usage"].get("completion_tokens")
                ch = obj.get("choices") or []
                if ch and (ch[0].get("delta") or {}).get("content"):
                    stamps.append(time.perf_counter())
    except Exception as ex:  # noqa: BLE001 - record any failure as an error sample
        err = f"{type(ex).__name__}: {ex}"
    e2e = (time.perf_counter() - t0) * 1000
    if err or not stamps:
        return {"ok": False, "error": err or "no tokens", "e2e_ms": e2e}
    return {"ok": True, "ttft_ms": (stamps[0] - t0) * 1000,
            "itl_ms": [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])],
            "tokens": usage_tokens or len(stamps), "e2e_ms": e2e}


async def run_level(args, concurrency: int, n_requests: int, seed0: int) -> dict:
    """Closed loop: `concurrency` workers each send requests back to back."""
    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ[args.api_key_env]}"} if args.api_key_env and os.environ.get(args.api_key_env) else {}
    queue = list(range(n_requests))
    results = []

    async def worker(client):
        while queue:
            i = queue.pop(0)
            payload = {"model": args.model, "stream": True, "max_tokens": args.max_tokens, "temperature": 0,
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": make_prompt(args.prompt_words, seed0 + i)}]}
            results.append(await one_request(client, url, headers, payload))

    async with httpx.AsyncClient(timeout=args.timeout, limits=httpx.Limits(max_connections=concurrency + 4)) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        wall = time.perf_counter() - t0
    ok = [r for r in results if r["ok"]]
    toks = sum(r["tokens"] for r in ok)
    return {"concurrency": concurrency, "requests": n_requests, "errors": len(results) - len(ok),
            "wall_s": wall, "throughput_tok_s": toks / wall if wall else 0.0, "req_s": len(ok) / wall if wall else 0.0,
            "output_tokens": toks,
            "ttft_ms": summarize([r["ttft_ms"] for r in ok]),
            "itl_ms": summarize([x for r in ok for x in r["itl_ms"]]),
            "e2e_ms": summarize([r["e2e_ms"] for r in ok])}


def sh(cmd: list) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def env_metadata() -> dict:
    """Record where the numbers came from. Missing tools give null, never a guess."""
    return {"platform": platform.platform(), "python": sys.version.split()[0], "cpu": platform.processor() or None,
            "cpu_count": os.cpu_count(),
            "gpu": sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
            "git_commit": sh(["git", "rev-parse", "--short", "HEAD"]), "httpx": httpx.__version__}


async def sweep(args) -> dict:
    levels = [int(x) for x in args.concurrency.split(",")]
    if args.warmup:  # warmup is discarded
        await run_level(args, 1, args.warmup, seed0=10_000)
    out = []
    for c in levels:
        n = max(args.requests_per_level, c * args.min_per_worker)
        print(f"concurrency={c} requests={n} ...", file=sys.stderr)
        out.append(await run_level(args, c, n, seed0=0))
    return {"schema_version": 1, "harness_version": HARNESS_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label": args.label, "validation_only": args.validation_only,
            "target": {"base_url": args.base_url, "model": args.model, "engine": args.engine, "quantization": args.quantization},
            "workload": {"prompt_words": args.prompt_words, "max_tokens": args.max_tokens, "warmup": args.warmup,
                         "requests_per_level": args.requests_per_level, "seed": 0},
            "environment": env_metadata(), "levels": out}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True, help="e.g. http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="", help="Name of env var holding the API key")
    p.add_argument("--concurrency", default="1,2,4,8,16")
    p.add_argument("--requests-per-level", type=int, default=32)
    p.add_argument("--min-per-worker", type=int, default=2)
    p.add_argument("--prompt-words", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--engine", default="unknown", help="e.g. vllm 0.6.3, llama.cpp b3000")
    p.add_argument("--quantization", default="unknown")
    p.add_argument("--label", default="", help="Free text note stored in the result")
    p.add_argument("--validation-only", action="store_true", help="Mark result as harness validation, not a perf claim")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    res = asyncio.run(sweep(args))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

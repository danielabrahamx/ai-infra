# 04 - Prefix-aware router

Roadmap stage 4 (AI Infrastructure Engineering series). A routing proxy that sends requests with the same system prompt to the same replica. The replica keeps a warm KV prefix cache, so prefill is skipped.

## What is here
- `hashing.py` - prefix key, rendezvous hashing, consistent-hash ring, bounded-load pick.
- `router.py` - FastAPI + httpx proxy. Strategies: `rendezvous`, `ring`, `round_robin`. `GET /router/stats`.
- `mock_replica.py` - mock replica with an LRU prefix cache. Hit = faster (simulated by sleep).
- `asgi_mux.py` - in-process transport so router and replicas run without sockets.
- `demo.py` - hit-rate comparison. `tests/test_router.py` - 8 tests.

## Quickstart
```
python -m pip install -r requirements.txt
python -m pytest -q
python demo.py
```

## How routing works
1. Key = system prompt (else first message), first 4096 chars.
2. Rank replicas by rendezvous hash (or ring walk) of the key.
3. Take the first replica with load <= ceil(1.25 x average in-flight). Else the next one. This is bounded-load hashing. A hot prefix spills to the next replica instead of overloading one.

## Demo results (SIMULATED: mock replicas, sleep-based latency, in-process)
Setup: 4 replicas, 24 prefixes with skewed popularity, cache capacity 8 prefixes per replica, 600 requests, concurrency 16.

| strategy | cache hit rate | mean latency ms | requests per replica |
|---|---|---|---|
| round_robin | 0.478 | 19.76 | 150/150/150/150 |
| ring | 0.770 | 11.51 | 163/90/188/159 |
| rendezvous | 0.810 | 10.38 | 121/152/162/165 |

Round robin gives each replica all 24 prefixes. Capacity is 8, so it thrashes. Hash routing gives each replica about 6 prefixes, which fit. p99 is the same (33.3 ms) because a cold miss costs the same on every strategy. Latency values come from the mock model, not hardware.

## Status
- Verified on CPU: 8 tests pass. `demo.py` runs. In-process only.
- NOT tested: real HTTP sockets, real vLLM replicas, streaming responses. The router does not stream yet.
- To use real servers: build `create_router(["http://host1:8000", ...])` in your own module and run it with uvicorn. No module-level `app` is provided.
- Real vLLM hit rates depend on its own prefix caching (`--enable-prefix-caching`). Not measured.

"""Prefix-aware routing proxy (FastAPI + httpx).
Same prefix -> same replica (warm KV prefix cache), unless that replica is overloaded."""
import itertools
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from hashing import HashRing, pick_bounded, prefix_key, rendezvous_rank

def create_router(replicas: list[str], client: httpx.AsyncClient | None = None,
                  strategy: str = "rendezvous", load_factor: float = 1.25) -> FastAPI:
    """strategy: rendezvous | ring | round_robin. client is injectable for tests."""
    app = FastAPI()
    http = client or httpx.AsyncClient(timeout=120)
    ring = HashRing(replicas) if strategy == "ring" else None
    rr = itertools.cycle(replicas)
    load = {r: 0 for r in replicas}      # in-flight requests per replica
    sent = {r: 0 for r in replicas}      # total routed per replica

    def choose(body: dict) -> str:
        if strategy == "round_robin":
            return next(rr)
        key = prefix_key(body.get("messages", []))
        ranking = ring.rank(key) if ring else rendezvous_rank(key, replicas)
        return pick_bounded(ranking, load, load_factor)

    @app.post("/v1/chat/completions")
    async def proxy(request: Request):
        body = await request.json()
        target = choose(body)
        load[target] += 1; sent[target] += 1
        try:
            r = await http.post(f"{target}/v1/chat/completions", json=body)
        finally:
            load[target] -= 1
        out = r.json()
        out["routed_to"] = target
        return JSONResponse(out, status_code=r.status_code)

    @app.get("/router/stats")
    async def stats():
        return {"strategy": strategy, "sent": sent, "inflight": load}

    return app

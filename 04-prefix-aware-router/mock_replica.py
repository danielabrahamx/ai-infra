"""Mock replica with a simulated prefix KV cache (LRU over prefix hashes).
Cache hit => prefix prefill is skipped => faster. Latency is SIMULATED with sleep."""
import asyncio
from collections import OrderedDict
from fastapi import FastAPI, Request
from hashing import h64, prefix_key

def create_replica(name: str, capacity: int = 8, ms_per_prefix_token: float = 0.05,
                   ms_base: float = 5.0) -> FastAPI:
    app = FastAPI()
    cache: OrderedDict[int, None] = OrderedDict()   # prefix hash -> present
    st = {"hits": 0, "misses": 0}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        prefix = prefix_key(body.get("messages", []))
        ph = h64(prefix)
        hit = ph in cache
        if hit: cache.move_to_end(ph); st["hits"] += 1
        else:
            st["misses"] += 1; cache[ph] = None
            if len(cache) > capacity: cache.popitem(last=False)   # evict LRU
        ptoks = len(prefix.split())
        latency_ms = ms_base + (0 if hit else ms_per_prefix_token * ptoks)
        await asyncio.sleep(latency_ms / 1000)
        return {"replica": name, "cache_hit": hit, "latency_ms": latency_ms,
                "choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    @app.get("/stats")
    async def stats():
        t = st["hits"] + st["misses"]
        return {**st, "hit_rate": st["hits"] / t if t else 0.0, "cached": len(cache)}

    return app

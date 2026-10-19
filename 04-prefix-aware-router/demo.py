"""Demo: cache hit rate, prefix-aware vs round-robin. SIMULATED (mock replicas, in-process).
Run: python demo.py"""
import asyncio, json, random
import httpx
from asgi_mux import MuxTransport
from mock_replica import create_replica
from router import create_router

N_REPL, N_PREFIX, N_REQ, CAP, CONC = 4, 24, 600, 8, 16
PREFIXES = [f"You are assistant #{i}. " + "Follow policy section words. " * 200 for i in range(N_PREFIX)]

def workload(seed=0):
    rng = random.Random(seed)
    w = [1 / (i + 1) ** 0.8 for i in range(N_PREFIX)]    # skewed popularity
    return [rng.choices(range(N_PREFIX), w)[0] for _ in range(N_REQ)]

async def run(strategy):
    names = [f"replica{i}" for i in range(N_REPL)]
    apps = {n: create_replica(n, capacity=CAP) for n in names}
    mux = httpx.AsyncClient(transport=MuxTransport(apps), timeout=60)
    router = create_router([f"http://{n}" for n in names], client=mux, strategy=strategy)
    front = httpx.AsyncClient(transport=httpx.ASGITransport(app=router), base_url="http://router")
    sem, lat = asyncio.Semaphore(CONC), []
    async def one(i, p):
        async with sem:
            r = await front.post("/v1/chat/completions", json={"messages": [
                {"role": "system", "content": PREFIXES[p]}, {"role": "user", "content": f"q{i}"}]})
            lat.append(r.json()["latency_ms"])
    await asyncio.gather(*(one(i, p) for i, p in enumerate(workload())))
    hits = misses = 0; per = []
    for n in names:
        s = (await mux.get(f"http://{n}/stats")).json()
        hits += s["hits"]; misses += s["misses"]; per.append(s["hits"] + s["misses"])
    lat.sort()
    return {"strategy": strategy, "hit_rate": round(hits / (hits + misses), 3),
            "mean_latency_ms": round(sum(lat) / len(lat), 2), "p99_latency_ms": round(lat[int(.99 * len(lat))], 2),
            "reqs_per_replica": per}

async def main():
    res = [await run(s) for s in ("round_robin", "ring", "rendezvous")]
    print("SIMULATED results (mock replicas; latency = sleep model, not hardware)")
    print(f"{N_REPL} replicas, {N_PREFIX} distinct prefixes, cache capacity {CAP}/replica, {N_REQ} requests")
    for r in res: print(json.dumps(r))

asyncio.run(main())

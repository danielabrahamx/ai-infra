import asyncio, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import httpx
from asgi_mux import MuxTransport
from hashing import HashRing, pick_bounded, prefix_key, rendezvous_rank
from mock_replica import create_replica
from router import create_router

R = [f"http://replica{i}" for i in range(4)]

def msgs(sys_, q="hi"):
    return {"messages": [{"role": "system", "content": sys_}, {"role": "user", "content": q}]}

def test_prefix_key_uses_system():
    assert prefix_key(msgs("SYS")["messages"]) == "SYS"

def test_rendezvous_deterministic_and_minimal_disruption():
    keys = [f"k{i}" for i in range(500)]
    before = {k: rendezvous_rank(k, R)[0] for k in keys}
    assert before == {k: rendezvous_rank(k, R)[0] for k in keys}
    after = {k: rendezvous_rank(k, R[:3])[0] for k in keys}
    # only keys that lived on the removed replica may move
    assert all(before[k] == after[k] for k in keys if before[k] != R[3])

def test_ring_deterministic_and_covers_all():
    ring = HashRing(R)
    assert ring.rank("x") == ring.rank("x") and set(ring.rank("x")) == set(R)

def test_distribution_roughly_even():
    counts = {r: 0 for r in R}
    for i in range(2000): counts[rendezvous_rank(f"k{i}", R)[0]] += 1
    assert min(counts.values()) > 350

def test_bounded_load_fallback():
    ranking = ["a", "b", "c", "d"]
    assert pick_bounded(ranking, {"a": 0, "b": 0, "c": 0, "d": 0}) == "a"
    # a is hot: cap = ceil(1.25*(10+1)/4)=4, a has 10 -> skip to b
    assert pick_bounded(ranking, {"a": 10, "b": 0, "c": 0, "d": 0}) == "b"

def _stack(strategy, cap=8):
    apps = {f"replica{i}": create_replica(f"replica{i}", capacity=cap, ms_base=0, ms_per_prefix_token=0) for i in range(4)}
    mux = httpx.AsyncClient(transport=MuxTransport(apps))
    app = create_router(R, client=mux, strategy=strategy)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router")

def test_same_prefix_same_replica_and_cache_hit():
    async def go():
        c = _stack("rendezvous")
        rs = [(await c.post("/v1/chat/completions", json=msgs("SAME PREFIX", f"q{i}"))).json() for i in range(5)]
        assert len({r["routed_to"] for r in rs}) == 1
        assert [r["cache_hit"] for r in rs] == [False, True, True, True, True]
    asyncio.run(go())

def test_round_robin_spreads_same_prefix():
    async def go():
        c = _stack("round_robin")
        rs = [(await c.post("/v1/chat/completions", json=msgs("SAME", f"q{i}"))).json() for i in range(4)]
        assert len({r["routed_to"] for r in rs}) == 4
        assert not any(r["cache_hit"] for r in rs)
    asyncio.run(go())

def test_concurrent_requests_all_succeed_and_counted():
    async def go():
        c = _stack("rendezvous")
        rs = await asyncio.gather(*(c.post("/v1/chat/completions", json=msgs("HOT", f"q{i}")) for i in range(16)))
        assert all(r.status_code == 200 for r in rs)
        st = (await c.get("/router/stats")).json()
        assert sum(st["sent"].values()) == 16
    asyncio.run(go())

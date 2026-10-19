import copy, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim import make_workload, simulate, Req, Cost

def test_all_tokens_emitted():
    reqs = make_workload(n=30, seed=1)
    simulate(reqs, 2048, True)
    assert all(len(r.token_times) == r.out for r in reqs)

def test_unchunked_all_tokens_emitted():
    reqs = make_workload(n=30, seed=1)
    simulate(reqs, 32768, False)
    assert all(len(r.token_times) == r.out for r in reqs)

def test_chunking_reduces_decode_stall():
    base = make_workload(n=120, seed=0)
    u = simulate(copy.deepcopy(base), 32768, False)
    c = simulate(copy.deepcopy(base), 512, True)
    assert c["itl_p999_ms"] < u["itl_p999_ms"]

def test_single_request_timing():
    r = [Req(0, 0.0, 1000, 3)]
    c = Cost(base_ms=1, per_decode_ms=0, per_prefill_tok_ms=0.01)
    out = simulate(r, 1000, True, cost=c)
    assert abs(r[0].token_times[0] - 11.0) < 1e-9   # 1 + 1000*0.01
    assert len(r[0].token_times) == 3

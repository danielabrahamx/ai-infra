import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import pytest
from kv_cost_model import *
from sweep import busbw_factor
from parse_plot import parse_nccl_tests, group

M = Model(layers=32, kv_heads=32, head_dim=128)


def test_kv_bytes_per_token_llama7b():
    assert M.kv_bytes_per_token() == 2 * 32 * 32 * 128 * 2 == 524288


def test_gqa_shrinks_kv():
    g = Model(layers=32, kv_heads=8, head_dim=128)
    assert g.kv_bytes_per_token() * 4 == M.kv_bytes_per_token()


def test_transfer_time_linear_in_bytes():
    l = Link(bandwidth_GBps=10, latency_us=0)
    assert transfer_time_s(2_000_000_000, l) == pytest.approx(0.2)


def test_latency_per_chunk():
    assert transfer_time_s(0, Link(10, 5, chunks=4)) == pytest.approx(20e-6)


def test_disagg_ttft_adds_transfer():
    l = Link(10, 0)
    base = prefill_time_s(M, 1024, 100)
    assert ttft_disagg_s(M, 1024, l, 100) == pytest.approx(base + kv_bytes(M, 1024) / 10e9)


def test_overlap_never_slower():
    l = Link(10, 5, chunks=32)
    assert ttft_disagg_s(M, 2048, l, 100, overlap=True) <= ttft_disagg_s(M, 2048, l, 100)


def test_breakeven_bandwidth():
    bw = breakeven_bandwidth_GBps(M, 2048, 100, budget_frac=0.1)
    l = Link(bw, 0)
    assert transfer_time_s(kv_bytes(M, 2048), l) == pytest.approx(0.1 * prefill_time_s(M, 2048, 100))


def test_busbw_factor():
    assert busbw_factor(1) == 0 and busbw_factor(2) == 1.0 and busbw_factor(8) == pytest.approx(1.75)


def test_parse_nccl_tests():
    txt = """#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
     1048576        262144     float     sum      -1    100.5   10.43   18.26      0    99.0   10.59   18.53      0
# Avg bus bandwidth : 18
"""
    r = parse_nccl_tests(txt)
    assert len(r) == 1 and r[0]["bytes"] == 1048576 and r[0]["busbw_GBps"] == 18.26


def test_group_sorted():
    rows = [dict(backend="g", topology="intra-node", label="", world_size=2, bytes=b) for b in (4, 1, 2)]
    assert [r["bytes"] for r in list(group(rows).values())[0]] == [1, 2, 4]

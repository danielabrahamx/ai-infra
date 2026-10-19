import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from commcost import (analyze, Plan, LLAMA_405B, MOE_405B,
                      ring_allreduce_bytes, alltoall_bytes)


def test_ring_formula():
    assert ring_allreduce_bytes(100, 1) == 0
    assert ring_allreduce_bytes(100, 2) == 100
    assert ring_allreduce_bytes(100, 8) == pytest.approx(175)


def test_alltoall_formula():
    assert alltoall_bytes(80, 8) == pytest.approx(70)


def test_tp8_msg_size_and_count():
    r = analyze(LLAMA_405B, Plan(tp=8, pp=1))
    assert r["tp_allreduce"]["count_per_token"] == 252
    assert r["tp_allreduce"]["msg_bytes"] == 16384 * 2
    assert r["tp_allreduce"]["bytes_per_token"] == pytest.approx(252 * 1.75 * 32768)
    assert r["pp_p2p"]["bytes_per_token"] == 0


def test_tp_across_nodes_slower_than_in_node():
    fast = analyze(LLAMA_405B, Plan(tp=8))["total_time_s"]
    slow = analyze(LLAMA_405B, Plan(tp=64))["total_time_s"]
    assert slow > 5 * fast


def test_pcie_slower_than_nvlink():
    a = analyze(LLAMA_405B, Plan(tp=8, intra="nvlink4"))["total_time_s"]
    b = analyze(LLAMA_405B, Plan(tp=8, intra="pcie4"))["total_time_s"]
    assert b > a


def test_pp_hops():
    r = analyze(LLAMA_405B, Plan(tp=8, pp=8))
    assert r["pp_p2p"]["count_per_token"] == 7
    assert r["tp_allreduce"]["count_per_token"] == 30  # 126//8 = 15 layers * 2


def test_ep_only_for_moe():
    assert analyze(LLAMA_405B, Plan(ep=8))["ep_alltoall"]["bytes_per_token"] == 0
    r = analyze(MOE_405B, Plan(ep=8))["ep_alltoall"]
    assert r["count_per_token"] == 252
    assert r["bytes_per_token"] == pytest.approx(252 * 7 / 8 * 2 * 32768)


def test_batch_scales_bytes():
    a = analyze(LLAMA_405B, Plan(tp=8))["tp_allreduce"]["bytes_per_token"]
    b = analyze(LLAMA_405B, Plan(tp=8, batch_tokens=4))["tp_allreduce"]["bytes_per_token"]
    assert b == pytest.approx(4 * a)

import pytest
from roofline import *

L8, L70, L405 = MODELS["llama3-8b"], MODELS["llama3-70b"], MODELS["llama3-405b"]


def test_param_counts_match_published():
    assert L8.params() / 1e9 == pytest.approx(8.03, abs=0.05)
    assert L70.params() / 1e9 == pytest.approx(70.55, abs=0.3)
    assert L405.params() / 1e9 == pytest.approx(405.8, abs=1.5)


def test_kv_per_token_8b():
    # 2 * 32 layers * 8 kv heads * 128 dim * 2 bytes = 131072 B = 128 KiB
    assert kv_bytes_per_token(L8) == 131072


def test_kv_per_token_70b_405b():
    assert kv_bytes_per_token(L70) == 2 * 80 * 8 * 128 * 2   # 320 KiB
    assert kv_bytes_per_token(L405) == 2 * 126 * 8 * 128 * 2  # 504 KiB


def test_kv_scales_linearly():
    assert kv_cache_bytes(L8, 4, 1000) == 4 * 1000 * kv_bytes_per_token(L8)
    assert kv_cache_bytes(L8, 1, 10, bytes_per=1) * 2 == kv_cache_bytes(L8, 1, 10)


def test_ridge_points():
    assert GPUS["h100"].ridge == pytest.approx(295, rel=0.01)
    assert GPUS["a100"].ridge == pytest.approx(153, rel=0.01)


def test_decode_batch1_is_memory_bound_everywhere():
    for g in GPUS.values():
        assert decode_intensity(L8, 1, 1024) < g.ridge / 10


def test_decode_intensity_about_one_flop_per_byte_at_fp16():
    # 2 FLOP per weight, 2 bytes per weight -> ~1 FLOP/B at batch 1.
    assert decode_intensity(L8, 1, 128) == pytest.approx(1.0, abs=0.1)


def test_intensity_grows_with_batch():
    assert decode_intensity(L8, 64, 512) > decode_intensity(L8, 1, 512)


def test_prefill_more_intense_than_decode():
    assert prefill_intensity(L8, 1, 2048) > 100 * decode_intensity(L8, 1, 2048)


def test_roofline_clamps():
    g = GPUS["a100"]
    assert attainable(g, 1.0) == g.bandwidth
    assert attainable(g, 1e6) == g.peak_flops


def test_decode_time_is_weight_read_at_batch1():
    g = GPUS["h100"]
    t = decode_time_per_token(L8, g, 1, 1)
    assert t == pytest.approx(weight_bytes(L8) / g.bandwidth, rel=0.02)


def test_405b_does_not_fit_one_h100_but_8b_fits_a100():
    assert max_batch(L405, GPUS["h100"], 4096) == 0
    assert max_batch(L8, GPUS["a100"], 4096) > 0
    assert max_batch(L8, GPUS["rtx2050"], 4096) == 0  # 16 GB fp16 weights > 4 GiB


def test_report_runs():
    assert "memory-bound" in report(L8, GPUS["h100"], 1, 2048)

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import torch
from quant import quantize, dequantize, pack_int4, unpack_int4, fake_quant, quantize_model


def test_int4_pack_roundtrip():
    q = torch.randint(-8, 8, (4, 16), dtype=torch.int8)
    assert torch.equal(unpack_int4(pack_int4(q)), q)


def test_code_range():
    w = torch.randn(8, 128)
    q8, _ = quantize(w, 8, 64); q4, _ = quantize(w, 4, 64)
    assert q8.min() >= -128 and q8.max() <= 127
    assert q4.min() >= -8 and q4.max() <= 7


def test_error_bounded_by_half_scale():
    w = torch.randn(16, 256)
    for bits in (4, 8):
        q, s = quantize(w, bits, 64)
        err = (dequantize(q, s, 64) - w).abs().reshape(16, 4, 64)
        assert (err <= s.unsqueeze(-1) / 2 + 1e-6).all()


def test_int8_better_than_int4_and_smaller_group_better():
    w = torch.randn(32, 256)
    e = lambda b, g: (fake_quant(w, b, g) - w).pow(2).mean().item()
    assert e(8, 64) < e(4, 64) and e(4, 32) < e(4, 256)


def test_zero_weights_safe():
    assert torch.equal(fake_quant(torch.zeros(2, 64), 4, 32), torch.zeros(2, 64))


def test_quantize_model_changes_linear_only():
    m = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.LayerNorm(64))
    ln = m[1].weight.clone(); lin = m[0].weight.clone()
    fp, q = quantize_model(m, 4, 32)
    assert torch.equal(m[1].weight, ln) and not torch.equal(m[0].weight, lin) and fp > q

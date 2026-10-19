"""From-scratch symmetric group-wise weight quantizer (INT8 / INT4) in torch.

Scheme: for each group of `group_size` consecutive input-channel weights,
scale = max|w| / qmax, q = clamp(round(w / scale), -qmax-1, qmax).
No zero point (symmetric). INT4 values can be packed two per byte.
"""
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D  # GPT-2 uses Conv1D (weight is in x out)


def quantize(w: torch.Tensor, bits: int = 4, group_size: int = 128):
    """w: (out, in). Returns int8 codes (out, in) and fp32 scales (out, in/group)."""
    assert bits in (4, 8) and w.dim() == 2
    out_f, in_f = w.shape
    assert in_f % group_size == 0, "in_features must divide by group_size"
    qmax = 2 ** (bits - 1) - 1
    g = w.float().reshape(out_f, in_f // group_size, group_size)
    scale = (g.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-8)
    q = torch.clamp(torch.round(g / scale), -qmax - 1, qmax).to(torch.int8)
    return q.reshape(out_f, in_f), scale.squeeze(-1)


def dequantize(q: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    out_f, in_f = q.shape
    g = q.float().reshape(out_f, in_f // group_size, group_size)
    return (g * scale.unsqueeze(-1)).reshape(out_f, in_f)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack signed 4-bit codes (-8..7) two per uint8 along the last dim."""
    u = (q + 8).to(torch.uint8)
    return u[..., 0::2] | (u[..., 1::2] << 4)


def unpack_int4(p: torch.Tensor) -> torch.Tensor:
    lo = (p & 0x0F).to(torch.int8) - 8
    hi = (p >> 4).to(torch.int8) - 8
    return torch.stack([lo, hi], -1).flatten(-2)


def fake_quant(w: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Quantize then dequantize. Gives the exact weights an INT kernel would see."""
    q, s = quantize(w, bits, group_size)
    return dequantize(q, s, group_size).to(w.dtype)


def stored_bytes(w: torch.Tensor, bits: int, group_size: int) -> int:
    """Real storage: packed codes + fp16 scales."""
    n = w.numel()
    return (n * bits) // 8 + (n // group_size) * 2


@torch.no_grad()
def quantize_model(model: nn.Module, bits: int, group_size: int, skip=("lm_head", "wte", "wpe")):
    """In-place fake-quantize every Linear/Conv1D weight. Returns (fp16_bytes, quant_bytes)."""
    fp_bytes = q_bytes = 0
    for name, m in model.named_modules():
        if any(s in name for s in skip):
            continue
        if isinstance(m, nn.Linear):
            w = m.weight                      # (out, in)
            m.weight.copy_(fake_quant(w, bits, group_size))
        elif isinstance(m, Conv1D):
            w = m.weight                      # (in, out): groups must run along `in`
            m.weight.copy_(fake_quant(w.t().contiguous(), bits, group_size).t())
        else:
            continue
        fp_bytes += w.numel() * 2             # fp16 baseline
        q_bytes += stored_bytes(w, bits, group_size)
    return fp_bytes, q_bytes

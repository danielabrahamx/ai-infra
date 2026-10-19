"""Fused RMSNorm and fused softmax: PyTorch references + Triton kernels.

The Triton import is guarded. Without Triton or CUDA only the references work.
Each kernel handles one row per program (row length <= ~64K floats fits in one block).
"""
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # Windows without triton-windows, or CPU-only installs
    triton = tl = None
    HAS_TRITON = False


# ---------------------------------------------------------------- references
def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """y = x / sqrt(mean(x^2) + eps) * weight. Math in fp32, output in x.dtype."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable row softmax written out by hand (3 passes over memory)."""
    xf = x.float()
    e = torch.exp(xf - xf.max(-1, keepdim=True).values)
    return (e / e.sum(-1, keepdim=True)).to(x.dtype)


# ------------------------------------------------------------------- kernels
if HAS_TRITON:

    @triton.jit
    def _rmsnorm_kernel(X, W, Y, stride, N, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)   # one reduction, data stays in registers
        tl.store(Y + row * stride + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)

    @triton.jit
    def _softmax_kernel(X, Y, stride, N, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=-float("inf")).to(tl.float32)
        e = tl.exp(x - tl.max(x, axis=0))
        tl.store(Y + row * stride + cols, (e / tl.sum(e, axis=0)).to(Y.dtype.element_ty), mask=mask)


def _launch_cfg(n: int):
    block = triton.next_power_of_2(n)
    return block, min(max(block // 256, 1), 16)  # block size, num_warps


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm over the last dim. Requires CUDA tensors and Triton."""
    if not HAS_TRITON:
        raise RuntimeError("Triton not installed. On Windows: pip install triton-windows, or use WSL.")
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    y = torch.empty_like(x2)
    block, warps = _launch_cfg(x2.shape[1])
    _rmsnorm_kernel[(x2.shape[0],)](x2, weight, y, x2.stride(0), x2.shape[1], eps,
                                    BLOCK=block, num_warps=warps)
    return y.view_as(x)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Fused row softmax over the last dim. Requires CUDA tensors and Triton."""
    if not HAS_TRITON:
        raise RuntimeError("Triton not installed. On Windows: pip install triton-windows, or use WSL.")
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    y = torch.empty_like(x2)
    block, warps = _launch_cfg(x2.shape[1])
    _softmax_kernel[(x2.shape[0],)](x2, y, x2.stride(0), x2.shape[1], BLOCK=block, num_warps=warps)
    return y.view_as(x)

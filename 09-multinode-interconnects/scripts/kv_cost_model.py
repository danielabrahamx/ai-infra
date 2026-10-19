"""Disaggregated prefill/decode KV-transfer cost model (pure python, no deps).

Question: does shipping the KV cache from a prefill GPU to a decode GPU
cost less than recomputing, and how much does it add to TTFT?
All inputs are parameters. No hardware numbers are built in as facts.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2          # fp16/bf16
    params: float = 7e9           # for the FLOP estimate

    def kv_bytes_per_token(self) -> int:
        # K and V, every layer.
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes


@dataclass(frozen=True)
class Link:
    bandwidth_GBps: float         # effective, one direction
    latency_us: float = 5.0       # per transfer setup
    chunks: int = 1               # layer-wise pipelining uses chunks > 1


def kv_bytes(m: Model, tokens: int) -> int:
    return m.kv_bytes_per_token() * tokens


def transfer_time_s(nbytes: int, link: Link) -> float:
    """Time to move nbytes. Each chunk pays the setup latency."""
    return link.chunks * link.latency_us * 1e-6 + nbytes / (link.bandwidth_GBps * 1e9)


def prefill_time_s(m: Model, tokens: int, gpu_tflops: float, mfu: float = 0.4) -> float:
    """Compute-bound prefill: about 2 * params FLOPs per token."""
    return 2 * m.params * tokens / (gpu_tflops * 1e12 * mfu)


def ttft_disagg_s(m: Model, tokens: int, link: Link, gpu_tflops: float, mfu: float = 0.4,
                  overlap: bool = False) -> float:
    """TTFT with disaggregation = prefill + transfer (minus overlap if layer-wise streaming).

    With overlap, transfer hides behind prefill; only the last chunk is exposed.
    """
    p = prefill_time_s(m, tokens, gpu_tflops, mfu)
    t = transfer_time_s(kv_bytes(m, tokens), link)
    if overlap:
        exposed = t / link.chunks
        return p + max(0.0, exposed) if t <= p else t + p / link.chunks
    return p + t


def breakeven_bandwidth_GBps(m: Model, tokens: int, gpu_tflops: float, mfu: float = 0.4,
                             budget_frac: float = 0.1) -> float:
    """Minimum link bandwidth so transfer adds at most budget_frac of prefill time."""
    budget = budget_frac * prefill_time_s(m, tokens, gpu_tflops, mfu)
    return kv_bytes(m, tokens) / budget / 1e9


if __name__ == "__main__":
    # Example shape only (Llama-2-7B-like, MHA). Link numbers are illustrative inputs.
    m = Model(layers=32, kv_heads=32, head_dim=128)
    for name, bw in [("example 12.5 GB/s link", 12.5), ("example 50 GB/s link", 50.0)]:
        for tok in (512, 4096):
            link = Link(bandwidth_GBps=bw)
            print(f"{name:24s} tokens={tok:5d} kv={kv_bytes(m, tok)/1e6:8.1f} MB "
                  f"xfer={transfer_time_s(kv_bytes(m, tok), link)*1e3:7.2f} ms "
                  f"prefill={prefill_time_s(m, tok, 100)*1e3:7.2f} ms (assumes 100 TFLOPS)")

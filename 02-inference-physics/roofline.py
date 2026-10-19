"""KV-cache size, arithmetic intensity and roofline calculator. Pure Python.

Run:  python roofline.py --model llama3-8b --gpu h100 --batch 1 --ctx 4096
"""
import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    ffn: int          # SwiGLU intermediate size
    vocab: int
    head_dim: int = 0

    @property
    def hd(self):
        return self.head_dim or self.hidden // self.n_heads

    def params(self):
        """Parameter count: attention (GQA) + SwiGLU MLP + embeddings + untied LM head."""
        h, hd = self.hidden, self.hd
        attn = h * self.n_heads * hd * 2 + h * self.n_kv_heads * hd * 2  # q,o + k,v
        mlp = 3 * h * self.ffn
        norms = 2 * h
        return self.layers * (attn + mlp + norms) + 2 * self.vocab * h + h


@dataclass(frozen=True)
class GPU:
    name: str
    peak_flops: float   # dense FP16/BF16 tensor FLOP/s
    bandwidth: float    # bytes/s
    mem_bytes: float

    @property
    def ridge(self):
        """Arithmetic intensity (FLOP/byte) where compute and memory limits meet."""
        return self.peak_flops / self.bandwidth


# Shapes from the public Llama 3 / 3.1 configs.
MODELS = {
    "llama3-8b": ModelConfig("Llama-3 8B", 32, 4096, 32, 8, 14336, 128256),
    "llama3-70b": ModelConfig("Llama-3 70B", 80, 8192, 64, 8, 28672, 128256),
    "llama3-405b": ModelConfig("Llama-3.1 405B", 126, 16384, 128, 8, 53248, 128256),
}
# Vendor spec-sheet numbers (dense, no sparsity). RTX 2050 laptop: tensor FP16 value is
# an approximation from public spec listings; check your own card.
GPUS = {
    "rtx2050": GPU("RTX 2050 (mobile)", 29.0e12 / 2, 112e9, 4 * 2**30),
    "a100": GPU("A100 80GB SXM", 312e12, 2.039e12, 80 * 2**30),
    "h100": GPU("H100 SXM", 989e12, 3.35e12, 80 * 2**30),
}


def kv_cache_bytes(m, batch, ctx, bytes_per=2):
    """K and V for every layer: 2 * layers * kv_heads * head_dim * ctx * batch * bytes."""
    return 2 * m.layers * m.n_kv_heads * m.hd * ctx * batch * bytes_per


def kv_bytes_per_token(m, bytes_per=2):
    return kv_cache_bytes(m, 1, 1, bytes_per)


def weight_bytes(m, bytes_per=2):
    return m.params() * bytes_per


def decode_flops_per_token(m, ctx):
    """2*params for matmuls + attention score/value work over ctx tokens."""
    attn = 4 * m.layers * m.n_heads * m.hd * ctx
    return 2 * (m.params() - m.vocab * m.hidden) + attn  # input embedding is a lookup


def decode_intensity(m, batch, ctx, bytes_per=2):
    """FLOP/byte for one decode step. Weights are read once per step, KV per sequence."""
    flops = batch * decode_flops_per_token(m, ctx)
    mem = weight_bytes(m, bytes_per) + kv_cache_bytes(m, batch, ctx, bytes_per)
    return flops / mem


def prefill_intensity(m, batch, seq, bytes_per=2):
    """FLOP/byte for processing `seq` prompt tokens at once (KV written once)."""
    flops = batch * seq * decode_flops_per_token(m, seq // 2 or 1)
    mem = weight_bytes(m, bytes_per) + kv_cache_bytes(m, batch, seq, bytes_per)
    return flops / mem


def attainable(gpu, intensity):
    """Roofline: min(peak compute, bandwidth * intensity), in FLOP/s."""
    return min(gpu.peak_flops, gpu.bandwidth * intensity)


def decode_time_per_token(m, gpu, batch, ctx, bytes_per=2):
    """Lower-bound step time (s) = max(compute time, memory time)."""
    flops = batch * decode_flops_per_token(m, ctx)
    mem = weight_bytes(m, bytes_per) + kv_cache_bytes(m, batch, ctx, bytes_per)
    return max(flops / gpu.peak_flops, mem / gpu.bandwidth)


def max_batch(m, gpu, ctx, bytes_per=2, n_gpus=1):
    """Largest batch whose weights + KV cache fit in memory (ignores activations)."""
    free = gpu.mem_bytes * n_gpus - weight_bytes(m, bytes_per)
    return max(0, int(free // kv_cache_bytes(m, 1, ctx, bytes_per)))


def report(m, gpu, batch, ctx, bytes_per=2):
    gib = 2**30
    ai = decode_intensity(m, batch, ctx, bytes_per)
    t = decode_time_per_token(m, gpu, batch, ctx, bytes_per)
    bound = "memory-bound" if ai < gpu.ridge else "compute-bound"
    fits = weight_bytes(m, bytes_per) + kv_cache_bytes(m, batch, ctx, bytes_per) <= gpu.mem_bytes
    return "\n".join([
        f"Model: {m.name}  params={m.params()/1e9:.2f}B  dtype_bytes={bytes_per}",
        f"GPU:   {gpu.name}  ridge={gpu.ridge:.1f} FLOP/B",
        f"Weights:          {weight_bytes(m, bytes_per)/gib:.2f} GiB",
        f"KV per token:     {kv_bytes_per_token(m, bytes_per)/1024:.1f} KiB",
        f"KV (b={batch}, ctx={ctx}): {kv_cache_bytes(m, batch, ctx, bytes_per)/gib:.2f} GiB",
        f"Fits on one GPU:  {fits}   max batch @ctx: {max_batch(m, gpu, ctx, bytes_per)}",
        f"Decode intensity: {ai:.2f} FLOP/B -> {bound}",
        f"Prefill intensity ({ctx} tok): {prefill_intensity(m, batch, ctx, bytes_per):.1f} FLOP/B",
        f"Decode step lower bound: {t*1e3:.2f} ms  ({1/t:.1f} tok/s per sequence)",
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, default="llama3-8b")
    p.add_argument("--gpu", choices=GPUS, default="h100")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--bytes", type=int, default=2, help="bytes per element (2=fp16, 1=int8)")
    a = p.parse_args()
    print(report(MODELS[a.model], GPUS[a.gpu], a.batch, a.ctx, a.bytes))


if __name__ == "__main__":
    main()

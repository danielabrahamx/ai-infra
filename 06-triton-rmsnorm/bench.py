"""Benchmark fused Triton kernels vs PyTorch native with triton.testing.do_bench.
NOT YET RUN on this machine (needs CUDA + Triton). Usage: python bench.py [--out bench_results]
"""
import argparse
import torch
from kernels import HAS_TRITON, rmsnorm, softmax

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="bench_results")
ap.add_argument("--dtype", default="float16")
a = ap.parse_args()
if not (HAS_TRITON and torch.cuda.is_available()):
    raise SystemExit("Needs CUDA and Triton (Windows: pip install triton-windows, or WSL).")

import json
from triton.testing import do_bench

dt = getattr(torch, a.dtype)
rows, rows_n = [], 4096
for n in (512, 1024, 2048, 4096, 8192, 16384):
    x = torch.randn(rows_n, n, device="cuda", dtype=dt)
    w = torch.randn(n, device="cuda", dtype=dt)
    rms_native = torch.nn.RMSNorm(n, eps=1e-6, device="cuda", dtype=dt)  # torch >= 2.4
    with torch.no_grad():
        rms_native.weight.copy_(w)
    cases = {
        "rmsnorm": (lambda: rmsnorm(x, w), lambda: rms_native(x), 2),
        "softmax": (lambda: softmax(x), lambda: torch.softmax(x, -1), 2),
    }
    for name, (tri, ref, rw) in cases.items():
        t_tri, t_ref = do_bench(tri), do_bench(ref)       # ms, median-ish of many runs
        gbs = lambda ms: rw * x.numel() * x.element_size() / ms * 1e-6
        rows.append(dict(op=name, rows=rows_n, cols=n, dtype=a.dtype, triton_ms=round(t_tri, 4),
                         torch_ms=round(t_ref, 4), speedup=round(t_ref / t_tri, 2),
                         triton_GBps=round(gbs(t_tri), 1), torch_GBps=round(gbs(t_ref), 1)))
        print(rows[-1], flush=True)

json.dump({"gpu": torch.cuda.get_device_name(), "results": rows}, open(a.out + ".json", "w"), indent=2)
cols = list(rows[0])
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
open(a.out + ".md", "w").write("\n".join(md) + "\n")

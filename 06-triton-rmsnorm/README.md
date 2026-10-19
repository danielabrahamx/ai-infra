# 06 - Triton RMSNorm and Softmax

Fused Triton kernels for RMSNorm and row softmax. Each has a PyTorch reference and a benchmark against PyTorch native.

**Roadmap stage:** 6 of 12 (AI Infrastructure Engineering) - custom GPU kernels.

## Why
Native PyTorch RMSNorm and hand-written softmax launch several kernels. Each one reads and writes global memory.
A fused kernel loads a row once, computes in registers, and writes once. These ops are memory-bound, so fewer bytes means less time.

## Files
| File | Purpose |
|---|---|
| `kernels.py` | `rmsnorm_ref`, `softmax_ref` (torch) and `rmsnorm`, `softmax` (Triton). Triton import is guarded. |
| `bench.py` | `triton.testing.do_bench` vs `torch.nn.RMSNorm` / `torch.softmax`. Writes `bench_results.json/.md`. |
| `tests/test_kernels.py` | CPU tests of the references. CUDA tests of the kernels (skipped without CUDA/Triton). |

## Quickstart
```bash
python -m pip install -r requirements.txt
python -m pytest -q -rs          # references run on CPU; Triton tests skip with a reason
python bench.py                  # needs CUDA + Triton
```

## Triton on Windows
Official Triton has no Windows wheels. Choose one:
- `python -m pip install triton-windows` (community fork; needs a CUDA torch build).
- Use WSL2 with a CUDA torch build and `pip install triton`.

## Kernel notes
- One program per row. Block size is the next power of two of the row length. `num_warps` scales with block size.
- RMSNorm: fp32 accumulate, one sum reduction, output cast back to input dtype.
- Softmax: max-subtract for stability, one load and one store.
- Limit: a row must fit in one block (about 64K elements). No backward pass.

## Status
| Item | State |
|---|---|
| Torch references vs `torch.nn.RMSNorm` / `torch.softmax` | Verified on CPU: `5 passed, 16 skipped`. |
| Triton kernel correctness tests | Verified on Modal H100: `20 passed, 1 skipped`. |
| Benchmark (H100, fp16, 4096 rows, widths 512 to 16384, single run) | Verified. Softmax speedup 1.24x to 2.75x. RMSNorm speedup 1.0x to 1.57x. Full table: `gpu_results/bench.md`. |

Reproduce on Modal: `modal run tools/modal_run.py --stage 06`.

## License
MIT - Daniel Abraham

# 01-cuda-matmul

Custom CUDA SGEMM kernels (naive and shared-memory tiled), a benchmark against cuBLAS, and a NumPy reference.

**Roadmap stage:** 1 of 12 (AI Infrastructure Engineering). Goal: learn the GPU memory hierarchy by writing the kernel that every transformer layer depends on.

## Why
Matmul is the core of LLM inference and training. The naive kernel is memory bound. Tiling in shared memory cuts global loads by a factor of `TILE`. cuBLAS shows how far that still is from a tuned library.

## Layout
- `src/kernels.cuh` - `matmul_naive` and `matmul_tiled` kernels.
- `src/bench.cu` - harness. Times each kernel with CUDA events. Checks error against `cublasSgemm`.
- `matmul_ref.py` - NumPy reference. `matmul_tiled` mirrors the CUDA tile loop.
- `test_matmul.py` - CPU correctness tests.
- `Makefile`, `CMakeLists.txt` - build files.

## Quickstart
CPU tests (no GPU needed):
```
python -m pip install -r requirements.txt
python -m pytest -q
```
GPU benchmark (needs CUDA Toolkit with `nvcc` and cuBLAS):
```
make ARCH=sm_86          # or: cmake -B build && cmake --build build
./bench 1024 1024 1024 20
```
Arguments: `M N K iters`. Edge sizes (not multiples of 16) work.

## Status
| Item | State |
|---|---|
| NumPy reference + tests | Verified on CPU |
| `.cu` kernels compile | Verified on Modal H100 (CUDA 12.4, `-arch=sm_90`) |
| Kernel correctness on GPU | Verified. `max_abs_err` is 0 against cuBLAS on random floats (1000x900x1100, 1024^3, 4096^3). All three accumulate along K in the same order, so bit-identical output is plausible. |
| Benchmark vs cuBLAS (fp32 SGEMM, H100 80GB HBM3, single runs) | See table below |

| M=N=K | cuBLAS GFLOP/s | tiled (16x16) | naive |
|---|---|---|---|
| 1024 | 37518 | 7626 | 5160 |
| 4096 | 52225 | 7990 | 5427 |

The tiled kernel reaches about 15% of cuBLAS at 4096. It is about 1.5x faster than the naive kernel. Raw output: `gpu_results/run.txt`. Reproduce on Modal: `modal run tools/modal_run.py --stage 01`.

## Notes
- cuBLAS is column-major. The harness computes `C^T = B^T * A^T` to get a row-major result.
- Next steps: register blocking (2D thread tiles), vectorized loads, double buffering.

## License
MIT, Daniel Abraham.

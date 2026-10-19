# 09 - Multi-node interconnects

Roadmap stage 9 of 12 (AI Infrastructure Engineering). Tools to measure all-reduce latency and bandwidth, intra-node vs inter-node. Plus a KV-transfer cost model for disaggregated prefill/decode.

## Why
Multi-node training and serving depend on the link. You must know the real busbw and latency before you choose TP/PP/disaggregation layouts.

## Contents
- `scripts/sweep.py` - torchrun all-reduce sweep (gloo or nccl). Reports time, algbw, busbw (nccl-tests formula: `busbw = algbw * 2(n-1)/n`). Tags rows intra-node or inter-node from `LOCAL_WORLD_SIZE` vs `WORLD_SIZE`.
- `scripts/parse_plot.py` - reads sweep JSONL or nccl-tests stdout. Plots latency and busbw.
- `scripts/kv_cost_model.py` - pure-python KV size, transfer time, TTFT, break-even bandwidth.
- `scripts/launch_local.py` - torchrun substitute for single-host runs (see Status).
- `slurm/nccl_sweep.sbatch`, `slurm/nccl_tests.sbatch` - 2-node jobs: sweep, nccl-tests, `ib_write_bw`.
- `docs/PROFILING.md` - how to run and read results.
- `tests/` - 10 unit tests.

## Quickstart
```bash
python -m pip install -r requirements.txt
python -m pytest -q
# CPU, 2 procs (single host):
python scripts/launch_local.py -n 2 scripts/sweep.py --backend gloo --max-bytes 67108864 --out results/sweep.jsonl
python scripts/parse_plot.py results/sweep.jsonl --out results/sweep.png
# Real cluster (torchrun):
torchrun --nproc-per-node=2 scripts/sweep.py --backend gloo
python scripts/kv_cost_model.py
```

## Status
**Verified on CPU**
- `pytest -q`: 10 passed.
- Real run: gloo, 2 processes, one Windows host (loopback), 20 iters, 5 warmup. Topology tag: intra-node.

| bytes | time (us) | algbw GB/s | busbw GB/s |
|---|---|---|---|
| 1024 | 1387.8 | 0.001 | 0.001 |
| 4096 | 994.6 | 0.004 | 0.004 |
| 16384 | 967.8 | 0.017 | 0.017 |
| 65536 | 1575.3 | 0.042 | 0.042 |
| 262144 | 6535.0 | 0.040 | 0.040 |
| 1048576 | 28731.2 | 0.036 | 0.036 |
| 4194304 | 69699.2 | 0.060 | 0.060 |
| 16777216 | 257699.1 | 0.065 | 0.065 |
| 67108864 | 1011456.1 | 0.066 | 0.066 |

These numbers are low and noisy. Gloo on this Windows machine warned that it could not connect via the hostname address (`kubernetes.docker.internal`), so it likely used a poor path. Treat them as a proof that the pipeline works, not as a hardware benchmark. They say nothing about NCCL, NVLink or InfiniBand.
- `torchrun` itself failed here: this PyTorch Windows build has no libuv, but the elastic rendezvous requests it. `launch_local.py` sets the same env vars and bypasses it. On Linux use `torchrun`.

**NOT YET RUN (needs GPUs / multi-node / InfiniBand)**
- NCCL sweep, inter-node: `sbatch slurm/nccl_sweep.sbatch`
- nccl-tests and `ib_write_bw`: `sbatch slurm/nccl_tests.sbatch`
- The nccl-tests parser is tested only on a hand-written sample row, not real output.
- The KV cost model is analytic. Its example links and 100 TFLOPS value are illustrative inputs, not measurements.

## License
MIT, Daniel Abraham.

## Real GPU results (Modal, 8xH100 single node, NCCL, NVLink)
Intra-node only. Modal has no multi-node InfiniBand, so inter-node numbers are **NOT YET RUN**.

| GPUs | 1 KiB latency | busbw at 16 MiB |
|---|---|---|
| 2 | 24.8 us | 222 GB/s |
| 8 | 28.5 us | 249 GB/s |

Raw: `gpu_results/sweep_2gpu.jsonl`, `sweep_8gpu.jsonl`. Run once. nccl-tests and `ib_write_bw` are NOT YET RUN.

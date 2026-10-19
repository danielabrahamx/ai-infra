# 07 - Distributed Inference: communication-cost analyzer + all-reduce benchmark

**Roadmap stage 7 of 12 (AI Infrastructure Engineering).**

## What and why
Large models do not fit on one GPU. TP, PP and EP split them. Each split adds network traffic.
This repo answers one question: *how many bytes per token, and how many milliseconds, does the network cost?*

- `commcost.py` - pure-python alpha-beta model. Inputs: model shape, TP/PP/EP sizes, link (NVLink4, IB NDR/HDR, PCIe4/5).
  Outputs: collective count, message size, bytes per token per GPU, time per token.
- `bench_allreduce.py` - `torch.distributed` all-reduce microbenchmark. `gloo` on CPU, `nccl` on GPU.

## Model rules
- TP: 2 ring all-reduces per layer. Bytes per GPU = `2(N-1)/N * hidden * dtype * batch`.
- PP: 1 point-to-point send of the activation per stage boundary.
- EP (MoE): 2 all-to-alls per MoE layer (dispatch, combine). Bytes per GPU = `(N-1)/N * top_k * hidden * dtype * batch`.
- Time = `steps * latency + bytes / bandwidth`. A group larger than one node uses the inter-node link.
- Defaults: Llama-3.1-405B shape (126 layers, hidden 16384, bf16). MoE case is a hypothetical 405B-class model (top-2).
- Bandwidths are nominal link numbers. Real systems get less. Treat output as a lower bound, not a prediction.

## Quickstart
```bash
python -m pip install -r requirements.txt
python -m pytest -q                       # 9 tests
python commcost.py                        # analysis tables
python bench_allreduce.py --nprocs 4      # gloo CPU benchmark (mp.spawn)
```

## Results (analytic, from `python commcost.py`)
Per decoded token, batch 1, 405B shape:

| Plan | All-reduces | Bytes/token/GPU | Comm time/token |
|---|---|---|---|
| TP=8, 1 node, NVLink | 252 x 32 KiB | 14.45 MB | 10.6 ms |
| TP=8 x PP=8, 8 nodes | 30 + 7 p2p | 1.72 MB + 0.23 MB | 1.30 ms |
| TP=64 across 8 nodes (IB) | 252 x 32 KiB | 16.26 MB | 159 ms |
| TP=8 on PCIe4 | 252 x 32 KiB | 14.45 MB | 35.7 ms |
| MoE EP=8 (NVLink) | 252 all-to-all | 14.45 MB | 5.3 ms |
| MoE EP=64 (crosses IB) | 252 all-to-all | 16.26 MB | 79.7 ms |

Reading: bytes barely change with group size. Latency and the slow link dominate at batch 1.
So keep TP inside a node. Use PP across nodes. The TP=8 x PP=8 row counts the comm of one pipeline stage plus hops.
These are model outputs, not measurements.

## Measured: gloo all-reduce on this CPU (Windows, mp.spawn, loopback)
Real run. Float32 buffers, 50 iters. `busbw = 2(N-1)/N * size / time`.

| bytes | 2 procs us | 2 procs busbw GB/s | 4 procs us | 4 procs busbw GB/s |
|---|---|---|---|---|
| 1024 | 1312 | 0.001 | 1970 | 0.001 |
| 16384 | 1677 | 0.010 | 1426 | 0.017 |
| 262144 | 2356 | 0.111 | 2639 | 0.149 |
| 1048576 | 18561 | 0.056 | 7917 | 0.199 |
| 4194304 | 59271 | 0.071 | 50518 | 0.125 |
| 16777216 | 244361 | 0.069 | 193502 | 0.130 |

These numbers describe loopback TCP between processes on one laptop CPU. They say nothing about NVLink or IB.
Raw JSON: `results_gloo_2p.json`, `results_gloo_4p.json`.

## Status
| Item | State |
|---|---|
| `commcost.py` + 8 unit tests | Verified on CPU |
| Smoke test (2-proc gloo benchmark) | Verified on CPU |
| gloo benchmark, 2 and 4 procs, via `mp.spawn` | **Run** (table above) |
| `torchrun` launch | **Not working on this machine.** Windows torch build has no libuv. torchrun's static rendezvous fails with `use_libuv was requested but PyTorch was built without libuv support`. Use `mp.spawn` here. Script supports torchrun on Linux (untested). |
| nccl benchmark, Modal 8xH100 single node, 2/4/8 GPUs via torchrun | **Run**. busbw at 16 MiB: 2 GPUs 215 GB/s, 4 GPUs 260 GB/s, 8 GPUs 249 GB/s. 8-GPU latency 16 us at 1 KiB, 118 us at 16 MiB. Raw: `gpu_results/`. |
| Real 405B run on 8 nodes | **NOT YET RUN.** |
| Analyzer vs measured NVLink/IB | **NOT YET RUN.** No calibration against real hardware. |

### Commands for the remaining NOT YET RUN items
nccl, one node (already run on Modal; reproduce with):
```bash
torchrun --standalone --nproc_per_node=8 bench_allreduce.py --backend nccl --out nccl_8p.json
```
nccl, 8 nodes x 8 GPUs (run on each node, `NODE_RANK` = 0..7):
```bash
torchrun --nnodes=8 --nproc_per_node=8 --node_rank=$NODE_RANK \
  --rdzv_backend=c10d --rdzv_endpoint=$HEAD_IP:29500 bench_allreduce.py --backend nccl
```
405B serving, TP=8 x PP=8 (vLLM, 8 nodes; start a Ray cluster first):
```bash
vllm serve meta-llama/Llama-3.1-405B-Instruct --tensor-parallel-size 8 --pipeline-parallel-size 8
```
Then compare vLLM's measured inter-token latency with the `TP=8 x PP=8` row. Fit link bandwidth and latency in `LINKS` to the nccl results.

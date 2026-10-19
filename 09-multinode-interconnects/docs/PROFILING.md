# Profiling guide

## Metrics
- Latency: time for one all-reduce. Dominates small messages.
- algbw = bytes / time.
- busbw = algbw * 2(n-1)/n. This makes the number comparable to link speed and to nccl-tests.

## Intra-node vs inter-node
- Intra-node (NVLink/PCIe/shared memory): high busbw, low latency.
- Inter-node (InfiniBand/RoCE/TCP): lower busbw, higher latency.
- `sweep.py` tags each row from `LOCAL_WORLD_SIZE` vs `WORLD_SIZE`.

## Steps on a cluster
1. `ib_write_bw` between two nodes. This is the raw RDMA ceiling.
2. `nccl-tests all_reduce_perf` on one node, then two nodes.
3. `sbatch slurm/nccl_sweep.sbatch`. Compare with step 2. Large gaps mean framework overhead.
4. Set `NCCL_DEBUG=INFO`. Check the log for `NET/IB` (RDMA) or `NET/Socket` (TCP fallback).
5. Force TCP with `NCCL_IB_DISABLE=1` to measure what RDMA gives you.

## Useful env vars
`NCCL_DEBUG`, `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `NCCL_IB_DISABLE`, `NCCL_ALGO` (Ring/Tree), `NCCL_P2P_DISABLE`.

## Parse nccl-tests output
`python scripts/parse_plot.py --nccl-tests results/nccl-tests-allreduce.log`

## KV transfer
`python scripts/kv_cost_model.py` prints the transfer time vs prefill time for example links.
Feed it your measured `ib_write_bw` number to decide if disaggregation pays off.

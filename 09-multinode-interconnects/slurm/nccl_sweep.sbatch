#!/bin/bash
# Two-node torchrun all-reduce sweep. NOT YET RUN (needs 2 GPU nodes).
#SBATCH --job-name=nccl-sweep
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-%j.out

set -euo pipefail
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export NCCL_DEBUG=INFO          # log chosen transport (IB/NET vs SHM/P2P)
# export NCCL_IB_HCA=mlx5       # pin HCAs if needed
# export NCCL_SOCKET_IFNAME=eth0 # force TCP fallback for comparison: also set NCCL_IB_DISABLE=1

# Inter-node: 2 nodes x 8 GPUs.
srun torchrun --nnodes=2 --nproc-per-node=8 --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER:29500" scripts/sweep.py --backend nccl \
  --max-bytes $((1<<30)) --label ib --out results/inter.jsonl

# Intra-node baseline: one node only.
srun --nodes=1 --ntasks=1 -w "$MASTER" torchrun --standalone --nproc-per-node=8 \
  scripts/sweep.py --backend nccl --max-bytes $((1<<30)) --label nvlink --out results/intra.jsonl

python scripts/parse_plot.py results/intra.jsonl results/inter.jsonl --out results/nccl.png

#!/bin/bash
# Reference nccl-tests + ib_write_bw. NOT YET RUN (needs GPUs and InfiniBand).
#SBATCH --job-name=nccl-tests
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=results/nccl-tests-%j.out

set -euo pipefail
# Build once: git clone https://github.com/NVIDIA/nccl-tests && make MPI=1 CUDA_HOME=... NCCL_HOME=...
NCCL_TESTS=${NCCL_TESTS:-$HOME/nccl-tests/build}

# -b/-e: min/max bytes, -f: step factor, -g: GPUs per task.
srun --mpi=pmix "$NCCL_TESTS/all_reduce_perf" -b 8 -e 4G -f 2 -g 1 | tee results/nccl-tests-allreduce.log

# Raw RDMA link test between the two nodes (perftest package).
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
srun --nodes=1 --ntasks=1 -w "${NODES[0]}" ib_write_bw -d mlx5_0 -a --report_gbits > results/ib_write_bw_server.log &
sleep 5
srun --nodes=1 --ntasks=1 -w "${NODES[1]}" ib_write_bw -d mlx5_0 -a --report_gbits "${NODES[0]}" | tee results/ib_write_bw_client.log
wait

python scripts/parse_plot.py --nccl-tests results/nccl-tests-allreduce.log --out results/nccl-tests.png

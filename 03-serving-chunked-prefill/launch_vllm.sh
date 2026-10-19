#!/usr/bin/env bash
# NOT YET RUN. Needs 4x80GB GPUs (e.g. 4xA100/H100) and vLLM installed.
# Usage: MAX_BATCHED=2048 CHUNKED=1 ./launch_vllm.sh
# Tune: sweep MAX_BATCHED in 512 1024 2048 4096 8192 with CHUNKED=1, plus CHUNKED=0.
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
TP="${TP:-4}"
MAX_BATCHED="${MAX_BATCHED:-2048}"
MAX_LEN="${MAX_LEN:-32768}"
PORT="${PORT:-8000}"
FLAGS=()
if [ "${CHUNKED:-1}" = "1" ]; then
  FLAGS+=(--enable-chunked-prefill)
else
  FLAGS+=(--no-enable-chunked-prefill)   # without chunking, MAX_BATCHED must be >= MAX_LEN
fi
exec vllm serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --max-num-batched-tokens "$MAX_BATCHED" \
  --gpu-memory-utilization 0.90 \
  --port "$PORT" "${FLAGS[@]}"

# ai-infra

Twelve small projects on LLM inference infrastructure, from a CUDA matmul up to a model gateway.
Each folder stands alone: short code, tests, one README, no framework.

Every number in this repo is one of three kinds, and the READMEs say which:
**measured** (ran on hardware), **simulated** (a model I wrote), or **not run**.

| # | Project | Topic | Status |
|---|---|---|---|
| 01 | [cuda-matmul](01-cuda-matmul) | Tiled CUDA matmul vs cuBLAS | Verified on H100. Tiled kernel ~15% of cuBLAS. |
| 02 | [inference-physics](02-inference-physics) | Prefill/decode profiling, roofline, KV math | Verified on CPU and H100 (GPT-2, Qwen2.5-7B). |
| 03 | [serving-chunked-prefill](03-serving-chunked-prefill) | Chunked prefill scheduler sim, vLLM tuning | Simulated + real vLLM runs on 7B and 72B. Small real stall (151 ms max) on 7B. |
| 04 | [prefix-aware-router](04-prefix-aware-router) | Prefix-hash routing for KV reuse | Simulation verified. |
| 05 | [quantization-bench](05-quantization-bench) | INT8/INT4/FP8 quality vs cost | CPU INT8/INT4 and H100 bf16/GPTQ/AWQ run. FP8 result invalid. |
| 06 | [triton-rmsnorm](06-triton-rmsnorm) | Fused Triton RMSNorm and softmax | Verified on H100. Softmax up to 2.75x vs PyTorch. |
| 07 | [distributed-inference](07-distributed-inference) | TP/PP/EP communication cost | Model + NCCL on 8xH100 single node. 405B not run. |
| 08 | [speculative-decoding](08-speculative-decoding) | Draft/verify decoding | CPU verified. |
| 09 | [multinode-interconnects](09-multinode-interconnects) | NCCL sweeps, KV transfer cost | NCCL sweep on 8xH100 single node. Multi-node/IB not run. |
| 10 | [gpu-queue-autoscaler](10-gpu-queue-autoscaler) | Queue-based GPU autoscaling | Simulation verified. |
| 11 | [llm-gateway](11-llm-gateway) | Complexity-based model routing | Mock upstreams verified. |
| 12 | [benchmark-teardowns](12-benchmark-teardowns) | Benchmark harness and method | Harness validated on a mock. |

## Run it
```
cd 02-inference-physics
python -m pip install -r requirements.txt
python -m pytest -q
```
CPU tests need no GPU. GPU results came from Modal. To reproduce them:
`modal run tools/modal_run.py --stage 01` (stages 01, 02, 05, 06) and `tools/modal_run2.py` (03, 07, 09) and `tools/modal_run3.py` (03, 05).

## What was and was not run
- GPU results are from one Modal H100, or one 8xH100 node for NCCL. Single runs, no confidence intervals.
- Stage 03 real stall is small: 151 ms max on a 7B model. The 72B run only reported p99, so it cannot show stalls.
- Stage 05 FP8 gave an invalid perplexity. The cause is not found.
- Not run: 405B (needs gated weights and about 8 H100), multi-node and InfiniBand (the Modal workspace allows 10 GPUs, a multi-node job needs 16), Nsight traces (`nsys` would not install).

MIT licence.

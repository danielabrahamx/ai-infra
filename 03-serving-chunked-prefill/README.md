# 03 - Serving: chunked prefill

Roadmap stage 3 (AI Infrastructure Engineering series). A simulator of why long prompts can stall decode and how chunked prefill trades that stall against TTFT, plus a real vLLM check. The real check (7B, one H100) did not reproduce the stall. See "Real GPU results".

## What is here
- `sim.py` - pure-python discrete-event simulator of continuous batching, chunked vs unchunked prefill.
- `sweep.py` - sweeps chunk budget. Writes `results.csv` and `plot.png` (plot needs matplotlib).
- `launch_vllm.sh`, `tune.sh`, `loadtest.py` - vLLM 70B launch, tuning loop, load client.
- `tests/test_sim.py` - simulator tests.

## Quickstart
```
python -m pip install -r requirements.txt
python -m pytest -q
python sweep.py
```

## Simulator model (all numbers SIMULATED)
- Step time = 8 ms + 0.05 ms x decode seqs + 0.12 ms x prefill tokens. The constants are assumptions, not measurements.
- Workload: 200 requests, Poisson 1 req/s, 10% long prompts (16k-32k tokens), 90% short (100-500), 64-256 output tokens, seed 0.
- Unchunked: waiting prompts run whole in a prefill step. Decodes stall.
- Chunked: each step runs all decodes first. Leftover token budget goes to prefill chunks.

## Simulated results (`sweep_output.txt`; SIMULATED, not hardware)
```
unchunked B= 32768 TTFTp50=     1479 TTFTp99=     7406 ITLp50=    8.4 ITLp99=     9.6 p99.9=  6041.7 tok/s=  149.4
chunked   B=   256 TTFTp50=     3231 TTFTp99=    10093 ITLp50=    8.4 ITLp99=    38.6 p99.9=    38.7 tok/s=  149.4
chunked   B=   512 TTFTp50=     2497 TTFTp99=     8442 ITLp50=    8.4 ITLp99=    69.3 p99.9=    69.4 tok/s=  149.4
chunked   B=  1024 TTFTp50=     1978 TTFTp99=     7938 ITLp50=    8.4 ITLp99=   130.7 p99.9=   130.8 tok/s=  149.4
chunked   B=  2048 TTFTp50=     1644 TTFTp99=     7814 ITLp50=    8.5 ITLp99=   253.5 p99.9=   253.7 tok/s=  149.4
chunked   B=  4096 TTFTp50=     1549 TTFTp99=     7641 ITLp50=    8.5 ITLp99=   499.1 p99.9=   499.4 tok/s=  149.4
chunked   B=  8192 TTFTp50=     1495 TTFTp99=     7601 ITLp50=    8.5 ITLp99=   989.6 p99.9=   990.9 tok/s=  149.4
chunked   B= 32768 TTFTp50=     1471 TTFTp99=     7407 ITLp50=    8.4 ITLp99=    47.0 p99.9=  3496.9 tok/s=  149.4
wrote plot.png
```

How to read:
- Unchunked has a normal ITL p50 and p99. Its tail is bad: ITL p99.9 is about 6 s. A long prefill freezes every decode.
- Chunked removes the freeze. ITL p99.9 is about the chunk step time (39 ms at B=256, 499 ms at B=4096).
- Small chunks cost TTFT: p50 TTFT 1.5 s (unchunked) vs 3.2 s (B=256). Long prompts need many steps.
- Throughput is the same in this model. Real GPUs differ (small chunks lose efficiency).
- p99 hides the stall because stalls are rare per token. Read p99.9.

## Status
- Verified on CPU: simulator, 4 tests pass, sweep, CSV, plot.
- **Run on Modal (1x H100, vLLM 0.19, Qwen2.5-7B-Instruct, not 70B)**: `launch_vllm.sh` and `loadtest.py`. See "Real GPU results". The 70B / 4-GPU run is **NOT YET RUN**.
  ```
  CHUNKED=1 MAX_BATCHED=2048 ./launch_vllm.sh
  python loadtest.py --label chunked-2048
  ./tune.sh      # full sweep
  ```
- vLLM flag names change between versions. Check `vllm serve --help`.

## Real GPU results (Modal, vLLM 0.19)
Single runs. Read max and p99.9 inter-token latency (ITL). p99 cannot see the stall: few tokens are affected.

**Qwen2.5-7B-Instruct, 1x H100.** 120 requests at 3 req/s, 20% long prompts (about 8000 words), 128 output tokens.

| Config | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | ITL p99.9 | ITL max | tok/s |
|---|---|---|---|---|---|---|---|
| unchunked | 23 ms | 202 ms | 6.1 ms | 6.8 ms | 8.7 ms | 151 ms | 343 |
| chunk 512 | 23 ms | 119 ms | 6.0 ms | 7.1 ms | 14.5 ms | 27 ms | 344 |
| chunk 2048 | 23 ms | 63 ms | 6.0 ms | 6.9 ms | 18.2 ms | 67 ms | 344 |

Result: unchunked prefill produced one 151 ms decode stall. Chunking cut the worst stall to 27 ms (512) and 67 ms (2048). Throughput did not change. Chunking did not raise TTFT here. The stall is real but small: a 7B model on an H100 prefills 8000 words in about 150 ms. The simulator's multi-second stalls come from its assumed step costs, not from this hardware.

**Qwen2.5-72B-Instruct, 4x H100 (TP=4).** 60 requests at 2 req/s, 20% long (14000 words). This run only reported p99, so it cannot show the stall. TTFT p99 was 1.06 s (unchunked), 1.21 s (chunk 1024), 1.13 s (chunk 4096). ITL p99 was 18.7, 21.9 and 18.6 ms. Throughput was 225 tok/s in all three. Not rerun with p99.9 to save cost.

Raw: `gpu_results/`. The first 7B run (`load_*.txt`, 60 requests) also lacks p99.9.

# Methodology

Short rules. Follow them for every benchmark in this repo.

## Metric definitions
- **Token.** One streamed SSE chunk with non-empty `delta.content`. If the server sends `usage.completion_tokens`, that count is used for throughput. Chunks may hold more than one real token on some engines. Check this per engine.
- **TTFT.** Time from sending the request to the first token. Includes network, queueing, and prefill.
- **ITL.** Time between two consecutive tokens of one request. All gaps from all requests are pooled, then percentiles are taken.
- **E2E latency.** Time from request send to stream end.
- **Throughput.** Total output tokens of successful requests divided by the wall time of the level. This is system throughput, not per-user speed.
- **Percentiles.** Nearest-rank. Report p50, p95, p99. Do not report only the mean.

## Load model
- **Closed loop.** At concurrency `c`, `c` workers each send the next request when the last one ends. Offered load follows server speed. This shows the saturation curve. It hides queueing delay that an open-loop (fixed arrival rate) test would show.
- **Sweep.** Run several concurrency levels in one run, lowest first. Each level has at least `max(requests-per-level, 2 x c)` requests.
- **Warmup.** Send a few requests first. They are discarded. This avoids cold-start, JIT, and cache effects.
- **Workload.** Synthetic prompts from a seeded generator. Same seed gives the same prompts. Prompt length (`--prompt-words`) and `--max-tokens` are stored in the result. Synthetic prompts do not model real traffic. State this in the teardown.
- **Errors.** Failed requests are counted, not hidden. A level with errors must be flagged in the teardown. Throughput counts successful requests only.

## What to record
The harness stores: timestamp, harness version, endpoint, model, engine, quantization, workload, OS, Python, CPU, GPU (from `nvidia-smi`, else null), git commit. Add these by hand in the teardown:
- Engine flags (batch size, max sequences, KV-cache dtype, context length).
- Driver and CUDA version.
- Power and clock limits.
- Whether the client ran on the same host as the server. Same-host runs share the CPU. Say so.

## Rules for honest claims
1. One run is an anecdote. Repeat at least 3 times and report the spread.
2. Compare engines only with the same model, quantization, prompts, and `max_tokens`.
3. Never compare a number from one GPU to a number from another without naming both.
4. Mark any run on a mock or toy setup `validation_only: true`. Do not quote it.
5. Do not extrapolate. If you ran c=16, do not state c=64 results.
6. Output length is capped by `max_tokens`. A model that stops early lowers tokens per request. Check `output_tokens`.

## Cost per token
`USD per 1M tokens = (GPU $/hr x n_gpus) / (tokens/s x 3600) x 1,000,000`

- It assumes the GPU is busy all hour at that throughput. Real cost is higher with idle time. Use `--utilization` to model this.
- It counts output tokens only. It ignores prefill cost, networking, storage, and people.
- Pick the concurrency level that meets your latency SLO. Cost at a level that breaks the SLO is not useful.

## Pareto frontier
Each concurrency level gives one point: (latency, throughput). A point is on the frontier if no other point is both faster and higher in throughput. Plot several engines or configs on one chart. Choose the point that meets your SLO at the lowest cost.

## Known limits of this harness
- Python `httpx` client. At very high concurrency the client can become the bottleneck. Watch client CPU.
- Windows timers are coarse (about 16 ms). Sub-20 ms values on Windows are less reliable. The mock uses a spin-wait to avoid this. A real server has its own timing.
- No open-loop mode yet. No prefix-cache or long-context sweep yet.

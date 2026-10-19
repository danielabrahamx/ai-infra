# 12 - Benchmark Teardowns

A reproducible way to benchmark any OpenAI-compatible LLM endpoint. It also holds the rules for how to report the results.

**Roadmap stage:** 12 of 12 (AI Infrastructure Engineering). It pairs with stage 11 (`11-llm-gateway`), which uses the same TTFT/ITL definitions.

## Why
Most LLM benchmark posts omit the hardware, the workload, or the percentile. This repo forces you to record all three, in one JSON file with a fixed schema.

## What is in it
| Path | Purpose |
|---|---|
| `bench/harness.py` | Streaming load test. Concurrency sweep. TTFT, ITL, e2e, tok/s, errors. Writes JSON with environment metadata. |
| `bench/plot.py` | Latency vs throughput plot with Pareto frontier. Several result files on one chart. |
| `bench/cost.py` | USD per 1M output tokens from measured tok/s and your GPU $/hr. |
| `bench/validate.py` | Checks a result file against `results/schema.json`. |
| `bench/mock_server.py` | Synthetic OpenAI-compatible server. For harness validation only. |
| `results/schema.json` | JSON schema of a result file. |
| `results/example-mock-validation.*` | Real harness run against the mock. **Not a performance claim.** |
| `templates/teardown.md` | Template for a written teardown. |
| `docs/methodology.md` | Metric definitions and rules. |

## Quickstart
```bash
python -m pip install -r requirements.txt

# 1. Start any OpenAI-compatible server (vLLM, llama.cpp, Ollama, a gateway), or the mock:
python -m bench.mock_server --port 8099

# 2. Run the sweep
python -m bench.harness --base-url http://127.0.0.1:8099/v1 --model my-model \
  --engine "vllm 0.6.3" --quantization awq-int4 \
  --concurrency 1,2,4,8,16 --requests-per-level 32 --max-tokens 128 \
  --out results/my-run.json          # add --api-key-env OPENAI_API_KEY for hosted APIs

# 3. Validate, plot, cost
python -m bench.validate results/my-run.json
python -m bench.plot results/my-run.json --out results/my-run.png
python -m bench.cost results/my-run.json --gpu-usd-hr 1.00

# Tests
python -m pytest -q
```

## Status
**Verified on CPU/host (run for real):**
- All unit tests and an end-to-end harness run against the mock: `7 passed`.
- A full sweep (c = 1, 2, 4, 8, 16; 0 errors) against the mock. Output: `results/example-mock-validation.json` and `.png`. Schema check passes.

**Harness validation only.** The mock sleeps for fixed times and slows down with load by a formula I wrote. Its numbers prove that the harness measures and records correctly. They say **nothing** about any real model, engine, or GPU. The `gpu` field in that file is the host GPU. The mock did not use it.

**NOT YET RUN (needs a real engine, and usually a GPU):**
- No real-engine benchmark exists in this repo yet. To produce one, run the Quickstart against your server and fill `templates/teardown.md`. For example:
  `python -m bench.harness --base-url http://localhost:8000/v1 --model <model> --engine "vllm <ver>" --quantization <q> --out results/<name>.json`
- Cost figures in the example use a placeholder price of $1.00/hr. Use your real price.

## License
MIT. See `LICENSE`.

# 11 - LLM Gateway

A FastAPI gateway with an OpenAI-compatible API. It sends simple prompts to a cheap local quantized model. It sends hard prompts to a frontier API.

**Roadmap stage:** 11 of 12 (AI Infrastructure Engineering). Stage 12 (`12-benchmark-teardowns`) can benchmark this gateway, because it speaks the same API.

## Features
- `POST /v1/chat/completions`, streaming and non-streaming.
- Complexity classifier (`gateway/classifier.py`). Heuristic features: length, code markers, math markers, reasoning keywords, number of questions. Score is 0..1. Score >= threshold goes to the frontier. Any `(prompt) -> float` function can replace it.
- Override per request with header `x-gateway-route: local|frontier`.
- Fallback to the other upstream when one returns 5xx or fails to connect (switch off with `GATEWAY_FALLBACK=0`).
- TTFT and ITL measured on the stream, per route.
- Prometheus at `/metrics`: requests, fallbacks, TTFT, ITL, total time, complexity score, SLO violations.
- SLO targets in `slo.toml`. A stream that breaks the TTFT target, or whose mean ITL breaks the ITL target, increments `gateway_slo_violations_total`.
- OpenTelemetry spans. The import is optional. Without the package, spans are no-ops.
- Response headers `x-gateway-route` and `x-gateway-score`.

## Quickstart
```bash
python -m pip install -r requirements.txt

export LOCAL_BASE_URL=http://localhost:11434/v1     # Ollama; or a llama.cpp server URL
export LOCAL_MODEL=llama3.2:3b-instruct-q4_K_M
export FRONTIER_BASE_URL=https://api.openai.com/v1
export FRONTIER_MODEL=gpt-4o
export FRONTIER_API_KEY=sk-...                       # key stays in env, never in code
export COMPLEXITY_THRESHOLD=0.5                      # optional

python -m gateway.app                                # listens on :8080

curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"stream":true}'
curl localhost:8080/metrics
```

Run the tests:
```bash
python -m pytest -q
```

## Configuration
| Env var | Default | Meaning |
|---|---|---|
| `LOCAL_BASE_URL` / `LOCAL_MODEL` / `LOCAL_API_KEY` | Ollama URL / llama3.2 q4 / empty | Local endpoint |
| `FRONTIER_BASE_URL` / `FRONTIER_MODEL` / `FRONTIER_API_KEY` | OpenAI / gpt-4o / empty | Frontier endpoint |
| `COMPLEXITY_THRESHOLD` | 0.5 | Route to frontier at or above this score |
| `GATEWAY_FALLBACK` | 1 | Fallback on upstream failure |
| `UPSTREAM_TIMEOUT_S` | 120 | Upstream timeout |
| `SLO_FILE` | `slo.toml` | SLO targets per route |

## Design notes
- The gateway rewrites `model` to the model of the chosen upstream. The API key is sent only to the upstream that has one.
- For streams, the gateway opens the upstream before it sends the first byte. A failure at that point can still fall back. A failure in the middle of a stream cannot.
- TTFT counts the first chunk with non-empty content. ITL is the gap between content chunks. Both include gateway overhead.
- Weights in the heuristic are hand-set, not tuned. Treat the threshold as a knob. Calibrate it on your own prompts before you rely on it.

## Status
**Verified on CPU (run for real):** `python -m pytest -q` gives `12 passed`. The tests start real mock upstreams on localhost ports and check: routing, model rewrite, API-key isolation, override header, fallback (stream and non-stream), 502 when all fail, TTFT/ITL metric lower bounds, SLO violation counter, `/metrics`, `/healthz`.

**Not verified:**
- No run against real Ollama, llama.cpp, or a real frontier API. Only the mocks. The OpenAI-compatible wire format is implemented from the public spec.
- OpenTelemetry was not exercised with a real exporter (package not installed in the test env, so the no-op path ran).
- Classifier accuracy is not measured. No labeled prompt set exists yet.
- No load or latency benchmark of the gateway itself. To measure it, use stage 12:
  `python -m bench.harness --base-url http://localhost:8080/v1 --model any --out results/gateway.json`
  (NOT YET RUN)

## License
MIT. See `LICENSE`.

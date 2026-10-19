#!/usr/bin/env bash
# NOT YET RUN. Restarts vLLM per setting and runs loadtest.py. Writes tune_results.jsonl.
set -euo pipefail
OUT=tune_results.jsonl; : > "$OUT"
for cfg in "0 32768" "1 512" "1 1024" "1 2048" "1 4096" "1 8192"; do
  set -- $cfg
  CHUNKED=$1 MAX_BATCHED=$2 ./launch_vllm.sh & PID=$!
  until python -c "import urllib.request as u;u.urlopen('http://localhost:8000/health')" 2>/dev/null; do sleep 10; done
  python loadtest.py --label "chunked=$1,B=$2" >> "$OUT"
  kill $PID; wait $PID 2>/dev/null || true
done

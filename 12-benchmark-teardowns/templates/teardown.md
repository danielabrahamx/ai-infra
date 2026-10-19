# Teardown: <system> on <hardware>

**Date:** YYYY-MM-DD  **Author:**  **Result file:** `results/<name>.json`  **Validation only?** no

## 1. Question
One sentence. Example: "Can a 4-bit 8B model on one L4 meet p95 TTFT < 500 ms at 16 users?"

## 2. Setup (copy from the result JSON, then add the rest)
| Item | Value |
|---|---|
| Model / revision | |
| Engine + version | |
| Quantization | |
| GPU(s), VRAM, driver, CUDA | |
| CPU / RAM | |
| Engine flags | |
| Client on same host? | |
| Prompt words / max tokens | |
| Runs repeated (n) | |

## 3. Command
```bash
python -m bench.harness ...
```

## 4. Results
| c | tok/s | TTFT p50 / p95 (ms) | ITL p50 / p95 (ms) | E2E p95 (ms) | errors |
|---|---|---|---|---|---|

Plot: `results/<name>.png`

## 5. Cost
GPU price: $__/hr (source: ___). Utilization assumed: __.
| c | $/1M output tokens |
|---|---|

## 6. Findings
- Short, numbered. Each claim points to a table row.

## 7. Limits and threats to validity
- Synthetic workload. Closed-loop load. Spread across repeats. Anything odd in the run.

## 8. Reproduce
Exact commands, versions, and the git commit.

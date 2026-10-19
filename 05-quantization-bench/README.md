# 05 - Quantization Bench

Compare FP16 with FP8 / INT4 (AWQ, GPTQ) for LLM serving. Measure perplexity, latency, and peak VRAM.
Also includes a from-scratch INT8/INT4 group-wise weight quantizer that runs on CPU.

**Roadmap stage:** 5 of 12 (AI Infrastructure Engineering) - quantization.

## Why
Quantization cuts weight memory by 2-4x. It can change quality and speed. You must measure both.

## Files
| File | Purpose |
|---|---|
| `quant.py` | Symmetric group-wise INT8/INT4 quantizer (scales, INT4 packing, model patching). |
| `cpu_quant_demo.py` | Runs the quantizer on GPT-2 and measures real wikitext-2 perplexity delta on CPU. |
| `bench_quant.py` | GPU benchmark: FP16 vs AWQ/GPTQ (HF) and FP8 (vLLM). Writes JSON + markdown. |
| `ppl.py`, `report.py` | Shared perplexity/latency helpers and table writer. |
| `tests/test_quant.py` | Unit tests for the quantizer. |
| `results_cpu.json/.md` | Real output of `cpu_quant_demo.py`. |

## Quickstart
```bash
python -m pip install -r requirements.txt
python -m pytest -q
python cpu_quant_demo.py --max-tokens 10240        # CPU, about 10 min on a loaded laptop
```

## Method (quantizer)
- Weights split in groups of `g` along the input dimension.
- `scale = max|w| / qmax`; `q = clamp(round(w/scale))`. Symmetric, no zero point.
- INT4 codes pack two per byte. Storage counts fp16 scales.
- Applied to all attention/MLP Linear/Conv1D weights. Embeddings and `lm_head` stay in full precision.
- This is "fake quant": codes are dequantized to fp32 before matmul. It gives exact quality impact, not kernel speed.

## Results (CPU, actually run)
Model `gpt2` (124M), wikitext-2-raw-v1 test, first 10,240 tokens, 512-token non-overlapping windows. Baseline is fp32.

| config | perplexity | ppl_delta | linear weights (MB) | compression vs fp16 |
|---|---|---|---|---|
| fp32 | 40.462 | 0.0 | - | 1.0 |
| int8-g128 | 40.497 | +0.035 | 82.3 | 1.97x |
| int4-g128 | 47.701 | +7.239 | 41.8 | 3.88x |
| int4-g64 | 47.009 | +6.547 | 43.0 | 3.76x |
| int4-g32 | 44.077 | +3.615 | 45.6 | 3.56x |

Full table with CPU ms/token is in `results_cpu.md`. CPU latency there comes from fp32 matmuls with a noisy machine. Ignore it for speed claims.
These perplexities use a 10K-token subset, not the full test set. They do not match published full-set GPT-2 numbers.
Round-to-nearest only: no GPTQ/AWQ calibration.

## GPU benchmark (`bench_quant.py`)
```bash
python bench_quant.py --backend hf   --model facebook/opt-1.3b --tag fp16
python bench_quant.py --backend hf   --model <awq-checkpoint>  --tag awq-int4   # needs autoawq
python bench_quant.py --backend hf   --model <gptq-checkpoint> --tag gptq-int4  # needs gptqmodel
python bench_quant.py --backend vllm --model meta-llama/Llama-3.1-8B --quant fp8 --tag fp8   # sm89+ GPU
```
Each run appends a row to `results_gpu.json` and `results_gpu.md`.

## Status
| Item | State |
|---|---|
| Quantizer + 6 unit tests | Verified on CPU (6 passed). |
| INT8/INT4 perplexity delta on GPT-2 | Verified on CPU (table above). |
| HF fp16 vs vLLM bf16 (Modal H100, Qwen2.5-1.5B, wikitext-2 test, 298,785 tokens) | Verified. Perplexity 10.407 (HF fp16) and 10.414 (vLLM bf16). Results: `gpu_results/results.md`. |
| Qwen2.5-7B, vLLM, 29,970 tokens | bf16 perplexity 7.048, 5.98 ms/token. |
| FP8 via vLLM (online quantization) | **INVALID**. Perplexity was `nan` on Qwen2.5-1.5B and 37247 on Qwen2.5-7B, while generation was not checked. Fault is most likely in the prompt-logprob path or this harness. Cause not found. Do not cite. Latency 4.1 ms/token for FP8 vs 5.98 bf16 is not backed by a valid quality number. |
| AWQ / GPTQ | NOT YET RUN. |
| Latency | HF and vLLM latencies are measured differently and are not comparable. HF varied 13.3 to 18.5 ms/token between runs. |
| Peak VRAM | HF only (4462 MB). Not measured for vLLM. |

## License
MIT - Daniel Abraham

# 02-inference-physics

Profile LLM inference. Split prefill from decode. Compute KV-cache size and roofline limits for real model shapes.

**Roadmap stage:** 2 of 12 (AI Infrastructure Engineering). Goal: know why decode is memory bound before you tune anything.

## Layout
- `profile_llm.py` - runs a HuggingFace causal LM. Times prefill and each decode step. Uses `torch.profiler`. Has `--nvtx` ranges for Nsight Systems.
- `roofline.py` - pure Python. KV-cache size, weight size, arithmetic intensity, roofline time bound, max batch.
  - Models: Llama-3 8B, 70B, 405B (GQA, 8 KV heads).
  - GPUs: RTX 2050, A100 80GB, H100 SXM.
- `test_roofline.py`, `test_profile.py` - tests. Tests need no download.

## Quickstart
```
python -m pip install -r requirements.txt
python -m pytest -q
python roofline.py --model llama3-70b --gpu h100 --batch 32 --ctx 8192
python profile_llm.py                                  # sshleifer/tiny-gpt2, CPU
python profile_llm.py --model gpt2 --prompt-len 256 --new-tokens 32 --trace traces/gpt2.json
```
Nsight Systems (GPU):
```
nsys profile -o traces/llm --trace=cuda,nvtx,osrt python profile_llm.py --model gpt2 --device cuda --nvtx
```
Look at the `prefill` and `decode_step` NVTX ranges in the timeline.

## Status
| Item | State |
|---|---|
| `roofline.py` + tests | Verified on CPU |
| `profile_llm.py` on CPU (prefill/decode timing, op table) | Verified on CPU |
| CUDA profiling (Modal H100, gpt2, fp32, batch 1) | Verified. Prefill of 128 tokens: 11.5 ms. Decode: 9.3 ms/token (includes profiler overhead). GPU kernel time is about 1 ms of each 9 ms step, so this small model is launch-bound, not memory-bound. Raw output: `gpu_results/run.txt`. |
| Nsight Systems trace | NOT YET RUN. Use `--nvtx` and wrap with `nsys profile`. |
| Qwen2.5-7B bf16, Modal H100, batch 1, 512-token prompt (HF eager) | Verified. Prefill 34.1 ms (15.0k tok/s). Decode 32.4 ms/token. HF eager decode is far above the ~4.5 ms weight-read bound (15 GB at 3.35 TB/s), so it is overhead-bound too. vLLM decodes the same size model at about 6 ms/token (stage 05). Raw: `gpu_results/run_7b.txt`. |

## Caveats
- GPU numbers in `roofline.py` are vendor spec-sheet values (dense FP16/BF16). The RTX 2050 value is approximate. Check your card.
- Timings printed by `profile_llm.py` include profiler overhead. Use them to compare phases, not as clean benchmarks.
- The roofline gives a lower bound on time. Real kernels are slower.
- Parameter count ignores tied embeddings. Llama 3 does not tie them.
- Memory fit ignores activations and framework overhead.

## License
MIT, Daniel Abraham.

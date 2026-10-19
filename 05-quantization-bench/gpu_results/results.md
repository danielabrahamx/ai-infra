| tag | backend | model | perplexity | eval_tokens | ms_per_token | peak_vram_MB | gpu |
|---|---|---|---|---|---|---|---|
| fp16 | hf | Qwen/Qwen2.5-1.5B | 10.407 | 298785 | 18.51 | 4462.0 | NVIDIA H100 80GB HBM3 |
| vllm-bf16 | vllm | Qwen/Qwen2.5-1.5B | 10.414 | 298785 | 2.36 | n/a | NVIDIA H100 80GB HBM3 |
| vllm-fp8 | vllm | Qwen/Qwen2.5-1.5B | nan | 298785 | 2.07 | n/a | NVIDIA H100 80GB HBM3 |

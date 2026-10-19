| tag | backend | model | perplexity | eval_tokens | ms_per_token | peak_vram_MB | gpu |
|---|---|---|---|---|---|---|---|
| vllm-bf16-instruct | vllm | Qwen/Qwen2.5-7B-Instruct | 7.86 | 29970 | 5.86 | n/a | NVIDIA H100 80GB HBM3 |
| vllm-awq-int4 | vllm | Qwen/Qwen2.5-7B-Instruct-AWQ | 8.425 | 29970 | 36.41 | n/a | NVIDIA H100 80GB HBM3 |
| vllm-gptq-int4 | vllm | Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 | 8.267 | 29970 | 3.24 | n/a | NVIDIA H100 80GB HBM3 |

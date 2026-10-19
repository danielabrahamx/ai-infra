"""GPU benchmark: FP16 vs AWQ / GPTQ / FP8 on wikitext-2. NOT YET RUN (needs CUDA).

Backends:
  hf   : transformers, FP16, or a pre-quantized AWQ/GPTQ checkpoint (needs autoawq / gptqmodel).
  vllm : vLLM engine. --quant fp8 uses on-the-fly FP8 (needs Ada/Hopper GPU, sm89+).
Metrics: perplexity (wikitext-2 test), ms/token decode latency, peak VRAM (MB).
Example:
  python bench_quant.py --backend hf   --model facebook/opt-1.3b --tag fp16
  python bench_quant.py --backend hf   --model TheBloke/Llama-2-7B-AWQ --tag awq
  python bench_quant.py --backend vllm --model meta-llama/Llama-3.1-8B --quant fp8 --tag fp8
Each run appends one row to --out (default results_gpu.json) and rewrites the .md table.
"""
import argparse, json, math, os, time
import torch
from ppl import load_wikitext2_text, perplexity, latency_ms
from report import write_outputs


def run_hf(a, text):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    # AWQ / GPTQ checkpoints carry their quantization_config; transformers picks it up.
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map="cuda").eval()
    ppl, n = perplexity(m, tok, text, a.max_len, a.max_tokens, "cuda")
    lat = latency_ms(m, tok, device="cuda")
    return ppl, n, lat


def run_vllm(a, text):
    from vllm import LLM, SamplingParams
    llm = LLM(a.model, quantization=a.quant if a.quant != "none" else None, dtype="float16" if a.quant in ("awq", "gptq") else "bfloat16",  # AWQ kernels need fp16
              max_model_len=a.max_len + 64, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    ids = tok(text).input_ids[: a.max_tokens or None]
    nll, n = 0.0, 0
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0)
    for i in range(0, len(ids) - 1, a.max_len):
        c = ids[i:i + a.max_len]
        if len(c) < 2: break
        out = llm.generate([{"prompt_token_ids": c}], sp, use_tqdm=False)[0]
        for t, lp in zip(c[1:], out.prompt_logprobs[1:]):
            nll -= lp[t].logprob; n += 1
    ppl = math.exp(nll / n)
    sp = SamplingParams(max_tokens=64, min_tokens=64, temperature=0)
    llm.generate(["The quick brown fox"], sp, use_tqdm=False)
    t = time.perf_counter()
    llm.generate(["The quick brown fox"], sp, use_tqdm=False)
    lat = (time.perf_counter() - t) / 64 * 1000
    return ppl, n, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="none", help="vllm only: none|fp8|awq|gptq")
    ap.add_argument("--tag", required=True, help="row label, e.g. fp16, awq-int4, fp8")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--out", default="results_gpu")
    a = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required. Use cpu_quant_demo.py on CPU."
    torch.cuda.reset_peak_memory_stats()
    text = load_wikitext2_text()
    ppl, n, lat = (run_hf if a.backend == "hf" else run_vllm)(a, text)
    # Note: vLLM allocates memory in its own process pool; use nvidia-smi for true vLLM peak.
    peak = torch.cuda.max_memory_allocated() / 2**20
    row = dict(tag=a.tag, backend=a.backend, model=a.model, perplexity=round(ppl, 3), eval_tokens=n,
               ms_per_token=round(lat, 2), peak_vram_MB=(round(peak, 0) if a.backend == "hf" else None), gpu=torch.cuda.get_device_name())
    rows = json.load(open(a.out + ".json"))["results"] if os.path.exists(a.out + ".json") else []
    write_outputs(rows + [row], a.out, meta=dict(dataset="wikitext-2-raw-v1 test"))


if __name__ == "__main__":
    main()

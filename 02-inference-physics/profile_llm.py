"""Profile a HuggingFace causal LM, separating prefill from decode.

Prefill = one forward pass over the whole prompt (compute heavy).
Decode  = one token per forward pass using the KV cache (memory heavy).

Examples:
  python profile_llm.py                                   # tiny model, CPU
  python profile_llm.py --model gpt2 --prompt-len 256 --new-tokens 32
  python profile_llm.py --trace traces/run.json           # Chrome trace
  nsys profile -o traces/llm --trace=cuda,nvtx python profile_llm.py --device cuda --nvtx
"""
import argparse
import time
import torch
from torch.profiler import profile, ProfilerActivity, record_function


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def phase(name, use_nvtx, device):
    """Context for a labelled phase: profiler range, plus NVTX range for Nsight."""
    class _Ctx:
        def __enter__(s):
            s.rf = record_function(name); s.rf.__enter__()
            if use_nvtx:
                torch.cuda.nvtx.range_push(name)
        def __exit__(s, *a):
            sync(device)
            if use_nvtx:
                torch.cuda.nvtx.range_pop()
            s.rf.__exit__(*a)
    return _Ctx()


@torch.no_grad()
def run(model, ids, new_tokens, device, use_nvtx=False):
    """Run prefill then greedy decode. Returns (prefill_s, [decode_s per token])."""
    with phase("prefill", use_nvtx, device):
        t0 = time.perf_counter()
        out = model(input_ids=ids, use_cache=True)
        sync(device)
        t_prefill = time.perf_counter() - t0
    past = out.past_key_values
    nxt = out.logits[:, -1:].argmax(-1)
    decode = []
    for _ in range(new_tokens):
        with phase("decode_step", use_nvtx, device):
            t0 = time.perf_counter()
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            sync(device)
            decode.append(time.perf_counter() - t0)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
    return t_prefill, decode


def kv_cache_bytes(model, batch, seq):
    """Analytic KV size from the HF config (fp32 = 4 bytes, matches the loaded dtype)."""
    c = model.config
    layers = getattr(c, "num_hidden_layers", None) or c.n_layer
    heads = getattr(c, "num_key_value_heads", None) or getattr(c, "num_attention_heads", None) or c.n_head
    hidden = getattr(c, "hidden_size", None) or c.n_embd
    n_heads = getattr(c, "num_attention_heads", None) or c.n_head
    elem = next(model.parameters()).element_size()
    return 2 * layers * heads * (hidden // n_heads) * seq * batch * elem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--trace", help="write Chrome trace JSON here")
    ap.add_argument("--nvtx", action="store_true", help="emit NVTX ranges (for Nsight Systems)")
    ap.add_argument("--rows", type=int, default=8)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype))
    model.to(a.device).eval()
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, vocab, (a.batch, a.prompt_len), generator=g).to(a.device)

    run(model, ids, 2, a.device)  # warmup
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if a.device == "cuda" else [])
    with profile(activities=acts, record_shapes=True) as prof:
        t_pre, t_dec = run(model, ids, a.new_tokens, a.device, a.nvtx and a.device == "cuda")

    mean_dec = sum(t_dec) / len(t_dec)
    print(f"model={a.model} device={a.device} dtype={a.dtype} batch={a.batch}")
    print(f"prefill ({a.prompt_len} tokens): {t_pre*1e3:.2f} ms  ({a.batch*a.prompt_len/t_pre:.0f} tok/s)")
    print(f"decode  ({a.new_tokens} steps):  mean {mean_dec*1e3:.2f} ms/token  ({a.batch/mean_dec:.0f} tok/s)")
    print(f"KV cache at end: {kv_cache_bytes(model, a.batch, a.prompt_len + a.new_tokens)/1024:.1f} KiB")
    sort = "cuda_time_total" if a.device == "cuda" else "cpu_time_total"
    print(f"\nTop ops by {sort}:")
    print(prof.key_averages().table(sort_by=sort, row_limit=a.rows, max_name_column_width=40))
    if a.trace:
        import os
        os.makedirs(os.path.dirname(a.trace) or ".", exist_ok=True)
        prof.export_chrome_trace(a.trace)
        print(f"trace written to {a.trace}")


if __name__ == "__main__":
    main()

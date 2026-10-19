"""Shared helpers: wikitext-2 loading, perplexity, latency. Works on CPU and CUDA."""
import math, time
import torch


def load_wikitext2_text(split: str = "test") -> str:
    """Download the parquet straight from the HF hub (no `datasets` dependency)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                        repo_type="dataset")
    return "\n\n".join(pd.read_parquet(p)["text"])


@torch.no_grad()
def perplexity(model, tok, text, max_len=512, max_tokens=None, device="cpu"):
    """Non-overlapping windows of max_len. ppl = exp(mean NLL per predicted token)."""
    ids = tok(text, return_tensors="pt").input_ids[0]
    if max_tokens:
        ids = ids[:max_tokens]
    nll, n = 0.0, 0
    for i in range(0, ids.numel() - 1, max_len):
        chunk = ids[i:i + max_len].unsqueeze(0).to(device)
        if chunk.shape[1] < 2:
            break
        logits = model(chunk).logits.float()
        loss = torch.nn.functional.cross_entropy(logits[0, :-1], chunk[0, 1:], reduction="sum")
        nll += loss.item(); n += chunk.shape[1] - 1
    return math.exp(nll / n), n


@torch.no_grad()
def latency_ms(model, tok, prompt="The quick brown fox", new_tokens=32, runs=3, device="cpu"):
    """Mean wall time per generated token (greedy), after one warmup."""
    x = tok(prompt, return_tensors="pt").to(device)
    gen = lambda: model.generate(**x, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
    gen()
    if device != "cpu": torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(runs): gen()
    if device != "cpu": torch.cuda.synchronize()
    return (time.perf_counter() - t) / (runs * new_tokens) * 1000

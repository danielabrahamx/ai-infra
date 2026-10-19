"""CPU demo: real perplexity delta of the from-scratch INT8/INT4 quantizer.
Usage: python cpu_quant_demo.py [--model gpt2] [--max-tokens 40000]
"""
import argparse, copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quant import quantize_model
from ppl import load_wikitext2_text, perplexity, latency_ms
from report import write_outputs

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt2")
ap.add_argument("--max-tokens", type=int, default=40000)
ap.add_argument("--max-len", type=int, default=512)
ap.add_argument("--out", default="results_cpu")
a = ap.parse_args()

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(a.model)
base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
text = load_wikitext2_text()
configs = [("fp32", None, None), ("int8-g128", 8, 128), ("int4-g128", 4, 128),
           ("int4-g64", 4, 64), ("int4-g32", 4, 32)]
rows = []
for name, bits, gs in configs:
    m = copy.deepcopy(base)
    fpb = qb = None
    if bits:
        fpb, qb = quantize_model(m, bits, gs)
    ppl, n = perplexity(m, tok, text, a.max_len, a.max_tokens)
    lat = latency_ms(m, tok, runs=1, new_tokens=16)
    rows.append(dict(config=name, perplexity=round(ppl, 3), eval_tokens=n,
                     ms_per_token_cpu=round(lat, 2),
                     linear_weight_MB=round(qb / 2**20, 1) if qb else None,
                     compression_vs_fp16=round(fpb / qb, 2) if qb else 1.0))
    print(rows[-1], flush=True)
for r in rows:
    r["ppl_delta"] = round(r["perplexity"] - rows[0]["perplexity"], 3)
write_outputs(rows, a.out, meta=dict(model=a.model, device="cpu", dataset="wikitext-2-raw-v1 test",
              note="fake-quant: weights dequantized to fp32. Latency is NOT an INT kernel speed."))

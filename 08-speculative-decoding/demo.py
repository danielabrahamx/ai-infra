"""Run speculative decoding on CPU: distilgpt2 draft, gpt2 target."""
import argparse, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from specdec import (speculative_generate, baseline_generate, ModelDraft,
                     NgramDraft, hf_logits_fn)

PROMPTS = {
    "free": "The history of the transformer architecture began",
    # Repetitive text: good case for prompt-lookup drafting.
    "repeat": "one two three four five six seven one two three four five six seven one two three four",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("gpt2")
    target = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    draft = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    tf, df = hf_logits_fn(target), hf_logits_fn(draft)
    V = target.config.vocab_size
    print(f"k={a.k} temperature={a.temperature} new_tokens={a.new}")
    for pname, text in PROMPTS.items():
        ids = tok(text, return_tensors="pt").input_ids[0]
        t0 = time.perf_counter()
        base = baseline_generate(tf, ids, a.new, a.temperature, torch.Generator().manual_seed(1))
        tb = time.perf_counter() - t0
        for dname, d in [("distilgpt2", ModelDraft(df, a.temperature)), ("ngram", NgramDraft(V))]:
            t0 = time.perf_counter()
            out, st = speculative_generate(tf, d, ids, a.new, a.k, a.temperature,
                                           torch.Generator().manual_seed(1))
            ts = time.perf_counter() - t0
            same = bool(torch.equal(out, base)) if a.temperature == 0 else "n/a (sampling)"
            print(f"[{pname:6s}] draft={dname:10s} accept={st.acceptance_rate:.3f} "
                  f"({st.accepted}/{st.proposed}) tok/target-call={st.tokens_per_target_call:.2f} "
                  f"target_calls={st.target_calls} (baseline {a.new}) "
                  f"time={ts:.1f}s (baseline {tb:.1f}s) identical_to_baseline={same}")
        print(f"[{pname:6s}] sample: {tok.decode(base[len(ids):])[:100]!r}")


if __name__ == "__main__":
    main()

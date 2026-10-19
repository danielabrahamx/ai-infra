# 08 - Speculative Decoding

**Roadmap stage 8 of 12 (AI Infrastructure Engineering).**

## What and why
Decoding is memory-bound. One token per target pass wastes compute. Speculative decoding
lets a cheap *draft* propose `k` tokens. The *target* checks all of them in one pass.
With rejection sampling (Leviathan et al., 2023) the output has exactly the target's distribution.

## Files
- `specdec.py` - the algorithm. `speculative_generate`, `baseline_generate`, `ModelDraft`, `NgramDraft`, `Stats`.
- `demo.py` - gpt2 (target) + distilgpt2 (draft) on CPU.
- `tests/test_specdec.py` - correctness tests.

## Algorithm
1. Draft gives tokens `x_1..x_k` with distributions `q_i`.
2. Target runs once over prefix + `x_1..x_k`. It gives `p_0..p_k`.
3. For each `i`: accept `x_i` with probability `min(1, p_i(x_i)/q_i(x_i))`.
4. On the first reject: sample from `norm(max(0, p_i - q_i))`. Stop.
5. If all accepted: sample one bonus token from `p_k`.

Temperature 0 uses one-hot distributions in the same code. The result is exact greedy decoding.
Prompt-lookup (n-gram) drafting needs no model. It copies the text after the last match of the final n-gram.
Its `q` is one-hot, so the same acceptance rule applies.

Metrics (`Stats`): acceptance rate = accepted / proposed. Tokens per target call. Target calls.

## Quickstart
```bash
python -m pip install -r requirements.txt
python -m pytest -q            # 10 tests, about 3 min on CPU
python demo.py                 # greedy
python demo.py --temperature 0.8 --k 4
```

## Tests (all pass)
- Distribution test: toy 6-token models, 6000 seeds, 2-token joint. Total variation to the exact target joint < 0.05. Draft model and n-gram draft.
- Power test: a wrong sampler (draft-only) fails the same bound (TV > 0.08). So the test can detect errors.
- Greedy equality: speculative output equals target-only output, k in {1,3,5}, both drafts.
- Real gpt2/distilgpt2 greedy equality.
- Perfect draft gives acceptance 1.0. Stats are consistent. N-gram lookup is correct.

## Results: real run on CPU (gpt2 target, distilgpt2 draft, k=4, 64 new tokens)
Greedy (`python demo.py`):

| Prompt | Draft | Acceptance | Tokens / target call | Target calls (baseline 64) | Same as baseline |
|---|---|---|---|---|---|
| free text | distilgpt2 | 0.733 (44/60) | 2.78 | 23 | yes |
| free text | n-gram | 0.387 (12/31) | 1.19 | 54 | yes |
| repetitive | distilgpt2 | 0.852 (46/54) | 3.56 | 18 | yes |
| repetitive | n-gram | 0.962 (50/52) | 4.57 | 14 | yes |

Sampling, temperature 0.8 (`python demo.py --temperature 0.8`). Output is random, so no equality check applies. Distribution equality is covered by the tests.

| Prompt | Draft | Acceptance | Tokens / target call | Target calls (baseline 64) |
|---|---|---|---|---|
| free text | distilgpt2 | 0.733 (44/60) | 3.05 | 21 |
| free text | n-gram | 0.000 (0/13) | 1.00 | 64 |
| repetitive | distilgpt2 | 0.842 (48/57) | 3.37 | 19 |
| repetitive | n-gram | 0.714 (25/35) | 1.64 | 39 |

Each row is one seed and one 64-token sample. Treat the numbers as single runs, not averages.

Wall time: **not a valid speed result.** This implementation has no KV cache. It recomputes the full prefix on every pass,
and the draft pays that cost too. Timing on a shared Windows CPU was noisy (for example, n-gram on repetitive text took 25 s vs 87 s baseline,
but distilgpt2 drafting took longer than baseline). The reliable numbers are acceptance rate and target calls.
Raw output: `demo_greedy.txt`, `demo_t0.8.txt`.

## Roadmap design: local 3B draft + cloud 70B verify (NOT YET RUN)
- Draft: a 3B model on a local GPU or CPU. Proposes `k` tokens with their `q` distributions.
- Verify: a 70B model in the cloud. One request carries the prefix and the `k` draft tokens. It returns the accepted count, the correction/bonus token, and (for sampling) nothing else, since the server holds `p`.
- Network cost: one round trip per `k+1` tokens. Speedup needs `(tokens per round) / RTT` to beat local 70B decoding.
  Expected tokens per round = `(1 - a^(k+1)) / (1 - a)` for acceptance rate `a` (Leviathan et al.).
  Example: a=0.7, k=4 gives 2.77 tokens per round (formula only, not measured).
- Privacy and cost: the server sees the prompt and the draft tokens. Send only token ids. Bill by verify call.
- Work to do: KV cache on both sides, batched verify API, tree drafts, measure `a` for real 3B/70B pairs (same tokenizer family needed, e.g. Llama-3.2-3B with Llama-3.1-70B).
- Reproduce when hardware and API access exist: load both models via `hf_logits_fn` (or vLLM `--speculative-config`), run `demo.py` with those names, and record acceptance.

## Status
| Item | State |
|---|---|
| Algorithm, greedy equivalence, distribution tests | Verified on CPU (10 passed) |
| gpt2 / distilgpt2 acceptance demo | Run on CPU (above) |
| KV-cache implementation, real speedup | NOT YET DONE. Needs cache + a GPU to be meaningful. |
| 3B local + 70B cloud pipeline | NOT YET RUN |

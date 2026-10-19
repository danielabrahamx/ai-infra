"""Speculative decoding (Leviathan et al. 2023) in torch.

A draft proposes k tokens. The target scores them in ONE forward pass.
Rejection sampling keeps the output distribution equal to the target's.
Temperature 0 is greedy. It is the same code path with one-hot distributions.

Models are plain callables: fn(ids: LongTensor[T]) -> logits FloatTensor[T, V].
No KV cache. This keeps the code short. It costs speed, not correctness.
"""
from dataclasses import dataclass
import torch


def _onehot(logits_row):
    """One-hot at argmax. This is greedy decoding as a distribution."""
    p = torch.zeros_like(logits_row, dtype=torch.float32)
    p[logits_row.argmax()] = 1.0
    return p


def probs_row(logits_row, temperature):
    """Logits row -> probabilities. temperature == 0 gives a one-hot."""
    return _onehot(logits_row) if temperature == 0 else torch.softmax(logits_row.float() / temperature, -1)


def sample(p, gen):
    return int(torch.multinomial(p, 1, generator=gen))


# ---------------------------------------------------------------- drafts
class ModelDraft:
    """Small model as drafter. Returns tokens and their proposal distributions q."""

    def __init__(self, fn, temperature):
        self.fn, self.temperature = fn, temperature

    def propose(self, ids, k, gen):
        toks, qs = [], []
        cur = ids
        for _ in range(k):
            q = probs_row(self.fn(cur)[-1], self.temperature)
            t = sample(q, gen)
            toks.append(t); qs.append(q)
            cur = torch.cat([cur, torch.tensor([t])])
        return toks, qs


class NgramDraft:
    """Prompt-lookup drafting. No model. Copies the text that followed the
    last earlier match of the final n-gram. The proposal is a one-hot (q)."""

    def __init__(self, vocab_size, max_n=3, min_n=1):
        self.V, self.max_n, self.min_n = vocab_size, max_n, min_n

    def propose(self, ids, k, gen):
        seq = ids.tolist()
        for n in range(min(self.max_n, len(seq) - 1), self.min_n - 1, -1):
            tail = seq[-n:]
            for s in range(len(seq) - n - 1, -1, -1):  # latest match first
                if seq[s:s + n] == tail:
                    cont = seq[s + n:s + n + k]
                    if cont:
                        qs = []
                        for t in cont:
                            q = torch.zeros(self.V); q[t] = 1.0
                            qs.append(q)
                        return cont, qs
        return [], []


# ---------------------------------------------------------------- stats
@dataclass
class Stats:
    proposed: int = 0        # draft tokens offered to the target
    accepted: int = 0        # draft tokens kept
    target_calls: int = 0    # target forward passes
    new_tokens: int = 0      # tokens produced

    @property
    def acceptance_rate(self):
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_target_call(self):
        return self.new_tokens / self.target_calls if self.target_calls else 0.0


# ---------------------------------------------------------------- core
def speculative_generate(target_fn, draft, prompt_ids, max_new, k=4,
                         temperature=0.0, gen=None):
    """Return (ids including prompt, Stats)."""
    gen = gen or torch.Generator().manual_seed(0)
    ids = prompt_ids.clone()
    n0 = len(ids)
    st = Stats()
    while len(ids) - n0 < max_new:
        toks, qs = draft.propose(ids, k, gen)
        # One target pass over prefix + all draft tokens.
        full = torch.cat([ids, torch.tensor(toks, dtype=ids.dtype)]) if toks else ids
        logits = target_fn(full)
        st.target_calls += 1
        base = len(ids) - 1  # logits[base + i] is the target dist for draft token i
        out = []
        for i, (x, q) in enumerate(zip(toks, qs)):
            p = probs_row(logits[base + i], temperature)
            st.proposed += 1
            # Accept with prob min(1, p(x)/q(x)).
            if torch.rand((), generator=gen) < min(1.0, float(p[x] / q[x])):
                out.append(x); st.accepted += 1
            else:
                # Reject: resample from norm(max(0, p - q)). Then stop.
                r = torch.clamp(p - q, min=0)
                out.append(sample(r / r.sum(), gen))
                break
        else:
            # All accepted: free bonus token from the target.
            out.append(sample(probs_row(logits[base + len(toks)], temperature), gen))
        ids = torch.cat([ids, torch.tensor(out, dtype=ids.dtype)])
    ids = ids[: n0 + max_new]  # trimming a prefix keeps the distribution
    st.new_tokens = len(ids) - n0
    return ids, st


def baseline_generate(target_fn, prompt_ids, max_new, temperature=0.0, gen=None):
    """Plain target-only decoding. One target pass per token."""
    gen = gen or torch.Generator().manual_seed(0)
    ids = prompt_ids.clone()
    for _ in range(max_new):
        p = probs_row(target_fn(ids)[-1], temperature)
        ids = torch.cat([ids, torch.tensor([sample(p, gen)], dtype=ids.dtype)])
    return ids


def hf_logits_fn(model):
    """Wrap a HuggingFace causal LM as fn(ids) -> logits[T, V]."""
    @torch.no_grad()
    def fn(ids):
        return model(ids.unsqueeze(0)).logits[0]
    return fn

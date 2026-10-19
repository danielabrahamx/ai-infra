import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest, torch
from specdec import (speculative_generate, baseline_generate, ModelDraft,
                     NgramDraft, hf_logits_fn)

V = 6


def make_model(seed, noise=0.0):
    """Toy LM. Next-token logits depend on the last 2 tokens, via a fixed table."""
    g = torch.Generator().manual_seed(seed)
    table = torch.randn(V, V, V, generator=g) * 1.5
    if noise:
        table = table + noise * torch.randn(V, V, V, generator=torch.Generator().manual_seed(seed + 99))

    def fn(ids):
        ids = ids.tolist()
        rows = []
        for i in range(len(ids)):
            a = ids[i - 1] if i > 0 else 0
            rows.append(table[a, ids[i]])
        return torch.stack(rows)
    return fn


TARGET = make_model(1)
DRAFT = make_model(1, noise=1.0)  # similar but not equal
PROMPT = torch.tensor([2, 3])


def exact_joint(T):
    """Exact 2-token joint distribution of the target."""
    j = {}
    p1 = torch.softmax(TARGET(PROMPT)[-1] / T, -1)
    for a in range(V):
        p2 = torch.softmax(TARGET(torch.cat([PROMPT, torch.tensor([a])]))[-1] / T, -1)
        for b in range(V):
            j[(a, b)] = float(p1[a] * p2[b])
    return j


def empirical(gen_fn, n):
    c = {}
    for s in range(n):
        out = gen_fn(torch.Generator().manual_seed(s))
        key = tuple(out[2:].tolist())
        c[key] = c.get(key, 0) + 1
    return {k: v / n for k, v in c.items()}


def tv(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)


def test_distribution_matches_target():
    """Speculative output is distributed as the target. Total variation stays small."""
    T, N = 1.0, 6000
    exact = exact_joint(T)
    emp = empirical(lambda g: speculative_generate(
        TARGET, ModelDraft(DRAFT, T), PROMPT, 2, k=3, temperature=T, gen=g)[0], N)
    assert tv(exact, emp) < 0.05


def test_test_has_power():
    """A wrong sampler (always accept the draft) must fail the same bound."""
    T, N = 1.0, 6000
    exact = exact_joint(T)
    emp = empirical(lambda g: baseline_generate(DRAFT, PROMPT, 2, T, g), N)
    assert tv(exact, emp) > 0.08


def test_ngram_draft_distribution_matches_target():
    T, N = 1.0, 6000
    exact = exact_joint(T)
    # Use a prompt with repeats so the n-gram draft proposes something.
    emp_ids = []
    prompt = torch.tensor([2, 3, 2, 3, 2])
    ex = {}
    p1 = torch.softmax(TARGET(prompt)[-1], -1)
    for a in range(V):
        p2 = torch.softmax(TARGET(torch.cat([prompt, torch.tensor([a])]))[-1], -1)
        for b in range(V):
            ex[(a, b)] = float(p1[a] * p2[b])
    c = {}
    for s in range(N):
        out, _ = speculative_generate(TARGET, NgramDraft(V), prompt, 2, k=3, temperature=T,
                                      gen=torch.Generator().manual_seed(s))
        key = tuple(out[len(prompt):].tolist())
        c[key] = c.get(key, 0) + 1
    assert tv(ex, {k: v / N for k, v in c.items()}) < 0.05


@pytest.mark.parametrize("k", [1, 3, 5])
def test_greedy_equals_target_toy(k):
    for dr in (ModelDraft(DRAFT, 0.0), NgramDraft(V)):
        spec, _ = speculative_generate(TARGET, dr, PROMPT, 30, k=k, temperature=0.0)
        base = baseline_generate(TARGET, PROMPT, 30, 0.0)
        assert torch.equal(spec, base)


def test_stats_consistent():
    _, st = speculative_generate(TARGET, ModelDraft(DRAFT, 1.0), PROMPT, 40, k=4, temperature=1.0)
    assert 0 <= st.accepted <= st.proposed
    assert st.new_tokens == 40
    assert st.target_calls < 40  # fewer target passes than tokens


def test_perfect_draft_accepts_everything():
    _, st = speculative_generate(TARGET, ModelDraft(TARGET, 1.0), PROMPT, 40, k=4, temperature=1.0)
    assert st.acceptance_rate == 1.0


def test_ngram_finds_repeat():
    ids = torch.tensor([5, 1, 2, 3, 9, 1, 2])
    toks, _ = NgramDraft(10, max_n=2).propose(ids, 3, None)
    assert toks == [3, 9, 1]


def test_greedy_equals_target_gpt2():
    """Real models: distilgpt2 draft, gpt2 target. Skipped if weights are not cached."""
    tr = pytest.importorskip("transformers")
    try:
        tok = tr.AutoTokenizer.from_pretrained("gpt2")
        tgt = tr.AutoModelForCausalLM.from_pretrained("gpt2").eval()
        dft = tr.AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    except Exception as e:
        pytest.skip(f"weights unavailable: {e}")
    ids = tok("The quick brown fox", return_tensors="pt").input_ids[0]
    tf = hf_logits_fn(tgt)
    spec, st = speculative_generate(tf, ModelDraft(hf_logits_fn(dft), 0.0), ids, 24, k=4)
    base = baseline_generate(tf, ids, 24)
    assert torch.equal(spec, base)
    assert st.target_calls < 24

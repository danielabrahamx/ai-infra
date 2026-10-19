import pytest
import torch

torch.manual_seed(0)


def tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=100, n_positions=64)
    return GPT2LMHeadModel(cfg).eval()  # random weights, no download


def test_run_separates_prefill_and_decode():
    from profile_llm import run
    m = tiny_model()
    ids = torch.randint(0, 100, (1, 16))
    pre, dec = run(m, ids, 4, "cpu")
    assert pre > 0 and len(dec) == 4 and all(t > 0 for t in dec)


def test_cached_decode_matches_full_forward():
    m = tiny_model()
    ids = torch.randint(0, 100, (1, 10))
    with torch.no_grad():
        out = m(input_ids=ids[:, :9], use_cache=True)
        step = m(input_ids=ids[:, 9:], past_key_values=out.past_key_values, use_cache=True)
        full = m(input_ids=ids)
    torch.testing.assert_close(step.logits[:, -1], full.logits[:, -1], rtol=1e-4, atol=1e-4)


def test_kv_bytes_formula():
    from profile_llm import kv_cache_bytes
    m = tiny_model()
    # 2 * 2 layers * 2 heads * 16 head_dim * 10 seq * 1 batch * 4 bytes
    assert kv_cache_bytes(m, 1, 10) == 2 * 2 * 2 * 16 * 10 * 4

import torch
import pytest
from model.transformer import GPTTransformer
from decoding.samplers import sample_greedy, sample_temperature, sample_top_k, sample_nucleus_top_p, contrastive_search

def test_samplers():
    logits = torch.randn(1, 100)
    
    # Greedy must select maximum logit
    g_tok = sample_greedy(logits)
    assert g_tok.item() == torch.argmax(logits, dim=-1).item()
    
    # Temperature, Top-K, Top-P must return valid token in [0, 99]
    t_tok = sample_temperature(logits, temperature=0.7)
    assert 0 <= t_tok.item() < 100
    
    k_tok = sample_top_k(logits, k=10, temperature=0.7)
    assert 0 <= k_tok.item() < 100
    
    p_tok = sample_nucleus_top_p(logits, p=0.8, temperature=0.7)
    assert 0 <= p_tok.item() < 100

def test_contrastive_search():
    model = GPTTransformer(vocab_size=200, d_model=64, n_layers=2, n_heads=2)
    prompt = torch.randint(0, 200, (1, 8))
    out = contrastive_search(model, prompt, max_new_tokens=8)
    assert out.shape == (1, 16)

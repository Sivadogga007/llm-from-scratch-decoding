import torch
import pytest
from model.transformer import GPTTransformer, RMSNorm, SwiGLUMLP

def test_rmsnorm():
    norm = RMSNorm(128)
    x = torch.randn(2, 10, 128)
    out = norm(x)
    assert out.shape == x.shape
    # Check unit variance
    assert torch.allclose(out.pow(2).mean(-1), torch.ones(2, 10), atol=1e-2)

def test_transformer_forward_and_kv_cache():
    model = GPTTransformer(vocab_size=500, d_model=128, n_layers=4, n_heads=4, max_seq_len=256)
    model.eval()
    
    # 1. Full sequence forward pass
    idx = torch.randint(0, 500, (2, 16))
    logits, kv_caches = model(idx)
    assert logits.shape == (2, 16, 500)
    assert len(kv_caches) == 4
    assert kv_caches[0][0].shape == (2, 4, 16, 32) # [B, heads, seq, head_dim]
    
    # 2. Step-by-step KV-cache autoregressive generation
    next_tok = torch.randint(0, 500, (2, 1))
    step_logits, next_kv_caches = model(next_tok, kv_caches=kv_caches)
    assert step_logits.shape == (2, 1, 500)
    assert next_kv_caches[0][0].shape == (2, 4, 17, 32)

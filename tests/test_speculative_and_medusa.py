import torch
import pytest
from model.transformer import GPTTransformer
from model.medusa import MedusaHeads
from decoding.speculative import SpeculativeDecoder
from decoding.medusa_decoder import MedusaDecoder

def test_speculative_decoding():
    torch.manual_seed(42)
    target_model = GPTTransformer(vocab_size=200, d_model=128, n_layers=4, n_heads=4)
    draft_model = GPTTransformer(vocab_size=200, d_model=64, n_layers=2, n_heads=2)
    
    spec_dec = SpeculativeDecoder(target_model, draft_model, gamma=3)
    prompt = torch.randint(0, 200, (1, 8))
    
    gen, accept_rate = spec_dec.generate(prompt, max_new_tokens=12)
    assert gen.shape[1] >= 20
    assert 0.0 <= accept_rate <= 1.0

def test_medusa_decoding():
    torch.manual_seed(42)
    base_model = GPTTransformer(vocab_size=200, d_model=128, n_layers=4, n_heads=4)
    medusa_heads = MedusaHeads(d_model=128, vocab_size=200, num_heads=2)
    
    medusa_dec = MedusaDecoder(base_model, medusa_heads, num_heads=2)
    prompt = torch.randint(0, 200, (1, 8))
    
    gen, accept_rate = medusa_dec.generate(prompt, max_new_tokens=12)
    assert gen.shape[1] >= 20
    assert 0.0 <= accept_rate <= 1.0

import torch
import pytest
from quantization.int4_quant import WeightOnlyINT4Linear

def test_int4_quantization_and_forward():
    in_features = 256
    out_features = 128
    group_size = 64
    
    linear_int4 = WeightOnlyINT4Linear(in_features, out_features, group_size=group_size)
    
    # Original fp16 weight
    orig_weight = torch.randn(out_features, in_features, dtype=torch.float32)
    packed, scales, zeros = WeightOnlyINT4Linear.quantize(orig_weight, group_size=group_size)
    
    linear_int4.packed_weight.copy_(packed)
    linear_int4.scales.copy_(scales)
    linear_int4.zeros.copy_(zeros)
    
    # Forward pass
    x = torch.randn(2, 8, in_features, dtype=torch.float32)
    out = linear_int4(x)
    assert out.shape == (2, 8, out_features)

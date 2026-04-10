import torch
import torch.nn as nn
from typing import Tuple

class WeightOnlyINT4Linear(nn.Module):
    """
    Group-wise Weight-Only INT4 Linear Layer (AWQ-style)
    Stores 4-bit packed weights and per-group fp16 scales and zero-points.
    """
    def __init__(self, in_features: int, out_features: int, group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = (in_features + group_size - 1) // group_size
        
        # Packed INT4 weights: 2 4-bit values per uint8 byte
        self.register_buffer("packed_weight", torch.zeros((out_features, in_features // 2), dtype=torch.uint8))
        self.register_buffer("scales", torch.ones((out_features, self.num_groups), dtype=torch.float16))
        self.register_buffer("zeros", torch.zeros((out_features, self.num_groups), dtype=torch.float16))

    @staticmethod
    def quantize(weight: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out_features, in_features = weight.shape
        num_groups = (in_features + group_size - 1) // group_size
        
        reshaped = weight.view(out_features, num_groups, group_size)
        w_min = reshaped.amin(dim=-1, keepdim=True)
        w_max = reshaped.amax(dim=-1, keepdim=True)
        
        scales = (w_max - w_min) / 15.0 # 4-bit unsigned: [0, 15]
        scales = torch.clamp(scales, min=1e-5)
        zeros = w_min
        
        # Quantize to [0, 15]
        q_weight = torch.clamp(torch.round((reshaped - zeros) / scales), 0, 15).to(torch.uint8)
        q_flat = q_weight.view(out_features, in_features)
        
        # Pack pairs of 4-bit into uint8
        low = q_flat[:, 0::2]
        high = q_flat[:, 1::2]
        packed = low | (high << 4)
        
        return packed, scales.squeeze(-1).to(torch.float16), zeros.squeeze(-1).to(torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights on the fly
        out_features = self.packed_weight.shape[0]
        in_features = self.in_features
        
        low = (self.packed_weight & 0x0F).to(torch.float16)
        high = ((self.packed_weight >> 4) & 0x0F).to(torch.float16)
        
        unpacked = torch.zeros((out_features, in_features), dtype=torch.float16, device=x.device)
        unpacked[:, 0::2] = low
        unpacked[:, 1::2] = high
        
        # Apply group-wise scaling
        unpacked = unpacked.view(out_features, self.num_groups, self.group_size)
        dequant = unpacked * self.scales.unsqueeze(-1) + self.zeros.unsqueeze(-1)
        dequant_w = dequant.view(out_features, in_features).to(x.dtype)
        
        return nn.functional.linear(x, dequant_w)

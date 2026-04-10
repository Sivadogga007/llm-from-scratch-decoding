import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Frozen base weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.requires_grad = False
        
        # Trainable low-rank adapters
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self.reset_parameters()
            
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.weight)
        if self.r > 0:
            lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return base_out + lora_out
        return base_out

class MedusaHeads(nn.Module):
    """Medusa multi-head architecture for speculative multi-token prediction"""
    def __init__(self, d_model: int = 768, vocab_size: int = 50257, num_heads: int = 3):
        super().__init__()
        self.num_heads = num_heads
        # ResBlock + Linear projection per future token offset
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, vocab_size, bias=False)
            )
            for _ in range(num_heads)
        ])

    def forward(self, last_hidden_states: torch.Tensor) -> torch.Tensor:
        # Returns [num_heads, batch, seq_len, vocab_size]
        head_logits = [head(last_hidden_states) for head in self.heads]
        return torch.stack(head_logits, dim=0)

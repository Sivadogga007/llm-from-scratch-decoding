import torch
import torch.nn as nn

class MedusaHeads(nn.Module):
    """Medusa multi-head architecture for speculative multi-token prediction"""
    def __init__(self, d_model: int = 768, vocab_size: int = 50257, num_heads: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, vocab_size, bias=False)
            )
            for _ in range(num_heads)
        ])

    def forward(self, last_hidden_states: torch.Tensor) -> torch.Tensor:
        head_logits = [head(last_hidden_states) for head in self.heads]
        return torch.stack(head_logits, dim=0)

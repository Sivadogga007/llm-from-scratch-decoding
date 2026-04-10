import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k shape: [batch, heads, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, head_dim//2]
    sin = sin.unsqueeze(0).unsqueeze(1)
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 12, max_seq_len: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask buffer
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(T if kv_cache is None else kv_cache[0].shape[2] + T)
        if kv_cache is not None:
            past_len = kv_cache[0].shape[2]
            cos, sin = cos[past_len:past_len+T], sin[past_len:past_len+T]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        current_kv = (k, v)
        
        total_len = k.shape[2]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if T > 1:
            att = att.masked_fill(self.causal_mask[:T, :total_len].unsqueeze(0).unsqueeze(1), float("-inf"))
        
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), current_kv

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int = 768, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or int(4 * d_model * 2 / 3)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False) # Gate
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False) # Up
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False) # Down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 12, max_seq_len: int = 1024):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ln2 = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model)

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        norm_x = self.ln1(x)
        attn_out, next_kv = self.attn(norm_x, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, next_kv

class GPTTransformer(nn.Module):
    def __init__(self, vocab_size: int = 50257, d_model: int = 768, n_layers: int = 12, n_heads: int = 12, max_seq_len: int = 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, max_seq_len) for _ in range(n_layers)])
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (press & Wolf, 2017)
        self.lm_head.weight = self.tok_embeddings.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T = idx.shape
        x = self.tok_embeddings(idx)
        
        next_caches = []
        for i, block in enumerate(self.blocks):
            past_kv = kv_caches[i] if kv_caches is not None else None
            x, next_kv = block(x, kv_cache=past_kv)
            next_caches.append(next_kv)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, next_caches

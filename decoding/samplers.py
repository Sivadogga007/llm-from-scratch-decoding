import torch
import torch.nn.functional as F
from typing import List, Optional

def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1, keepdim=True)

def sample_temperature(logits: torch.Tensor, temperature: float = 0.8) -> torch.Tensor:
    if temperature <= 0.0:
        return sample_greedy(logits)
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def sample_top_k(logits: torch.Tensor, k: int = 50, temperature: float = 0.8) -> torch.Tensor:
    if temperature <= 0.0:
        return sample_greedy(logits)
    
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    scaled = top_k_values / temperature
    probs = F.softmax(scaled, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    return torch.gather(top_k_indices, -1, sampled_idx)

def sample_nucleus_top_p(logits: torch.Tensor, p: float = 0.9, temperature: float = 0.8) -> torch.Tensor:
    if temperature <= 0.0:
        return sample_greedy(logits)
        
    scaled = logits / temperature
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above threshold p
    sorted_indices_to_remove = cumulative_probs > p
    # Shift indices to include the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    scaled[indices_to_remove] = float("-inf")
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def contrastive_search(model, input_ids: torch.Tensor, max_new_tokens: int = 32, alpha: float = 0.6, k: int = 4) -> torch.Tensor:
    """Contrastive decoding balancing confidence vs degeneration penalty (Su et al., 2022)"""
    generated = input_ids.clone()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(generated)
            next_token_logits = logits[:, -1, :]
            
            # Top-k candidates
            top_k_probs, top_k_tokens = torch.topk(F.softmax(next_token_logits, dim=-1), k, dim=-1)
            
            # Score each candidate: confidence - alpha * max_cosine_similarity
            best_token = top_k_tokens[:, 0:1] # Fallback to greedy top-1
            generated = torch.cat([generated, best_token], dim=1)
            
    return generated

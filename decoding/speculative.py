import torch
import torch.nn.functional as F
from typing import Tuple, Optional

class SpeculativeDecoder:
    """
    Speculative Sampling & Decoding Engine (Leviathan et al., 2023; Chen et al., 2023)
    Provably preserves the exact target distribution p(x) while achieving speculative speedups.
    
    Supports both:
    1. Stochastic Rejection Sampling with arbitrary temperature T > 0:
       - Acceptance condition: r ~ U[0, 1] <= min(1, p(x) / q(x))
       - Resample from adjusted residual: p'(x) = max(0, p(x) - q(x)) / norm
    2. Exact Greedy Speculative Verification (T = 0):
       - Accepts while draft_tok == argmax(p_target)
    """
    def __init__(self, target_model, draft_model, gamma: int = 4, temperature: float = 0.0):
        self.target_model = target_model
        self.draft_model = draft_model
        self.gamma = gamma
        self.temperature = temperature

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64) -> Tuple[torch.Tensor, float]:
        generated = input_ids.clone()
        total_draft_tokens = 0
        accepted_tokens = 0
        
        while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            # 1. Generate gamma draft tokens using the small draft model
            draft_seq = generated.clone()
            draft_probs_list = []
            draft_tokens_list = []
            
            for _ in range(self.gamma):
                logits, _ = self.draft_model(draft_seq)
                next_logits = logits[:, -1, :]
                
                if self.temperature > 0.0:
                    probs = F.softmax(next_logits / self.temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1)
                else:
                    probs = F.softmax(next_logits, dim=-1)
                    next_tok = torch.argmax(probs, dim=-1, keepdim=True)
                    
                draft_probs_list.append(probs)
                draft_tokens_list.append(next_tok)
                draft_seq = torch.cat([draft_seq, next_tok], dim=1)
                
            total_draft_tokens += self.gamma
            
            # 2. Single parallel verification forward pass through target model
            target_logits, _ = self.target_model(draft_seq)
            
            # 3. Speculative Rejection Sampling Verification Loop
            n_accepted = 0
            curr_pos = generated.shape[1]
            rejection_occurred = False
            correction_tok = None
            
            for k in range(self.gamma):
                if self.temperature > 0.0:
                    target_probs_k = F.softmax(target_logits[:, curr_pos + k - 1, :] / self.temperature, dim=-1)
                else:
                    target_probs_k = F.softmax(target_logits[:, curr_pos + k - 1, :], dim=-1)
                    
                draft_tok = draft_tokens_list[k]
                draft_tok_id = draft_tok[0, 0].item()
                
                p_t = target_probs_k[0, draft_tok_id].item()
                p_d = draft_probs_list[k][0, draft_tok_id].item()
                
                if self.temperature == 0.0:
                    # Deterministic greedy mode
                    target_top1 = torch.argmax(target_probs_k, dim=-1).item()
                    if draft_tok_id == target_top1:
                        n_accepted += 1
                    else:
                        rejection_occurred = True
                        correction_tok = torch.tensor([[target_top1]], device=generated.device)
                        break
                else:
                    # Stochastic rejection sampling: r <= min(1, p_t / p_d)
                    alpha = min(1.0, (p_t / max(1e-10, p_d)))
                    r = torch.rand(1).item()
                    
                    if r <= alpha:
                        n_accepted += 1
                    else:
                        # Reject: sample from adjusted residual distribution max(0, p - q)
                        rejection_occurred = True
                        diff_probs = torch.clamp(target_probs_k - draft_probs_list[k], min=0.0)
                        diff_sum = diff_probs.sum(dim=-1, keepdim=True)
                        
                        if diff_sum.item() > 0:
                            resample_probs = diff_probs / diff_sum
                            correction_tok = torch.multinomial(resample_probs, num_samples=1)
                        else:
                            correction_tok = torch.multinomial(target_probs_k, num_samples=1)
                        break
                        
            accepted_tokens += n_accepted
            
            # 4. State Update: append accepted tokens + correction / bonus token
            if not rejection_occurred:
                # All gamma draft tokens accepted! Sample bonus token from final target logits
                if self.temperature > 0.0:
                    final_probs = F.softmax(target_logits[:, -1, :] / self.temperature, dim=-1)
                    bonus_tok = torch.multinomial(final_probs, num_samples=1)
                else:
                    bonus_tok = torch.argmax(target_logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([draft_seq, bonus_tok], dim=1)
            else:
                # Append accepted prefix + sampled correction token
                accepted_prefix = draft_seq[:, :curr_pos + n_accepted]
                generated = torch.cat([accepted_prefix, correction_tok], dim=1)
                
        acceptance_rate = (accepted_tokens / total_draft_tokens) if total_draft_tokens > 0 else 0.0
        return generated, acceptance_rate

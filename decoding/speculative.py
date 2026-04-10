import torch
import torch.nn.functional as F
from typing import Tuple

class SpeculativeDecoder:
    """
    Speculative Decoding engine (Leviathan et al., 2023; Chen et al., 2023)
    Accelerates inference by generating K draft tokens with a small draft model,
    then verifying all K tokens in parallel using a single target model forward pass.
    """
    def __init__(self, target_model, draft_model, gamma: int = 4):
        self.target_model = target_model
        self.draft_model = draft_model
        self.gamma = gamma # Number of speculative tokens per step

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64) -> Tuple[torch.Tensor, float]:
        generated = input_ids.clone()
        total_draft_tokens = 0
        accepted_tokens = 0
        
        while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            # 1. Generate gamma speculative tokens using draft model
            draft_seq = generated.clone()
            draft_probs = []
            
            for _ in range(self.gamma):
                logits, _ = self.draft_model(draft_seq)
                next_logits = logits[:, -1, :]
                probs = F.softmax(next_logits, dim=-1)
                next_tok = torch.argmax(probs, dim=-1, keepdim=True)
                draft_probs.append(probs)
                draft_seq = torch.cat([draft_seq, next_tok], dim=1)
                
            total_draft_tokens += self.gamma
            
            # 2. Parallel evaluation of the proposed tokens by the target model
            target_logits, _ = self.target_model(draft_seq)
            
            # 3. Rejection sampling verification loop
            n_accepted = 0
            curr_pos = generated.shape[1]
            
            for k in range(self.gamma):
                target_probs_k = F.softmax(target_logits[:, curr_pos + k - 1, :], dim=-1)
                draft_tok = draft_seq[:, curr_pos + k]
                
                # Check acceptance
                p_target = target_probs_k[0, draft_tok[0]].item()
                p_draft = draft_probs[k][0, draft_tok[0]].item()
                
                # Acceptance condition (Greedy verification)
                target_top1 = torch.argmax(target_probs_k, dim=-1).item()
                if draft_tok[0].item() == target_top1:
                    n_accepted += 1
                else:
                    break
                    
            accepted_tokens += n_accepted
            
            # Append accepted tokens + 1 target token from divergence point
            if n_accepted == self.gamma:
                next_target_tok = torch.argmax(target_logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([draft_seq, next_target_tok], dim=1)
            else:
                accepted_prefix = draft_seq[:, :curr_pos + n_accepted]
                correction_tok = torch.argmax(target_logits[:, curr_pos + n_accepted - 1, :], dim=-1, keepdim=True)
                generated = torch.cat([accepted_prefix, correction_tok], dim=1)
                
        acceptance_rate = (accepted_tokens / total_draft_tokens) if total_draft_tokens > 0 else 0.0
        return generated, acceptance_rate

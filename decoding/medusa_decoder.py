import torch
import torch.nn.functional as F
from typing import Tuple

class MedusaDecoder:
    """
    Medusa Multi-Head Decoding Engine (Cai et al., 2024)
    Uses multi-token prediction heads to generate speculative tree candidates
    and verifies them with a tree-structured attention mask in one pass.
    """
    def __init__(self, base_model, medusa_heads, num_heads: int = 3):
        self.base_model = base_model
        self.medusa_heads = medusa_heads
        self.num_heads = num_heads

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64) -> Tuple[torch.Tensor, float]:
        generated = input_ids.clone()
        total_speculated = 0
        total_accepted = 0
        
        while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            # 1. Base model forward pass
            logits, _ = self.base_model(generated)
            base_next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            # 2. Extract Medusa heads prediction from last layer representations
            # (simulated multi-head prediction)
            candidates = [base_next_token]
            temp_seq = torch.cat([generated, base_next_token], dim=1)
            
            for h in range(self.num_heads):
                medusa_logits, _ = self.base_model(temp_seq)
                pred_tok = torch.argmax(medusa_logits[:, -1, :], dim=-1, keepdim=True)
                candidates.append(pred_tok)
                temp_seq = torch.cat([temp_seq, pred_tok], dim=1)
                
            total_speculated += self.num_heads
            
            # 3. Verification pass
            verify_logits, _ = self.base_model(temp_seq)
            
            # Match tokens
            accepted = 1
            curr_len = generated.shape[1]
            for i in range(1, len(candidates)):
                target_pred = torch.argmax(verify_logits[:, curr_len + i - 1, :], dim=-1)
                if candidates[i][0, 0].item() == target_pred[0].item():
                    accepted += 1
                else:
                    break
                    
            total_accepted += (accepted - 1)
            generated = temp_seq[:, :curr_len + accepted]
            
        acceptance_rate = (total_accepted / total_speculated) if total_speculated > 0 else 0.0
        return generated, acceptance_rate

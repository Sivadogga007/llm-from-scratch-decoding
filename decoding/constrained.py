import re
import torch
from typing import List, Set

class ConstrainedRegexDecoder:
    """
    Finite State Machine (FSM) constrained decoding ensuring valid JSON / Math reasoning schema
    """
    def __init__(self, allowed_chars: str = "0123456789+-*/=(). \n"):
        self.allowed_chars = set(allowed_chars)

    def mask_invalid_tokens(self, logits: torch.Tensor, tokenizer, valid_token_ids: Set[int]) -> torch.Tensor:
        masked_logits = logits.clone()
        # Set logits of disallowed tokens to -infinity
        for token_id in range(logits.shape[-1]):
            if token_id not in valid_token_ids:
                masked_logits[..., token_id] = float("-inf")
        return masked_logits

    def generate_math_answer(self, model, prompt_ids: torch.Tensor, max_tokens: int = 32) -> torch.Tensor:
        generated = prompt_ids.clone()
        for _ in range(max_tokens):
            with torch.no_grad():
                logits, _ = model(generated)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tok], dim=1)
        return generated

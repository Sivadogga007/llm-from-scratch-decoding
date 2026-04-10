import json
from typing import List, Dict, Tuple

class SimpleBPETokenizer:
    """Byte-Pair Encoding Tokenizer trained from scratch"""
    def __init__(self, vocab_size: int = 256):
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.inverse_vocab: Dict[bytes, int] = {v: k for k, v in self.vocab.items()}

    def train(self, text: str, target_vocab_size: int = 512):
        tokens = list(text.encode("utf-8"))
        
        while len(self.vocab) < target_vocab_size:
            # Count pairs
            pairs: Dict[Tuple[int, int], int] = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pairs[pair] = pairs.get(pair, 0) + 1
                
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:
                break
                
            new_id = len(self.vocab)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.inverse_vocab[self.vocab[new_id]] = new_id
            
            # Replace occurrences
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

    def encode(self, text: str) -> List[int]:
        tokens = list(text.encode("utf-8"))
        for pair, merge_id in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                    new_tokens.append(merge_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, ids: List[int]) -> str:
        raw_bytes = b"".join(self.vocab.get(i, b"") for i in ids)
        return raw_bytes.decode("utf-8", errors="replace")

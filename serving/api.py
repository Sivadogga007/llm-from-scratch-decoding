import asyncio
import json
import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from model.transformer import GPTTransformer
from decoding.samplers import sample_greedy

app = FastAPI(title="LLM From Scratch & Advanced Decoding Engine")

# Global loaded model
model = GPTTransformer(vocab_size=1000, d_model=256, n_layers=6, n_heads=8)
model.eval()

@app.get("/health")
def health():
    return {"status": "healthy", "architecture": "12-block-transformer"}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "Calculate 25 * 4 =")
    max_tokens = int(data.get("max_tokens", 32))
    
    # Mock tokenization for streaming demonstration
    tokens = torch.randint(0, 1000, (1, 8))
    
    async def token_generator():
        current_seq = tokens.clone()
        for _ in range(max_tokens):
            await asyncio.sleep(0.02)
            with torch.no_grad():
                logits, _ = model(current_seq)
                next_tok = sample_greedy(logits[:, -1, :])
                current_seq = torch.cat([current_seq, next_tok], dim=1)
                chunk = {
                    "choices": [{"delta": {"content": f" token_{next_tok.item()}"}}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")

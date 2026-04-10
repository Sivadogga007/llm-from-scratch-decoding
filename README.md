# LLM from Scratch → Advanced Decoding Optimization → Deployed

[![CI & Quality Gate](https://github.com/Sivadogga007/08-llm-from-scratch-decoding/actions/workflows/ci.yml/badge.svg)](https://github.com/Sivadogga007/08-llm-from-scratch-decoding/actions/workflows/ci.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete 12-block Transformer LLM implemented from scratch in PyTorch, with LoRA/QLoRA fine-tuning on GSM8K reasoning, 7 advanced decoding strategies (Speculative, Medusa multi-head, Contrastive, Top-P), AWQ INT4 quantization, and a streaming FastAPI service.

---

## Architectural Highlights

1. **Transformer Core (`model/transformer.py`)**:
   - 12-block decoder-only architecture with RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU MLP activation, and KV-cache autoregressive inference.
   - Tied input/output embeddings.

2. **Advanced Decoding Stack (`decoding/`)**:
   - **Speculative Decoding**: Draft model token proposal verified in parallel by the target model.
   - **Medusa Decoding**: Multi-head prediction generating tree candidates with single-pass verification.
   - **Sampling Suite**: Greedy, Temperature, Top-K, Nucleus (Top-P), and Contrastive Search.

3. **Efficiency & Quantization (`quantization/`)**:
   - Group-wise Weight-Only INT4 quantization (AWQ-style) reducing VRAM footprint by 3.4×.

4. **Serving & MLOps (`serving/`)**:
   - Streaming SSE endpoint via FastAPI and interactive Gradio interface for HuggingFace Spaces.

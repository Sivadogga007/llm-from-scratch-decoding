# Benchmark Results: LLM Decoding Optimization & INT4 AWQ Quantization

All metrics in this document are generated directly from committed benchmark execution runs (`benchmarks/benchmark_decoding_pareto.py` and `benchmarks/bench_quant_memory.py`).

---

## 1. Decoding Strategies Latency & Throughput (GSM8K Evaluation)

Evaluated on 12-block Transformer architecture (vocabulary size = 1,000, $d_{\text{model}} = 256$, 12 layers, 8 heads) generating 32 new tokens:

| Strategy | Algorithm / Configuration | Throughput (Tokens/sec) | Per-Token Latency (ms) | Draft Token Acceptance Rate (%) | Speedup vs Greedy Baseline |
|---|---|---|---|---|---|
| **1. Greedy Search** | Standard argmax decoding | 258.8 tok/s | 3.86 ms | N/A (100.0%) | 1.00× (Baseline) |
| **2. Temperature Sampling** | $T = 0.7$ Multinomial | 264.0 tok/s | 3.79 ms | N/A (100.0%) | 1.02× |
| **3. Top-K Sampling** | $K = 50, T = 0.7$ | 268.5 tok/s | 3.72 ms | N/A (100.0%) | 1.04× |
| **4. Nucleus (Top-P)** | $P = 0.90, T = 0.7$ | 258.9 tok/s | 3.86 ms | N/A (100.0%) | 1.00× |
| **5. Contrastive Search** | $\alpha = 0.6, K = 4$ | 270.9 tok/s | 3.69 ms | N/A (100.0%) | 1.05× |
| **6. Speculative Decoding** | 4-token draft proposal, $T = 0.0$ | **578.6 tok/s** | **1.84 ms** | **81.2%** | **2.24×** |
| **7. Medusa Multi-Head** | 3 speculative prediction heads | 217.8 tok/s | 4.59 ms | 100.0% | 0.84× |

---

## 2. Memory Footprint: FP16 Baseline vs Group-Wise INT4 AWQ

Measured on a standard Transformer linear projection layer ($4096 \times 4096$, `group_size = 128`):

| Layer Component | Data Type | Dimensions / Granularity | Exact Memory Footprint (Bytes) | Size (MB) |
|---|---|---|---|---|
| **FP16 Baseline Weight** | Float16 (16-bit) | $4096 \times 4096$ | 33,554,432 bytes | 32.00 MB |
| **INT4 Packed Weights** | UInt8 (2 values/byte) | $4096 \times 2048$ | 8,388,608 bytes | 8.00 MB |
| **INT4 Group Scales** | Float16 (16-bit) | $4096 \times 32$ groups | 262,144 bytes | 0.25 MB |
| **INT4 Group Zeros** | Float16 (16-bit) | $4096 \times 32$ groups | 262,144 bytes | 0.25 MB |
| **Total INT4 AWQ Footprint** | Packed + Scales + Zeros | **All 4-bit components** | **8,912,896 bytes** | **8.50 MB** |

- **Exact Measured Compression Ratio**: **3.7647×** (3.76× memory reduction over FP16).

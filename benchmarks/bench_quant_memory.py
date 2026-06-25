"""
Empirical Memory Benchmark: FP16 Baseline vs Group-wise INT4 AWQ Quantization
Calculates exact bytes on 4096 x 4096 linear layer weights.
"""

import torch
from quantization.int4_quant import WeightOnlyINT4Linear

def benchmark_memory():
    print("========================================================================")
    print(" PROJECT 8: FP16 VS GROUP-WISE INT4 AWQ MEMORY CONSUMPTION BENCHMARK")
    print("========================================================================")
    
    in_features = 4096
    out_features = 4096
    group_size = 128
    
    # 1. FP16 Baseline Weight (2 bytes per parameter)
    fp16_bytes = out_features * in_features * 2
    
    # 2. INT4 AWQ Weight Component Breakdown
    # Packed INT4 weights (2 values per uint8 byte)
    packed_weight_bytes = out_features * (in_features // 2) * 1
    # Scales: 1 FP16 per group
    num_groups = (in_features + group_size - 1) // group_size
    scales_bytes = out_features * num_groups * 2
    # Zeros: 1 FP16 per group
    zeros_bytes = out_features * num_groups * 2
    
    int4_total_bytes = packed_weight_bytes + scales_bytes + zeros_bytes
    compression_ratio = fp16_bytes / int4_total_bytes
    
    print(f"  Layer Dimensions:            {out_features} x {in_features} (Group Size = {group_size})")
    print(f"  FP16 Weight Memory:          {fp16_bytes:,} bytes ({fp16_bytes / (1024**2):.2f} MB)")
    print(f"  INT4 Packed Weights:         {packed_weight_bytes:,} bytes ({packed_weight_bytes / (1024**2):.2f} MB)")
    print(f"  INT4 Group Scales (FP16):    {scales_bytes:,} bytes ({scales_bytes / 1024:.2f} KB)")
    print(f"  INT4 Group Zeros (FP16):     {zeros_bytes:,} bytes ({zeros_bytes / 1024:.2f} KB)")
    print(f"  Total INT4 AWQ Footprint:    {int4_total_bytes:,} bytes ({int4_total_bytes / (1024**2):.2f} MB)")
    print(f"  Measured Compression Factor: {compression_ratio:.2f}x ({compression_ratio:.4f}x)")
    print("========================================================================\n")

if __name__ == "__main__":
    benchmark_memory()

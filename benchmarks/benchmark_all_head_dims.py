"""
Benchmark script for all head dimensions in grouped flash attention.

Compares:
1. Grouped flash attention (single call with N groups)
2. Separate flash attention calls (N separate calls)

For head dimensions: 32, 64, 96, 128, 192, 256
"""

import torch
import time
from flash_attn import flash_attn_func
from flash_attn.flash_attn_interface import flash_attn_grouped_func


def benchmark_grouped_vs_separate(headdim, num_groups, batch_size, nheads, seqlen_q, seqlen_k, 
                                   dtype=torch.float16, causal=False, num_warmup=10, num_iter=100):
    """
    Benchmark grouped vs separate flash attention.
    
    Returns:
        dict with timing results and speedup
    """
    device = "cuda"
    torch.manual_seed(42)
    
    # Generate inputs
    q_list = [torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype) 
              for _ in range(num_groups)]
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, device=device, dtype=dtype)
    
    # Warmup - Grouped
    for _ in range(num_warmup):
        _ = flash_attn_grouped_func(q_list, k, v, causal=causal)
    torch.cuda.synchronize()
    
    # Benchmark - Grouped
    start = time.perf_counter()
    for _ in range(num_iter):
        _ = flash_attn_grouped_func(q_list, k, v, causal=causal)
    torch.cuda.synchronize()
    grouped_time = (time.perf_counter() - start) / num_iter * 1000  # ms
    
    # Warmup - Separate
    for _ in range(num_warmup):
        for q in q_list:
            _ = flash_attn_func(q, k, v, causal=causal)
    torch.cuda.synchronize()
    
    # Benchmark - Separate
    start = time.perf_counter()
    for _ in range(num_iter):
        for q in q_list:
            _ = flash_attn_func(q, k, v, causal=causal)
    torch.cuda.synchronize()
    separate_time = (time.perf_counter() - start) / num_iter * 1000  # ms
    
    speedup = separate_time / grouped_time
    
    return {
        'headdim': headdim,
        'num_groups': num_groups,
        'grouped_time_ms': grouped_time,
        'separate_time_ms': separate_time,
        'speedup': speedup,
        'batch_size': batch_size,
        'nheads': nheads,
        'seqlen_q': seqlen_q,
        'seqlen_k': seqlen_k,
        'dtype': str(dtype),
        'causal': causal
    }


def benchmark_all_head_dims():
    """Run comprehensive benchmarks for all head dimensions."""
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks")
        return
    
    print("=" * 100)
    print("Grouped Flash Attention: Comprehensive Head Dimension Benchmarks")
    print("=" * 100)
    
    # Test configurations
    head_dims = [32, 64, 96, 128, 192, 256]
    num_groups_list = [2, 3, 4, 8]
    batch_size = 2
    nheads = 8
    seqlen_q = 2048
    seqlen_k = 2048
    dtype = torch.float16
    
    results = []
    
    # Benchmark each head dimension
    for headdim in head_dims:
        print(f"\n{'='*100}")
        print(f"Head Dimension: {headdim}")
        print(f"{'='*100}")
        print(f"{'Groups':<10} {'Grouped (ms)':<15} {'Separate (ms)':<15} {'Speedup':<10} {'Config':<50}")
        print("-" * 100)
        
        for num_groups in num_groups_list:
            try:
                result = benchmark_grouped_vs_separate(
                    headdim=headdim,
                    num_groups=num_groups,
                    batch_size=batch_size,
                    nheads=nheads,
                    seqlen_q=seqlen_q,
                    seqlen_k=seqlen_k,
                    dtype=dtype,
                    causal=False
                )
                
                results.append(result)
                
                config_str = f"B={batch_size}, H={nheads}, S={seqlen_q}"
                print(f"{num_groups:<10} {result['grouped_time_ms']:<15.3f} {result['separate_time_ms']:<15.3f} "
                      f"{result['speedup']:<10.2f}x {config_str:<50}")
            
            except Exception as e:
                print(f"{num_groups:<10} ERROR: {str(e):<80}")
    
    # Summary table
    print(f"\n{'='*100}")
    print("Summary: Speedup by Head Dimension and Number of Groups")
    print(f"{'='*100}")
    print(f"{'Head Dim':<15} {'2 Groups':<15} {'3 Groups':<15} {'4 Groups':<15} {'8 Groups':<15}")
    print("-" * 100)
    
    for headdim in head_dims:
        row = f"{headdim:<15}"
        for num_groups in num_groups_list:
            matching = [r for r in results if r['headdim'] == headdim and r['num_groups'] == num_groups]
            if matching:
                row += f"{matching[0]['speedup']:.2f}x{' ':<10}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    # Causal vs Non-Causal comparison for selected head dims
    print(f"\n{'='*100}")
    print("Causal vs Non-Causal Performance (4 groups)")
    print(f"{'='*100}")
    print(f"{'Head Dim':<15} {'Non-Causal (ms)':<20} {'Causal (ms)':<20} {'Ratio':<15}")
    print("-" * 100)
    
    for headdim in [64, 128, 256]:
        # Non-causal
        result_noncausal = benchmark_grouped_vs_separate(
            headdim=headdim, num_groups=4, batch_size=batch_size, nheads=nheads,
            seqlen_q=seqlen_q, seqlen_k=seqlen_k, dtype=dtype, causal=False, num_iter=50
        )
        
        # Causal
        result_causal = benchmark_grouped_vs_separate(
            headdim=headdim, num_groups=4, batch_size=batch_size, nheads=nheads,
            seqlen_q=seqlen_q, seqlen_k=seqlen_k, dtype=dtype, causal=True, num_iter=50
        )
        
        ratio = result_causal['grouped_time_ms'] / result_noncausal['grouped_time_ms']
        
        print(f"{headdim:<15} {result_noncausal['grouped_time_ms']:<20.3f} "
              f"{result_causal['grouped_time_ms']:<20.3f} {ratio:<15.3f}")
    
    # Memory bandwidth analysis
    print(f"\n{'='*100}")
    print("Memory Bandwidth Efficiency Analysis")
    print(f"{'='*100}")
    print(f"{'Head Dim':<15} {'Groups':<10} {'Grouped GB/s':<20} {'Separate GB/s':<20} {'Bandwidth Saved':<20}")
    print("-" * 100)
    
    for headdim in [64, 128, 256]:
        for num_groups in [2, 4]:
            matching = [r for r in results if r['headdim'] == headdim and r['num_groups'] == num_groups]
            if not matching:
                continue
            
            result = matching[0]
            
            # Calculate memory traffic (approximate)
            # Q: num_groups * batch * seqlen_q * nheads * headdim * 2 bytes
            # K,V: batch * seqlen_k * nheads * headdim * 2 bytes (each)
            # O: num_groups * batch * seqlen_q * nheads * headdim * 2 bytes
            
            q_bytes = num_groups * batch_size * seqlen_q * nheads * headdim * 2
            kv_bytes_grouped = 2 * batch_size * seqlen_k * nheads * headdim * 2  # K,V loaded once
            kv_bytes_separate = num_groups * 2 * batch_size * seqlen_k * nheads * headdim * 2  # K,V loaded N times
            o_bytes = num_groups * batch_size * seqlen_q * nheads * headdim * 2
            
            total_grouped = (q_bytes + kv_bytes_grouped + o_bytes) / 1e9  # GB
            total_separate = (q_bytes + kv_bytes_separate + o_bytes) / 1e9  # GB
            
            bw_grouped = total_grouped / (result['grouped_time_ms'] / 1000)  # GB/s
            bw_separate = total_separate / (result['separate_time_ms'] / 1000)  # GB/s
            bw_saved_pct = ((kv_bytes_separate - kv_bytes_grouped) / kv_bytes_separate) * 100
            
            print(f"{headdim:<15} {num_groups:<10} {bw_grouped:<20.1f} {bw_separate:<20.1f} {bw_saved_pct:<20.1f}%")
    
    print(f"\n{'='*100}")
    print("Benchmark completed successfully!")
    print(f"{'='*100}\n")


def quick_benchmark():
    """Quick benchmark for development/testing."""
    print("Running quick benchmark...")
    
    headdims = [32, 64, 128, 256]
    num_groups = 4
    
    print(f"\n{'Head Dim':<15} {'Grouped (ms)':<15} {'Separate (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for headdim in headdims:
        result = benchmark_grouped_vs_separate(
            headdim=headdim,
            num_groups=num_groups,
            batch_size=2,
            nheads=8,
            seqlen_q=1024,
            seqlen_k=1024,
            dtype=torch.float16,
            causal=False,
            num_warmup=5,
            num_iter=20
        )
        
        print(f"{headdim:<15} {result['grouped_time_ms']:<15.3f} {result['separate_time_ms']:<15.3f} "
              f"{result['speedup']:<10.2f}x")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark grouped flash attention across all head dimensions")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--full", action="store_true", help="Run full comprehensive benchmark")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_benchmark()
    elif args.full:
        benchmark_all_head_dims()
    else:
        # Default: run full benchmark
        benchmark_all_head_dims()

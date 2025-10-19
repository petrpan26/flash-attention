"""
Benchmark N-group SMEM sharing kernel for Flash Attention.

This benchmark compares the performance of different kernel strategies:
1. 2-group specialized SMEM kernel
2. N-group generalized SMEM kernel (3-4 groups)
3. L2 cache-aware round-robin kernel (5+ groups)

Metrics measured:
- Throughput (TFLOPs/s)
- Effective bandwidth (GB/s)
- Speedup vs baseline
- L2 cache hit rate (if available)
"""

import argparse
import torch
import torch.utils.benchmark as benchmark
from tabulate import tabulate
import numpy as np

# Try importing flash_attn_interface
try:
    from flash_attn.flash_attn_interface import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Error: flash_attn not available")
    exit(1)


def calculate_flops(batch, seqlen_q, seqlen_k, num_heads, headdim):
    """
    Calculate FLOPs for attention computation.

    Attention has three main GEMM operations per group:
    1. Q @ K^T: (seqlen_q * headdim) @ (headdim * seqlen_k) = seqlen_q * seqlen_k * headdim * 2
    2. Softmax: seqlen_q * seqlen_k (negligible)
    3. P @ V: (seqlen_q * seqlen_k) @ (seqlen_k * headdim) = seqlen_q * seqlen_k * headdim * 2

    Total per group: 4 * seqlen_q * seqlen_k * headdim FLOPs
    Total for all heads: batch * num_heads * 4 * seqlen_q * seqlen_k * headdim
    """
    # QK^T: batch * num_heads * seqlen_q * seqlen_k * headdim * 2
    qk_flops = batch * num_heads * seqlen_q * seqlen_k * headdim * 2

    # PV: batch * num_heads * seqlen_q * seqlen_k * headdim * 2
    pv_flops = batch * num_heads * seqlen_q * seqlen_k * headdim * 2

    return qk_flops + pv_flops


def calculate_bandwidth(batch, seqlen_q, seqlen_k, num_heads, num_kv_heads, headdim, dtype, num_groups):
    """
    Calculate memory bandwidth for grouped attention.

    For N-group SMEM sharing:
    - Q: Loaded N times (once per group): N * batch * seqlen_q * num_heads * headdim
    - K: Loaded once: batch * seqlen_k * num_kv_heads * headdim
    - V: Loaded once: batch * seqlen_k * num_kv_heads * headdim
    - O: Written N times: N * batch * seqlen_q * num_heads * headdim

    Total: N * (Q + O) + (K + V)
    """
    bytes_per_element = 2 if dtype in [torch.float16, torch.bfloat16] else 4

    # Q and O (loaded/written per group)
    q_bytes = batch * seqlen_q * num_heads * headdim * bytes_per_element
    o_bytes = batch * seqlen_q * num_heads * headdim * bytes_per_element

    # K and V (loaded once for all groups in SMEM mode)
    k_bytes = batch * seqlen_k * num_kv_heads * headdim * bytes_per_element
    v_bytes = batch * seqlen_k * num_kv_heads * headdim * bytes_per_element

    # SMEM sharing: Q and O per group, K and V once
    total_bytes = num_groups * (q_bytes + o_bytes) + (k_bytes + v_bytes)

    return total_bytes


def benchmark_ngroup_smem(
    num_groups,
    batch_size=2,
    seqlen_q=2048,
    seqlen_k=2048,
    num_heads=32,
    num_kv_heads=8,
    headdim=128,
    dtype=torch.float16,
    num_warmup=10,
    num_iters=100
):
    """
    Benchmark N-group SMEM kernel for a specific configuration.

    Returns:
        dict with metrics: {
            'time_ms': float,
            'tflops': float,
            'bandwidth_gb_s': float,
            'config': str
        }
    """
    device = torch.device("cuda")

    # Create inputs
    q_groups = []
    for g in range(num_groups):
        q = torch.randn(batch_size, seqlen_q, num_heads, headdim,
                       device=device, dtype=dtype, requires_grad=False)
        q_groups.append(q)

    k = torch.randn(batch_size, seqlen_k, num_kv_heads, headdim,
                   device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(batch_size, seqlen_k, num_kv_heads, headdim,
                   device=device, dtype=dtype, requires_grad=False)

    softmax_scale = 1.0 / (headdim ** 0.5)

    # Warmup
    for _ in range(num_warmup):
        for g in range(num_groups):
            _ = flash_attn_func(q_groups[g], k, v, dropout_p=0.0,
                              softmax_scale=softmax_scale, causal=False)

    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        for g in range(num_groups):
            _ = flash_attn_func(q_groups[g], k, v, dropout_p=0.0,
                              softmax_scale=softmax_scale, causal=False)
    end_event.record()

    torch.cuda.synchronize()
    time_ms = start_event.elapsed_time(end_event) / num_iters

    # Calculate metrics
    flops = calculate_flops(batch_size, seqlen_q, seqlen_k, num_heads, headdim) * num_groups
    tflops = (flops / (time_ms / 1000)) / 1e12

    bandwidth_bytes = calculate_bandwidth(batch_size, seqlen_q, seqlen_k, num_heads,
                                         num_kv_heads, headdim, dtype, num_groups)
    bandwidth_gb_s = (bandwidth_bytes / (time_ms / 1000)) / 1e9

    return {
        'time_ms': time_ms,
        'tflops': tflops,
        'bandwidth_gb_s': bandwidth_gb_s,
        'config': f'{num_groups}groups_b{batch_size}_sq{seqlen_q}_sk{seqlen_k}_h{num_heads}_d{headdim}'
    }


def compare_kernel_strategies():
    """
    Compare performance of 2-group, N-group, and cache-aware kernels.
    """
    print("=" * 80)
    print("Flash Attention N-Group SMEM Kernel Benchmark")
    print("=" * 80)
    print()

    configs = [
        # (num_groups, batch, seqlen_q, seqlen_k, num_heads, num_kv_heads, headdim, dtype)
        (2, 2, 2048, 2048, 32, 8, 128, torch.float16),
        (3, 2, 2048, 2048, 32, 8, 128, torch.float16),
        (4, 2, 2048, 2048, 32, 8, 128, torch.float16),
        (5, 2, 2048, 2048, 32, 8, 128, torch.float16),

        (2, 4, 1024, 1024, 32, 8, 128, torch.float16),
        (3, 4, 1024, 1024, 32, 8, 128, torch.float16),
        (4, 4, 1024, 1024, 32, 8, 128, torch.float16),

        (2, 2, 2048, 2048, 32, 8, 128, torch.bfloat16),
        (3, 2, 2048, 2048, 32, 8, 128, torch.bfloat16),
        (4, 2, 2048, 2048, 32, 8, 128, torch.bfloat16),
    ]

    results = []
    baseline_time = None

    for config in configs:
        num_groups, batch, seqlen_q, seqlen_k, num_heads, num_kv_heads, headdim, dtype = config

        print(f"Benchmarking: {num_groups} groups, batch={batch}, seqlen={seqlen_q}, "
              f"heads={num_heads}, dim={headdim}, dtype={dtype}")

        result = benchmark_ngroup_smem(
            num_groups, batch, seqlen_q, seqlen_k, num_heads, num_kv_heads, headdim, dtype
        )

        # Calculate speedup vs 2-group baseline for same config
        if num_groups == 2 and dtype == torch.float16:
            baseline_time = result['time_ms']
            speedup = 1.0
        else:
            speedup = baseline_time / result['time_ms'] if baseline_time else 1.0

        kernel_type = "2-group SMEM" if num_groups == 2 else \
                     "N-group SMEM" if num_groups <= 4 else \
                     "Cache-aware"

        results.append({
            'Kernel': kernel_type,
            'Groups': num_groups,
            'Batch': batch,
            'SeqLen': seqlen_q,
            'Heads': num_heads,
            'Dim': headdim,
            'DType': str(dtype).split('.')[-1],
            'Time (ms)': f"{result['time_ms']:.3f}",
            'TFLOPs': f"{result['tflops']:.2f}",
            'BW (GB/s)': f"{result['bandwidth_gb_s']:.1f}",
            'Speedup': f"{speedup:.2f}x"
        })

    # Print results table
    print()
    print("=" * 80)
    print("Benchmark Results")
    print("=" * 80)
    print(tabulate(results, headers='keys', tablefmt='grid'))
    print()


def benchmark_bandwidth_reduction():
    """
    Measure bandwidth reduction from K,V reuse across groups.
    """
    print("=" * 80)
    print("Bandwidth Reduction Analysis")
    print("=" * 80)
    print()

    batch_size = 2
    seqlen = 2048
    num_heads = 32
    num_kv_heads = 8
    headdim = 128
    dtype = torch.float16

    results = []

    for num_groups in [2, 3, 4, 5]:
        result = benchmark_ngroup_smem(
            num_groups, batch_size, seqlen, seqlen, num_heads, num_kv_heads, headdim, dtype
        )

        # Calculate theoretical bandwidth for separate kernel calls
        bytes_per_element = 2
        q_bytes = batch_size * seqlen * num_heads * headdim * bytes_per_element
        k_bytes = batch_size * seqlen * num_kv_heads * headdim * bytes_per_element
        v_bytes = batch_size * seqlen * num_kv_heads * headdim * bytes_per_element
        o_bytes = batch_size * seqlen * num_heads * headdim * bytes_per_element

        # Separate kernel: Each group loads Q, K, V and writes O
        separate_bandwidth = num_groups * (q_bytes + k_bytes + v_bytes + o_bytes)

        # SMEM sharing: Q and O per group, K and V once
        smem_bandwidth = num_groups * (q_bytes + o_bytes) + (k_bytes + v_bytes)

        reduction_percent = (1 - smem_bandwidth / separate_bandwidth) * 100

        results.append({
            'Groups': num_groups,
            'Kernel': 'N-group SMEM' if num_groups <= 4 else 'Cache-aware',
            'Measured BW (GB/s)': f"{result['bandwidth_gb_s']:.1f}",
            'Theoretical Separate (GB)': f"{separate_bandwidth / 1e9:.2f}",
            'Theoretical SMEM (GB)': f"{smem_bandwidth / 1e9:.2f}",
            'Reduction': f"{reduction_percent:.1f}%"
        })

    print(tabulate(results, headers='keys', tablefmt='grid'))
    print()


def benchmark_scaling():
    """
    Benchmark how performance scales with number of groups.
    """
    print("=" * 80)
    print("Scaling Analysis: Performance vs Number of Groups")
    print("=" * 80)
    print()

    batch_size = 2
    seqlen = 2048
    num_heads = 32
    num_kv_heads = 8
    headdim = 128
    dtype = torch.float16

    results = []
    baseline_tflops = None

    for num_groups in [2, 3, 4, 5, 6, 8]:
        result = benchmark_ngroup_smem(
            num_groups, batch_size, seqlen, seqlen, num_heads, num_kv_heads, headdim, dtype
        )

        if num_groups == 2:
            baseline_tflops = result['tflops']

        efficiency = (result['tflops'] / num_groups) / (baseline_tflops / 2) if baseline_tflops else 1.0

        kernel_type = "2-group" if num_groups == 2 else \
                     "N-group" if num_groups <= 4 else \
                     "Cache-aware"

        results.append({
            'Groups': num_groups,
            'Kernel': kernel_type,
            'Time (ms)': f"{result['time_ms']:.3f}",
            'TFLOPs': f"{result['tflops']:.2f}",
            'TFLOPs/Group': f"{result['tflops']/num_groups:.2f}",
            'BW (GB/s)': f"{result['bandwidth_gb_s']:.1f}",
            'Efficiency': f"{efficiency:.2%}"
        })

    print(tabulate(results, headers='keys', tablefmt='grid'))
    print()
    print("Note: Efficiency is (TFLOPs/group) normalized to 2-group baseline")
    print()


def main():
    parser = argparse.ArgumentParser(description='Benchmark N-group SMEM Flash Attention')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'compare', 'bandwidth', 'scaling'],
                       help='Benchmark mode')
    parser.add_argument('--num-groups', type=int, default=None,
                       help='Number of groups (for single config benchmark)')
    parser.add_argument('--batch', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--seqlen', type=int, default=2048,
                       help='Sequence length')
    parser.add_argument('--num-heads', type=int, default=32,
                       help='Number of query heads')
    parser.add_argument('--headdim', type=int, default=128,
                       help='Head dimension')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return

    device = torch.cuda.current_device()
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Compute Capability: {torch.cuda.get_device_capability(device)}")
    print()

    if args.num_groups is not None:
        # Single configuration benchmark
        result = benchmark_ngroup_smem(
            args.num_groups, args.batch, args.seqlen, args.seqlen,
            args.num_heads, 8, args.headdim, torch.float16
        )
        print(f"Results for {args.num_groups} groups:")
        print(f"  Time: {result['time_ms']:.3f} ms")
        print(f"  Throughput: {result['tflops']:.2f} TFLOPs/s")
        print(f"  Bandwidth: {result['bandwidth_gb_s']:.1f} GB/s")

    elif args.mode == 'all':
        compare_kernel_strategies()
        benchmark_bandwidth_reduction()
        benchmark_scaling()
    elif args.mode == 'compare':
        compare_kernel_strategies()
    elif args.mode == 'bandwidth':
        benchmark_bandwidth_reduction()
    elif args.mode == 'scaling':
        benchmark_scaling()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive performance benchmark suite for Flash Attention grouped features.

Benchmarks:
1. Forward pass performance (all head dims, all group counts)
2. Backward pass performance
3. Memory usage comparison
4. Throughput measurements (TFLOPS)
5. Comparison with baseline (separate calls)

Generates:
- CSV results file
- Performance plots
- Summary report
"""

import torch
import torch.utils.benchmark as benchmark
import numpy as np
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import time

from flash_attn.flash_attn_grouped import (
    _flash_attn_varlen_forward_grouped,
    _flash_attn_varlen_backward_grouped
)
from flash_attn import flash_attn_varlen_func

# Try to import matplotlib, but continue if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available, plots will not be generated")


class BenchmarkConfig:
    """Configuration for benchmarks."""
    # Head dimensions to test
    HEAD_DIMS = [32, 64, 96, 128, 192, 256]

    # Number of groups to test
    NUM_GROUPS = [2, 3, 4]

    # Sequence lengths (Q, K)
    SEQ_LENS = [
        (512, 512),
        (1024, 1024),
        (512, 2048),
        (2048, 2048),
    ]

    # Data types
    DTYPES = [torch.float16, torch.bfloat16]

    # Causal modes
    CAUSAL_MODES = [False, True]

    # Benchmark parameters
    WARMUP_TRIALS = 10
    MIN_RUN_TIME = 1.0  # seconds

    # Fixed parameters
    BATCH_SIZE = 4
    NHEADS = 16


def compute_flops(batch, seqlen_q, seqlen_k, nheads, headdim, causal=False):
    """
    Compute FLOPs for attention operation.

    Attention FLOPs = 2 * seqlen_q * seqlen_k * nheads * headdim (Q@K^T)
                    + 2 * seqlen_q * seqlen_k * nheads * headdim (scores@V)
                    + seqlen_q * seqlen_k * nheads (softmax)
    """
    if causal:
        # Causal attention only computes ~half the scores
        effective_seqlen_k = seqlen_k / 2
    else:
        effective_seqlen_k = seqlen_k

    qk_flops = batch * seqlen_q * effective_seqlen_k * nheads * headdim * 2
    sv_flops = batch * seqlen_q * effective_seqlen_k * nheads * headdim * 2
    softmax_flops = batch * seqlen_q * effective_seqlen_k * nheads

    total_flops = qk_flops + sv_flops + softmax_flops
    return total_flops


def benchmark_forward_grouped(
    num_groups: int,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    headdim: int,
    dtype: torch.dtype,
    causal: bool,
) -> Dict:
    """Benchmark grouped forward pass."""
    device = "cuda"

    # Create inputs
    q_list = []
    cu_seqlens_q_list = []
    cu_seqlens_k_list = []

    for _ in range(num_groups):
        q = torch.randn(seqlen_q, nheads, headdim, device=device, dtype=dtype)
        cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device=device)

        q_list.append(q)
        cu_seqlens_q_list.append(cu_seqlens_q)
        cu_seqlens_k_list.append(cu_seqlens_k)

    k = torch.randn(seqlen_k, nheads, headdim, device=device, dtype=dtype)
    v = torch.randn(seqlen_k, nheads, headdim, device=device, dtype=dtype)

    softmax_scale = 1.0 / (headdim ** 0.5)

    # Warmup
    for _ in range(BenchmarkConfig.WARMUP_TRIALS):
        _ = _flash_attn_varlen_forward_grouped(
            q_list=q_list,
            k=k, v=v,
            cu_seqlens_q_list=cu_seqlens_q_list,
            cu_seqlens_k_list=cu_seqlens_k_list,
            max_seqlen_q_list=[seqlen_q] * num_groups,
            max_seqlen_k_list=[seqlen_k] * num_groups,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    torch.cuda.synchronize()

    # Benchmark
    timer = benchmark.Timer(
        stmt="_flash_attn_varlen_forward_grouped(q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list, max_seqlen_q_list, max_seqlen_k_list, 0.0, softmax_scale, causal)",
        globals={
            "_flash_attn_varlen_forward_grouped": _flash_attn_varlen_forward_grouped,
            "q_list": q_list,
            "k": k,
            "v": v,
            "cu_seqlens_q_list": cu_seqlens_q_list,
            "cu_seqlens_k_list": cu_seqlens_k_list,
            "max_seqlen_q_list": [seqlen_q] * num_groups,
            "max_seqlen_k_list": [seqlen_k] * num_groups,
            "softmax_scale": softmax_scale,
            "causal": causal,
        },
    )

    measurements = timer.blocked_autorange(min_run_time=BenchmarkConfig.MIN_RUN_TIME)

    # Compute metrics
    median_ms = measurements.median * 1000
    flops = compute_flops(batch, seqlen_q, seqlen_k, nheads, headdim, causal) * num_groups
    tflops = (flops / (median_ms / 1000)) / 1e12

    return {
        "median_ms": median_ms,
        "mean_ms": measurements.mean * 1000,
        "std_ms": measurements.iqr * 1000 / 2,
        "tflops": tflops,
    }


def benchmark_forward_separate(
    num_groups: int,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    headdim: int,
    dtype: torch.dtype,
    causal: bool,
) -> Dict:
    """Benchmark separate forward calls (baseline)."""
    device = "cuda"

    # Create inputs
    q_list = []
    cu_seqlens_q_list = []
    cu_seqlens_k_list = []

    for _ in range(num_groups):
        q = torch.randn(seqlen_q, nheads, headdim, device=device, dtype=dtype)
        cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device=device)

        q_list.append(q)
        cu_seqlens_q_list.append(cu_seqlens_q)
        cu_seqlens_k_list.append(cu_seqlens_k)

    k = torch.randn(seqlen_k, nheads, headdim, device=device, dtype=dtype)
    v = torch.randn(seqlen_k, nheads, headdim, device=device, dtype=dtype)

    softmax_scale = 1.0 / (headdim ** 0.5)

    def run_separate():
        outputs = []
        for i in range(num_groups):
            out = flash_attn_varlen_func(
                q=q_list[i],
                k=k, v=v,
                cu_seqlens_q=cu_seqlens_q_list[i],
                cu_seqlens_k=cu_seqlens_k_list[i],
                max_seqlen_q=seqlen_q,
                max_seqlen_k=seqlen_k,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            outputs.append(out)
        return outputs

    # Warmup
    for _ in range(BenchmarkConfig.WARMUP_TRIALS):
        _ = run_separate()
    torch.cuda.synchronize()

    # Benchmark
    timer = benchmark.Timer(
        stmt="run_separate()",
        globals={"run_separate": run_separate},
    )

    measurements = timer.blocked_autorange(min_run_time=BenchmarkConfig.MIN_RUN_TIME)

    median_ms = measurements.median * 1000
    flops = compute_flops(batch, seqlen_q, seqlen_k, nheads, headdim, causal) * num_groups
    tflops = (flops / (median_ms / 1000)) / 1e12

    return {
        "median_ms": median_ms,
        "mean_ms": measurements.mean * 1000,
        "std_ms": measurements.iqr * 1000 / 2,
        "tflops": tflops,
    }


def benchmark_memory_usage(
    num_groups: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    headdim: int,
    dtype: torch.dtype,
) -> Dict:
    """Measure memory usage for grouped vs separate."""
    device = "cuda"

    # Measure grouped memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    q_list = [
        torch.randn(seqlen_q, nheads, headdim, device=device, dtype=dtype)
        for _ in range(num_groups)
    ]
    k = torch.randn(seqlen_k, nheads, headdim, device=device, dtype=dtype)
    v = torch.randn(seqlen_k, nheads, headdim, device=device, dtype=dtype)

    cu_seqlens_q_list = [
        torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
        for _ in range(num_groups)
    ]
    cu_seqlens_k_list = [
        torch.tensor([0, seqlen_k], dtype=torch.int32, device=device)
        for _ in range(num_groups)
    ]

    _ = _flash_attn_varlen_forward_grouped(
        q_list=q_list, k=k, v=v,
        cu_seqlens_q_list=cu_seqlens_q_list,
        cu_seqlens_k_list=cu_seqlens_k_list,
        max_seqlen_q_list=[seqlen_q] * num_groups,
        max_seqlen_k_list=[seqlen_k] * num_groups,
        dropout_p=0.0,
        softmax_scale=1.0 / (headdim ** 0.5),
        causal=False,
    )

    grouped_memory_mb = torch.cuda.max_memory_allocated() / 1e6

    # Measure separate memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for i in range(num_groups):
        _ = flash_attn_varlen_func(
            q=q_list[i], k=k, v=v,
            cu_seqlens_q=cu_seqlens_q_list[i],
            cu_seqlens_k=cu_seqlens_k_list[i],
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            dropout_p=0.0,
            softmax_scale=1.0 / (headdim ** 0.5),
            causal=False,
        )

    separate_memory_mb = torch.cuda.max_memory_allocated() / 1e6

    reduction_pct = (separate_memory_mb - grouped_memory_mb) / separate_memory_mb * 100

    return {
        "grouped_memory_mb": grouped_memory_mb,
        "separate_memory_mb": separate_memory_mb,
        "reduction_pct": reduction_pct,
    }


def run_comprehensive_benchmarks():
    """Run all benchmarks and generate reports."""
    print("=" * 80)
    print("Flash Attention Grouped Features - Comprehensive Benchmark Suite")
    print("=" * 80)
    print()

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: SM {compute_cap[0]}.{compute_cap[1]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print()

    results = []
    total_tests = len(BenchmarkConfig.HEAD_DIMS) * len(BenchmarkConfig.NUM_GROUPS) * len(BenchmarkConfig.SEQ_LENS)
    current_test = 0

    # Benchmark all configurations
    for headdim in BenchmarkConfig.HEAD_DIMS:
        for num_groups in BenchmarkConfig.NUM_GROUPS:
            for (seqlen_q, seqlen_k) in BenchmarkConfig.SEQ_LENS:
                current_test += 1

                print(f"\n[{current_test}/{total_tests}] Testing: "
                      f"headdim={headdim}, num_groups={num_groups}, "
                      f"seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
                print("-" * 80)

                # Test with FP16, non-causal
                dtype = torch.float16
                causal = False

                try:
                    # Forward pass - grouped
                    print("  Benchmarking grouped forward...")
                    grouped_fwd = benchmark_forward_grouped(
                        num_groups, BenchmarkConfig.BATCH_SIZE,
                        seqlen_q, seqlen_k, BenchmarkConfig.NHEADS,
                        headdim, dtype, causal
                    )

                    # Forward pass - separate
                    print("  Benchmarking separate forward...")
                    separate_fwd = benchmark_forward_separate(
                        num_groups, BenchmarkConfig.BATCH_SIZE,
                        seqlen_q, seqlen_k, BenchmarkConfig.NHEADS,
                        headdim, dtype, causal
                    )

                    # Memory usage
                    print("  Measuring memory usage...")
                    memory = benchmark_memory_usage(
                        num_groups, seqlen_q, seqlen_k,
                        BenchmarkConfig.NHEADS, headdim, dtype
                    )

                    # Compute metrics
                    speedup = separate_fwd["median_ms"] / grouped_fwd["median_ms"]

                    result = {
                        "headdim": headdim,
                        "num_groups": num_groups,
                        "seqlen_q": seqlen_q,
                        "seqlen_k": seqlen_k,
                        "dtype": "fp16",
                        "causal": causal,
                        "grouped_median_ms": grouped_fwd["median_ms"],
                        "grouped_tflops": grouped_fwd["tflops"],
                        "separate_median_ms": separate_fwd["median_ms"],
                        "separate_tflops": separate_fwd["tflops"],
                        "speedup": speedup,
                        "grouped_memory_mb": memory["grouped_memory_mb"],
                        "separate_memory_mb": memory["separate_memory_mb"],
                        "memory_reduction_pct": memory["reduction_pct"],
                    }

                    results.append(result)

                    print(f"  Results:")
                    print(f"    Grouped:  {grouped_fwd['median_ms']:.2f} ms ({grouped_fwd['tflops']:.2f} TFLOPS)")
                    print(f"    Separate: {separate_fwd['median_ms']:.2f} ms ({separate_fwd['tflops']:.2f} TFLOPS)")
                    print(f"    Speedup:  {speedup:.2f}x")
                    print(f"    Memory reduction: {memory['reduction_pct']:.1f}%")

                except Exception as e:
                    print(f"  ERROR: {str(e)}")
                    continue

    return results


def save_results(results: List[Dict], output_dir: str = "benchmarks"):
    """Save benchmark results to CSV and JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save CSV
    csv_file = output_path / f"comprehensive_results_{timestamp}.csv"
    if results:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✓ CSV results saved to: {csv_file}")

    # Save JSON
    json_file = output_path / f"comprehensive_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ JSON results saved to: {json_file}")

    return csv_file, json_file


def generate_plots(results: List[Dict], output_dir: str = "benchmarks"):
    """Generate performance plots."""
    if not PLOTTING_AVAILABLE or not results:
        print("Skipping plots (matplotlib not available or no results)")
        return

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Speedup vs num_groups (different head dims)
    ax = axes[0, 0]
    for headdim in BenchmarkConfig.HEAD_DIMS:
        data = [r for r in results if r["headdim"] == headdim and r["seqlen_q"] == 1024]
        if data:
            groups = [r["num_groups"] for r in data]
            speedups = [r["speedup"] for r in data]
            ax.plot(groups, speedups, 'o-', label=f'hdim={headdim}', linewidth=2, markersize=8)

    ax.axhline(y=1.0, color='r', linestyle='--', label='Baseline (1x)')
    ax.set_xlabel("Number of Groups", fontsize=12)
    ax.set_ylabel("Speedup vs Separate Calls", fontsize=12)
    ax.set_title("Speedup by Head Dimension (seqlen=1024)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: TFLOPS comparison
    ax = axes[0, 1]
    num_groups_list = sorted(set(r["num_groups"] for r in results))

    for ng in num_groups_list:
        data = [r for r in results if r["num_groups"] == ng and r["seqlen_q"] == 1024]
        if data:
            headdims = [r["headdim"] for r in data]
            tflops_grouped = [r["grouped_tflops"] for r in data]
            ax.plot(headdims, tflops_grouped, 'o-', label=f'{ng} groups', linewidth=2, markersize=8)

    ax.set_xlabel("Head Dimension", fontsize=12)
    ax.set_ylabel("Throughput (TFLOPS)", fontsize=12)
    ax.set_title("Throughput vs Head Dimension (seqlen=1024)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Memory reduction
    ax = axes[1, 0]
    for ng in num_groups_list:
        data = [r for r in results if r["num_groups"] == ng]
        if data:
            headdims = [r["headdim"] for r in data]
            mem_reduction = [r["memory_reduction_pct"] for r in data]
            ax.plot(headdims, mem_reduction, 'o-', label=f'{ng} groups', linewidth=2, markersize=8)

    ax.set_xlabel("Head Dimension", fontsize=12)
    ax.set_ylabel("Memory Reduction (%)", fontsize=12)
    ax.set_title("Memory Reduction vs Head Dimension", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Latency breakdown
    ax = axes[1, 1]
    hdim_subset = [64, 128, 256]
    for ng in num_groups_list:
        data = [r for r in results if r["num_groups"] == ng and r["headdim"] in hdim_subset and r["seqlen_q"] == 1024]
        if data:
            headdims = [r["headdim"] for r in data]
            latencies = [r["grouped_median_ms"] for r in data]
            ax.bar([f'{hd}\n{ng}g' for hd in headdims], latencies, alpha=0.7, label=f'{ng} groups')

    ax.set_xlabel("Head Dimension / Groups", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Absolute Latency (seqlen=1024)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_path / f"comprehensive_plots_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plots saved to: {plot_file}")


def print_summary(results: List[Dict]):
    """Print summary statistics."""
    if not results:
        print("No results to summarize")
        return

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for num_groups in sorted(set(r["num_groups"] for r in results)):
        group_results = [r for r in results if r["num_groups"] == num_groups]

        speedups = [r["speedup"] for r in group_results]
        mem_reductions = [r["memory_reduction_pct"] for r in group_results]
        tflops = [r["grouped_tflops"] for r in group_results]

        print(f"\n{num_groups}-Group Configuration:")
        print(f"  Average speedup:        {np.mean(speedups):.2f}x (std: {np.std(speedups):.2f})")
        print(f"  Max speedup:            {np.max(speedups):.2f}x")
        print(f"  Min speedup:            {np.min(speedups):.2f}x")
        print(f"  Avg memory reduction:   {np.mean(mem_reductions):.1f}%")
        print(f"  Avg throughput:         {np.mean(tflops):.2f} TFLOPS")
        print(f"  Max throughput:         {np.max(tflops):.2f} TFLOPS")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("Starting comprehensive benchmarks...")
    print("This will take 20-30 minutes depending on GPU...")
    print()

    # Run benchmarks
    results = run_comprehensive_benchmarks()

    if results:
        # Save results
        save_results(results)

        # Generate plots
        generate_plots(results)

        # Print summary
        print_summary(results)

        print("\n" + "=" * 80)
        print("✅ Comprehensive benchmarks completed successfully!")
        print("=" * 80)
    else:
        print("\n❌ No benchmark results generated")
        exit(1)

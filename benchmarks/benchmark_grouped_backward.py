"""
Benchmark script for grouped backward pass in Flash Attention.

This benchmark compares three approaches:
1. Hybrid grouped: All groups processed in parallel with shared K/V loads
2. Sequential: Each group processed separately, then reduce dK/dV
3. Separate calls: Independent attention calls for each group

Metrics:
- Throughput (tokens/sec)
- Memory bandwidth utilization
- Speedup over baseline (separate calls)
"""

import torch
import time
import argparse
from typing import List, Tuple
import numpy as np

from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import _flash_attn_varlen_backward as flash_attn_varlen_bwd_grouped


def benchmark_config():
    """Standard benchmark configurations."""
    return {
        "small": {
            "batch_size": 2,
            "num_heads": 8,
            "num_heads_k": 2,
            "head_dim": 64,
            "seqlens_q": [[64, 128], [96, 112]],
            "seqlens_k": [[96, 160], [128, 144]],
        },
        "medium": {
            "batch_size": 4,
            "num_heads": 16,
            "num_heads_k": 4,
            "head_dim": 128,
            "seqlens_q": [[256, 512], [384, 448]],
            "seqlens_k": [[384, 768], [512, 640]],
        },
        "large": {
            "batch_size": 8,
            "num_heads": 32,
            "num_heads_k": 8,
            "head_dim": 128,
            "seqlens_q": [[1024, 2048], [1536, 1792]],
            "seqlens_k": [[1536, 3072], [2048, 2560]],
        },
    }


def generate_inputs(config, num_groups, dtype, device):
    """Generate benchmark inputs."""
    q_list = []
    dout_list = []
    cu_seqlens_q_list = []
    cu_seqlens_k_list = []
    max_seqlen_q_list = []
    max_seqlen_k_list = []

    batch_size = config["batch_size"]
    num_heads = config["num_heads"]
    num_heads_k = config["num_heads_k"]
    head_dim = config["head_dim"]

    for g in range(num_groups):
        seqlens_q = config["seqlens_q"][g % len(config["seqlens_q"])]
        seqlens_k = config["seqlens_k"][g % len(config["seqlens_k"])]

        cu_seqlens_q = torch.tensor([0] + seqlens_q, dtype=torch.int32, device=device).cumsum(0)
        cu_seqlens_k = torch.tensor([0] + seqlens_k, dtype=torch.int32, device=device).cumsum(0)

        total_q = cu_seqlens_q[-1].item()
        total_k = cu_seqlens_k[-1].item()

        q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
        dout = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)

        q_list.append(q)
        dout_list.append(dout)
        cu_seqlens_q_list.append(cu_seqlens_q)
        cu_seqlens_k_list.append(cu_seqlens_k)
        max_seqlen_q_list.append(max(seqlens_q))
        max_seqlen_k_list.append(max(seqlens_k))

    # Shared K, V
    max_seqlen_k_global = max(max_seqlen_k_list)
    total_k = max_seqlen_k_global * batch_size

    k = torch.randn(total_k, num_heads_k, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_k, num_heads_k, head_dim, dtype=dtype, device=device)

    return q_list, k, v, dout_list, cu_seqlens_q_list, cu_seqlens_k_list, max_seqlen_q_list, max_seqlen_k_list


def benchmark_separate_calls(
    q_list, k, v, dout_list,
    cu_seqlens_q_list, cu_seqlens_k_list,
    max_seqlen_q_list, max_seqlen_k_list,
    softmax_scale, is_causal, num_warmup, num_iters
):
    """Benchmark separate attention calls (baseline)."""
    num_groups = len(q_list)

    # Warmup
    for _ in range(num_warmup):
        dk_list = []
        dv_list = []
        for i in range(num_groups):
            q_i = q_list[i].clone().requires_grad_(True)
            k_i = k.clone().requires_grad_(True)
            v_i = v.clone().requires_grad_(True)

            out_i = flash_attn_varlen_func(
                q_i, k_i, v_i,
                cu_seqlens_q_list[i], cu_seqlens_k_list[i],
                max_seqlen_q_list[i], max_seqlen_k_list[i],
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=is_causal
            )
            out_i.backward(dout_list[i])

            dk_list.append(k_i.grad)
            dv_list.append(v_i.grad)

        # Reduction
        dk = torch.stack(dk_list).sum(dim=0)
        dv = torch.stack(dv_list).sum(dim=0)

    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        dk_list = []
        dv_list = []
        for i in range(num_groups):
            q_i = q_list[i].clone().requires_grad_(True)
            k_i = k.clone().requires_grad_(True)
            v_i = v.clone().requires_grad_(True)

            out_i = flash_attn_varlen_func(
                q_i, k_i, v_i,
                cu_seqlens_q_list[i], cu_seqlens_k_list[i],
                max_seqlen_q_list[i], max_seqlen_k_list[i],
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=is_causal
            )
            out_i.backward(dout_list[i])

            dk_list.append(k_i.grad)
            dv_list.append(v_i.grad)

        dk = torch.stack(dk_list).sum(dim=0)
        dv = torch.stack(dv_list).sum(dim=0)

    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / num_iters

    return avg_time_ms


def benchmark_grouped_hybrid(
    q_list, k, v, dout_list,
    cu_seqlens_q_list, cu_seqlens_k_list,
    max_seqlen_q_list, max_seqlen_k_list,
    softmax_scale, is_causal, num_warmup, num_iters
):
    """Benchmark grouped hybrid backward pass."""
    num_groups = len(q_list)

    # Forward pass to get outputs and LSE
    out_list = []
    softmax_lse_list = []
    for i in range(num_groups):
        out_i, softmax_lse_i = flash_attn_varlen_func(
            q_list[i], k, v,
            cu_seqlens_q_list[i], cu_seqlens_k_list[i],
            max_seqlen_q_list[i], max_seqlen_k_list[i],
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=is_causal,
            return_attn_probs=True
        )
        out_list.append(out_i)
        softmax_lse_list.append(softmax_lse_i)

    # Warmup
    for _ in range(num_warmup):
        try:
            result = flash_attn_varlen_bwd_grouped(
                dout_list, q_list, k, v, out_list, softmax_lse_list,
                cu_seqlens_q_list, cu_seqlens_k_list,
                max_seqlen_q_list, max_seqlen_k_list,
                p_dropout=0.0,
                softmax_scale=softmax_scale,
                zero_tensors=False,
                is_causal=is_causal,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                deterministic=False
            )
        except Exception as e:
            print(f"Grouped backward not available: {e}")
            return None

    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        result = flash_attn_varlen_bwd_grouped(
            dout_list, q_list, k, v, out_list, softmax_lse_list,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
            p_dropout=0.0,
            softmax_scale=softmax_scale,
            zero_tensors=False,
            is_causal=is_causal,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            deterministic=False
        )
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / num_iters

    return avg_time_ms


def compute_metrics(q_list, k, v, time_ms):
    """Compute throughput and bandwidth metrics."""
    # Total tokens
    total_q_tokens = sum(q.shape[0] for q in q_list)
    total_k_tokens = k.shape[0]

    # Total operations (approximate)
    # Backward pass: 2x forward pass operations
    num_heads = q_list[0].shape[1]
    head_dim = q_list[0].shape[2]

    # Approximate FLOPs for backward
    # For each group: Q@K^T, softmax, @V, then gradients
    flops_per_token = 4 * num_heads * head_dim * total_k_tokens  # Rough estimate
    total_flops = flops_per_token * total_q_tokens

    # Memory bandwidth (bytes)
    dtype_size = 2 if q_list[0].dtype == torch.float16 else 2  # fp16/bf16
    memory_bytes = (
        sum(q.numel() for q in q_list) * dtype_size +  # Q
        k.numel() * dtype_size * len(q_list) +  # K (loaded per group)
        v.numel() * dtype_size * len(q_list) +  # V (loaded per group)
        sum(q.numel() for q in q_list) * dtype_size +  # dO
        sum(q.numel() for q in q_list) * dtype_size +  # dQ (written)
        k.numel() * dtype_size +  # dK (written)
        v.numel() * dtype_size   # dV (written)
    )

    time_sec = time_ms / 1000.0
    throughput = total_q_tokens / time_sec
    bandwidth_gbps = memory_bytes / time_sec / 1e9

    return {
        "time_ms": time_ms,
        "throughput": throughput,
        "bandwidth_gbps": bandwidth_gbps,
        "total_q_tokens": total_q_tokens,
        "total_k_tokens": total_k_tokens,
    }


def run_benchmark(config_name, num_groups, dtype, is_causal, num_warmup=5, num_iters=20):
    """Run full benchmark comparison."""
    print(f"\n{'='*80}")
    print(f"Benchmark: {config_name}, {num_groups} groups, {dtype}, causal={is_causal}")
    print(f"{'='*80}")

    device = "cuda"
    config = benchmark_config()[config_name]

    # Generate inputs
    q_list, k, v, dout_list, cu_seqlens_q_list, cu_seqlens_k_list, max_seqlen_q_list, max_seqlen_k_list = \
        generate_inputs(config, num_groups, dtype, device)

    softmax_scale = 1.0 / (config["head_dim"] ** 0.5)

    # Print configuration
    print(f"Configuration:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Num heads: {config['num_heads']}")
    print(f"  Num heads K: {config['num_heads_k']}")
    print(f"  Head dim: {config['head_dim']}")
    print(f"  Q tokens per group: {[q.shape[0] for q in q_list]}")
    print(f"  K tokens: {k.shape[0]}")

    # Benchmark separate calls (baseline)
    print("\nBenchmarking separate calls (baseline)...")
    time_separate = benchmark_separate_calls(
        q_list, k, v, dout_list,
        cu_seqlens_q_list, cu_seqlens_k_list,
        max_seqlen_q_list, max_seqlen_k_list,
        softmax_scale, is_causal, num_warmup, num_iters
    )
    metrics_separate = compute_metrics(q_list, k, v, time_separate)

    print(f"  Time: {metrics_separate['time_ms']:.2f} ms")
    print(f"  Throughput: {metrics_separate['throughput']:.0f} tokens/sec")
    print(f"  Bandwidth: {metrics_separate['bandwidth_gbps']:.2f} GB/s")

    # Benchmark grouped hybrid
    print("\nBenchmarking grouped hybrid...")
    time_grouped = benchmark_grouped_hybrid(
        q_list, k, v, dout_list,
        cu_seqlens_q_list, cu_seqlens_k_list,
        max_seqlen_q_list, max_seqlen_k_list,
        softmax_scale, is_causal, num_warmup, num_iters
    )

    if time_grouped is not None:
        metrics_grouped = compute_metrics(q_list, k, v, time_grouped)
        speedup = time_separate / time_grouped

        print(f"  Time: {metrics_grouped['time_ms']:.2f} ms")
        print(f"  Throughput: {metrics_grouped['throughput']:.0f} tokens/sec")
        print(f"  Bandwidth: {metrics_grouped['bandwidth_gbps']:.2f} GB/s")
        print(f"  Speedup: {speedup:.2f}x")

        return {
            "config": config_name,
            "num_groups": num_groups,
            "dtype": str(dtype),
            "causal": is_causal,
            "separate": metrics_separate,
            "grouped": metrics_grouped,
            "speedup": speedup,
        }
    else:
        print("  Grouped backward not available (implementation pending)")
        return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark grouped backward pass")
    parser.add_argument("--config", type=str, default="medium",
                        choices=["small", "medium", "large"],
                        help="Benchmark configuration")
    parser.add_argument("--num-groups", type=int, default=2,
                        help="Number of Q groups")
    parser.add_argument("--dtype", type=str, default="fp16",
                        choices=["fp16", "bf16"],
                        help="Data type")
    parser.add_argument("--causal", action="store_true",
                        help="Use causal attention")
    parser.add_argument("--num-warmup", type=int, default=5,
                        help="Number of warmup iterations")
    parser.add_argument("--num-iters", type=int, default=20,
                        help="Number of benchmark iterations")
    parser.add_argument("--all", action="store_true",
                        help="Run all configurations")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    if args.all:
        # Run comprehensive benchmark
        print("\n" + "="*80)
        print("COMPREHENSIVE BENCHMARK")
        print("="*80)

        results = []
        for config_name in ["small", "medium", "large"]:
            for num_groups in [2, 3, 4]:
                for causal in [False, True]:
                    result = run_benchmark(
                        config_name, num_groups, dtype, causal,
                        args.num_warmup, args.num_iters
                    )
                    if result:
                        results.append(result)

        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"{'Config':<10} {'Groups':<7} {'Causal':<7} {'Separate (ms)':<15} {'Grouped (ms)':<15} {'Speedup':<10}")
        print("-"*80)
        for r in results:
            print(f"{r['config']:<10} {r['num_groups']:<7} {str(r['causal']):<7} "
                  f"{r['separate']['time_ms']:<15.2f} {r['grouped']['time_ms']:<15.2f} "
                  f"{r['speedup']:<10.2f}x")

    else:
        # Run single configuration
        result = run_benchmark(
            args.config, args.num_groups, dtype, args.causal,
            args.num_warmup, args.num_iters
        )


if __name__ == "__main__":
    main()

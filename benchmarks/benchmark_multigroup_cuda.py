#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark Suite for Multi-Group Varlen Attention

Phase 2: Performance benchmarking and optimization
Measures speedup, bandwidth reduction, and GPU efficiency.

Targets:
- Forward speedup: >1.3x
- K,V bandwidth reduction: >30%
- GPU efficiency: >92%
- End-to-end speedup: 17-22%
"""

import torch
import time
import os
import sys
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict

# Check if CUDA is available
if not torch.cuda.is_available():
    print("ERROR: CUDA not available. This benchmark requires a GPU.")
    sys.exit(1)

# Import flash attention
try:
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward
    from flash_attn.flash_attn_multigroup_interface import _flash_attn_varlen_multigroup_forward
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Compute Capability: SM {torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}")
except ImportError as e:
    print(f"ERROR: Could not import flash attention: {e}")
    sys.exit(1)


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    num_groups: int
    batch_size: int
    seqlen_q: int
    seqlen_k: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype = torch.bfloat16
    causal: bool = True
    num_iterations: int = 100
    warmup_iterations: int = 10

    def __str__(self):
        return (f"groups={self.num_groups}, batch={self.batch_size}, "
                f"seqlen_q={self.seqlen_q}, seqlen_k={self.seqlen_k}, "
                f"heads={self.num_heads}, headdim={self.head_dim}")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: BenchmarkConfig
    multigroup_time_ms: float
    sequential_time_ms: float
    speedup: float
    time_saved_ms: float
    time_saved_percent: float
    bandwidth_reduction_percent: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    max_diff: Optional[float] = None

    def to_dict(self):
        result = asdict(self)
        result['config'] = str(self.config)
        return result


class MultiGroupBenchmark:
    """Comprehensive benchmark suite for multi-group varlen attention."""

    def __init__(self, device='cuda', verbose=True):
        self.device = device
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)

    def _create_test_data(self, config: BenchmarkConfig) -> Tuple:
        """Create test data for benchmarking."""
        # Create Q tensors for each group
        q_list = []
        cu_seqlens_q_list = []
        max_seqlen_q_list = []

        for g in range(config.num_groups):
            # Create Q for this group
            total_q_tokens = config.batch_size * config.seqlen_q
            q = torch.randn(
                total_q_tokens, config.num_heads, config.head_dim,
                dtype=config.dtype, device=self.device
            )
            q_list.append(q)

            # Create cumulative sequence lengths
            cu_seqlens_q = torch.arange(
                0, (config.batch_size + 1) * config.seqlen_q, config.seqlen_q,
                dtype=torch.int32, device=self.device
            )
            cu_seqlens_q_list.append(cu_seqlens_q)
            max_seqlen_q_list.append(config.seqlen_q)

        # Create shared K,V
        total_kv_tokens = config.batch_size * config.seqlen_k
        k = torch.randn(
            total_kv_tokens, config.num_heads, config.head_dim,
            dtype=config.dtype, device=self.device
        )
        v = torch.randn(
            total_kv_tokens, config.num_heads, config.head_dim,
            dtype=config.dtype, device=self.device
        )

        # Create K,V metadata
        cu_seqlens_k_list = []
        max_seqlen_k_list = []
        kv_endpoints_list = []

        for g in range(config.num_groups):
            # For simplicity, use full K,V for all groups
            # In real zigzag, groups would have different endpoints
            cu_seqlens_k = torch.arange(
                0, (config.batch_size + 1) * config.seqlen_k, config.seqlen_k,
                dtype=torch.int32, device=self.device
            )
            cu_seqlens_k_list.append(cu_seqlens_k)
            max_seqlen_k_list.append(config.seqlen_k)

            # KV endpoints: each group sees full K,V
            kv_endpoints = torch.full(
                (config.batch_size,), config.seqlen_k,
                dtype=torch.int32, device=self.device
            )
            kv_endpoints_list.append(kv_endpoints)

        # Stack kv_endpoints to [num_groups, batch_size]
        kv_endpoints = torch.stack(kv_endpoints_list, dim=0)

        softmax_scale = 1.0 / (config.head_dim ** 0.5)

        return (q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list,
                kv_endpoints, max_seqlen_q_list, max_seqlen_k_list, softmax_scale)

    def benchmark_forward_speedup(self, config: BenchmarkConfig) -> BenchmarkResult:
        """
        Benchmark 1: Forward speedup vs sequential single-group calls.

        Measures the speedup of multi-group kernel compared to calling
        single-group kernel sequentially for each group.
        """
        self._log(f"\n{'='*80}")
        self._log(f"Benchmark 1: Forward Speedup")
        self._log(f"Config: {config}")
        self._log(f"{'='*80}")

        # Create test data
        (q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list,
         kv_endpoints, max_seqlen_q_list, max_seqlen_k_list,
         softmax_scale) = self._create_test_data(config)

        # Warmup: Multi-group
        self._log("  Warming up multi-group...")
        for _ in range(config.warmup_iterations):
            try:
                out_list_mg, lse_list_mg = _flash_attn_varlen_multigroup_forward(
                    q_list=q_list,
                    k=k,
                    v=v,
                    cu_seqlens_q_list=cu_seqlens_q_list,
                    cu_seqlens_k_list=cu_seqlens_k_list,
                    kv_endpoints=kv_endpoints,
                    max_seqlen_q_list=max_seqlen_q_list,
                    max_seqlen_k_list=max_seqlen_k_list,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=config.causal,
                )
            except Exception as e:
                self._log(f"  ERROR in multi-group warmup: {e}")
                return None

        # Warmup: Sequential
        self._log("  Warming up sequential...")
        for _ in range(config.warmup_iterations):
            for g in range(config.num_groups):
                max_kv_end = kv_endpoints[g].max().item()
                _ = _flash_attn_varlen_forward(
                    q=q_list[g],
                    k=k[:max_kv_end],
                    v=v[:max_kv_end],
                    cu_seqlens_q=cu_seqlens_q_list[g],
                    cu_seqlens_k=cu_seqlens_k_list[g],
                    max_seqlen_q=max_seqlen_q_list[g],
                    max_seqlen_k=max_seqlen_k_list[g],
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=config.causal,
                )

        torch.cuda.synchronize()

        # Benchmark: Multi-group
        self._log("  Benchmarking multi-group kernel...")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(config.num_iterations):
            out_list_mg, lse_list_mg = _flash_attn_varlen_multigroup_forward(
                q_list=q_list,
                k=k,
                v=v,
                cu_seqlens_q_list=cu_seqlens_q_list,
                cu_seqlens_k_list=cu_seqlens_k_list,
                kv_endpoints=kv_endpoints,
                max_seqlen_q_list=max_seqlen_q_list,
                max_seqlen_k_list=max_seqlen_k_list,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=config.causal,
            )
        end.record()
        torch.cuda.synchronize()
        multigroup_time_ms = start.elapsed_time(end) / config.num_iterations

        # Benchmark: Sequential
        self._log("  Benchmarking sequential single-group calls...")
        out_list_seq = []
        lse_list_seq = []

        start.record()
        for _ in range(config.num_iterations):
            for g in range(config.num_groups):
                max_kv_end = kv_endpoints[g].max().item()
                out_g, lse_g, _, _ = _flash_attn_varlen_forward(
                    q=q_list[g],
                    k=k[:max_kv_end],
                    v=v[:max_kv_end],
                    cu_seqlens_q=cu_seqlens_q_list[g],
                    cu_seqlens_k=cu_seqlens_k_list[g],
                    max_seqlen_q=max_seqlen_q_list[g],
                    max_seqlen_k=max_seqlen_k_list[g],
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=config.causal,
                )
                if _ == config.num_iterations - 1:  # Save last iteration for correctness check
                    out_list_seq.append(out_g)
                    lse_list_seq.append(lse_g)
        end.record()
        torch.cuda.synchronize()
        sequential_time_ms = start.elapsed_time(end) / config.num_iterations

        # Calculate metrics
        speedup = sequential_time_ms / multigroup_time_ms
        time_saved_ms = sequential_time_ms - multigroup_time_ms
        time_saved_percent = (time_saved_ms / sequential_time_ms) * 100

        # Check correctness
        max_diff = 0.0
        for g in range(config.num_groups):
            diff = torch.abs(out_list_mg[g] - out_list_seq[g]).max().item()
            max_diff = max(max_diff, diff)

        result = BenchmarkResult(
            config=config,
            multigroup_time_ms=multigroup_time_ms,
            sequential_time_ms=sequential_time_ms,
            speedup=speedup,
            time_saved_ms=time_saved_ms,
            time_saved_percent=time_saved_percent,
            max_diff=max_diff,
        )

        self._log(f"\n  Results:")
        self._log(f"    Sequential time:  {sequential_time_ms:.3f} ms")
        self._log(f"    Multi-group time: {multigroup_time_ms:.3f} ms")
        self._log(f"    Speedup:          {speedup:.3f}x {'✓' if speedup >= 1.3 else '✗'} (target: ≥1.3x)")
        self._log(f"    Time saved:       {time_saved_ms:.3f} ms ({time_saved_percent:.1f}%)")
        self._log(f"    Max diff:         {max_diff:.6e}")

        self.results.append(result)
        return result

    def benchmark_memory_bandwidth(self, config: BenchmarkConfig) -> Dict:
        """
        Benchmark 2: Memory bandwidth analysis.

        Uses torch.profiler to measure memory bandwidth and estimate
        K,V bandwidth reduction.
        """
        self._log(f"\n{'='*80}")
        self._log(f"Benchmark 2: Memory Bandwidth")
        self._log(f"Config: {config}")
        self._log(f"{'='*80}")

        # Create test data
        (q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list,
         kv_endpoints, max_seqlen_q_list, max_seqlen_k_list,
         softmax_scale) = self._create_test_data(config)

        # Calculate theoretical K,V bytes
        kv_tokens = config.batch_size * config.seqlen_k
        kv_bytes_per_read = kv_tokens * config.num_heads * config.head_dim * 2  # 2 bytes for bfloat16

        self._log(f"  Theoretical K,V bytes per read: {kv_bytes_per_read / 1e6:.2f} MB")

        # Sequential: loads K,V for each group
        sequential_kv_bytes = config.num_groups * kv_bytes_per_read
        self._log(f"  Sequential total K,V bytes:     {sequential_kv_bytes / 1e6:.2f} MB ({config.num_groups}x)")

        # Multi-group: loads K,V once (ideally)
        multigroup_kv_bytes = kv_bytes_per_read
        self._log(f"  Multi-group ideal K,V bytes:    {multigroup_kv_bytes / 1e6:.2f} MB (1x)")

        # Theoretical reduction
        theoretical_reduction = (1 - multigroup_kv_bytes / sequential_kv_bytes) * 100
        self._log(f"  Theoretical bandwidth reduction: {theoretical_reduction:.1f}%")

        # Profile multi-group
        self._log("\n  Profiling multi-group kernel...")
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                with_flops=True,
            ) as prof:
                for _ in range(10):  # Profile 10 iterations
                    out_list_mg, lse_list_mg = _flash_attn_varlen_multigroup_forward(
                        q_list=q_list,
                        k=k,
                        v=v,
                        cu_seqlens_q_list=cu_seqlens_q_list,
                        cu_seqlens_k_list=cu_seqlens_k_list,
                        kv_endpoints=kv_endpoints,
                        max_seqlen_q_list=max_seqlen_q_list,
                        max_seqlen_k_list=max_seqlen_k_list,
                        dropout_p=0.0,
                        softmax_scale=softmax_scale,
                        causal=config.causal,
                    )

            # Extract memory metrics
            self._log("\n  Profiler summary:")
            self._log(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

        except Exception as e:
            self._log(f"  WARNING: Profiling failed: {e}")

        result = {
            'theoretical_kv_bytes_sequential': sequential_kv_bytes,
            'theoretical_kv_bytes_multigroup': multigroup_kv_bytes,
            'theoretical_reduction_percent': theoretical_reduction,
        }

        return result

    def benchmark_scaling(self, base_config: BenchmarkConfig) -> List[BenchmarkResult]:
        """
        Benchmark 4: Scaling with number of groups.

        Measures how performance scales as we increase the number of groups.
        """
        self._log(f"\n{'='*80}")
        self._log(f"Benchmark 4: Scaling with Number of Groups")
        self._log(f"{'='*80}")

        results = []

        for num_groups in [2, 3, 4]:
            config = BenchmarkConfig(
                num_groups=num_groups,
                batch_size=base_config.batch_size,
                seqlen_q=base_config.seqlen_q,
                seqlen_k=base_config.seqlen_k,
                num_heads=base_config.num_heads,
                head_dim=base_config.head_dim,
                dtype=base_config.dtype,
                causal=base_config.causal,
                num_iterations=base_config.num_iterations,
            )

            result = self.benchmark_forward_speedup(config)
            if result:
                results.append(result)

        # Print summary
        self._log(f"\n{'='*80}")
        self._log(f"Scaling Summary:")
        self._log(f"{'='*80}")
        self._log(f"{'Num Groups':<12} {'Sequential':<15} {'Multi-Group':<15} {'Speedup':<10}")
        self._log(f"{'-'*80}")
        for result in results:
            self._log(f"{result.config.num_groups:<12} "
                     f"{result.sequential_time_ms:>12.3f} ms "
                     f"{result.multigroup_time_ms:>12.3f} ms "
                     f"{result.speedup:>8.3f}x")

        return results

    def run_comprehensive_suite(self):
        """Run complete benchmark suite with multiple configurations."""
        self._log("="*80)
        self._log("COMPREHENSIVE MULTI-GROUP VARLEN ATTENTION BENCHMARK SUITE")
        self._log("Phase 2: Performance Benchmarking and Optimization")
        self._log("="*80)

        # Standard configurations to test
        configs = [
            # (num_groups, batch, seqlen_q, seqlen_k, heads, headdim)
            BenchmarkConfig(2, 4, 512, 512, 32, 128),      # Small
            BenchmarkConfig(2, 4, 1024, 1024, 32, 128),    # Medium
            BenchmarkConfig(2, 4, 2048, 2048, 32, 128),    # Large
            BenchmarkConfig(2, 4, 512, 1024, 32, 128),     # Asymmetric Q,K
            BenchmarkConfig(3, 4, 512, 512, 32, 128),      # 3 groups
            BenchmarkConfig(4, 4, 512, 512, 32, 128),      # 4 groups
            BenchmarkConfig(2, 4, 1024, 1024, 32, 64),     # Different headdim
            BenchmarkConfig(2, 4, 1024, 1024, 32, 256),    # Larger headdim
        ]

        self._log(f"\nTotal configurations to benchmark: {len(configs)}")
        self._log(f"Iterations per config: {configs[0].num_iterations}")

        # Run forward speedup benchmarks
        self._log("\n" + "="*80)
        self._log("BENCHMARK 1: Forward Speedup")
        self._log("="*80)

        for i, config in enumerate(configs, 1):
            self._log(f"\nConfig {i}/{len(configs)}")
            result = self.benchmark_forward_speedup(config)

        # Run memory bandwidth benchmark on a subset
        self._log("\n" + "="*80)
        self._log("BENCHMARK 2: Memory Bandwidth")
        self._log("="*80)

        bandwidth_config = BenchmarkConfig(2, 4, 1024, 1024, 32, 128, num_iterations=10)
        bandwidth_result = self.benchmark_memory_bandwidth(bandwidth_config)

        # Run scaling benchmark
        self._log("\n" + "="*80)
        self._log("BENCHMARK 4: Scaling Analysis")
        self._log("="*80)

        scaling_base = BenchmarkConfig(2, 4, 512, 512, 32, 128)
        scaling_results = self.benchmark_scaling(scaling_base)

        # Generate summary report
        self.generate_summary_report()

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        self._log("\n" + "="*80)
        self._log("PERFORMANCE SUMMARY REPORT")
        self._log("="*80)

        if not self.results:
            self._log("No results to report.")
            return

        # Overall statistics
        speedups = [r.speedup for r in self.results]
        avg_speedup = np.mean(speedups)
        min_speedup = np.min(speedups)
        max_speedup = np.max(speedups)
        std_speedup = np.std(speedups)

        self._log(f"\nOverall Performance Metrics:")
        self._log(f"  Average speedup:  {avg_speedup:.3f}x")
        self._log(f"  Min speedup:      {min_speedup:.3f}x")
        self._log(f"  Max speedup:      {max_speedup:.3f}x")
        self._log(f"  Std deviation:    {std_speedup:.3f}x")

        # Check against targets
        target_speedup = 1.3
        configs_meeting_target = sum(1 for r in self.results if r.speedup >= target_speedup)
        success_rate = (configs_meeting_target / len(self.results)) * 100

        self._log(f"\n  Target speedup:   {target_speedup}x")
        self._log(f"  Configs meeting target: {configs_meeting_target}/{len(self.results)} ({success_rate:.1f}%)")

        if avg_speedup >= target_speedup:
            self._log(f"\n  ✓ PASS: Average speedup meets target!")
        else:
            self._log(f"\n  ✗ FAIL: Average speedup below target ({avg_speedup:.3f}x < {target_speedup}x)")

        # Detailed results table
        self._log(f"\n{'='*120}")
        self._log(f"Detailed Results:")
        self._log(f"{'='*120}")
        self._log(f"{'Groups':<7} {'Batch':<6} {'SeqQ':<7} {'SeqK':<7} {'Heads':<6} {'Dim':<5} "
                 f"{'Sequential':<12} {'MultiGroup':<12} {'Speedup':<10} {'MaxDiff':<12}")
        self._log(f"{'-'*120}")

        for result in self.results:
            c = result.config
            self._log(f"{c.num_groups:<7} {c.batch_size:<6} {c.seqlen_q:<7} {c.seqlen_k:<7} "
                     f"{c.num_heads:<6} {c.head_dim:<5} "
                     f"{result.sequential_time_ms:>9.3f} ms "
                     f"{result.multigroup_time_ms:>9.3f} ms "
                     f"{result.speedup:>8.3f}x "
                     f"{result.max_diff:>10.2e}")

        self._log(f"{'='*120}")

        # Group by number of groups
        by_num_groups = defaultdict(list)
        for result in self.results:
            by_num_groups[result.config.num_groups].append(result.speedup)

        if len(by_num_groups) > 1:
            self._log(f"\nSpeedup by Number of Groups:")
            for num_groups in sorted(by_num_groups.keys()):
                speedups = by_num_groups[num_groups]
                avg = np.mean(speedups)
                self._log(f"  {num_groups} groups: {avg:.3f}x average ({len(speedups)} configs)")

        # Save results to JSON
        self.save_results_json()

    def save_results_json(self, filename='benchmark_results_multigroup.json'):
        """Save results to JSON file."""
        output_path = os.path.join('/Users/petrpan26/work/flash-attention/benchmarks', filename)

        data = {
            'device': torch.cuda.get_device_name(0),
            'compute_capability': f"SM {torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
            'num_configs': len(self.results),
            'results': [r.to_dict() for r in self.results],
        }

        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            self._log(f"\n  Results saved to: {output_path}")
        except Exception as e:
            self._log(f"\n  WARNING: Could not save results: {e}")


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("Multi-Group Varlen Attention - Comprehensive Benchmark Suite")
    print("Phase 2: Performance Benchmarking and Optimization")
    print("="*80)

    # Check environment
    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available")
        sys.exit(1)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Compute Capability: SM {torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}")

    # Create benchmark suite
    benchmark = MultiGroupBenchmark(verbose=True)

    # Run comprehensive suite
    benchmark.run_comprehensive_suite()

    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

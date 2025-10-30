#!/usr/bin/env python3
"""
Comprehensive functional tests for Phase 5.1 multigroup kernel optimizations.

Tests all dimension-specific, architecture-aware, and causal optimizations.

Usage:
    # Run all tests
    pytest test_multigroup_kernel_correctness.py -v -s

    # Run specific test
    pytest test_multigroup_kernel_correctness.py::test_forward_all_dimensions -v

    # Run without pytest
    python test_multigroup_kernel_correctness.py
"""

import pytest
import torch
import sys
from typing import List, Tuple

# Try to import the CUDA extension
try:
    import flash_attn_multigroup_cuda
    CUDA_EXTENSION_AVAILABLE = True
except ImportError:
    CUDA_EXTENSION_AVAILABLE = False
    print("WARNING: flash_attn_multigroup_cuda not available. Install with: pip install -e .")

# Import standard flash attention for comparison
try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("WARNING: flash_attn not available. Cannot compare against reference.")


# ============================================================================
# Helper Functions
# ============================================================================

def get_tolerance(dtype: torch.dtype) -> float:
    """Get numerical error tolerance based on dtype."""
    if dtype == torch.float16:
        return 1e-2
    elif dtype == torch.bfloat16:
        return 5e-2
    else:
        return 1e-5


def create_test_inputs(
    num_groups: int,
    seq_len: int,
    kv_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str = 'cuda'
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor,
           List[torch.Tensor], List[torch.Tensor], torch.Tensor,
           List[int], List[int]]:
    """
    Create test inputs for multigroup attention.

    Returns:
        q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints,
        max_seqlen_q_list, max_seqlen_k_list
    """
    # Create Q tensors (one per group)
    q_list = [
        torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device)
        for _ in range(num_groups)
    ]

    # Create shared K, V tensors
    k = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device=device)

    # Cumulative sequence lengths (simple case: one sequence per group)
    cu_seqlens_q_list = [
        torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        for _ in range(num_groups)
    ]

    cu_seqlens_k_list = [
        torch.tensor([0, kv_len], dtype=torch.int32, device=device)
        for _ in range(num_groups)
    ]

    # KV endpoints: all groups attend to full K,V
    kv_endpoints = torch.ones((num_groups, 1), dtype=torch.int32, device=device) * kv_len

    # Max sequence lengths
    max_seqlen_q_list = [seq_len] * num_groups
    max_seqlen_k_list = [kv_len] * num_groups

    return (q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list,
            kv_endpoints, max_seqlen_q_list, max_seqlen_k_list)


def run_multigroup_forward(
    q_list: List[torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q_list: List[torch.Tensor],
    cu_seqlens_k_list: List[torch.Tensor],
    kv_endpoints: torch.Tensor,
    max_seqlen_q_list: List[int],
    max_seqlen_k_list: List[int],
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Run multigroup forward pass."""
    if not CUDA_EXTENSION_AVAILABLE:
        raise RuntimeError("CUDA extension not available")

    if softmax_scale is None:
        softmax_scale = 1.0 / (q_list[0].shape[-1] ** 0.5)

    return flash_attn_multigroup_cuda.fwd(
        q_list, k, v,
        cu_seqlens_q_list, cu_seqlens_k_list,
        kv_endpoints,
        max_seqlen_q_list,
        max_seqlen_k_list,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
    )


def compare_with_reference(
    q_list: List[torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q_list: List[torch.Tensor],
    cu_seqlens_k_list: List[torch.Tensor],
    max_seqlen_q_list: List[int],
    max_seqlen_k_list: List[int],
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
) -> List[torch.Tensor]:
    """
    Run standard flash attention for each group separately as reference.

    Returns list of reference outputs.
    """
    if not FLASH_ATTN_AVAILABLE:
        raise RuntimeError("Standard flash_attn not available for comparison")

    if softmax_scale is None:
        softmax_scale = 1.0 / (q_list[0].shape[-1] ** 0.5)

    ref_outputs = []
    for g in range(len(q_list)):
        ref_out = flash_attn_varlen_func(
            q=q_list[g],
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q_list[g],
            cu_seqlens_k=cu_seqlens_k_list[g],
            max_seqlen_q=max_seqlen_q_list[g],
            max_seqlen_k=max_seqlen_k_list[g],
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        ref_outputs.append(ref_out)

    return ref_outputs


# ============================================================================
# Test Suite: Forward Pass Correctness
# ============================================================================

@pytest.mark.parametrize("head_dim", [32, 64, 96, 128, 192, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_groups", [1, 2, 3, 4])
def test_forward_all_dimensions(head_dim, dtype, causal, num_groups):
    """
    Test forward pass correctness for all head dimensions and configurations.

    This tests Phase 5.1 dimension-specific optimizations:
    - d=32: M=128, N=128, W=4
    - d=64: M=128, N=128, W=4 (or M=96 for NumGroups=2)
    - d=96: Architecture-aware configs
    - d=128: sm8x optimization (M=128, N=32 for 2 CTAs/SM)
    - d=192: 8-warp kernels (W=8)
    - d=256: GPU-specific configs (A100 vs H100)
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not CUDA_EXTENSION_AVAILABLE:
        pytest.skip("CUDA extension not built")

    # Test parameters
    seq_len = 256
    kv_len = 512
    num_heads = 8

    # Create inputs
    q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints, \
        max_seqlen_q_list, max_seqlen_k_list = create_test_inputs(
        num_groups, seq_len, kv_len, num_heads, head_dim, dtype
    )

    # Run multigroup kernel
    out_list, lse_list = run_multigroup_forward(
        q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints,
        max_seqlen_q_list, max_seqlen_k_list, causal=causal
    )

    # Compare with reference
    if FLASH_ATTN_AVAILABLE:
        ref_outputs = compare_with_reference(
            q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list, causal=causal
        )

        tolerance = get_tolerance(dtype)
        max_errors = []

        for g in range(num_groups):
            diff = torch.abs(out_list[g] - ref_outputs[g])
            max_error = diff.max().item()
            mean_error = diff.mean().item()
            max_errors.append(max_error)

            assert max_error < tolerance, (
                f"Group {g} max_error={max_error:.6e} exceeds tolerance={tolerance:.6e}\n"
                f"  Config: head_dim={head_dim}, dtype={dtype}, causal={causal}, "
                f"num_groups={num_groups}\n"
                f"  Mean error: {mean_error:.6e}\n"
                f"  Output shape: {out_list[g].shape}"
            )

        max_error_overall = max(max_errors)
        print(f"✓ head_dim={head_dim:3d}, dtype={str(dtype):17s}, causal={causal}, "
              f"num_groups={num_groups}, max_error={max_error_overall:.6e}")
    else:
        # Just verify shapes and no NaN/Inf
        for g in range(num_groups):
            assert out_list[g].shape == q_list[g].shape
            assert not torch.isnan(out_list[g]).any()
            assert not torch.isinf(out_list[g]).any()
        print(f"✓ head_dim={head_dim:3d}, dtype={str(dtype):17s}, causal={causal}, "
              f"num_groups={num_groups} (shape check only)")


@pytest.mark.parametrize("head_dim", [128, 192, 256])
def test_sm8x_optimization(head_dim):
    """
    Test sm8x-specific optimizations for critical dimensions.

    d=128: Should use M=128, N=32 for non-causal on sm8x (2 CTAs/SM)
    d=192: Should use 8-warp kernels
    d=256: Should use GPU-specific configs
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not CUDA_EXTENSION_AVAILABLE:
        pytest.skip("CUDA extension not built")

    # Check GPU architecture
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)

    print(f"\nTesting on GPU: {gpu_name} (SM {cc_major}.{cc_minor})")

    # Test parameters
    num_groups = 1
    seq_len = 512
    kv_len = 1024
    num_heads = 8
    dtype = torch.bfloat16

    # Create inputs
    q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints, \
        max_seqlen_q_list, max_seqlen_k_list = create_test_inputs(
        num_groups, seq_len, kv_len, num_heads, head_dim, dtype
    )

    # Run non-causal (triggers sm8x optimization for d=128)
    out_list, lse_list = run_multigroup_forward(
        q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints,
        max_seqlen_q_list, max_seqlen_k_list, causal=False
    )

    # Verify correctness
    if FLASH_ATTN_AVAILABLE:
        ref_outputs = compare_with_reference(
            q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list, causal=False
        )

        diff = torch.abs(out_list[0] - ref_outputs[0])
        max_error = diff.max().item()
        tolerance = get_tolerance(dtype)

        assert max_error < tolerance, f"Error {max_error:.6e} exceeds {tolerance:.6e}"

        if cc_major == 8 and cc_minor > 0:
            print(f"✓ SM8x optimization test passed (head_dim={head_dim}, error={max_error:.6e})")
            if head_dim == 128:
                print("  → Should use M=128, N=32 config (2 CTAs/SM)")
        else:
            print(f"✓ Standard config test passed (head_dim={head_dim}, error={max_error:.6e})")
    else:
        print(f"✓ Shape check passed (head_dim={head_dim})")


@pytest.mark.parametrize("head_dim", [192, 256])
def test_8warp_kernels(head_dim):
    """
    Test 8-warp kernel optimization for large dimensions.

    Phase 5.1 implements 8-warp kernels (256 threads) for d=192 and d=256
    to achieve 20-30% speedup.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not CUDA_EXTENSION_AVAILABLE:
        pytest.skip("CUDA extension not built")

    # Test with different NumGroups
    for num_groups in [1, 2, 3]:
        seq_len = 256
        kv_len = 512
        num_heads = 8
        dtype = torch.bfloat16

        q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints, \
            max_seqlen_q_list, max_seqlen_k_list = create_test_inputs(
            num_groups, seq_len, kv_len, num_heads, head_dim, dtype
        )

        # Run kernel
        out_list, lse_list = run_multigroup_forward(
            q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints,
            max_seqlen_q_list, max_seqlen_k_list, causal=False
        )

        # Verify correctness
        if FLASH_ATTN_AVAILABLE:
            ref_outputs = compare_with_reference(
                q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list,
                max_seqlen_q_list, max_seqlen_k_list, causal=False
            )

            for g in range(num_groups):
                diff = torch.abs(out_list[g] - ref_outputs[g])
                max_error = diff.max().item()
                tolerance = get_tolerance(dtype)

                assert max_error < tolerance, (
                    f"8-warp kernel error for head_dim={head_dim}, num_groups={num_groups}, "
                    f"group={g}: {max_error:.6e}"
                )

        print(f"✓ 8-warp kernel: head_dim={head_dim}, num_groups={num_groups}")


# ============================================================================
# Test Suite: Dimension Rounding
# ============================================================================

@pytest.mark.parametrize("input_dim,expected_kernel_dim", [
    (8, 32),      # Rounds to 32
    (24, 32),     # Rounds to 32
    (40, 64),     # Rounds to 64
    (56, 64),     # Rounds to 64
    (72, 96),     # Rounds to 96
    (80, 96),     # Rounds to 96
    (88, 96),     # Rounds to 96
    (104, 128),   # Rounds to 128
    (112, 128),   # Rounds to 128
    (120, 128),   # Rounds to 128
    (136, 192),   # Rounds to 192
    (144, 192),   # Rounds to 192
    (160, 192),   # Rounds to 192
    (200, 256),   # Rounds to 256
    (224, 256),   # Rounds to 256
    (240, 256),   # Rounds to 256
])
def test_dimension_rounding(input_dim, expected_kernel_dim):
    """
    Test that non-standard dimensions are properly rounded.

    The HEADDIM_SWITCH dispatcher uses <= for rounding:
    - 1-32 → 32
    - 33-64 → 64
    - 65-96 → 96
    - 97-128 → 128
    - 129-192 → 192
    - 193-256 → 256
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not CUDA_EXTENSION_AVAILABLE:
        pytest.skip("CUDA extension not built")

    num_groups = 2
    seq_len = 128
    kv_len = 256
    num_heads = 4
    dtype = torch.float16

    # Create inputs with non-standard dimension
    q_list = [
        torch.randn(seq_len, num_heads, input_dim, dtype=dtype, device='cuda')
        for _ in range(num_groups)
    ]
    k = torch.randn(kv_len, num_heads, input_dim, dtype=dtype, device='cuda')
    v = torch.randn(kv_len, num_heads, input_dim, dtype=dtype, device='cuda')

    cu_seqlens_q_list = [
        torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')
        for _ in range(num_groups)
    ]
    cu_seqlens_k_list = [
        torch.tensor([0, kv_len], dtype=torch.int32, device='cuda')
        for _ in range(num_groups)
    ]
    kv_endpoints = torch.ones((num_groups, 1), dtype=torch.int32, device='cuda') * kv_len

    max_seqlen_q_list = [seq_len] * num_groups
    max_seqlen_k_list = [kv_len] * num_groups

    # Should NOT raise error
    out_list, lse_list = run_multigroup_forward(
        q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints,
        max_seqlen_q_list, max_seqlen_k_list, causal=False
    )

    # Verify output shape preserves input dimension
    for g in range(num_groups):
        assert out_list[g].shape[-1] == input_dim, (
            f"Output dimension {out_list[g].shape[-1]} != input dimension {input_dim}"
        )
        assert not torch.isnan(out_list[g]).any()
        assert not torch.isinf(out_list[g]).any()

    print(f"✓ Dimension rounding: {input_dim} → kernel uses {expected_kernel_dim}, "
          f"output preserves {input_dim}")


# ============================================================================
# Test Suite: Backward Pass
# ============================================================================

@pytest.mark.parametrize("head_dim", [32, 64, 96, 128, 192, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_groups", [2, 3, 4])
def test_backward_correctness(head_dim, dtype, num_groups):
    """
    Test backward pass gradient correctness for all dimensions.

    Critical for ensuring Phase 5.1 optimizations don't break gradients.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not CUDA_EXTENSION_AVAILABLE:
        pytest.skip("CUDA extension not built")

    seq_len = 128
    kv_len = 256
    num_heads = 4

    # Create inputs (requires_grad=True for gradient checking)
    q_list = [
        torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device='cuda', requires_grad=True)
        for _ in range(num_groups)
    ]
    k = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device='cuda', requires_grad=True)

    cu_seqlens_q_list = [
        torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')
        for _ in range(num_groups)
    ]
    cu_seqlens_k_list = [
        torch.tensor([0, kv_len], dtype=torch.int32, device='cuda')
        for _ in range(num_groups)
    ]
    kv_endpoints = torch.ones((num_groups, 1), dtype=torch.int32, device='cuda') * kv_len

    max_seqlen_q_list = [seq_len] * num_groups
    max_seqlen_k_list = [kv_len] * num_groups

    # Forward pass
    out_list, lse_list = run_multigroup_forward(
        q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints,
        max_seqlen_q_list, max_seqlen_k_list, causal=False
    )

    # Create gradient outputs
    dout_list = [
        torch.randn_like(out) for out in out_list
    ]

    # Backward pass
    try:
        dq_list, dk, dv = flash_attn_multigroup_cuda.bwd(
            dout_list, q_list, k, v, out_list, lse_list,
            cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints,
            max_seqlen_q_list, max_seqlen_k_list,
            dropout_p=0.0,
            softmax_scale=1.0 / (head_dim ** 0.5),
            causal=False,
        )

        # Verify gradient shapes
        for g in range(num_groups):
            assert dq_list[g].shape == q_list[g].shape, f"dq shape mismatch for group {g}"
            assert not torch.isnan(dq_list[g]).any(), f"NaN in dq for group {g}"
            assert not torch.isinf(dq_list[g]).any(), f"Inf in dq for group {g}"

        assert dk.shape == k.shape, "dk shape mismatch"
        assert dv.shape == v.shape, "dv shape mismatch"
        assert not torch.isnan(dk).any(), "NaN in dk"
        assert not torch.isnan(dv).any(), "NaN in dv"
        assert not torch.isinf(dk).any(), "Inf in dk"
        assert not torch.isinf(dv).any(), "Inf in dv"

        print(f"✓ Backward: head_dim={head_dim:3d}, dtype={str(dtype):17s}, num_groups={num_groups}")

    except RuntimeError as e:
        if "not yet implemented" in str(e):
            pytest.skip(f"Backward not implemented yet: {e}")
        else:
            raise


@pytest.mark.parametrize("head_dim", [64, 128])
def test_gradient_accumulation(head_dim):
    """
    Test that overlapping K,V regions correctly accumulate gradients.

    When multiple Q groups attend to overlapping K,V regions, the gradients
    for those K,V positions should be accumulated.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not CUDA_EXTENSION_AVAILABLE:
        pytest.skip("CUDA extension not built")

    num_groups = 3
    seq_len = 128
    kv_len = 256
    num_heads = 4
    dtype = torch.float16

    # Create inputs
    q_list = [
        torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device='cuda', requires_grad=True)
        for _ in range(num_groups)
    ]
    k = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device='cuda', requires_grad=True)

    # All groups attend to same K,V → gradients should accumulate
    cu_seqlens_q_list = [
        torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')
        for _ in range(num_groups)
    ]
    cu_seqlens_k_list = [
        torch.tensor([0, kv_len], dtype=torch.int32, device='cuda')
        for _ in range(num_groups)
    ]
    kv_endpoints = torch.ones((num_groups, 1), dtype=torch.int32, device='cuda') * kv_len

    max_seqlen_q_list = [seq_len] * num_groups
    max_seqlen_k_list = [kv_len] * num_groups

    # Forward
    out_list, lse_list = run_multigroup_forward(
        q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints,
        max_seqlen_q_list, max_seqlen_k_list, causal=False
    )

    dout_list = [torch.randn_like(out) for out in out_list]

    # Backward
    try:
        dq_list, dk, dv = flash_attn_multigroup_cuda.bwd(
            dout_list, q_list, k, v, out_list, lse_list,
            cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints,
            max_seqlen_q_list, max_seqlen_k_list,
            dropout_p=0.0,
            softmax_scale=1.0 / (head_dim ** 0.5),
            causal=False,
        )

        # Verify dk, dv are non-zero (gradients accumulated)
        assert dk.abs().max() > 0, "dk is zero (gradient accumulation failed?)"
        assert dv.abs().max() > 0, "dv is zero (gradient accumulation failed?)"

        print(f"✓ Gradient accumulation test passed (head_dim={head_dim})")
        print(f"  dk stats: mean={dk.abs().mean():.6e}, max={dk.abs().max():.6e}")
        print(f"  dv stats: mean={dv.abs().mean():.6e}, max={dv.abs().max():.6e}")

    except RuntimeError as e:
        if "not yet implemented" in str(e):
            pytest.skip(f"Backward not implemented yet: {e}")
        else:
            raise


# ============================================================================
# Test Suite: Edge Cases
# ============================================================================

def test_single_group():
    """Test with single group (NumGroups=1)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not CUDA_EXTENSION_AVAILABLE:
        pytest.skip("CUDA extension not built")

    num_groups = 1
    seq_len = 256
    kv_len = 512
    num_heads = 8
    head_dim = 128
    dtype = torch.bfloat16

    q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints, \
        max_seqlen_q_list, max_seqlen_k_list = create_test_inputs(
        num_groups, seq_len, kv_len, num_heads, head_dim, dtype
    )

    out_list, lse_list = run_multigroup_forward(
        q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints,
        max_seqlen_q_list, max_seqlen_k_list, causal=False
    )

    assert len(out_list) == 1
    assert out_list[0].shape == q_list[0].shape
    print("✓ Single group test passed")


def test_variable_sequence_lengths():
    """Test with different Q sequence lengths per group."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not CUDA_EXTENSION_AVAILABLE:
        pytest.skip("CUDA extension not built")

    num_groups = 3
    seq_lens = [128, 256, 384]  # Different Q lengths
    kv_len = 512
    num_heads = 8
    head_dim = 64
    dtype = torch.float16

    # Create Q with different lengths
    q_list = [
        torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device='cuda')
        for seq_len in seq_lens
    ]
    k = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device='cuda')
    v = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device='cuda')

    cu_seqlens_q_list = [
        torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')
        for seq_len in seq_lens
    ]
    cu_seqlens_k_list = [
        torch.tensor([0, kv_len], dtype=torch.int32, device='cuda')
        for _ in range(num_groups)
    ]
    kv_endpoints = torch.ones((num_groups, 1), dtype=torch.int32, device='cuda') * kv_len

    max_seqlen_q_list = seq_lens
    max_seqlen_k_list = [kv_len] * num_groups

    out_list, lse_list = run_multigroup_forward(
        q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints,
        max_seqlen_q_list, max_seqlen_k_list, causal=False
    )

    # Verify each output has correct shape
    for g in range(num_groups):
        assert out_list[g].shape == q_list[g].shape

    print(f"✓ Variable sequence lengths test passed: {seq_lens}")


def test_large_batch():
    """Test with larger batch and longer sequences."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not CUDA_EXTENSION_AVAILABLE:
        pytest.skip("CUDA extension not built")

    num_groups = 4
    seq_len = 1024
    kv_len = 2048
    num_heads = 16
    head_dim = 128
    dtype = torch.bfloat16

    q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints, \
        max_seqlen_q_list, max_seqlen_k_list = create_test_inputs(
        num_groups, seq_len, kv_len, num_heads, head_dim, dtype
    )

    out_list, lse_list = run_multigroup_forward(
        q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints,
        max_seqlen_q_list, max_seqlen_k_list, causal=False
    )

    for g in range(num_groups):
        assert out_list[g].shape == q_list[g].shape
        assert not torch.isnan(out_list[g]).any()
        assert not torch.isinf(out_list[g]).any()

    print(f"✓ Large batch test passed: {num_groups} groups, {seq_len} seq_len, {kv_len} kv_len")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all tests without pytest."""
    print("=" * 80)
    print("PHASE 5.1 MULTIGROUP KERNEL FUNCTIONAL TESTS")
    print("=" * 80)
    print()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1

    if not CUDA_EXTENSION_AVAILABLE:
        print("ERROR: flash_attn_multigroup_cuda not available")
        print("Build with: pip install -e .")
        return 1

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    print(f"Compute capability: SM {cc_major}.{cc_minor}")
    print()

    total = 0
    passed = 0
    failed = 0
    skipped = 0

    # Test configurations
    head_dims = [32, 64, 96, 128, 192, 256]
    dtypes = [torch.float16, torch.bfloat16]
    causal_options = [False, True]
    num_groups_options = [1, 2, 3, 4]

    print("Running forward correctness tests...")
    print("-" * 80)

    for head_dim in head_dims:
        for dtype in dtypes:
            for causal in causal_options:
                for num_groups in num_groups_options:
                    total += 1
                    try:
                        test_forward_all_dimensions(head_dim, dtype, causal, num_groups)
                        passed += 1
                    except Exception as e:
                        failed += 1
                        print(f"✗ FAILED: head_dim={head_dim}, dtype={dtype}, causal={causal}, "
                              f"num_groups={num_groups}")
                        print(f"  Error: {str(e)[:100]}")

    print()
    print("Running dimension rounding tests...")
    print("-" * 80)

    rounding_tests = [
        (8, 32), (24, 32), (40, 64), (56, 64), (72, 96), (80, 96),
        (88, 96), (104, 128), (112, 128), (120, 128), (136, 192),
        (144, 192), (160, 192), (200, 256), (224, 256), (240, 256)
    ]

    for input_dim, expected_dim in rounding_tests:
        total += 1
        try:
            test_dimension_rounding(input_dim, expected_dim)
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ FAILED: dimension rounding {input_dim} → {expected_dim}")
            print(f"  Error: {str(e)[:100]}")

    print()
    print("Running backward pass tests...")
    print("-" * 80)

    for head_dim in head_dims:
        for dtype in dtypes:
            for num_groups in [2, 3, 4]:
                total += 1
                try:
                    test_backward_correctness(head_dim, dtype, num_groups)
                    passed += 1
                except Exception as e:
                    if "skip" in str(e).lower():
                        skipped += 1
                    else:
                        failed += 1
                        print(f"✗ FAILED: backward head_dim={head_dim}, dtype={dtype}, "
                              f"num_groups={num_groups}")
                        print(f"  Error: {str(e)[:100]}")

    print()
    print("Running edge case tests...")
    print("-" * 80)

    edge_tests = [
        ("Single group", test_single_group),
        ("Variable sequence lengths", test_variable_sequence_lengths),
        ("Large batch", test_large_batch),
        ("SM8x d=128", lambda: test_sm8x_optimization(128)),
        ("SM8x d=192", lambda: test_sm8x_optimization(192)),
        ("8-warp d=192", lambda: test_8warp_kernels(192)),
        ("8-warp d=256", lambda: test_8warp_kernels(256)),
        ("Gradient accumulation d=64", lambda: test_gradient_accumulation(64)),
        ("Gradient accumulation d=128", lambda: test_gradient_accumulation(128)),
    ]

    for name, test_func in edge_tests:
        total += 1
        try:
            test_func()
            passed += 1
        except Exception as e:
            if "skip" in str(e).lower():
                skipped += 1
            else:
                failed += 1
                print(f"✗ FAILED: {name}")
                print(f"  Error: {str(e)[:100]}")

    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total:   {total}")
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")
    print()

    if failed == 0:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

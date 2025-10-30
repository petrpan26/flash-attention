#!/usr/bin/env python3
"""
Quick smoke test for multigroup kernel.

Run this first to verify basic functionality before running comprehensive tests.

Usage:
    python test_multigroup_quick.py
"""

import torch
import sys

def test_basic_forward():
    """Quick test of basic forward pass."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Forward Pass")
    print("=" * 60)

    try:
        import flash_attn_multigroup_cuda
    except ImportError:
        print("✗ FAILED: flash_attn_multigroup_cuda not available")
        print("  Build with: pip install -e .")
        return False

    # Simple test case
    num_groups = 2
    seq_len = 128
    kv_len = 256
    num_heads = 4
    head_dim = 64
    dtype = torch.float16

    print(f"Config: {num_groups} groups, {seq_len} Q tokens, {kv_len} KV tokens")
    print(f"        {num_heads} heads, dim={head_dim}, dtype={dtype}")

    # Create inputs
    q_list = [
        torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device='cuda')
        for _ in range(num_groups)
    ]
    k = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device='cuda')
    v = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device='cuda')

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

    try:
        # Run forward
        out_list, lse_list = flash_attn_multigroup_cuda.fwd(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            kv_endpoints,
            max_seqlen_q_list,
            max_seqlen_k_list,
            dropout_p=0.0,
            softmax_scale=1.0 / (head_dim ** 0.5),
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
        )

        # Check outputs
        assert len(out_list) == num_groups
        for g in range(num_groups):
            assert out_list[g].shape == q_list[g].shape
            assert not torch.isnan(out_list[g]).any()
            assert not torch.isinf(out_list[g]).any()

        print("✓ Forward pass succeeded")
        print(f"  Output shapes: {[o.shape for o in out_list]}")
        print(f"  LSE shapes: {[l.shape for l in lse_list]}")
        return True

    except RuntimeError as e:
        if "not yet implemented" in str(e):
            print("✗ FAILED: Kernel dispatcher not implemented")
            print("  The CUDA extension compiled but kernels are not hooked up yet")
            print("  Check flash_api_multigroup.cpp for TODO/TORCH_CHECK")
        else:
            print(f"✗ FAILED: Runtime error: {e}")
        return False


def test_all_head_dimensions():
    """Test all head dimensions to verify Phase 5.1 optimizations."""
    print("\n" + "=" * 60)
    print("TEST 2: All Head Dimensions")
    print("=" * 60)

    try:
        import flash_attn_multigroup_cuda
    except ImportError:
        print("✗ FAILED: flash_attn_multigroup_cuda not available")
        return False

    head_dims = [32, 64, 96, 128, 192, 256]
    num_groups = 2
    seq_len = 128
    kv_len = 256
    num_heads = 4
    dtype = torch.bfloat16

    all_passed = True

    for head_dim in head_dims:
        try:
            q_list = [
                torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device='cuda')
                for _ in range(num_groups)
            ]
            k = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device='cuda')
            v = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device='cuda')

            cu_seqlens_q_list = [
                torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')
                for _ in range(num_groups)
            ]
            cu_seqlens_k_list = [
                torch.tensor([0, kv_len], dtype=torch.int32, device='cuda')
                for _ in range(num_groups)
            ]
            kv_endpoints = torch.ones((num_groups, 1), dtype=torch.int32, device='cuda') * kv_len

            out_list, lse_list = flash_attn_multigroup_cuda.fwd(
                q_list, k, v,
                cu_seqlens_q_list, cu_seqlens_k_list,
                kv_endpoints,
                [seq_len] * num_groups,
                [kv_len] * num_groups,
                dropout_p=0.0,
                softmax_scale=1.0 / (head_dim ** 0.5),
                causal=False,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
            )

            # Verify
            for g in range(num_groups):
                assert out_list[g].shape[-1] == head_dim
                assert not torch.isnan(out_list[g]).any()
                assert not torch.isinf(out_list[g]).any()

            print(f"  ✓ head_dim={head_dim:3d}")

        except Exception as e:
            print(f"  ✗ head_dim={head_dim:3d} - {str(e)[:60]}")
            all_passed = False

    if all_passed:
        print("✓ All head dimensions passed")
    else:
        print("✗ Some head dimensions failed")

    return all_passed


def test_dimension_rounding():
    """Test non-standard dimension rounding."""
    print("\n" + "=" * 60)
    print("TEST 3: Dimension Rounding")
    print("=" * 60)

    try:
        import flash_attn_multigroup_cuda
    except ImportError:
        print("✗ FAILED: flash_attn_multigroup_cuda not available")
        return False

    # Test a few non-standard dimensions
    test_dims = [
        (80, 96),    # Should round to 96
        (112, 128),  # Should round to 128
        (144, 192),  # Should round to 192
    ]

    num_groups = 2
    seq_len = 64
    kv_len = 128
    num_heads = 4
    dtype = torch.float16

    all_passed = True

    for input_dim, expected_kernel_dim in test_dims:
        try:
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

            out_list, lse_list = flash_attn_multigroup_cuda.fwd(
                q_list, k, v,
                cu_seqlens_q_list, cu_seqlens_k_list,
                kv_endpoints,
                [seq_len] * num_groups,
                [kv_len] * num_groups,
                dropout_p=0.0,
                softmax_scale=1.0 / (input_dim ** 0.5),
                causal=False,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
            )

            # Output should preserve input dimension
            for g in range(num_groups):
                assert out_list[g].shape[-1] == input_dim
                assert not torch.isnan(out_list[g]).any()

            print(f"  ✓ input_dim={input_dim:3d} → kernel uses {expected_kernel_dim:3d}, output preserves {input_dim}")

        except Exception as e:
            print(f"  ✗ input_dim={input_dim:3d} - {str(e)[:60]}")
            all_passed = False

    if all_passed:
        print("✓ Dimension rounding works correctly")
    else:
        print("✗ Dimension rounding failed")

    return all_passed


def test_correctness_vs_reference():
    """Test correctness against standard flash attention."""
    print("\n" + "=" * 60)
    print("TEST 4: Correctness vs Standard Flash Attention")
    print("=" * 60)

    try:
        import flash_attn_multigroup_cuda
        from flash_attn import flash_attn_varlen_func
    except ImportError as e:
        print(f"✗ FAILED: Missing import - {e}")
        return False

    num_groups = 2
    seq_len = 128
    kv_len = 256
    num_heads = 4
    head_dim = 64
    dtype = torch.float16

    print(f"Comparing multigroup kernel vs standard flash attention")
    print(f"Config: {num_groups} groups, head_dim={head_dim}")

    # Create inputs
    q_list = [
        torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device='cuda')
        for _ in range(num_groups)
    ]
    k = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device='cuda')
    v = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device='cuda')

    cu_seqlens_q_list = [
        torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')
        for _ in range(num_groups)
    ]
    cu_seqlens_k_list = [
        torch.tensor([0, kv_len], dtype=torch.int32, device='cuda')
        for _ in range(num_groups)
    ]
    kv_endpoints = torch.ones((num_groups, 1), dtype=torch.int32, device='cuda') * kv_len

    try:
        # Run multigroup kernel
        out_list, lse_list = flash_attn_multigroup_cuda.fwd(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            kv_endpoints,
            [seq_len] * num_groups,
            [kv_len] * num_groups,
            dropout_p=0.0,
            softmax_scale=1.0 / (head_dim ** 0.5),
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
        )

        # Run standard flash attention for each group
        ref_outputs = []
        for g in range(num_groups):
            ref_out = flash_attn_varlen_func(
                q=q_list[g],
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q_list[g],
                cu_seqlens_k=cu_seqlens_k_list[g],
                max_seqlen_q=seq_len,
                max_seqlen_k=kv_len,
                dropout_p=0.0,
                softmax_scale=1.0 / (head_dim ** 0.5),
                causal=False,
            )
            ref_outputs.append(ref_out)

        # Compare
        tolerance = 1e-2  # float16 tolerance
        max_errors = []

        for g in range(num_groups):
            diff = torch.abs(out_list[g] - ref_outputs[g])
            max_error = diff.max().item()
            mean_error = diff.mean().item()
            max_errors.append(max_error)

            if max_error < tolerance:
                print(f"  ✓ Group {g}: max_error={max_error:.6e}, mean_error={mean_error:.6e}")
            else:
                print(f"  ✗ Group {g}: max_error={max_error:.6e} exceeds tolerance {tolerance:.6e}")
                return False

        print(f"✓ Correctness test passed (max error across groups: {max(max_errors):.6e})")
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def main():
    """Run quick smoke tests."""
    print("=" * 60)
    print("MULTIGROUP KERNEL QUICK TEST")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\n✗ CUDA not available")
        return 1

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    print(f"Compute capability: SM {cc_major}.{cc_minor}")

    results = []

    # Run tests
    results.append(("Basic forward", test_basic_forward()))
    results.append(("All head dimensions", test_all_head_dimensions()))
    results.append(("Dimension rounding", test_dimension_rounding()))
    results.append(("Correctness vs reference", test_correctness_vs_reference()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:30s}: {status}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All quick tests passed!")
        print("\nNext: Run comprehensive tests with:")
        print("  pytest test_multigroup_kernel_correctness.py -v -s")
        print("  or: python test_multigroup_kernel_correctness.py")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        print("\nDebug the failures before running comprehensive tests")
        return 1


if __name__ == "__main__":
    sys.exit(main())

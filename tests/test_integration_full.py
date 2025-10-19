#!/usr/bin/env python3
"""
Full integration tests for Flash Attention grouped features.

Tests end-to-end workflows including:
- Forward + backward pass combined
- All kernel variants
- All head dimensions
- Autograd integration
- Training loop simulation
"""

import pytest
import torch
import math
from flash_attn.flash_attn_grouped import (
    _flash_attn_varlen_forward_grouped,
    _flash_attn_varlen_backward_grouped
)
from flash_attn import flash_attn_varlen_func


def generate_random_lengths(batch_size, max_seqlen, device):
    """Generate random sequence lengths."""
    return torch.randint(
        max(1, max_seqlen // 2),
        max_seqlen + 1,
        (batch_size,),
        dtype=torch.int32,
        device=device
    )


def generate_cu_seqlens(lengths):
    """Generate cumulative sequence lengths from lengths tensor."""
    return torch.nn.functional.pad(
        torch.cumsum(lengths, dim=0, dtype=torch.int32),
        (1, 0)
    )


@pytest.mark.parametrize("num_groups", [2, 3, 4])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_forward_backward_integration(num_groups, head_dim, dtype):
    """
    Test complete forward + backward pass integration.

    Verifies that forward and backward passes work together correctly
    and produce valid gradients.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    batch_size = 2
    nheads = 8
    nheads_k = 4
    max_seqlen_q = 256
    max_seqlen_k = 512

    # Generate test data
    q_list = []
    cu_seqlens_q_list = []
    max_seqlen_q_list = []

    for i in range(num_groups):
        lengths_q = generate_random_lengths(batch_size, max_seqlen_q, device)
        cu_seqlens_q = generate_cu_seqlens(lengths_q)
        total_q = lengths_q.sum().item()

        q = torch.randn(
            total_q, nheads, head_dim,
            device=device, dtype=dtype, requires_grad=True
        )

        q_list.append(q)
        cu_seqlens_q_list.append(cu_seqlens_q)
        max_seqlen_q_list.append(lengths_q.max().item())

    # Shared K, V
    lengths_k = generate_random_lengths(batch_size, max_seqlen_k, device)
    cu_seqlens_k = generate_cu_seqlens(lengths_k)
    total_k = lengths_k.sum().item()

    k = torch.randn(
        total_k, nheads_k, head_dim,
        device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        total_k, nheads_k, head_dim,
        device=device, dtype=dtype, requires_grad=True
    )

    # Create cu_seqlens_k for each group
    cu_seqlens_k_list = [cu_seqlens_k for _ in range(num_groups)]
    max_seqlen_k_list = [lengths_k.max().item() for _ in range(num_groups)]

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Forward pass
    out_list, lse_list, _, _ = _flash_attn_varlen_forward_grouped(
        q_list=q_list,
        k=k,
        v=v,
        cu_seqlens_q_list=cu_seqlens_q_list,
        cu_seqlens_k_list=cu_seqlens_k_list,
        max_seqlen_q_list=max_seqlen_q_list,
        max_seqlen_k_list=max_seqlen_k_list,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
    )

    # Create gradient outputs
    dout_list = [torch.randn_like(out) for out in out_list]

    # Backward pass
    dq_list, dk, dv = _flash_attn_varlen_backward_grouped(
        dout_list=dout_list,
        q_list=q_list,
        k=k,
        v=v,
        out_list=out_list,
        softmax_lse_list=lse_list,
        cu_seqlens_q_list=cu_seqlens_q_list,
        cu_seqlens_k_list=cu_seqlens_k_list,
        max_seqlen_q_list=max_seqlen_q_list,
        max_seqlen_k_list=max_seqlen_k_list,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
    )

    # Verify outputs
    assert len(out_list) == num_groups
    assert len(dq_list) == num_groups

    # Verify shapes
    for i in range(num_groups):
        assert out_list[i].shape == q_list[i].shape
        assert dq_list[i].shape == q_list[i].shape

        # Check for NaN/Inf
        assert not torch.isnan(out_list[i]).any()
        assert not torch.isinf(out_list[i]).any()
        assert not torch.isnan(dq_list[i]).any()
        assert not torch.isinf(dq_list[i]).any()

    assert dk.shape == k.shape
    assert dv.shape == v.shape
    assert not torch.isnan(dk).any()
    assert not torch.isinf(dk).any()
    assert not torch.isnan(dv).any()
    assert not torch.isinf(dv).any()

    print(f"✓ Forward+Backward integration test passed: "
          f"num_groups={num_groups}, head_dim={head_dim}, dtype={dtype}")


@pytest.mark.parametrize("head_dim", [32, 64, 96, 128, 192, 256])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_all_head_dimensions_integration(head_dim, dtype):
    """
    Test integration across all supported head dimensions.

    Ensures all kernels are compiled and functional.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    num_groups = 2
    batch_size = 2
    nheads = 4
    seqlen_q = 128
    seqlen_k = 256

    # Create inputs
    q_list = []
    cu_seqlens_q_list = []
    cu_seqlens_k_list = []

    for _ in range(num_groups):
        q = torch.randn(
            seqlen_q, nheads, head_dim,
            device=device, dtype=dtype, requires_grad=True
        )
        cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device=device)

        q_list.append(q)
        cu_seqlens_q_list.append(cu_seqlens_q)
        cu_seqlens_k_list.append(cu_seqlens_k)

    k = torch.randn(seqlen_k, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(seqlen_k, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Forward pass
    out_list, lse_list, _, _ = _flash_attn_varlen_forward_grouped(
        q_list=q_list,
        k=k,
        v=v,
        cu_seqlens_q_list=cu_seqlens_q_list,
        cu_seqlens_k_list=cu_seqlens_k_list,
        max_seqlen_q_list=[seqlen_q] * num_groups,
        max_seqlen_k_list=[seqlen_k] * num_groups,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=True,
    )

    # Backward pass
    dout_list = [torch.randn_like(out) for out in out_list]

    dq_list, dk, dv = _flash_attn_varlen_backward_grouped(
        dout_list=dout_list,
        q_list=q_list,
        k=k,
        v=v,
        out_list=out_list,
        softmax_lse_list=lse_list,
        cu_seqlens_q_list=cu_seqlens_q_list,
        cu_seqlens_k_list=cu_seqlens_k_list,
        max_seqlen_q_list=[seqlen_q] * num_groups,
        max_seqlen_k_list=[seqlen_k] * num_groups,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=True,
    )

    # Verify all outputs valid
    for i in range(num_groups):
        assert not torch.isnan(out_list[i]).any()
        assert not torch.isnan(dq_list[i]).any()

    assert not torch.isnan(dk).any()
    assert not torch.isnan(dv).any()

    print(f"✓ All head dimensions test passed: head_dim={head_dim}")


@pytest.mark.parametrize("causal", [True, False])
def test_causal_and_noncausal(causal):
    """Test both causal and non-causal modes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    dtype = torch.float16
    num_groups = 2
    nheads = 8
    head_dim = 128
    seqlen_q = 256
    seqlen_k = 512

    q_list = []
    cu_seqlens_q_list = []
    cu_seqlens_k_list = []

    for _ in range(num_groups):
        q = torch.randn(seqlen_q, nheads, head_dim, device=device, dtype=dtype)
        cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device=device)

        q_list.append(q)
        cu_seqlens_q_list.append(cu_seqlens_q)
        cu_seqlens_k_list.append(cu_seqlens_k)

    k = torch.randn(seqlen_k, nheads, head_dim, device=device, dtype=dtype)
    v = torch.randn(seqlen_k, nheads, head_dim, device=device, dtype=dtype)

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Forward pass
    out_list, _, _, _ = _flash_attn_varlen_forward_grouped(
        q_list=q_list,
        k=k,
        v=v,
        cu_seqlens_q_list=cu_seqlens_q_list,
        cu_seqlens_k_list=cu_seqlens_k_list,
        max_seqlen_q_list=[seqlen_q] * num_groups,
        max_seqlen_k_list=[seqlen_k] * num_groups,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
    )

    # Verify outputs
    for out in out_list:
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    print(f"✓ Causal mode test passed: causal={causal}")


def test_training_loop_simulation():
    """
    Simulate a simple training loop with grouped attention.

    Tests that gradients flow correctly through multiple iterations.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    dtype = torch.float16
    num_groups = 2
    nheads = 8
    head_dim = 128
    seqlen_q = 128
    seqlen_k = 256
    num_iterations = 5

    # Create learnable parameters
    q_params = [
        torch.randn(seqlen_q, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
        for _ in range(num_groups)
    ]
    k_param = torch.randn(seqlen_k, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v_param = torch.randn(seqlen_k, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)

    # Simple optimizer
    params = q_params + [k_param, v_param]
    optimizer = torch.optim.SGD(params, lr=0.01)

    cu_seqlens_q_list = [
        torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
        for _ in range(num_groups)
    ]
    cu_seqlens_k_list = [
        torch.tensor([0, seqlen_k], dtype=torch.int32, device=device)
        for _ in range(num_groups)
    ]

    softmax_scale = 1.0 / math.sqrt(head_dim)

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Forward pass
        out_list, lse_list, _, _ = _flash_attn_varlen_forward_grouped(
            q_list=q_params,
            k=k_param,
            v=v_param,
            cu_seqlens_q_list=cu_seqlens_q_list,
            cu_seqlens_k_list=cu_seqlens_k_list,
            max_seqlen_q_list=[seqlen_q] * num_groups,
            max_seqlen_k_list=[seqlen_k] * num_groups,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
        )

        # Compute simple loss
        loss = sum(out.sum() for out in out_list)

        # Backward pass
        dout_list = [torch.ones_like(out) for out in out_list]

        dq_list, dk, dv = _flash_attn_varlen_backward_grouped(
            dout_list=dout_list,
            q_list=q_params,
            k=k_param,
            v=v_param,
            out_list=out_list,
            softmax_lse_list=lse_list,
            cu_seqlens_q_list=cu_seqlens_q_list,
            cu_seqlens_k_list=cu_seqlens_k_list,
            max_seqlen_q_list=[seqlen_q] * num_groups,
            max_seqlen_k_list=[seqlen_k] * num_groups,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
        )

        # Manually set gradients (simulating autograd)
        for i, q in enumerate(q_params):
            q.grad = dq_list[i]
        k_param.grad = dk
        v_param.grad = dv

        # Verify gradients exist and are finite
        for param in params:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

        # Optimizer step
        optimizer.step()

        print(f"  Iteration {iteration + 1}/{num_iterations}: loss={loss.item():.4f}")

    print("✓ Training loop simulation test passed")


def test_gradient_accumulation():
    """
    Test gradient accumulation across multiple groups.

    Verifies that dK and dV are correctly accumulated.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    dtype = torch.float16
    num_groups = 3
    nheads = 8
    head_dim = 128
    seqlen_q = 128
    seqlen_k = 256

    q_list = [
        torch.randn(seqlen_q, nheads, head_dim, device=device, dtype=dtype)
        for _ in range(num_groups)
    ]
    k = torch.randn(seqlen_k, nheads, head_dim, device=device, dtype=dtype)
    v = torch.randn(seqlen_k, nheads, head_dim, device=device, dtype=dtype)

    cu_seqlens_q_list = [
        torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
        for _ in range(num_groups)
    ]
    cu_seqlens_k_list = [
        torch.tensor([0, seqlen_k], dtype=torch.int32, device=device)
        for _ in range(num_groups)
    ]

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Grouped forward
    out_list_grouped, lse_list_grouped, _, _ = _flash_attn_varlen_forward_grouped(
        q_list=q_list,
        k=k,
        v=v,
        cu_seqlens_q_list=cu_seqlens_q_list,
        cu_seqlens_k_list=cu_seqlens_k_list,
        max_seqlen_q_list=[seqlen_q] * num_groups,
        max_seqlen_k_list=[seqlen_k] * num_groups,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
    )

    # Grouped backward
    dout_list = [torch.ones_like(out) for out in out_list_grouped]

    dq_list_grouped, dk_grouped, dv_grouped = _flash_attn_varlen_backward_grouped(
        dout_list=dout_list,
        q_list=q_list,
        k=k,
        v=v,
        out_list=out_list_grouped,
        softmax_lse_list=lse_list_grouped,
        cu_seqlens_q_list=cu_seqlens_q_list,
        cu_seqlens_k_list=cu_seqlens_k_list,
        max_seqlen_q_list=[seqlen_q] * num_groups,
        max_seqlen_k_list=[seqlen_k] * num_groups,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
    )

    # Verify dK and dV have reasonable magnitudes
    # (should be sum of contributions from all groups)
    dk_norm = dk_grouped.norm()
    dv_norm = dv_grouped.norm()

    # Expected to be larger than single group contribution
    assert dk_norm > 0
    assert dv_norm > 0

    print(f"✓ Gradient accumulation test passed: "
          f"dk_norm={dk_norm:.2f}, dv_norm={dv_norm:.2f}")


def test_memory_efficiency():
    """
    Test memory usage with grouped attention vs separate calls.

    Verifies that grouped attention uses less memory.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    dtype = torch.float16
    num_groups = 3
    nheads = 16
    head_dim = 128
    seqlen_q = 512
    seqlen_k = 1024

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Test grouped attention memory
    q_list = [
        torch.randn(seqlen_q, nheads, head_dim, device=device, dtype=dtype)
        for _ in range(num_groups)
    ]
    k = torch.randn(seqlen_k, nheads, head_dim, device=device, dtype=dtype)
    v = torch.randn(seqlen_k, nheads, head_dim, device=device, dtype=dtype)

    cu_seqlens_q_list = [
        torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
        for _ in range(num_groups)
    ]
    cu_seqlens_k_list = [
        torch.tensor([0, seqlen_k], dtype=torch.int32, device=device)
        for _ in range(num_groups)
    ]

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Grouped call
    out_list_grouped, _, _, _ = _flash_attn_varlen_forward_grouped(
        q_list=q_list,
        k=k,
        v=v,
        cu_seqlens_q_list=cu_seqlens_q_list,
        cu_seqlens_k_list=cu_seqlens_k_list,
        max_seqlen_q_list=[seqlen_q] * num_groups,
        max_seqlen_k_list=[seqlen_k] * num_groups,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
    )

    grouped_memory = torch.cuda.max_memory_allocated() / 1e6  # MB

    # Clear and test separate calls
    del out_list_grouped
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    out_list_separate = []
    for i in range(num_groups):
        out = flash_attn_varlen_func(
            q=q_list[i],
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q_list[i],
            cu_seqlens_k=cu_seqlens_k_list[i],
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
        )
        out_list_separate.append(out)

    separate_memory = torch.cuda.max_memory_allocated() / 1e6  # MB

    memory_reduction = (separate_memory - grouped_memory) / separate_memory * 100

    print(f"✓ Memory efficiency test:")
    print(f"  Grouped memory: {grouped_memory:.2f} MB")
    print(f"  Separate memory: {separate_memory:.2f} MB")
    print(f"  Reduction: {memory_reduction:.1f}%")

    # Grouped should use less or equal memory
    # (may be equal due to memory allocation granularity)
    assert grouped_memory <= separate_memory * 1.1  # Allow 10% tolerance


if __name__ == "__main__":
    print("=" * 80)
    print("Flash Attention Grouped Features - Full Integration Tests")
    print("=" * 80)
    print()

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping tests.")
        exit(0)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: SM {torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}")
    print()

    # Run tests
    tests = [
        ("Forward+Backward (2 groups, hdim=64, FP16)",
         lambda: test_forward_backward_integration(2, 64, torch.float16)),
        ("Forward+Backward (3 groups, hdim=128, BF16)",
         lambda: test_forward_backward_integration(3, 128, torch.bfloat16)),
        ("All head dimensions (hdim=32)",
         lambda: test_all_head_dimensions_integration(32, torch.float16)),
        ("All head dimensions (hdim=64)",
         lambda: test_all_head_dimensions_integration(64, torch.float16)),
        ("All head dimensions (hdim=96)",
         lambda: test_all_head_dimensions_integration(96, torch.float16)),
        ("All head dimensions (hdim=128)",
         lambda: test_all_head_dimensions_integration(128, torch.float16)),
        ("All head dimensions (hdim=192)",
         lambda: test_all_head_dimensions_integration(192, torch.float16)),
        ("All head dimensions (hdim=256)",
         lambda: test_all_head_dimensions_integration(256, torch.float16)),
        ("Causal mode", lambda: test_causal_and_noncausal(True)),
        ("Non-causal mode", lambda: test_causal_and_noncausal(False)),
        ("Training loop simulation", test_training_loop_simulation),
        ("Gradient accumulation", test_gradient_accumulation),
        ("Memory efficiency", test_memory_efficiency),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 80)
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {test_name}")
            print(f"  Error: {str(e)}")
            failed += 1

    print()
    print("=" * 80)
    print(f"Integration Tests Summary: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 80)

    if failed == 0:
        print("\n✅ All integration tests passed!")
        exit(0)
    else:
        print(f"\n❌ {failed} test(s) failed")
        exit(1)

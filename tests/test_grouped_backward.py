# Copyright (c) 2024, Tri Dao.
# Test suite for grouped flash attention backward pass

import pytest
import torch
import math
from flash_attn.flash_attn_grouped import _flash_attn_varlen_forward_grouped, _flash_attn_varlen_backward_grouped
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, _flash_attn_varlen_backward


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    """Generate random padding mask for variable length sequences."""
    if mode == "random":
        lengths = torch.randint(1, max_seqlen + 1, (batch_size,), device=device)
    elif mode == "full":
        lengths = torch.full((batch_size,), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "mixed":
        lengths = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device=device)
    else:
        raise ValueError(f"Unknown mode {mode}")
    return lengths


def generate_cu_seqlens(lengths):
    """Generate cumulative sequence lengths from lengths."""
    return torch.nn.functional.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_groups", [2, 3])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_grouped_backward_correctness(dtype, num_groups, causal, head_dim):
    """
    Test that grouped backward pass produces correct gradients by comparing
    with separate backward passes.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    batch_size = 4
    nheads = 8
    nheads_k = 4
    max_seqlen_q = 256
    base_seqlen_k = 512

    # Create different K,V lengths for each group
    max_seqlen_k_list = [base_seqlen_k // (i + 1) for i in range(num_groups)]

    # Generate Q groups
    q_list = []
    cu_seqlens_q_list = []
    max_seqlen_q_list = []
    out_list = []
    softmax_lse_list = []

    for i in range(num_groups):
        # Generate random lengths for this group
        lengths_q = generate_random_padding_mask(max_seqlen_q, batch_size, device, mode="random")
        cu_seqlens_q = generate_cu_seqlens(lengths_q)
        total_q = lengths_q.sum().item()

        q = torch.randn(total_q, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
        q_list.append(q)
        cu_seqlens_q_list.append(cu_seqlens_q)
        max_seqlen_q_list.append(lengths_q.max().item())

    # Generate shared K, V (use longest K,V length)
    max_seqlen_k_global = max(max_seqlen_k_list)
    lengths_k = generate_random_padding_mask(max_seqlen_k_global, batch_size, device, mode="full")
    cu_seqlens_k_global = generate_cu_seqlens(lengths_k)
    total_k = lengths_k.sum().item()

    k = torch.randn(total_k, nheads_k, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(total_k, nheads_k, head_dim, device=device, dtype=dtype, requires_grad=True)

    # Generate cu_seqlens_k for each group (different endpoints)
    cu_seqlens_k_list = []
    for max_k in max_seqlen_k_list:
        # Scale down cu_seqlens_k to match this group's max length
        lengths_k_group = torch.clamp(lengths_k, max=max_k)
        cu_seqlens_k_group = generate_cu_seqlens(lengths_k_group)
        cu_seqlens_k_list.append(cu_seqlens_k_group)

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # ========== Run grouped forward pass ==========
    results_fwd_grouped = _flash_attn_varlen_forward_grouped(
        q_list=q_list,
        k=k,
        v=v,
        cu_seqlens_q_list=cu_seqlens_q_list,
        cu_seqlens_k_list=cu_seqlens_k_list,
        max_seqlen_q_list=max_seqlen_q_list,
        max_seqlen_k_list=max_seqlen_k_list,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    out_list_grouped, lse_list_grouped, _, _ = results_fwd_grouped

    # ========== Run separate forward passes for reference ==========
    out_list_separate = []
    lse_list_separate = []
    for i in range(num_groups):
        k_end = cu_seqlens_k_list[i][-1].item()
        k_slice = k[:k_end]
        v_slice = v[:k_end]

        out_sep, lse_sep, _, _ = _flash_attn_varlen_forward(
            q=q_list[i],
            k=k_slice,
            v=v_slice,
            cu_seqlens_q=cu_seqlens_q_list[i],
            cu_seqlens_k=cu_seqlens_k_list[i],
            max_seqlen_q=max_seqlen_q_list[i],
            max_seqlen_k=max_seqlen_k_list[i],
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        out_list_separate.append(out_sep)
        lse_list_separate.append(lse_sep)

    # Verify forward pass outputs match
    for i in range(num_groups):
        assert torch.allclose(out_list_grouped[i], out_list_separate[i], rtol=1e-3, atol=1e-3), \
            f"Forward output mismatch for group {i}"
        assert torch.allclose(lse_list_grouped[i], lse_list_separate[i], rtol=1e-3, atol=1e-3), \
            f"Forward LSE mismatch for group {i}"

    # ========== Create gradient outputs ==========
    dout_list = [torch.randn_like(out) for out in out_list_grouped]

    # ========== Run grouped backward pass ==========
    dq_list_grouped, dk_grouped, dv_grouped = _flash_attn_varlen_backward_grouped(
        dout_list=dout_list,
        q_list=q_list,
        k=k,
        v=v,
        out_list=out_list_grouped,
        softmax_lse_list=lse_list_grouped,
        cu_seqlens_q_list=cu_seqlens_q_list,
        cu_seqlens_k_list=cu_seqlens_k_list,
        max_seqlen_q_list=max_seqlen_q_list,
        max_seqlen_k_list=max_seqlen_k_list,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
    )

    # ========== Run separate backward passes for reference ==========
    dk_accumulator = torch.zeros_like(k)
    dv_accumulator = torch.zeros_like(v)
    dq_list_separate = []

    for i in range(num_groups):
        k_end = cu_seqlens_k_list[i][-1].item()
        k_slice = k[:k_end]
        v_slice = v[:k_end]

        # Create dummy gradients for k_slice and v_slice
        dk_slice = torch.zeros_like(k_slice)
        dv_slice = torch.zeros_like(v_slice)

        dq_sep, dk_sep, dv_sep, _ = _flash_attn_varlen_backward(
            dout=dout_list[i],
            q=q_list[i],
            k=k_slice,
            v=v_slice,
            out=out_list_separate[i],
            softmax_lse=lse_list_separate[i],
            dq=None,
            dk=dk_slice,
            dv=dv_slice,
            cu_seqlens_q=cu_seqlens_q_list[i],
            cu_seqlens_k=cu_seqlens_k_list[i],
            max_seqlen_q=max_seqlen_q_list[i],
            max_seqlen_k=max_seqlen_k_list[i],
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
        )

        dq_list_separate.append(dq_sep)

        # Accumulate dk, dv
        dk_accumulator[:k_end] += dk_slice
        dv_accumulator[:k_end] += dv_slice

    # ========== Verify gradients ==========
    for i in range(num_groups):
        assert torch.allclose(dq_list_grouped[i], dq_list_separate[i], rtol=1e-3, atol=1e-3), \
            f"dQ mismatch for group {i}"

    assert torch.allclose(dk_grouped, dk_accumulator, rtol=1e-3, atol=1e-3), "dK mismatch"
    assert torch.allclose(dv_grouped, dv_accumulator, rtol=1e-3, atol=1e-3), "dV mismatch"

    print(f"✓ Test passed: dtype={dtype}, num_groups={num_groups}, causal={causal}, head_dim={head_dim}")


@pytest.mark.parametrize("dtype", [torch.float16])
def test_grouped_backward_edge_cases(dtype):
    """Test edge cases like empty groups, single token, etc."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    nheads = 4
    nheads_k = 2
    head_dim = 64
    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Test with one group having very few tokens
    batch_size = 2
    q1 = torch.randn(1, nheads, head_dim, device=device, dtype=dtype)  # Single token
    q2 = torch.randn(100, nheads, head_dim, device=device, dtype=dtype)
    k = torch.randn(200, nheads_k, head_dim, device=device, dtype=dtype)
    v = torch.randn(200, nheads_k, head_dim, device=device, dtype=dtype)

    cu_seqlens_q1 = torch.tensor([0, 1], device=device, dtype=torch.int32)
    cu_seqlens_q2 = torch.tensor([0, 50, 100], device=device, dtype=torch.int32)
    cu_seqlens_k1 = torch.tensor([0, 100], device=device, dtype=torch.int32)
    cu_seqlens_k2 = torch.tensor([0, 100, 200], device=device, dtype=torch.int32)

    q_list = [q1, q2]
    cu_seqlens_q_list = [cu_seqlens_q1, cu_seqlens_q2]
    cu_seqlens_k_list = [cu_seqlens_k1, cu_seqlens_k2]
    max_seqlen_q_list = [1, 50]
    max_seqlen_k_list = [100, 100]

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
        max_seqlen_q_list=max_seqlen_q_list,
        max_seqlen_k_list=max_seqlen_k_list,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
    )

    # Verify shapes
    assert dq_list[0].shape == q1.shape
    assert dq_list[1].shape == q2.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape

    # Verify no NaN or Inf
    for i, dq in enumerate(dq_list):
        assert not torch.isnan(dq).any(), f"dQ[{i}] contains NaN"
        assert not torch.isinf(dq).any(), f"dQ[{i}] contains Inf"

    assert not torch.isnan(dk).any(), "dK contains NaN"
    assert not torch.isinf(dk).any(), "dK contains Inf"
    assert not torch.isnan(dv).any(), "dV contains NaN"
    assert not torch.isinf(dv).any(), "dV contains Inf"

    print("✓ Edge cases test passed")


@pytest.mark.parametrize("head_dim", [64, 96, 128, 192])
def test_grouped_backward_all_head_dims(head_dim):
    """Test grouped backward with various head dimensions."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    dtype = torch.float16
    batch_size = 2
    nheads = 4
    nheads_k = 2
    num_groups = 2

    q_list = []
    cu_seqlens_q_list = []
    max_seqlen_q_list = []
    cu_seqlens_k_list = []
    max_seqlen_k_list = []

    for i in range(num_groups):
        total_q = 64
        q = torch.randn(total_q, nheads, head_dim, device=device, dtype=dtype)
        cu_seqlens_q = torch.tensor([0, 32, 64], device=device, dtype=torch.int32)
        q_list.append(q)
        cu_seqlens_q_list.append(cu_seqlens_q)
        max_seqlen_q_list.append(32)

        cu_seqlens_k = torch.tensor([0, 64, 128], device=device, dtype=torch.int32)
        cu_seqlens_k_list.append(cu_seqlens_k)
        max_seqlen_k_list.append(64)

    k = torch.randn(128, nheads_k, head_dim, device=device, dtype=dtype)
    v = torch.randn(128, nheads_k, head_dim, device=device, dtype=dtype)

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Forward
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
        causal=True,
    )

    # Backward
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
        max_seqlen_q_list=max_seqlen_q_list,
        max_seqlen_k_list=max_seqlen_k_list,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=True,
    )

    # Verify shapes
    for i in range(num_groups):
        assert dq_list[i].shape == q_list[i].shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape

    print(f"✓ Test passed for head_dim={head_dim}")


if __name__ == "__main__":
    # Run tests
    print("Running grouped backward pass tests...")

    # Test correctness
    for dtype in [torch.float16, torch.bfloat16]:
        for num_groups in [2, 3]:
            for causal in [True, False]:
                for head_dim in [64, 128]:
                    try:
                        test_grouped_backward_correctness(dtype, num_groups, causal, head_dim)
                    except Exception as e:
                        print(f"✗ Test failed: {e}")

    # Test edge cases
    try:
        test_grouped_backward_edge_cases(torch.float16)
    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")

    # Test all head dims
    for head_dim in [64, 96, 128, 192]:
        try:
            test_grouped_backward_all_head_dims(head_dim)
        except Exception as e:
            print(f"✗ Test failed for head_dim={head_dim}: {e}")

    print("\nAll tests completed!")

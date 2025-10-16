"""
Unit tests for grouped flash attention.

This tests the new _flash_attn_varlen_forward_grouped function that enables
multiple Q groups to share K,V loads for better memory efficiency.
"""

import torch
import pytest
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, _flash_attn_varlen_forward_grouped


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    """Generate cumulative sequence lengths for variable-length sequences."""
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size,), max_seqlen, dtype=torch.int32, device=device)
    elif mode == "random":
        lengths = torch.randint(max(1, max_seqlen - 20), max_seqlen + 1, (batch_size,), dtype=torch.int32, device=device)
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size,), dtype=torch.int32, device=device)
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(lengths, dim=0), (1, 0))
    return cu_seqlens


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128])
def test_flash_attn_varlen_grouped_correctness(dtype, causal, d):
    """
    Test that grouped attention produces the same results as separate calls.

    This validates that the API works correctly and returns the same outputs
    as calling flash attention separately for each group.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    batch_size = 4
    nheads = 8
    nheads_k = 2  # Test GQA
    seqlen_q = 128
    seqlen_k_early = 256
    seqlen_k_late = 512

    # Generate variable-length sequences
    cu_seqlens_q = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    cu_seqlens_k_early = generate_random_padding_mask(seqlen_k_early, batch_size, device, mode="random")
    cu_seqlens_k_late = generate_random_padding_mask(seqlen_k_late, batch_size, device, mode="random")

    # Ensure late includes early (late seqlens >= early seqlens)
    cu_seqlens_k_late[1:] = torch.maximum(cu_seqlens_k_late[1:], cu_seqlens_k_early[1:])

    # Get max sequence lengths
    max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    max_seqlen_k_early = (cu_seqlens_k_early[1:] - cu_seqlens_k_early[:-1]).max().item()
    max_seqlen_k_late = (cu_seqlens_k_late[1:] - cu_seqlens_k_late[:-1]).max().item()

    # Create Q tensors for early and late groups
    total_q = cu_seqlens_q[-1].item()
    q_early = torch.randn(total_q, nheads, d, device=device, dtype=dtype, requires_grad=False)
    q_late = torch.randn(total_q, nheads, d, device=device, dtype=dtype, requires_grad=False)

    # Create shared K,V tensors (full length)
    total_k = cu_seqlens_k_late[-1].item()
    k = torch.randn(total_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(total_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False)

    # Reference: Call flash attention separately for each group
    k_early_end = cu_seqlens_k_early[-1].item()
    out_early_ref, lse_early_ref, _, _ = _flash_attn_varlen_forward(
        q=q_early,
        k=k[:k_early_end],
        v=v[:k_early_end],
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k_early,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k_early,
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
    )

    out_late_ref, lse_late_ref, _, _ = _flash_attn_varlen_forward(
        q=q_late,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k_late,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k_late,
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
    )

    # Test: Call grouped flash attention
    out_list, lse_list, _, _ = _flash_attn_varlen_forward_grouped(
        q_list=[q_early, q_late],
        k=k,
        v=v,
        cu_seqlens_q_list=[cu_seqlens_q, cu_seqlens_q],
        cu_seqlens_k_list=[cu_seqlens_k_early, cu_seqlens_k_late],
        max_seqlen_q_list=[max_seqlen_q, max_seqlen_q],
        max_seqlen_k_list=[max_seqlen_k_early, max_seqlen_k_late],
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
    )

    # Verify outputs match
    assert len(out_list) == 2, "Should have 2 output tensors"
    assert len(lse_list) == 2, "Should have 2 LSE tensors"

    # Check early group
    assert torch.allclose(out_list[0], out_early_ref, rtol=1e-3, atol=1e-3), \
        f"Early output mismatch: max diff = {(out_list[0] - out_early_ref).abs().max()}"
    assert torch.allclose(lse_list[0], lse_early_ref, rtol=1e-2, atol=1e-2), \
        f"Early LSE mismatch: max diff = {(lse_list[0] - lse_early_ref).abs().max()}"

    # Check late group
    assert torch.allclose(out_list[1], out_late_ref, rtol=1e-3, atol=1e-3), \
        f"Late output mismatch: max diff = {(out_list[1] - out_late_ref).abs().max()}"
    assert torch.allclose(lse_list[1], lse_late_ref, rtol=1e-2, atol=1e-2), \
        f"Late LSE mismatch: max diff = {(lse_list[1] - lse_late_ref).abs().max()}"

    print(f"✓ Grouped attention test passed: dtype={dtype}, causal={causal}, d={d}")


if __name__ == "__main__":
    # Run basic smoke test
    print("Running grouped attention smoke tests...")
    test_flash_attn_varlen_grouped_correctness(torch.float16, False, 64)
    test_flash_attn_varlen_grouped_correctness(torch.float16, True, 64)
    test_flash_attn_varlen_grouped_correctness(torch.bfloat16, False, 128)
    print("\n✓ All smoke tests passed!")

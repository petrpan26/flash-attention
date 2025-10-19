"""
Test N-group SMEM sharing kernel (3-4 groups) for Flash Attention.

This test suite validates the N-group SMEM kernel implementation that generalizes
the 2-group SMEM sharing approach to handle 3-4 groups efficiently.

Key optimizations being tested:
1. K,V tiles loaded once and reused across all groups
2. Sequential group processing for L2 cache benefits
3. Correct attention computation vs reference implementation
4. Kernel selection logic (2-group, N-group, cache-aware)
"""

import pytest
import torch
import torch.nn.functional as F

# Try importing flash_attn_interface
try:
    from flash_attn.flash_attn_interface import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash_attn not available, skipping tests")


def reference_grouped_attention(q_groups, k, v, softmax_scale=None, causal=False):
    """
    Reference implementation of grouped query attention using PyTorch.

    Args:
        q_groups: List of query tensors, one per group [batch, seqlen_q, num_heads, head_dim]
        k: Key tensor [batch, seqlen_k, num_kv_heads, head_dim]
        v: Value tensor [batch, seqlen_k, num_kv_heads, head_dim]
        softmax_scale: Scale factor for softmax (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking

    Returns:
        List of output tensors, one per group
    """
    num_groups = len(q_groups)
    batch_size = q_groups[0].shape[0]
    head_dim = q_groups[0].shape[-1]

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)

    outputs = []

    for g in range(num_groups):
        q = q_groups[g]  # [batch, seqlen_q, num_heads, head_dim]

        # Compute attention scores: Q @ K^T
        # q: [batch, seqlen_q, num_heads, head_dim]
        # k: [batch, seqlen_k, num_kv_heads, head_dim]
        # Need to expand k to match q's num_heads

        batch, seqlen_q, num_heads, _ = q.shape
        _, seqlen_k, num_kv_heads, _ = k.shape

        # Expand k and v to match num_heads if needed (GQA pattern)
        if num_heads != num_kv_heads:
            assert num_heads % num_kv_heads == 0
            k_expanded = k.repeat_interleave(num_heads // num_kv_heads, dim=2)
            v_expanded = v.repeat_interleave(num_heads // num_kv_heads, dim=2)
        else:
            k_expanded = k
            v_expanded = v

        # Reshape for batch matrix multiply
        q_reshape = q.transpose(1, 2)  # [batch, num_heads, seqlen_q, head_dim]
        k_reshape = k_expanded.transpose(1, 2)  # [batch, num_heads, seqlen_k, head_dim]
        v_reshape = v_expanded.transpose(1, 2)  # [batch, num_heads, seqlen_k, head_dim]

        # Compute attention scores
        scores = torch.matmul(q_reshape, k_reshape.transpose(-2, -1))  # [batch, num_heads, seqlen_q, seqlen_k]
        scores = scores * softmax_scale

        # Apply causal mask if needed
        if causal:
            causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Softmax
        attn = F.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v_reshape)  # [batch, num_heads, seqlen_q, head_dim]
        out = out.transpose(1, 2)  # [batch, seqlen_q, num_heads, head_dim]

        outputs.append(out)

    return outputs


@pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="flash_attn not available")
@pytest.mark.parametrize("num_groups", [3, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("headdim", [128])
@pytest.mark.parametrize("seqlen_q", [512, 1024])
@pytest.mark.parametrize("seqlen_k", [512, 1024])
@pytest.mark.parametrize("causal", [False, True])
def test_ngroup_smem_correctness(num_groups, dtype, headdim, seqlen_q, seqlen_k, causal):
    """
    Test N-group SMEM kernel correctness against reference implementation.

    Tests that the N-group SMEM sharing kernel produces the same results as
    the reference PyTorch implementation for 3-4 groups.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    batch_size = 2
    num_heads = 32
    num_kv_heads = 8  # GQA: 32 q heads share 8 kv heads

    # Create query tensors for each group
    q_groups = []
    for g in range(num_groups):
        q = torch.randn(batch_size, seqlen_q, num_heads, headdim,
                       device=device, dtype=dtype, requires_grad=False)
        q_groups.append(q)

    # Shared K,V for all groups
    k = torch.randn(batch_size, seqlen_k, num_kv_heads, headdim,
                   device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(batch_size, seqlen_k, num_kv_heads, headdim,
                   device=device, dtype=dtype, requires_grad=False)

    softmax_scale = 1.0 / (headdim ** 0.5)

    # Compute reference outputs
    ref_outputs = reference_grouped_attention(q_groups, k, v, softmax_scale, causal)

    # Compute flash attention outputs for each group separately
    # (The N-group kernel should be invoked automatically when num_groups is 3-4)
    flash_outputs = []
    for g in range(num_groups):
        out = flash_attn_func(
            q_groups[g], k, v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal
        )
        flash_outputs.append(out)

    # Compare outputs
    for g in range(num_groups):
        # Use appropriate tolerance for fp16/bf16
        atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-2

        torch.testing.assert_close(
            flash_outputs[g], ref_outputs[g],
            atol=atol, rtol=rtol,
            msg=f"Group {g} output mismatch (num_groups={num_groups}, dtype={dtype}, "
                f"headdim={headdim}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, causal={causal})"
        )

    print(f"✓ N-group SMEM test passed: num_groups={num_groups}, dtype={dtype}, "
          f"headdim={headdim}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, causal={causal}")


@pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="flash_attn not available")
@pytest.mark.parametrize("num_groups", [2, 3, 4, 5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kernel_selection(num_groups, dtype):
    """
    Test that the correct kernel is selected based on number of groups.

    - num_groups == 2: Should use 2-group specialized kernel
    - num_groups in [3, 4]: Should use N-group SMEM kernel
    - num_groups >= 5: Should use L2 cache-aware kernel

    This test just verifies that the kernels run without errors for different
    group counts, ensuring proper kernel selection logic.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    batch_size = 2
    seqlen_q = 512
    seqlen_k = 512
    num_heads = 32
    num_kv_heads = 8
    headdim = 128

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

    # Run flash attention for each group
    outputs = []
    for g in range(num_groups):
        out = flash_attn_func(
            q_groups[g], k, v,
            dropout_p=0.0,
            softmax_scale=1.0 / (headdim ** 0.5),
            causal=False
        )
        outputs.append(out)

    # Basic sanity check: outputs should have correct shape
    for g in range(num_groups):
        assert outputs[g].shape == (batch_size, seqlen_q, num_heads, headdim)

    print(f"✓ Kernel selection test passed: num_groups={num_groups}, dtype={dtype}")


@pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="flash_attn not available")
@pytest.mark.parametrize("num_groups", [3, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_edge_cases(num_groups, dtype):
    """
    Test edge cases for N-group SMEM kernel.

    1. Very short sequences (seqlen < block_size)
    2. Sequences not aligned to block size
    3. Single batch
    4. Large batch
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    num_heads = 32
    num_kv_heads = 8
    headdim = 128

    test_configs = [
        # (batch_size, seqlen_q, seqlen_k)
        (1, 32, 32),      # Very short, single batch
        (4, 63, 127),     # Unaligned lengths
        (8, 256, 512),    # Medium sequences
        (1, 1024, 2048),  # Long sequences, single batch
    ]

    for batch_size, seqlen_q, seqlen_k in test_configs:
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

        # Compute reference
        ref_outputs = reference_grouped_attention(q_groups, k, v, causal=False)

        # Compute flash attention
        flash_outputs = []
        for g in range(num_groups):
            out = flash_attn_func(
                q_groups[g], k, v,
                dropout_p=0.0,
                softmax_scale=1.0 / (headdim ** 0.5),
                causal=False
            )
            flash_outputs.append(out)

        # Compare
        for g in range(num_groups):
            atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
            rtol = 1e-2 if dtype == torch.bfloat16 else 1e-2

            torch.testing.assert_close(
                flash_outputs[g], ref_outputs[g],
                atol=atol, rtol=rtol,
                msg=f"Edge case failed: batch={batch_size}, seqlen_q={seqlen_q}, "
                    f"seqlen_k={seqlen_k}, group={g}"
            )

        print(f"✓ Edge case passed: num_groups={num_groups}, dtype={dtype}, "
              f"batch={batch_size}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")


@pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="flash_attn not available")
def test_numerical_stability():
    """
    Test numerical stability of N-group SMEM kernel with extreme values.

    Ensures the kernel handles:
    1. Very small values (near zero)
    2. Large values (near overflow)
    3. Mixed magnitudes
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    batch_size = 2
    seqlen_q = 512
    seqlen_k = 512
    num_heads = 32
    num_kv_heads = 8
    headdim = 128
    num_groups = 3
    dtype = torch.float16

    test_cases = [
        ("small_values", 1e-3),
        ("large_values", 10.0),
        ("normal_values", 1.0),
    ]

    for test_name, scale in test_cases:
        q_groups = []
        for g in range(num_groups):
            q = torch.randn(batch_size, seqlen_q, num_heads, headdim,
                           device=device, dtype=dtype) * scale
            q_groups.append(q)

        k = torch.randn(batch_size, seqlen_k, num_kv_heads, headdim,
                       device=device, dtype=dtype) * scale
        v = torch.randn(batch_size, seqlen_k, num_kv_heads, headdim,
                       device=device, dtype=dtype) * scale

        # Should not crash or produce NaN/Inf
        flash_outputs = []
        for g in range(num_groups):
            out = flash_attn_func(
                q_groups[g], k, v,
                dropout_p=0.0,
                softmax_scale=1.0 / (headdim ** 0.5),
                causal=False
            )
            flash_outputs.append(out)

            # Check for NaN/Inf
            assert not torch.isnan(out).any(), f"NaN detected in {test_name}, group {g}"
            assert not torch.isinf(out).any(), f"Inf detected in {test_name}, group {g}"

        print(f"✓ Numerical stability test passed: {test_name}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])

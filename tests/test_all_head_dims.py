"""
Comprehensive tests for all head dimensions in grouped flash attention.
Tests all head dims: 32, 64, 96, 128, 192, 256
Tests all dtypes: fp16, bf16
Tests causal and non-causal
Tests various num_groups: 2, 3, 4
"""

import pytest
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from flash_attn.flash_attn_interface import flash_attn_grouped_func


def attention_ref(q, k, v, causal=False):
    """
    Reference attention implementation using PyTorch.
    Args:
        q: (batch, seqlen_q, nheads, headdim)
        k: (batch, seqlen_k, nheads, headdim)
        v: (batch, seqlen_k, nheads, headdim)
        causal: whether to use causal masking
    Returns:
        out: (batch, seqlen_q, nheads, headdim)
    """
    q = q.transpose(1, 2)  # (batch, nheads, seqlen_q, headdim)
    k = k.transpose(1, 2)  # (batch, nheads, seqlen_k, headdim)
    v = v.transpose(1, 2)  # (batch, nheads, seqlen_k, headdim)
    
    seqlen_q, seqlen_k = q.shape[2], k.shape[2]
    headdim = q.shape[3]
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / (headdim ** 0.5)
    
    if causal:
        # Create causal mask
        mask = torch.triu(torch.ones(seqlen_q, seqlen_k, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
    
    # Softmax and weighted sum
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    
    return out.transpose(1, 2)  # (batch, seqlen_q, nheads, headdim)


# Test parameters
HEAD_DIMS = [32, 64, 96, 128, 192, 256]
DTYPES = [torch.float16, torch.bfloat16]
NUM_GROUPS = [2, 3, 4]
CAUSAL_OPTIONS = [False, True]


@pytest.mark.parametrize("headdim", HEAD_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", CAUSAL_OPTIONS)
@pytest.mark.parametrize("num_groups", NUM_GROUPS)
def test_flash_attn_grouped_all_head_dims(headdim, dtype, causal, num_groups):
    """
    Test grouped flash attention for all combinations of parameters.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = "cuda"
    batch_size = 2
    nheads = 8
    seqlen_q = 256
    seqlen_k = 256
    
    # Generate random inputs
    torch.manual_seed(42)
    
    # Create Q groups - multiple Q tensors with same K,V
    q_list = []
    for _ in range(num_groups):
        q = torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype, requires_grad=False)
        q_list.append(q)
    
    # Shared K,V
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, device=device, dtype=dtype, requires_grad=False)
    
    # Run grouped flash attention
    try:
        out_grouped_list = flash_attn_grouped_func(
            q_list, k, v,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False
        )
    except Exception as e:
        pytest.fail(f"Grouped flash attention failed with headdim={headdim}, dtype={dtype}, causal={causal}, num_groups={num_groups}: {e}")
    
    # Run reference attention for each group
    ref_out_list = []
    for q_group in q_list:
        ref_out = attention_ref(q_group, k, v, causal=causal)
        ref_out_list.append(ref_out)
    
    # Compare outputs
    rtol = 1e-2 if dtype == torch.float16 else 2e-2
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    
    for i, (out_grouped, ref_out) in enumerate(zip(out_grouped_list, ref_out_list)):
        torch.testing.assert_close(
            out_grouped, ref_out,
            rtol=rtol, atol=atol,
            msg=f"Group {i} output mismatch for headdim={headdim}, dtype={dtype}, causal={causal}, num_groups={num_groups}"
        )
    
    print(f"PASS: headdim={headdim}, dtype={dtype}, causal={causal}, num_groups={num_groups}")


@pytest.mark.parametrize("headdim", HEAD_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_flash_attn_grouped_vs_separate(headdim, dtype):
    """
    Test that grouped attention with 1 group matches separate flash attention calls.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = "cuda"
    batch_size = 2
    nheads = 8
    seqlen_q = 256
    seqlen_k = 256
    
    torch.manual_seed(42)
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, device=device, dtype=dtype, requires_grad=False)
    
    # Run grouped attention with 1 group
    out_grouped = flash_attn_grouped_func(
        [q], k, v,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False
    )[0]
    
    # Run regular flash attention
    out_separate = flash_attn_func(
        q, k, v,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False
    )
    
    # Should match exactly
    torch.testing.assert_close(
        out_grouped, out_separate,
        rtol=1e-5, atol=1e-5,
        msg=f"Grouped vs separate mismatch for headdim={headdim}, dtype={dtype}"
    )
    
    print(f"PASS: Grouped vs separate match for headdim={headdim}, dtype={dtype}")


@pytest.mark.parametrize("headdim", HEAD_DIMS)
def test_flash_attn_grouped_large_num_groups(headdim):
    """
    Test that grouped attention works with many groups (tests L2 cache-aware path).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = "cuda"
    batch_size = 1
    nheads = 4
    seqlen_q = 128
    seqlen_k = 128
    num_groups = 8  # More than 4 to trigger L2 cache-aware path
    dtype = torch.float16
    
    torch.manual_seed(42)
    
    q_list = [torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype, requires_grad=False) for _ in range(num_groups)]
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, device=device, dtype=dtype, requires_grad=False)
    
    # Run grouped flash attention
    try:
        out_grouped_list = flash_attn_grouped_func(
            q_list, k, v,
            causal=False,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False
        )
    except Exception as e:
        pytest.fail(f"Grouped flash attention failed with {num_groups} groups and headdim={headdim}: {e}")
    
    # Run reference
    ref_out_list = [attention_ref(q_group, k, v, causal=False) for q_group in q_list]
    
    # Compare
    rtol, atol = 1e-2, 1e-2
    for i, (out_grouped, ref_out) in enumerate(zip(out_grouped_list, ref_out_list)):
        torch.testing.assert_close(
            out_grouped, ref_out,
            rtol=rtol, atol=atol,
            msg=f"Group {i} output mismatch for headdim={headdim} with {num_groups} groups"
        )
    
    print(f"PASS: {num_groups} groups with headdim={headdim}")


if __name__ == "__main__":
    # Run a subset of tests for quick verification
    print("Running quick verification tests...")
    
    for headdim in [32, 64, 128, 256]:
        for dtype in [torch.float16, torch.bfloat16]:
            for causal in [False, True]:
                for num_groups in [2, 4]:
                    try:
                        test_flash_attn_grouped_all_head_dims(headdim, dtype, causal, num_groups)
                    except Exception as e:
                        print(f"FAILED: headdim={headdim}, dtype={dtype}, causal={causal}, num_groups={num_groups}")
                        print(f"Error: {e}")
    
    print("\nAll quick verification tests completed!")

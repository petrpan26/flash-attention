# Copyright (c) 2023, Tri Dao.
# Grouped Flash Attention - Python Prototype
# This module provides a Python wrapper for grouped attention, enabling multiple Q groups
# to share K,V loads for improved memory efficiency.

from typing import List, Optional, Tuple
import torch
from .flash_attn_interface import _flash_attn_varlen_forward, _flash_attn_varlen_backward
import flash_attn_2_cuda as flash_attn_cuda


def _flash_attn_varlen_forward_grouped(
    q_list: List[torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q_list: List[torch.Tensor],
    cu_seqlens_k_list: List[torch.Tensor],
    max_seqlen_q_list: List[int],
    max_seqlen_k_list: List[int],
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
    deterministic: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Flash attention with multiple Q groups sharing K,V loads.

    This is a Python prototype that calls existing _flash_attn_varlen_forward multiple times,
    once per Q group. Each group can attend to a different-length prefix of K,V.

    The key optimization is that all calls run on the same CUDA stream, allowing the GPU's
    L2 cache to potentially share K,V data between consecutive kernel launches, reducing
    redundant HBM reads.

    Args:
        q_list: List of Q tensors, one per group. Each has shape [total_q_tokens, nheads, head_dim]
        k: Shared K tensor [total_k_tokens, nheads_k, head_dim]
        v: Shared V tensor [total_k_tokens, nheads_k, head_dim]
        cu_seqlens_q_list: List of cu_seqlens_q tensors, one per group
        cu_seqlens_k_list: List of cu_seqlens_k tensors, one per group.
                           These can have different endpoint values, allowing each group
                           to attend to different K,V slice lengths.
        max_seqlen_q_list: List of max sequence lengths for Q, one per group
        max_seqlen_k_list: List of max sequence lengths for K, one per group.
                           THESE CAN BE DIFFERENT! This is the key feature.
                           Example: [4096, 8192] means group 0 attends to K,V[:4096]
                           and group 1 attends to K,V[:8192]
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for softmax. If None, uses 1/sqrt(head_dim)
        causal: Whether to apply causal masking
        window_size_left: Left window size for sliding window attention (-1 = infinite)
        window_size_right: Right window size for sliding window attention (-1 = infinite)
        softcap: Softcap value (<=0.0 means deactivated)
        alibi_slopes: ALiBi slopes tensor
        return_softmax: Whether to return softmax outputs (for dropout)
        deterministic: Whether to use deterministic implementation

    Returns:
        out_list: List of output tensors, one per group. Each has shape [q_tokens, nheads, head_dim]
        lse_list: List of LSE (log-sum-exp) tensors, one per group
        S_dmask: Dropout mask from last group (if return_softmax=True)
        rng_state: RNG state from last group

    Example:
        # Two groups with different K,V lengths
        q_early = torch.randn(1000, 32, 128, device='cuda', dtype=torch.float16)
        q_late = torch.randn(1000, 32, 128, device='cuda', dtype=torch.float16)
        k = torch.randn(2000, 8, 128, device='cuda', dtype=torch.float16)
        v = torch.randn(2000, 8, 128, device='cuda', dtype=torch.float16)

        out_list, lse_list, _, _ = _flash_attn_varlen_forward_grouped(
            q_list=[q_early, q_late],
            k=k,
            v=v,
            cu_seqlens_q_list=[cu_seqlens_q_early, cu_seqlens_q_late],
            cu_seqlens_k_list=[cu_seqlens_k_early, cu_seqlens_k_late],
            max_seqlen_q_list=[max_seqlen_q, max_seqlen_q],
            max_seqlen_k_list=[1000, 2000],  # Group 0 attends to first 1000, group 1 to all 2000
            softmax_scale=1.0/math.sqrt(128),
            causal=True,
        )
    """
    assert len(q_list) > 0, "Must provide at least one Q group"
    assert len(q_list) == len(cu_seqlens_q_list) == len(cu_seqlens_k_list), \
        "q_list, cu_seqlens_q_list, and cu_seqlens_k_list must have same length"
    assert len(max_seqlen_q_list) == len(max_seqlen_k_list) == len(q_list), \
        "max_seqlen_q_list and max_seqlen_k_list must match number of groups"

    # Validate that all Q tensors are on the same device as K,V
    device = k.device
    for i, q_group in enumerate(q_list):
        assert q_group.device == device, f"q_list[{i}] must be on same device as K,V"

    # Initialize output lists
    out_list = []
    lse_list = []
    S_dmask = None
    rng_state = None

    # Process each group sequentially on the same CUDA stream
    # This allows L2 cache sharing between consecutive kernel launches
    for i, q_group in enumerate(q_list):
        # Skip empty groups
        if q_group.shape[0] == 0:
            # Create empty outputs with correct shape
            out_empty = torch.zeros_like(q_group)
            # LSE shape depends on the version of flash attention
            # Try to infer from a dummy call or use standard shape
            lse_empty = torch.zeros(
                (q_group.shape[1], q_group.shape[0]),  # [nheads, total_q]
                dtype=torch.float32,
                device=device
            )
            out_list.append(out_empty)
            lse_list.append(lse_empty)
            continue

        # Get K,V slice for this group
        k_end = cu_seqlens_k_list[i][-1].item()
        k_slice = k[:k_end]
        v_slice = v[:k_end]

        # Call flash attention for this group
        out_group, lse_group, S_dmask_group, rng_state_group = _flash_attn_varlen_forward(
            q=q_group,
            k=k_slice,
            v=v_slice,
            cu_seqlens_q=cu_seqlens_q_list[i],
            cu_seqlens_k=cu_seqlens_k_list[i],
            max_seqlen_q=max_seqlen_q_list[i],
            max_seqlen_k=max_seqlen_k_list[i],
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )

        out_list.append(out_group)
        lse_list.append(lse_group)

        # Keep last group's dropout mask and rng_state for return value
        S_dmask = S_dmask_group
        rng_state = rng_state_group

    return out_list, lse_list, S_dmask, rng_state


def _flash_attn_varlen_backward_grouped(
    dout_list: List[torch.Tensor],
    q_list: List[torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    out_list: List[torch.Tensor],
    softmax_lse_list: List[torch.Tensor],
    cu_seqlens_q_list: List[torch.Tensor],
    cu_seqlens_k_list: List[torch.Tensor],
    max_seqlen_q_list: List[int],
    max_seqlen_k_list: List[int],
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    rng_state: Optional[torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Flash attention backward pass with multiple Q groups sharing K,V loads.

    This function computes gradients for grouped flash attention, where multiple Q groups
    share the same K,V tensors. Each group can attend to different-length prefixes of K,V.

    The key optimization is that dK and dV are accumulated across all groups, reducing
    redundant gradient computations.

    Args:
        dout_list: List of gradient tensors w.r.t. outputs [total_q_tokens, nheads, head_dim] per group
        q_list: List of Q tensors, one per group [total_q_tokens, nheads, head_dim]
        k: Shared K tensor [total_k_tokens, nheads_k, head_dim]
        v: Shared V tensor [total_k_tokens, nheads_k, head_dim]
        out_list: List of forward pass outputs, one per group
        softmax_lse_list: List of LSE tensors from forward pass, one per group
        cu_seqlens_q_list: List of cu_seqlens_q tensors, one per group
        cu_seqlens_k_list: List of cu_seqlens_k tensors, one per group
        max_seqlen_q_list: List of max Q sequence lengths, one per group
        max_seqlen_k_list: List of max K,V lengths per group (can be different!)
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for softmax. If None, uses 1/sqrt(head_dim)
        causal: Whether to apply causal masking
        window_size_left: Left window size for sliding window attention (-1 = infinite)
        window_size_right: Right window size for sliding window attention (-1 = infinite)
        softcap: Softcap value (<=0.0 means deactivated)
        alibi_slopes: ALiBi slopes tensor
        deterministic: Whether to use deterministic implementation
        rng_state: RNG state from forward pass (for dropout)

    Returns:
        dq_list: List of gradient tensors w.r.t. Q, one per group [q_tokens, nheads, head_dim]
        dk: Gradient tensor w.r.t. K [total_k_tokens, nheads_k, head_dim]
        dv: Gradient tensor w.r.t. V [total_k_tokens, nheads_k, head_dim]

    Example:
        # Two groups with different K,V lengths
        dout_early = torch.randn_like(out_early)
        dout_late = torch.randn_like(out_late)

        dq_list, dk, dv = _flash_attn_varlen_backward_grouped(
            dout_list=[dout_early, dout_late],
            q_list=[q_early, q_late],
            k=k,
            v=v,
            out_list=[out_early, out_late],
            softmax_lse_list=[lse_early, lse_late],
            cu_seqlens_q_list=[cu_seqlens_q_early, cu_seqlens_q_late],
            cu_seqlens_k_list=[cu_seqlens_k_early, cu_seqlens_k_late],
            max_seqlen_q_list=[max_seqlen_q, max_seqlen_q],
            max_seqlen_k_list=[1000, 2000],  # Group 0: first 1000, group 1: all 2000
            softmax_scale=1.0/math.sqrt(128),
            causal=True,
        )
    """
    assert len(q_list) > 0, "Must provide at least one Q group"
    assert len(dout_list) == len(q_list), "dout_list and q_list must have same length"
    assert len(out_list) == len(q_list), "out_list and q_list must have same length"
    assert len(softmax_lse_list) == len(q_list), "softmax_lse_list and q_list must have same length"
    assert len(cu_seqlens_q_list) == len(cu_seqlens_k_list) == len(q_list), \
        "cu_seqlens lists must match number of groups"
    assert len(max_seqlen_q_list) == len(max_seqlen_k_list) == len(q_list), \
        "max_seqlen lists must match number of groups"

    # Validate device consistency
    device = k.device
    for i, (q_group, dout_group) in enumerate(zip(q_list, dout_list)):
        assert q_group.device == device, f"q_list[{i}] must be on same device as K,V"
        assert dout_group.device == device, f"dout_list[{i}] must be on same device as K,V"

    if softmax_scale is None:
        softmax_scale = q_list[0].shape[-1] ** (-0.5)

    # Call the C++ grouped backward implementation
    results = flash_attn_cuda.varlen_bwd_grouped(
        dout_list,
        q_list,
        k,
        v,
        out_list,
        softmax_lse_list,
        cu_seqlens_q_list,
        cu_seqlens_k_list,
        max_seqlen_q_list,
        max_seqlen_k_list,
        dropout_p,
        softmax_scale,
        False,  # zero_tensors
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        None,  # gen
    )

    # Results format: [dq0, dq1, ..., dqN, dk, dv]
    num_groups = len(q_list)
    dq_list = results[:num_groups]
    dk = results[num_groups]
    dv = results[num_groups + 1]

    return dq_list, dk, dv

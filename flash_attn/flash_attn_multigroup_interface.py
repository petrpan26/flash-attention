"""
Multi-group varlen attention interface.

This module provides multi-group flash attention for variable-length sequences,
optimized for zigzag attention patterns where different Q groups need different
K,V slices.
"""

from typing import List, Optional, Tuple
import torch


def _flash_attn_varlen_multigroup_forward(
    q_list: List[torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q_list: List[torch.Tensor],
    cu_seqlens_k_list: List[torch.Tensor],
    kv_endpoints: torch.Tensor,
    max_seqlen_q_list: List[int],
    max_seqlen_k_list: List[int],
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Multi-group varlen flash attention forward pass.

    Processes multiple Q groups against shared K,V with group-specific endpoints.
    In the mock implementation, this calls flash attention sequentially per group.
    The real CUDA implementation will load each K,V tile once.

    Args:
        q_list: List of Q tensors, one per group
                Each has shape [tokens_in_group, nheads, head_dim]
        k: Shared K tensor in contiguous format
           Shape: [total_tokens, nheads_k, head_dim]
        v: Shared V tensor in contiguous format
           Shape: [total_tokens, nheads_k, head_dim]
        cu_seqlens_q_list: Cumulative sequence lengths for each Q group
                          Each is int32 tensor of shape [num_seqs+1]
        cu_seqlens_k_list: Cumulative sequence lengths for KV slice per group
                          Each is int32 tensor of shape [num_seqs+1]
        kv_endpoints: Maximum K,V position each group/sequence needs
                     Shape: [num_groups, num_seqs], dtype: int32
                     kv_endpoints[g, s] = end position for group g, seq s
        max_seqlen_q_list: List of maximum sequence lengths in each Q group
        max_seqlen_k_list: List of maximum KV sequence lengths for each group
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for attention scores (default: 1/sqrt(d))
        causal: Whether to use causal masking
        window_size: Sliding window size (left, right)
        alibi_slopes: ALiBi positional encoding slopes
        return_softmax: Whether to return softmax probabilities

    Returns:
        out_list: List of output tensors, one per group
                  Each has shape [tokens_in_group, nheads, head_dim]
        lse_list: List of LSE tensors, one per group
                  Each has shape [nheads, tokens_in_group] or [tokens_in_group, nheads]
    """
    # Import here to avoid circular dependency
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward

    num_groups = len(q_list)
    out_list = []
    lse_list = []

    # Mock implementation: Sequential calls to standard flash attention
    for g in range(num_groups):
        # Extract K,V slice for this group based on kv_endpoints
        # For simplicity in mock, extract full range [0:max_endpoint]
        max_kv_end = kv_endpoints[g].max().item()

        k_slice = k[:max_kv_end]
        v_slice = v[:max_kv_end]

        # Call standard flash attention
        out_g, lse_g, _, _ = _flash_attn_varlen_forward(
            q=q_list[g],
            k=k_slice,
            v=v_slice,
            cu_seqlens_q=cu_seqlens_q_list[g],
            cu_seqlens_k=cu_seqlens_k_list[g],
            max_seqlen_q=max_seqlen_q_list[g],
            max_seqlen_k=max_seqlen_k_list[g],
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )

        out_list.append(out_g)
        lse_list.append(lse_g)

    return out_list, lse_list


def _flash_attn_varlen_multigroup_backward(
    dout_list: List[torch.Tensor],
    q_list: List[torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    out_list: List[torch.Tensor],
    softmax_lse_list: List[torch.Tensor],
    cu_seqlens_q_list: List[torch.Tensor],
    cu_seqlens_k_list: List[torch.Tensor],
    kv_endpoints: torch.Tensor,
    max_seqlen_q_list: List[int],
    max_seqlen_k_list: List[int],
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Multi-group varlen flash attention backward pass.

    Mock implementation: Calls flash attention backward sequentially per group
    and accumulates dK, dV gradients.

    Args:
        dout_list: List of output gradients, one per group
        q_list, k, v: Forward inputs
        out_list: Forward outputs
        softmax_lse_list: Forward LSE outputs
        cu_seqlens_q_list, cu_seqlens_k_list: Sequence length metadata
        kv_endpoints: KV endpoints from forward
        max_seqlen_q_list, max_seqlen_k_list: Max sequence lengths
        (other args same as forward)

    Returns:
        dq_list: List of Q gradients, one per group
        dk: K gradient [total_tokens, nheads_k, head_dim] - accumulated
        dv: V gradient [total_tokens, nheads_k, head_dim] - accumulated
    """
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward

    num_groups = len(q_list)

    # Initialize dK, dV accumulators
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    dq_list = []

    # Process each group
    for g in range(num_groups):
        # Extract K,V slice
        max_kv_end = kv_endpoints[g].max().item()
        k_slice = k[:max_kv_end]
        v_slice = v[:max_kv_end]

        # Call standard flash attention backward
        dq_g, dk_slice, dv_slice = _flash_attn_varlen_backward(
            dout=dout_list[g],
            q=q_list[g],
            k=k_slice,
            v=v_slice,
            out=out_list[g],
            softmax_lse=softmax_lse_list[g],
            cu_seqlens_q=cu_seqlens_q_list[g],
            cu_seqlens_k=cu_seqlens_k_list[g],
            max_seqlen_q=max_seqlen_q_list[g],
            max_seqlen_k=max_seqlen_k_list[g],
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
        )

        dq_list.append(dq_g)

        # Accumulate dK, dV (overlapping regions get contributions from multiple groups)
        dk[:max_kv_end] += dk_slice
        dv[:max_kv_end] += dv_slice

    return dq_list, dk, dv


class MultiGroupFlashAttnVarlenFunc(torch.autograd.Function):
    """
    AutoGrad wrapper for multi-group varlen flash attention.

    Usage:
        out_list, lse_list = MultiGroupFlashAttnVarlenFunc.apply(
            *q_list, k, v, *cu_seqlens_q_list, *cu_seqlens_k_list,
            kv_endpoints, max_seqlen_q_list, max_seqlen_k_list,
            dropout_p, softmax_scale, causal, window_size, alibi_slopes, deterministic
        )
    """

    @staticmethod
    def forward(
        ctx,
        k: torch.Tensor,
        v: torch.Tensor,
        q_list: List[torch.Tensor],
        cu_seqlens_q_list: List[torch.Tensor],
        cu_seqlens_k_list: List[torch.Tensor],
        kv_endpoints: torch.Tensor,
        max_seqlen_q_list: List[int],
        max_seqlen_k_list: List[int],
        dropout_p: float,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
    ):
        if softmax_scale is None:
            softmax_scale = q_list[0].shape[-1] ** (-0.5)

        # Forward pass
        out_list, lse_list = _flash_attn_varlen_multigroup_forward(
            q_list=q_list,
            k=k,
            v=v,
            cu_seqlens_q_list=cu_seqlens_q_list,
            cu_seqlens_k_list=cu_seqlens_k_list,
            kv_endpoints=kv_endpoints,
            max_seqlen_q_list=max_seqlen_q_list,
            max_seqlen_k_list=max_seqlen_k_list,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            return_softmax=False,
        )

        # Save for backward
        ctx.save_for_backward(k, v, *q_list, *out_list, *lse_list, *cu_seqlens_q_list, *cu_seqlens_k_list, kv_endpoints)
        ctx.num_groups = len(q_list)
        ctx.max_seqlen_q_list = max_seqlen_q_list
        ctx.max_seqlen_k_list = max_seqlen_k_list
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic

        return out_list, lse_list

    @staticmethod
    def backward(ctx, *grad_outputs):
        num_groups = ctx.num_groups

        # Unpack gradient outputs
        grad_out_list = list(grad_outputs[:num_groups])
        # grad_lse_list = list(grad_outputs[num_groups:2*num_groups])  # Not used

        # Unpack saved tensors
        saved = ctx.saved_tensors
        k = saved[0]
        v = saved[1]
        q_list = list(saved[2:2+num_groups])
        out_list = list(saved[2+num_groups:2+2*num_groups])
        lse_list = list(saved[2+2*num_groups:2+3*num_groups])
        cu_seqlens_q_list = list(saved[2+3*num_groups:2+4*num_groups])
        cu_seqlens_k_list = list(saved[2+4*num_groups:2+5*num_groups])
        kv_endpoints = saved[2+5*num_groups]

        # Backward pass
        dq_list, dk, dv = _flash_attn_varlen_multigroup_backward(
            dout_list=grad_out_list,
            q_list=q_list,
            k=k,
            v=v,
            out_list=out_list,
            softmax_lse_list=lse_list,
            cu_seqlens_q_list=cu_seqlens_q_list,
            cu_seqlens_k_list=cu_seqlens_k_list,
            kv_endpoints=kv_endpoints,
            max_seqlen_q_list=ctx.max_seqlen_q_list,
            max_seqlen_k_list=ctx.max_seqlen_k_list,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )

        # Return gradients
        # Order: k, v, q_list, cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints, ...
        return (
            dk, dv,                              # k, v gradients
            *dq_list,                            # q_list gradients
            *([None] * num_groups),              # cu_seqlens_q_list (no grad)
            *([None] * num_groups),              # cu_seqlens_k_list (no grad)
            None,                                # kv_endpoints (no grad)
            None, None,                          # max_seqlen_q_list, max_seqlen_k_list
            None, None, None, None, None, None,  # dropout_p, softmax_scale, causal, window_size, alibi_slopes, deterministic
        )

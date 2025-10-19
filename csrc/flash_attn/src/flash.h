/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <cuda.h>
#include <vector>

#include <ATen/cuda/CUDAGeneratorImpl.h> // For at::Generator and at::PhiloxCudaState

namespace FLASH_NAMESPACE {
constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ leftpad_k;

    // If provided, the actual length of each k sequence.
    int * __restrict__ seqused_k;

    int *__restrict__ blockmask;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    // The indices to index into the KV cache.
    int * __restrict__ cache_batch_idx;

    // Paged KV cache
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_softmax_rp_dropout;

    // Local window size
    int window_size_left, window_size_right;
    float softcap;

    // Random state.
    at::PhiloxCudaState philox_args;

    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    uint64_t * rng_state;

    bool is_bf16;
    bool is_causal;

    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    bool is_seqlens_k_cumulative;

    bool is_rotary_interleaved;

    int num_splits;  // For split-KV version

    void * __restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;

    bool unpadded_lse;  // For varlen paths: LSE is in [nheads, total_seqlen_q] format instead of [b, nheads, seqlen_q].
    bool seqlenq_ngroups_swapped;  // q has been transposed from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d).

    // Grouped attention support: multiple Q groups sharing K,V loads
    int num_groups;                          // Number of Q groups (e.g., 2 for early/late split)
    void** group_q_ptrs;                     // Device pointer: array of Q pointers, one per group
    int** group_cu_seqlens_q;                // Device pointer: array of cu_seqlens_q pointers, one per group
    int** group_cu_seqlens_k;                // Device pointer: array of cu_seqlens_k pointers, one per group
    int* group_num_m_blocks;                 // Device pointer: number of M blocks per group [blocks_g0, blocks_g1, ...]
    int* group_max_seqlen_k;                 // Device pointer: max K,V length per group [tokens_early, tokens_late]
    void** group_o_ptrs;                     // Device pointer: array of output pointers, one per group
    void** group_softmax_lse_ptrs;           // Device pointer: array of LSE pointers, one per group
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params {

    // The dO and dQKV matrices.
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // To accumulate dQ
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

    // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
    // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
    // dv_accum_ptr;

    // The stride between rows of the dO, dQ, dK and dV matrices.
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    // The pointer to the softmax d sum.
    void *__restrict__ dsoftmax_sum;

    bool deterministic;
    index_t dq_accum_split_stride;

    // ============================================================================
    // Grouped Attention Backward Pass Support
    // ============================================================================
    // Multiple Q groups share K,V loads during backward pass to compute gradients
    // efficiently. Each group has separate Q, dO, O, dQ, LSE, and softmax_d, while
    // K, V, dK, dV are shared across all groups.
    //
    // Memory Layout:
    //   - All group pointer arrays are allocated in device memory
    //   - Host code allocates arrays and copies pointers to device
    //   - Kernel accesses via params.group_*_ptrs[group_id]
    //
    // Usage Example:
    //   num_groups = 2 (e.g., early tokens vs late tokens)
    //   group_do_ptrs[0] -> dO for group 0
    //   group_do_ptrs[1] -> dO for group 1
    //   dk_ptr -> shared dK output (accumulated from both groups)
    //
    // Gradient Flow:
    //   For each K/V block n:
    //     Load K[n], V[n] once (shared)
    //     Initialize dK_accum = 0, dV_accum = 0
    //     For each group g:
    //       For each Q block m in group g:
    //         Load Q_g[m], dO_g[m], O_g[m] (needed for recomputation)
    //         Recompute S_g = Q_g[m] @ K[n]^T
    //         Recompute P_g = softmax(S_g) using saved LSE_g[m]
    //         Compute dP_g = dO_g[m] @ V[n]^T
    //         Compute dS_g = P_g * (dP_g - softmax_d_g[m])
    //         Accumulate dK += dS_g^T @ Q_g[m]  (shared accumulator)
    //         Accumulate dV += P_g^T @ dO_g[m]  (shared accumulator)
    //         Compute dQ_g[m] += dS_g @ K[n]    (per-group output)
    //     Write accumulated dK[n], dV[n]
    //
    // Key Differences from Forward Pass:
    //   - Forward: iterate K/V blocks for each Q block → accumulate O
    //   - Backward: iterate Q blocks for each K/V block → accumulate dK/dV
    //   - Forward: compute LSE for numerical stability
    //   - Backward: use saved LSE to recompute attention probabilities P
    //   - Forward: direct computation
    //   - Backward: need softmax_d = sum(dO * O) for gradient through softmax
    // ============================================================================

    // Input gradients (per-group)
    void** group_do_ptrs;               // Device pointer: [num_groups] array of dO pointers
                                        // Each points to gradient w.r.t. output for one group
                                        // Shape per group: [B, seqlen_q_group, H, D]
                                        // Used in main backward kernel to compute dP = dO @ V^T

    // Output gradients (per-group for dQ, shared for dK/dV)
    void** group_dq_ptrs;               // Device pointer: [num_groups] array of dQ pointers
                                        // Each points to gradient w.r.t. Q for one group
                                        // Shape per group: [B, seqlen_q_group, H, D]
                                        // Updated via: dQ_g[m] += dS_g @ K

    // Note: dk_ptr and dv_ptr (already defined above) are SHARED across all groups
    // They accumulate gradients from all groups:
    //   dK = sum over groups g: dS_g^T @ Q_g
    //   dV = sum over groups g: P_g^T @ dO_g
    // Shape: [B, seqlen_k, H, D] (same K/V for all groups)

    // Intermediate values (per-group)
    void** group_dsoftmax_sum_ptrs;     // Device pointer: [num_groups] array of softmax_d pointers
                                        // softmax_d[i] = sum_j(dO[i,j] * O[i,j]) for each query row
                                        // Required for gradient through softmax: dS = P * (dP - softmax_d)
                                        // Shape per group: [B, H, seqlen_q_rounded_group] or [H, total_q + 128*B]
                                        // Computed by preprocessing kernel (compute_dot_do_o)
                                        // Layout matches LSE for consistency (unpadded_lse flag)

    // Accumulator buffers (per-group)
    void** group_dq_accum_ptrs;         // Device pointer: [num_groups] array of dQ accumulator pointers
                                        // Used in sequence-parallel and deterministic modes
                                        // Stores fp32 gradients before final conversion to fp16/bf16
                                        // Shape per group: [B, seqlen_q_rounded + 128*B, H, D]
                                        // Padding (128*B) prevents false sharing in atomic operations
                                        // If deterministic: each seqk split writes to separate region
                                        // Converted to final dQ by convert_dQ kernel

    // Intermediate buffers for hybrid approach (optional, may not be used)
    void** group_dk_intermediate_ptrs;  // Device pointer: intermediate dK buffers [num_groups]
                                        // Only used if implementing separate-then-sum strategy
                                        // Each buffer: [B, seqlen_k, H, D] in fp32
                                        // Final dK = sum over groups: dk_intermediate[g]
                                        // Typically NULL - prefer direct accumulation in registers

    void** group_dv_intermediate_ptrs;  // Device pointer: intermediate dV buffers [num_groups]
                                        // Only used if implementing separate-then-sum strategy
                                        // Each buffer: [B, seqlen_k, H, D] in fp32
                                        // Final dV = sum over groups: dv_intermediate[g]
                                        // Typically NULL - prefer direct accumulation in registers

    // Note: The following group-specific metadata is inherited from Flash_fwd_params:
    //   - num_groups: Number of Q groups
    //   - group_q_ptrs: [num_groups] Q pointers (needed for dK computation)
    //   - group_o_ptrs: [num_groups] O pointers (needed for softmax_d computation)
    //   - group_cu_seqlens_q: [num_groups] cumulative sequence length pointers
    //   - group_num_m_blocks: [num_groups] number of M blocks per group
    //   - group_softmax_lse_ptrs: [num_groups] LSE pointers (from forward pass, for recomputing P)
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_grouped_(Flash_fwd_params &params, cudaStream_t stream);

template<typename T, int Headdim, bool Is_causal> void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);
template<typename T, int Headdim, bool Is_causal> void run_mha_bwd_grouped_(Flash_bwd_params &params, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE

/******************************************************************************
 * Copyright (c) 2024, Multi-Group Flash Attention Implementation.
 *
 * Multi-group varlen attention parameter structures.
 *
 * This file defines the parameter structures for multi-group flash attention
 * kernels that process multiple Q groups with shared K,V tensors in a single
 * kernel launch, reducing redundant memory loads.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <cuda.h>
#include <vector>

#include <ATen/cuda/CUDAGeneratorImpl.h>

namespace FLASH_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-Group Forward Parameters
////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_multigroup_params {
    using index_t = int64_t;

    // ============================================================================
    // Per-Group Q Pointers (Array of Pointers)
    // ============================================================================

    // Array of Q tensor pointers, one per group
    // Each Q tensor: [total_q_tokens_for_group, nheads, head_dim]
    void **q_ptr_list;
    int num_groups;  // Number of Q groups (e.g., 2 for zigzag_llama3)

    // Array of Q strides for each group
    // For varlen: q_batch_stride is unused, indexing via cu_seqlens_q
    index_t *q_row_stride_list;    // Array of row strides
    index_t *q_head_stride_list;   // Array of head strides

    // ============================================================================
    // Shared K, V Pointers (Single Tensor per Type)
    // ============================================================================

    // Shared K tensor: [total_k_tokens, nheads_k, head_dim]
    // All groups attend to (potentially different subsets of) this K
    void *__restrict__ k_ptr;

    // Shared V tensor: [total_v_tokens, nheads_k, head_dim]
    // All groups attend to (potentially different subsets of) this V
    void *__restrict__ v_ptr;

    // K, V strides (shared across all groups)
    index_t k_row_stride;
    index_t k_head_stride;
    index_t v_row_stride;
    index_t v_head_stride;

    // ============================================================================
    // Per-Group Output Pointers (Array of Pointers)
    // ============================================================================

    // Array of output tensor pointers, one per group
    // Each O tensor: [total_q_tokens_for_group, nheads, head_dim]
    void **out_ptr_list;

    // Array of output strides for each group
    index_t *o_row_stride_list;
    index_t *o_head_stride_list;

    // ============================================================================
    // Per-Group LSE Pointers (Array of Pointers)
    // ============================================================================

    // Array of LSE (log-sum-exp) tensor pointers, one per group
    // Format depends on unpadded_lse flag:
    //   If unpadded_lse: [nheads, total_q_tokens_for_group]
    //   Otherwise:       [batch_size, nheads, max_seqlen_q]
    void **softmax_lse_ptr_list;

    // ============================================================================
    // Sequence Length Metadata (Per-Group Arrays)
    // ============================================================================

    // Array of cu_seqlens_q pointers, one per group
    // Each cu_seqlens_q: [batch_size + 1], cumulative Q sequence lengths
    int **cu_seqlens_q_list;

    // Array of cu_seqlens_k pointers, one per group
    // Each cu_seqlens_k: [batch_size + 1], cumulative K sequence lengths
    // Note: K,V data is shared, but different groups may have different valid ranges
    int **cu_seqlens_k_list;

    // Array of max sequence lengths per group (for allocation sizing)
    int *max_seqlen_q_list;   // [num_groups]
    int *max_seqlen_k_list;   // [num_groups]

    // ============================================================================
    // KV Endpoint Control (Per-Group, Per-Batch)
    // ============================================================================

    // KV endpoint tensor: [num_groups, batch_size]
    // kv_endpoints[g][b] = maximum K,V position that group g should attend to
    //                      for batch b
    // Example: Group 0 attends to K[0:256], Group 1 attends to K[0:512]
    //          kv_endpoints = [[256, 256, ...], [512, 512, ...]]
    int *kv_endpoints;

    // Batch size (same across all groups)
    int batch_size;

    // ============================================================================
    // Common Attention Parameters
    // ============================================================================

    // Number of query heads
    int h;

    // Number of key/value heads (for MQA/GQA)
    int h_k;

    // Ratio h / h_k (precomputed for efficiency)
    int h_h_k_ratio;

    // Head dimension (must be same for all groups)
    int head_size;

    // Head dimension rounded up to multiple of 32
    int head_size_rounded;

    // Total number of Q tokens across all sequences (per group)
    // Used for unpadded_lse indexing
    int *total_q_list;  // [num_groups]

    // ============================================================================
    // Softmax Scaling
    // ============================================================================

    // Softmax scaling factor: 1/sqrt(head_dim)
    float scale_softmax;

    // Softmax scaling in log2 domain for exp2 optimization
    float scale_softmax_log2;

    // Softcap value (0.0 = disabled)
    float softcap;

    // ============================================================================
    // Masking Parameters
    // ============================================================================

    // Causal masking: true = only attend to past/present positions
    bool is_causal;

    // Local attention window sizes (-1 = infinite)
    int window_size_left;
    int window_size_right;

    // ============================================================================
    // Dropout Parameters
    // ============================================================================

    // Dropout probability (probability of keeping an element)
    float p_dropout;
    uint8_t p_dropout_in_uint8_t;

    // Reciprocal dropout scaling factor
    float rp_dropout;
    float scale_softmax_rp_dropout;

    // Random number generator state
    at::PhiloxCudaState philox_args;
    uint64_t *rng_state;

    // ============================================================================
    // Alibi Slopes (Optional)
    // ============================================================================

    // Alibi slopes pointer (nullptr if not used)
    void *__restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;

    // ============================================================================
    // Format Flags
    // ============================================================================

    // Data type flags
    bool is_bf16;  // true = BFloat16, false = Float16

    // LSE format flag
    // If true: LSE is [nheads, total_q] (for varlen)
    // If false: LSE is [batch, nheads, max_seqlen_q]
    bool unpadded_lse;

    // Sequence length format flag
    // If true: cu_seqlens_k stores cumulative sums
    // If false: cu_seqlens_k stores individual lengths
    bool is_seqlens_k_cumulative;

    // ============================================================================
    // Debug and Profiling (Optional)
    // ============================================================================

    // Pointer to debug buffer for kernel diagnostics (nullptr = disabled)
    void *debug_buffer;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-Group Backward Parameters
////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_multigroup_params : public Flash_fwd_multigroup_params {

    // ============================================================================
    // Per-Group Gradient Input Pointers (dO)
    // ============================================================================

    // Array of dO (gradient of output) pointers, one per group
    // Each dO tensor: [total_q_tokens_for_group, nheads, head_dim]
    void **do_ptr_list;

    // Array of dO strides for each group
    index_t *do_row_stride_list;
    index_t *do_head_stride_list;

    // ============================================================================
    // Per-Group Q Gradient Output Pointers (dQ)
    // ============================================================================

    // Array of dQ (gradient of Q) pointers, one per group
    // Each dQ tensor: [total_q_tokens_for_group, nheads, head_dim]
    void **dq_ptr_list;

    // Array of dQ strides for each group
    index_t *dq_row_stride_list;
    index_t *dq_head_stride_list;

    // Optional: dQ accumulator for split-backward parallelization
    void **dq_accum_ptr_list;

    // ============================================================================
    // Shared K,V Gradient Output Pointers (dK, dV)
    // ============================================================================

    // Gradient of K: [total_k_tokens, nheads_k, head_dim]
    // IMPORTANT: Multiple groups may contribute gradients to same K positions
    //            Requires accumulation (atomic or two-pass reduction)
    void *__restrict__ dk_ptr;

    // Gradient of V: [total_v_tokens, nheads_k, head_dim]
    // IMPORTANT: Multiple groups may contribute gradients to same V positions
    //            Requires accumulation (atomic or two-pass reduction)
    void *__restrict__ dv_ptr;

    // K, V gradient strides
    index_t dk_row_stride;
    index_t dk_head_stride;
    index_t dv_row_stride;
    index_t dv_head_stride;

    // Optional: Accumulators for two-pass gradient reduction
    // If nullptr: use atomic adds directly to dk_ptr, dv_ptr
    // If non-null: write per-group gradients separately, then reduce
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

    // ============================================================================
    // Softmax Backward State
    // ============================================================================

    // Pointer to dsoftmax_sum (D_i = rowsum(dO * O) for each row)
    // Array of pointers, one per group
    void **dsoftmax_sum_list;

    // ============================================================================
    // Backward-Specific Flags
    // ============================================================================

    // Deterministic mode: use two-pass reduction instead of atomics
    // Slower but reproducible across runs
    bool deterministic;

    // Stride for dq_accum split accumulation
    index_t dq_accum_split_stride;

    // ============================================================================
    // Gradient Accumulation Strategy
    // ============================================================================

    // Strategy for dK, dV gradient accumulation:
    //   0 = Atomic adds (fast but non-deterministic)
    //   1 = Two-pass reduction (deterministic, requires extra memory)
    //   2 = Warp-level reduction before atomics (balanced)
    int grad_accum_strategy;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Launch Declarations (to be implemented in Phase 2)
////////////////////////////////////////////////////////////////////////////////////////////////////

// Forward kernel launcher
template<typename T, int Headdim, int NumGroups, bool Is_causal>
void run_mha_fwd_multigroup(Flash_fwd_multigroup_params &params, cudaStream_t stream);

// Backward kernel launcher
template<typename T, int Headdim, int NumGroups, bool Is_causal>
void run_mha_bwd_multigroup(Flash_bwd_multigroup_params &params, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

// Validate parameters before kernel launch
inline void validate_multigroup_params(const Flash_fwd_multigroup_params &params) {
    assert(params.num_groups >= 1 && params.num_groups <= 8 &&
           "num_groups must be between 1 and 8");
    assert(params.q_ptr_list != nullptr && "q_ptr_list cannot be null");
    assert(params.k_ptr != nullptr && "k_ptr cannot be null");
    assert(params.v_ptr != nullptr && "v_ptr cannot be null");
    assert(params.kv_endpoints != nullptr && "kv_endpoints cannot be null");
    assert(params.batch_size > 0 && "batch_size must be positive");
    assert(params.h > 0 && params.h_k > 0 && "h and h_k must be positive");
    assert(params.head_size > 0 && params.head_size % 32 == 0 &&
           "head_size must be positive and multiple of 32");
}

// Compute shared memory requirements for given configuration
inline size_t compute_smem_size_multigroup(
    int num_groups,
    int kBlockM,
    int kBlockN,
    int head_dim,
    bool share_q_k_smem = false
) {
    // Element size (2 bytes for FP16/BF16)
    const size_t elem_size = 2;

    // Per-group Q tiles
    size_t smem_q = num_groups * kBlockM * head_dim * elem_size;

    // Shared K,V tiles
    size_t smem_kv = 2 * kBlockN * head_dim * elem_size;

    // Total
    size_t total = share_q_k_smem ? std::max(smem_q, smem_kv) : smem_q + smem_kv;

    return total;
}

}  // namespace FLASH_NAMESPACE

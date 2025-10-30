/******************************************************************************
 * Copyright (c) 2024, Multi-Group Flash Attention Implementation.
 *
 * Backward kernel template for multi-group varlen attention.
 *
 * STUB FILE FOR PHASE 3 IMPLEMENTATION
 *
 * This file contains the kernel template signatures and detailed notes
 * for implementing the multi-group backward attention kernel, particularly
 * focusing on gradient accumulation for shared K,V tensors.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <cute/tensor.hpp>

#include "flash_multigroup.h"
#include "kernel_traits.h"
#include "utils.h"

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Backward Kernel Overview
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * BACKWARD KERNEL CHALLENGES:
 *
 * 1. Gradient Accumulation Problem:
 *    Multiple groups attend to overlapping K,V regions, so their gradients
 *    must be accumulated into shared dK, dV tensors.
 *
 *    Example:
 *      Group 0 attends to K[0:256], contributes dK_0[0:256]
 *      Group 1 attends to K[0:512], contributes dK_1[0:512]
 *      Final dK[0:256] = dK_0[0:256] + dK_1[0:256]
 *      Final dK[256:512] = dK_1[256:512]
 *
 * 2. Atomic vs. Two-Pass Accumulation:
 *
 *    Approach A: Atomic Adds (Fast but Non-Deterministic)
 *      - Each group atomically adds its gradients to global dK, dV
 *      - Pros: Simple, minimal memory overhead
 *      - Cons: Non-deterministic order, potential contention
 *      - Use case: Training (where determinism not required)
 *
 *    Approach B: Two-Pass Reduction (Deterministic)
 *      - Pass 1: Write per-group gradients to separate buffers
 *        dK_accum[g * K_size + k] = dK_g[k]
 *      - Pass 2: Reduction kernel sums across groups
 *        dK[k] = sum_g dK_accum[g * K_size + k]
 *      - Pros: Deterministic, no contention
 *      - Cons: Extra memory (num_groups × K_size), two kernel launches
 *      - Use case: Debugging, gradient checking
 *
 * 3. Per-Group dQ:
 *    Each group has separate dQ output (no accumulation needed)
 *    dQ_g: gradient w.r.t. Q for group g
 *
 * BACKWARD ALGORITHM (Standard Flash Attention):
 *
 * Given: dO (gradient of output), Q, K, V, O (output), LSE (from forward)
 * Compute: dQ, dK, dV
 *
 * 1. Compute D_i = rowsum(dO_i * O_i) for each row i
 * 2. For each Q block (i):
 *      dQ_i = 0
 *      For each K,V block (j):
 *        P_ij = exp(S_ij - LSE_i)  (recompute from forward)
 *        dV_j += P_ij^T @ dO_i
 *        dP_ij = dO_i @ V_j^T
 *        dS_ij = P_ij * (dP_ij - D_i)  (softmax backward)
 *        dQ_i += dS_ij @ K_j
 *        dK_j += dS_ij^T @ Q_i
 * 3. Final normalization and scaling
 *
 * MULTI-GROUP BACKWARD ALGORITHM:
 *
 * For each group g:
 *   1. Compute D_g,i = rowsum(dO_g,i * O_g,i)
 *   2. For each Q block i in group g:
 *        dQ_g,i = 0
 *        For each K,V block j that group g attends to:
 *          P_g,ij = exp(S_g,ij - LSE_g,i)  (recompute)
 *          dV_local_j += P_g,ij^T @ dO_g,i
 *          dP_g,ij = dO_g,i @ V_j^T
 *          dS_g,ij = P_g,ij * (dP_g,ij - D_g,i)
 *          dQ_g,i += dS_g,ij @ K_j
 *          dK_local_j += dS_g,ij^T @ Q_g,i
 *
 *        Write dQ_g,i to global memory
 *
 *   3. Accumulate dK_local, dV_local to global dK, dV
 *      (using atomics or two-pass reduction)
 *
 * KEY INSIGHT: Each thread block processes one (group, batch, head, Q block)
 *              and accumulates to shared dK, dV buffers.
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
// Gradient Accumulation Strategies
////////////////////////////////////////////////////////////////////////////////////////////////////

// Strategy 1: Atomic Adds (for FP32)
template<typename ElementAccum>
__device__ __forceinline__ void accumulate_gradients_atomic_fp32(
    ElementAccum* global_grad_ptr,  // Global dK or dV pointer (FP32)
    const ElementAccum* local_grad, // Local gradient values (FP32)
    int num_elements,               // Number of elements to accumulate
    int tidx,                       // Thread index
    int nthreads                    // Number of threads
) {
    // Each thread atomically adds its portion of gradients
    #pragma unroll 4
    for (int i = tidx; i < num_elements; i += nthreads) {
        atomicAdd(&global_grad_ptr[i], local_grad[i]);
    }
}

// Strategy 2: Warp-level reduction before atomic (reduces contention by 32x)
template<typename ElementAccum>
__device__ __forceinline__ void accumulate_gradients_atomic_warp_reduce(
    ElementAccum* global_grad_ptr,  // Global dK or dV pointer
    const ElementAccum* local_grad, // Local gradient values
    int num_elements,               // Number of elements to accumulate
    int tidx,                       // Thread index
    int nthreads                    // Number of threads
) {
    // Each warp reduces locally first, then one thread does atomic
    const int lane_id = tidx % 32;
    const int warp_id = tidx / 32;
    const int num_warps = nthreads / 32;

    // Shared memory for warp reduction (small footprint)
    __shared__ ElementAccum warp_reduced[32];  // One slot per warp

    for (int i = warp_id; i < num_elements; i += num_warps) {
        ElementAccum val = (i < num_elements) ? local_grad[i] : ElementAccum(0.0f);

        // Warp-level reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        // First thread in warp does atomic add
        if (lane_id == 0) {
            atomicAdd(&global_grad_ptr[i], val);
        }
    }
}

// Strategy 3: Atomic add for Element type (FP16/BF16)
// Uses intermediate conversion to FP32 for atomic operations
template<typename Element>
__device__ __forceinline__ void atomic_add_half(Element* address, Element val) {
    // Convert to FP32, do atomic add, no conversion back needed
    // Multiple threads may race, but the atomic ensures correctness
    atomicAdd(reinterpret_cast<float*>(address), static_cast<float>(val));
}

// Optimized: Write to shared memory first, then coalesce atomic writes
template<typename Element, typename Kernel_traits>
__device__ __forceinline__ void write_dkv_with_atomics(
    Element* gdK_ptr,
    Element* gdV_ptr,
    const Tensor auto& rdK,
    const Tensor auto& rdV,
    const Tensor auto& tdKgdK,
    const Tensor auto& tdVgdV,
    const Tensor auto& tdKVcdKV,
    const Tensor auto& tdKVpdKV,
    int actual_seqlen_k,
    int n_block,
    int tidx
) {
    constexpr int kBlockN = Kernel_traits::kBlockN;

    // Atomically accumulate dK and dV
    // Use FP32 accumulation for better precision
    #pragma unroll
    for (int i = 0; i < size(tdKgdK); ++i) {
        if (get<0>(tdKVcdKV(_0{}, i, _0{})) < actual_seqlen_k - n_block * kBlockN && tdKVpdKV(get<1>(tdKVcdKV(_0{}, i, _0{})))) {
            atomicAdd(reinterpret_cast<float*>(&tdKgdK(i)), static_cast<float>(rdK(i)));
        }
    }
    #pragma unroll
    for (int i = 0; i < size(tdVgdV); ++i) {
        if (get<0>(tdKVcdKV(_0{}, i, _0{})) < actual_seqlen_k - n_block * kBlockN && tdKVpdKV(get<1>(tdKVcdKV(_0{}, i, _0{})))) {
            atomicAdd(reinterpret_cast<float*>(&tdVgdV(i)), static_cast<float>(rdV(i)));
        }
    }
}

// Strategy 2: Two-Pass Reduction
template<typename Element, typename index_t>
__device__ __forceinline__ void write_gradients_separate(
    Element* accum_buffer,         // Per-group gradient buffer
    const Element* local_grad,     // Local gradient tile
    int group_id,                  // Which group this is
    int num_elements,              // Number of elements
    index_t buffer_stride          // Stride between groups
) {
    // Write to separate per-group buffer for later reduction
    // accum_buffer layout: [num_groups, K_size, num_heads_k, head_dim]
    // Each group writes to its own slice

    // Calculate base offset for this group's slice
    const index_t group_offset = group_id * buffer_stride;

    // Each thread writes its portion of gradients
    #pragma unroll 4
    for (int i = threadIdx.x; i < num_elements; i += blockDim.x) {
        accum_buffer[group_offset + i] = local_grad[i];
    }
}

// Reduction kernel (separate launch)
template<typename Element, typename index_t>
__global__ void reduce_multigroup_gradients_kernel(
    Element* output_grad,          // Final dK or dV [K_size, num_heads_k, head_dim]
    const Element* accum_buffer,   // Per-group buffers [num_groups, K_size, ...]
    int num_groups,
    int K_size,
    int num_heads_k,
    int head_dim
) {
    // Reduce across groups deterministically
    // Each thread handles one output element
    // output_grad[idx] = sum_g accum_buffer[g][idx]

    const int total_elements = K_size * num_heads_k * head_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        // Use FP32 accumulation for better precision
        float sum = 0.0f;

        // Sum contributions from all groups
        // Buffer layout: [num_groups, K_size, num_heads_k, head_dim]
        const index_t group_stride = K_size * num_heads_k * head_dim;

        #pragma unroll
        for (int g = 0; g < num_groups; g++) {
            const index_t offset = g * group_stride + idx;
            sum += static_cast<float>(accum_buffer[offset]);
        }

        // Write final result
        output_grad[idx] = static_cast<Element>(sum);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Backward Kernel Template (Multi-Group)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, int NumGroups, bool Is_causal, bool Is_local,
         bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_dropout>
__device__ __forceinline__ void compute_dqkv_multigroup_1colblock(
    const Flash_bwd_multigroup_params &params,
    const int bidb,
    const int bidh,
    const int n_block
) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory
    extern __shared__ char smem_[];

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNThreads = Kernel_traits::kNThreads;

    // Check if this K,V block is valid for at least one group
    bool any_group_active = false;
    int m_block_max_global = 0;

    #pragma unroll
    for (int g = 0; g < NumGroups; g++) {
        // Get KV endpoint for this group
        int kv_endpoint = params.kv_endpoints[g * params.batch_size + bidb];

        if (n_block * kBlockN < kv_endpoint) {
            any_group_active = true;

            // Compute m_block_max for this group
            const int cu_seqlens_q_start = params.cu_seqlens_q_list[g][bidb];
            const int cu_seqlens_q_end = params.cu_seqlens_q_list[g][bidb + 1];
            int actual_seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start;

            int m_block_max = cute::ceil_div(actual_seqlen_q, kBlockM);
            if (Is_local) {
                const int cu_seqlens_k_start = params.cu_seqlens_k_list[g][bidb];
                const int cu_seqlens_k_end = params.cu_seqlens_k_list[g][bidb + 1];
                int actual_seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start;
                m_block_max = std::min(m_block_max,
                    cute::ceil_div((n_block + 1) * kBlockN + actual_seqlen_q - actual_seqlen_k + params.window_size_left, kBlockM));
            }
            m_block_max_global = max(m_block_max_global, m_block_max);
        }
    }

    if (!any_group_active) {
        // No group needs this K,V block, write zeros to dK, dV and return
        // (Implementation similar to standard backward kernel early exit)
        return;
    }

    // ========================================================================
    // Setup shared memory for K, V (shared across groups)
    // ========================================================================

    const int cu_seqlens_k_start = params.cu_seqlens_k_list[0][bidb];  // K,V are shared
    const int cu_seqlens_k_end = params.cu_seqlens_k_list[0][bidb + 1];
    int actual_seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start;

    const index_t row_offset_k = cu_seqlens_k_start * params.k_row_stride
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = cu_seqlens_k_start * params.v_row_stride
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));

    // Shared memory tensors
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // Load K, V (shared across all groups)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) {
            tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.head_size;
        }
    }

    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, actual_seqlen_k - n_block * kBlockN
    );
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, actual_seqlen_k - n_block * kBlockN
    );
    FLASH_NAMESPACE::cp_async_fence();

    // ========================================================================
    // Initialize per-group accumulators for dK, dV (in registers, will accumulate in FP32)
    // ========================================================================

    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);

    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    clear(acc_dk);
    clear(acc_dv);

    // ========================================================================
    // Process each group and accumulate gradients
    // ========================================================================

    FLASH_NAMESPACE::cp_async_wait<0>();
    __syncthreads();

    // Main loop over Q blocks (iterate backward through m_blocks)
    for (int m_block = m_block_max_global - 1; m_block >= 0; --m_block) {

        // Process each group
        #pragma unroll
        for (int g = 0; g < NumGroups; g++) {
            // Check if this group is active for this (m_block, n_block) pair
            const int cu_seqlens_q_start = params.cu_seqlens_q_list[g][bidb];
            const int cu_seqlens_q_end = params.cu_seqlens_q_list[g][bidb + 1];
            int actual_seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start;

            // Check m_block range
            if (m_block * kBlockM >= actual_seqlen_q) continue;

            // Check n_block range (KV endpoint)
            int kv_endpoint = params.kv_endpoints[g * params.batch_size + bidb];
            if (n_block * kBlockN >= kv_endpoint) continue;

            // Check causal/local masking constraints
            int m_block_min = 0;
            if (Is_causal || Is_local) {
                const int cu_seqlens_k_start_g = params.cu_seqlens_k_list[g][bidb];
                const int cu_seqlens_k_end_g = params.cu_seqlens_k_list[g][bidb + 1];
                int actual_seqlen_k_g = cu_seqlens_k_end_g - cu_seqlens_k_start_g;

                m_block_min = std::max(0,
                    (n_block * kBlockN + actual_seqlen_q - actual_seqlen_k_g -
                     (Is_local ? params.window_size_right : 0)) / kBlockM);
            }

            if (m_block < m_block_min) continue;

            // This group is active, compute gradients

            // ===================================================================
            // 1. Setup group-specific tensors
            // ===================================================================

            const index_t row_offset_q_g = params.cu_seqlens_q_list[g][bidb] * params.q_row_stride_list[g]
                + m_block * kBlockM * params.q_row_stride_list[g] + bidh * params.q_head_stride_list[g];
            const index_t row_offset_do_g = params.cu_seqlens_q_list[g][bidb] * params.do_row_stride_list[g]
                + m_block * kBlockM * params.do_row_stride_list[g] + bidh * params.do_head_stride_list[g];
            const index_t row_offset_o_g = params.cu_seqlens_q_list[g][bidb] * params.o_row_stride_list[0]
                + m_block * kBlockM * params.o_row_stride_list[0] + bidh * params.o_head_stride_list[0];

            Tensor gQ_g = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr_list[g]) + row_offset_q_g),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.q_row_stride_list[g], _1{}));
            Tensor gdO_g = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr_list[g]) + row_offset_do_g),
                                     Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                     make_stride(params.do_row_stride_list[g], _1{}));
            Tensor gO_g = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.out_ptr_list[g]) + row_offset_o_g),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.o_row_stride_list[0], _1{}));

            // Allocate shared memory for this group's Q, dO
            Element* sQ_g_ptr = reinterpret_cast<Element*>(smem_) +
                                (2 * size(sK)) +  // After sK and sV
                                g * kBlockM * kHeadDim;
            Element* sdO_g_ptr = sQ_g_ptr + NumGroups * kBlockM * kHeadDim;

            Tensor sQ_g = make_tensor(make_smem_ptr(sQ_g_ptr),
                                    typename Kernel_traits::SmemLayoutQdO{});
            Tensor sdO_g = make_tensor(make_smem_ptr(sdO_g_ptr),
                                     typename Kernel_traits::SmemLayoutQdO{});

            // Load Q and dO for this group
            auto gmem_thr_copy_QKV_g = gmem_tiled_copy_QKV.get_thread_slice(tidx);
            Tensor tQgQ_g = gmem_thr_copy_QKV_g.partition_S(gQ_g);
            Tensor tQsQ_g = gmem_thr_copy_QKV_g.partition_D(sQ_g);
            Tensor tdOgdO_g = gmem_thr_copy_QKV_g.partition_S(gdO_g);
            Tensor tdOsdO_g = gmem_thr_copy_QKV_g.partition_D(sdO_g);

            // Predicates
            Tensor cQ_g = make_identity_tensor(make_shape(size<0>(sQ_g), size<1>(sQ_g)));
            Tensor tQcQ_g = gmem_thr_copy_QKV_g.partition_D(cQ_g);
            Tensor tQpQ_g = make_tensor<bool>(make_shape(size<2>(tQsQ_g)));

            if (!Is_even_K) {
                #pragma unroll
                for (int k = 0; k < size(tQpQ_g); ++k) {
                    tQpQ_g(k) = get<1>(tQcQ_g(0, 0, k)) < params.head_size;
                }
            }

            // Load Q and dO with boundary checks
            FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tQgQ_g, tQsQ_g, tQcQ_g, tQpQ_g, actual_seqlen_q - m_block * kBlockM
            );
            FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tdOgdO_g, tdOsdO_g, tQcQ_g, tQpQ_g, actual_seqlen_q - m_block * kBlockM
            );
            FLASH_NAMESPACE::cp_async_fence();
            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();

            // ===================================================================
            // 2. Compute S = Q @ K^T (attention scores)
            // ===================================================================

            typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
            auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
            Tensor tSrQ_g = thr_mma_sdp.partition_fragment_A(sQ_g);
            Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);

            Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
            clear(acc_s);

            gemm(tiled_mma_sdp, tSrQ_g, tSrK, acc_s);

            // ===================================================================
            // 3. Apply masking and recompute softmax (P)
            // ===================================================================

            // Apply KV boundary mask
            int kv_valid_end = min((n_block + 1) * kBlockN, kv_endpoint);
            if ((n_block + 1) * kBlockN > kv_valid_end) {
                // Mask out positions beyond KV endpoint
                #pragma unroll
                for (int i = 0; i < size(acc_s); ++i) {
                    // Get position in K,V dimension
                    // This is simplified - actual implementation needs proper layout extraction
                    // For now, placeholder logic
                    acc_s(i) = acc_s(i);  // TODO: Apply mask based on kv_valid_end
                }
            }

            // Get LSE for this group's m_block
            const index_t row_offset_lse_g = (params.unpadded_lse
                ? bidh * params.total_q_list[g] + params.cu_seqlens_q_list[g][bidb]
                : (bidb * params.h + bidh) * params.max_seqlen_q_list[g])
                + m_block * kBlockM;

            Tensor gLSE_g = make_tensor(make_gmem_ptr(
                reinterpret_cast<float*>(params.softmax_lse_ptr_list[g]) + row_offset_lse_g),
                Shape<Int<kBlockM>>{}, Stride<_1>{});

            // Load LSE (already computed in forward pass)
            float lse_g[kBlockM];  // Simplified - actual would be per-thread
            // TODO: Load LSE values properly

            // Recompute P = exp(S - LSE)
            // Apply scale and exponential
            FLASH_NAMESPACE::scale_apply_exp2</*scale_max=*/false>(
                acc_s, lse_g, params.scale_softmax_log2);

            // Convert to element type
            Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);

            // ===================================================================
            // 4. Compute dP = dO @ V^T
            // ===================================================================

            Tensor tdPrdO_g = thr_mma_sdp.partition_fragment_A(sdO_g);
            Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);

            Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
            clear(acc_dp);

            gemm(tiled_mma_sdp, tdPrdO_g, tdPrV, acc_dp);

            // ===================================================================
            // 5. Compute D_i = rowsum(dO * O) and dS = P * (dP - D_i)
            // ===================================================================

            // Load D_i from pre-computed dPsum (computed in separate pass before main kernel)
            // For now, we'll compute D_i inline - this should be optimized in Phase 5
            // D_i(row) = sum_k(dO(row, k) * O(row, k))

            // Get dP_sum (D_i) for this group's m_block
            // In a full implementation, this would be pre-computed like in the standard kernel
            // For Phase 2, we compute it inline
            Tensor dP_sum = make_fragment_like(acc_s);
            clear(dP_sum);

            // Simplified D_i computation - in practice would use dot_do_o helper
            // For each row, compute sum of element-wise product of dO and O
            // This is a placeholder - actual implementation needs proper per-thread partitioning
            float D_i[kBlockM / kNWarps];  // Simplified per-thread D values

            #pragma unroll
            for (int mi = 0; mi < kBlockM / kNWarps; ++mi) {
                D_i[mi] = 0.0f;
                // In practice, would load O, compute dO * O, and sum across head dimension
                // For now, setting to 0 as placeholder (will cause incorrect gradients but won't crash)
            }

            // Compute dS = P * (dP - D_i)
            // Reshape acc_dp from (MMA=4, MMA_N, MMA_N) to (row=(2, MMA_N), col=(2, MMA_N))
            Tensor dS = make_tensor(acc_dp.data(), make_tensor(acc_s.data(), acc_s.layout()).layout());

            // Apply softmax backward: dS = P * (dP - D_i)
            #pragma unroll
            for (int mi = 0; mi < size<0>(dS); ++mi) {
                #pragma unroll
                for (int ni = 0; ni < size<1>(dS); ++ni) {
                    // acc_s contains P (already softmaxed)
                    // acc_dp contains dP
                    // Use simplified D_i index (this needs proper coordinate extraction)
                    int d_idx = mi / (size<0>(dS) / (kBlockM / kNWarps));
                    if (d_idx < kBlockM / kNWarps) {
                        dS(mi, ni) = acc_s(mi, ni) * (acc_dp(mi, ni) - D_i[d_idx]);
                    }
                }
            }

            // ===================================================================
            // 6. Accumulate gradients
            // ===================================================================

            // dV += P^T @ dO (shared K,V, accumulate to acc_dv)
            Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
            Tensor tPrP = make_tensor(rP.data(),
                FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMmaSdP>(rP.layout()));

            // Setup for dV accumulation
            Tensor tVrPt = tPrP;  // Simplified - actual needs transpose setup
            Tensor tVrdO = thr_mma_sdp.partition_fragment_A(sdO_g);

            // Accumulate: acc_dv += P^T @ dO
            // Using simplified gemm - actual implementation needs proper tiling
            // FLASH_NAMESPACE::gemm(acc_dv, tVrPt, tVrdO, ...);
            // Placeholder: assume gemm is called here

            // dK += dS^T @ Q (shared K,V, accumulate to acc_dk)
            Tensor tdSrdS = FLASH_NAMESPACE::convert_type<Element>(dS);
            Tensor tKrdSt = tdSrdS;  // Simplified - actual needs transpose
            Tensor tKrQt = thr_mma_sdp.partition_fragment_B(sQ_g);

            // Accumulate: acc_dk += dS^T @ Q
            // FLASH_NAMESPACE::gemm(acc_dk, tKrdSt, tKrQt, ...);
            // Placeholder: assume gemm is called here

            // dQ for this group (not shared, compute and write separately)
            Tensor acc_dq_g = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kHeadDim>>{});
            clear(acc_dq_g);

            // acc_dq_g = dS @ K
            Tensor tQrdS = tdSrdS;
            Tensor tQrK = thr_mma_sdp.partition_fragment_B(sK);
            // FLASH_NAMESPACE::gemm(acc_dq_g, tQrdS, tQrK, ...);

            // Write dQ for this group to global memory
            // (Implementation similar to standard backward kernel)
        }
    }

    // ========================================================================
    // Write accumulated dK, dV to global memory (with atomic adds)
    // ========================================================================

    const index_t row_offset_dk = cu_seqlens_k_start * params.dk_row_stride
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = cu_seqlens_k_start * params.dv_row_stride
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;

    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dv_row_stride, _1{}));

    // Convert acc_dk, acc_dv from FP32 to Element type
    Tensor rdK = FLASH_NAMESPACE::convert_type<Element>(acc_dk);
    Tensor rdV = FLASH_NAMESPACE::convert_type<Element>(acc_dv);

    // Copy to shared memory first
    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{});

    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);

    __syncthreads();
    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();

    // Use atomic adds to accumulate gradients (since multiple groups may contribute)
    // Strategy: Convert to FP32, use atomicAdd, rely on hardware atomics for correctness
    if (!params.deterministic) {
        // Atomic accumulation mode (fast, non-deterministic order)
        // Write dK and dV using atomics
        Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);

        #pragma unroll
        for (int i = 0; i < size(tdKgdK); ++i) {
            // Check bounds
            if (get<0>(tdKVcdKV(_0{}, i, _0{})) < actual_seqlen_k - n_block * kBlockN) {
                // Convert to FP32 and atomically add
                // Note: This assumes dK and dV are FP32 or we can safely cast
                atomicAdd(reinterpret_cast<float*>(&tdKgdK(i)), static_cast<float>(tdKsdK(i)));
            }
        }
        #pragma unroll
        for (int i = 0; i < size(tdVgdV); ++i) {
            if (get<0>(tdKVcdKV(_0{}, i, _0{})) < actual_seqlen_k - n_block * kBlockN) {
                atomicAdd(reinterpret_cast<float*>(&tdVgdV(i)), static_cast<float>(tdVsdV(i)));
            }
        }
    } else {
        // Deterministic mode: write to separate per-threadblock buffers, then reduce
        // Each threadblock processes a unique (n_block, batch, head) combination,
        // so we write to a unique slice of the accumulation buffer

        // Calculate unique threadblock ID
        const int tb_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
        const int elements_per_block = kBlockN * kHeadDim;

        // Use params.dk_accum_ptr and params.dv_accum_ptr if available
        if (params.dk_accum_ptr != nullptr && params.dv_accum_ptr != nullptr) {
            // Write to accumulation buffers (no atomics needed, unique location per TB)
            const index_t accum_offset = tb_id * elements_per_block;

            Element* dk_accum = reinterpret_cast<Element*>(params.dk_accum_ptr) + accum_offset;
            Element* dv_accum = reinterpret_cast<Element*>(params.dv_accum_ptr) + accum_offset;

            // Create tensors pointing to accumulation buffers
            Tensor gdK_accum = make_tensor(make_gmem_ptr(dk_accum),
                                          Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                          make_stride(params.dk_row_stride, _1{}));
            Tensor gdV_accum = make_tensor(make_gmem_ptr(dv_accum),
                                          Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                          make_stride(params.dv_row_stride, _1{}));

            Tensor tdKgdK_accum = gmem_thr_copy_dKV.partition_D(gdK_accum);
            Tensor tdVgdV_accum = gmem_thr_copy_dKV.partition_D(gdV_accum);

            // Write without atomics (each TB has unique location)
            Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));
            Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);

            cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKgdK_accum);
            cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVgdV_accum);
        } else {
            // Fallback: use atomics if accumulation buffers not provided
            // This maintains correctness but loses determinism
            Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));
            Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);

            #pragma unroll
            for (int i = 0; i < size(tdKgdK); ++i) {
                if (get<0>(tdKVcdKV(_0{}, i, _0{})) < actual_seqlen_k - n_block * kBlockN) {
                    atomicAdd(reinterpret_cast<float*>(&tdKgdK(i)), static_cast<float>(tdKsdK(i)));
                }
            }
            #pragma unroll
            for (int i = 0; i < size(tdVgdV); ++i) {
                if (get<0>(tdKVcdKV(_0{}, i, _0{})) < actual_seqlen_k - n_block * kBlockN) {
                    atomicAdd(reinterpret_cast<float*>(&tdVgdV(i)), static_cast<float>(tdVsdV(i)));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Backward Kernel Launch Wrapper
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, int NumGroups, bool Is_causal, bool Is_local,
         bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_dropout>
__global__ void flash_bwd_multigroup_kernel(
    KERNEL_PARAM_MODIFIER const Flash_bwd_multigroup_params params
) {
    // Block indices
    const int bidb = blockIdx.y;  // Batch index
    const int bidh = blockIdx.z;  // Head index

    // Process all K,V blocks for this (batch, head) pair
    // Each thread block processes one K,V column block
    const int n_block = blockIdx.x;

    compute_dqkv_multigroup_1colblock<Kernel_traits, NumGroups, Is_causal, Is_local,
                                       Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Is_dropout>(
        params, bidb, bidh, n_block
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Main Backward Computation (processes all K,V blocks)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, int NumGroups, bool Is_causal, bool Is_local,
         bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_dropout>
__device__ __forceinline__ void compute_dqkv_multigroup(
    const Flash_bwd_multigroup_params &params
) {
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // Get maximum K,V sequence length across all groups
    int max_seqlen_k = 0;
    #pragma unroll
    for (int g = 0; g < NumGroups; g++) {
        const int cu_seqlens_k_start = params.cu_seqlens_k_list[g][bidb];
        const int cu_seqlens_k_end = params.cu_seqlens_k_list[g][bidb + 1];
        int actual_seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start;
        max_seqlen_k = max(max_seqlen_k, actual_seqlen_k);
    }

    const int n_block_max = (max_seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;

    // Iterate over all K,V blocks
    // Each iteration processes one K,V block for all groups that need it
    for (int n_block = 0; n_block < n_block_max; n_block++) {
        compute_dqkv_multigroup_1colblock<Kernel_traits, NumGroups, Is_causal, Is_local,
                                           Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Is_dropout>(
            params, bidb, bidh, n_block
        );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Template Instantiation Helper
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, int NumGroups, bool Is_causal>
void run_flash_bwd_multigroup(
    Flash_bwd_multigroup_params &params,
    cudaStream_t stream
) {
    // Determine grid dimensions
    // Grid: (num_n_blocks, batch_size, num_heads)
    // Each thread block processes one (n_block, batch, head) combination

    // Get maximum K,V sequence length to determine n_block_max
    int max_seqlen_k = 0;
    for (int g = 0; g < NumGroups; g++) {
        if (params.max_seqlen_k_list) {
            max_seqlen_k = std::max(max_seqlen_k, params.max_seqlen_k_list[g]);
        }
    }

    const int num_n_blocks = (max_seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;

    dim3 grid(num_n_blocks, params.batch_size, params.h);
    dim3 block(Kernel_traits::kNThreads);

    // Compute shared memory size
    // For backward: need K, V, dK, dV in shared memory
    // Plus per-group Q, dO, dQ (can be kept in registers mostly)
    const int smem_size = Kernel_traits::kSmemSize;

    // Template parameters
    constexpr bool Is_local = false;  // TODO: support local attention
    constexpr bool Has_alibi = false; // TODO: support alibi
    constexpr bool Is_even_MN = false; // Be conservative, check at runtime
    constexpr bool Is_even_K = Kernel_traits::kHeadDim % Kernel_traits::kBlockKSmem == 0;
    constexpr bool Is_softcap = false; // TODO: support softcap
    constexpr bool Is_dropout = false; // TODO: support dropout

    // Launch kernel
    if (!params.deterministic) {
        // Atomic accumulation mode (fast but non-deterministic)
        flash_bwd_multigroup_kernel<Kernel_traits, NumGroups, Is_causal, Is_local,
                                     Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Is_dropout>
            <<<grid, block, smem_size, stream>>>(params);
    } else {
        // Deterministic mode: two-pass reduction
        // 1. First pass: write per-threadblock gradients to separate buffers
        flash_bwd_multigroup_kernel<Kernel_traits, NumGroups, Is_causal, Is_local,
                                     Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Is_dropout>
            <<<grid, block, smem_size, stream>>>(params);

        // Check for errors from first pass
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("First pass kernel failed: ") + cudaGetErrorString(err));
        }

        // 2. Second pass: reduction kernel sums from accumulation buffers to final outputs
        // Only needed if accumulation buffers are separate from output buffers
        if (params.dk_accum_ptr != nullptr && params.dv_accum_ptr != nullptr &&
            params.dk_accum_ptr != params.dk_ptr && params.dv_accum_ptr != params.dv_ptr) {

            // Calculate total number of thread blocks from first pass
            const int total_tbs = num_n_blocks * params.batch_size * params.h;
            const int elements_per_tb = Kernel_traits::kBlockN * Kernel_traits::kHeadDim;
            const int total_elements_per_grad = max_seqlen_k * params.h_k * params.head_size;

            // Launch reduction kernel for dK
            const int threads_per_block = 256;
            const int blocks_for_dk = (total_elements_per_grad + threads_per_block - 1) / threads_per_block;

            using Element = typename Kernel_traits::Element;
            using index_t = typename Kernel_traits::index_t;

            reduce_multigroup_gradients_kernel<Element, index_t>
                <<<blocks_for_dk, threads_per_block, 0, stream>>>(
                    reinterpret_cast<Element*>(params.dk_ptr),
                    reinterpret_cast<const Element*>(params.dk_accum_ptr),
                    total_tbs,  // treated as num_groups for reduction
                    max_seqlen_k,
                    params.h_k,
                    params.head_size
                );

            // Launch reduction kernel for dV
            const int blocks_for_dv = (total_elements_per_grad + threads_per_block - 1) / threads_per_block;

            reduce_multigroup_gradients_kernel<Element, index_t>
                <<<blocks_for_dv, threads_per_block, 0, stream>>>(
                    reinterpret_cast<Element*>(params.dv_ptr),
                    reinterpret_cast<const Element*>(params.dv_accum_ptr),
                    total_tbs,  // treated as num_groups for reduction
                    max_seqlen_k,
                    params.h_k,
                    params.head_size
                );

            // Check for errors from reduction kernels
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("Reduction kernel failed: ") + cudaGetErrorString(err));
            }
        }
    }

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Public API
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim, int NumGroups, bool Is_causal>
void run_mha_bwd_multigroup(Flash_bwd_multigroup_params &params, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Gradient Accumulation Notes
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * GRADIENT ACCUMULATION CONSIDERATIONS:
 *
 * 1. Atomic Performance:
 *    - FP32 atomicAdd: Fast, widely supported
 *    - FP16/BF16 atomicAdd: Hardware support varies (H100+)
 *    - Workaround: Convert to FP32, atomic add, convert back
 *    - Contention: Multiple TBs may write to same K,V positions
 *      → Use warp-level aggregation before atomic (reduce contention)
 *
 * 2. Two-Pass Memory Requirements:
 *    - Extra memory: num_groups × (K_size × num_heads_k × head_dim × 2 bytes)
 *    - Example: 2 groups, K_size=2048, h_k=8, d=128
 *      → 2 × 2048 × 8 × 128 × 2 = 8.4 MB per dK
 *      → Total (dK + dV): ~17 MB (acceptable)
 *    - Trade-off: Memory cost vs. determinism
 *
 * 3. Warp-Level Reduction Before Atomic:
 *    - Each warp accumulates locally, then single thread does atomic
 *    - Reduces atomic contention by 32×
 *    - Example:
 *      __shared__ float warp_accum[32];
 *      float local_grad = ...;
 *      warp_accum[threadIdx.x % 32] = local_grad;
 *      __syncwarp();
 *      if (threadIdx.x % 32 == 0) {
 *          float warp_sum = 0.0f;
 *          for (int i = 0; i < 32; i++) warp_sum += warp_accum[i];
 *          atomicAdd(&global_grad[idx], warp_sum);
 *      }
 *
 * 4. Testing Strategy:
 *    - Test both atomic and two-pass modes
 *    - Compare against sequential kernel calls (ground truth)
 *    - Use torch.autograd.gradcheck for correctness
 *    - Profile atomic contention with Nsight Compute:
 *      → Metric: global_atomic_store_issued vs. global_atomic_store_executed
 *
 * 5. Backward Kernel Optimization (Phase 5):
 *    - Fuse D_i computation with first K,V block iteration
 *    - Reuse forward pass's P_ij computation (if cached)
 *    - Tune tile sizes separately for backward (may differ from forward)
 *    - Profile memory bandwidth: backward is more memory-bound than forward
 */

}  // namespace FLASH_NAMESPACE

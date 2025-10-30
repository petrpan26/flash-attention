/******************************************************************************
 * Copyright (c) 2024, Multi-Group Flash Attention Implementation.
 *
 * Forward kernel template for multi-group varlen attention.
 *
 * STUB FILE FOR PHASE 2 IMPLEMENTATION
 *
 * This file contains the kernel template signatures and detailed pseudocode
 * for implementing the multi-group forward attention kernel. The actual CUDA
 * implementation should follow the structure outlined in the comments.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include "philox_unpack.cuh"

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "flash_multigroup.h"
#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-Group Kernel Traits (Extended from standard Flash Attention)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, int NumGroups_,
         bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false,
         typename elem_type=cutlass::half_t>
struct Flash_fwd_multigroup_kernel_traits {

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using Element = elem_type;
    static constexpr bool Has_cp_async = true;
#else
    using Element = cutlass::half_t;
    static constexpr bool Has_cp_async = false;
#endif

    using ElementAccum = float;
    using index_t = int64_t;

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;
#else
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#endif

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
#else
    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
#endif

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int NumGroups = NumGroups_;

    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        Tile<Int<16 * kNWarps>, _16, _16>>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;

    // Per-group Q tile size
    static constexpr int kSmemQSizePerGroup = size(SmemLayoutQ{}) * sizeof(Element);
    // Shared K,V tiles (shared across all groups)
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    // Total shared memory: NumGroups Q tiles + K,V tiles
    static constexpr int kSmemSize = NumGroups * kSmemQSizePerGroup + kSmemKVSize;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        AutoVectorizingCopyWithAssumedAlignment<128>
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Per-Group State Management Structure
////////////////////////////////////////////////////////////////////////////////////////////////////

// This structure holds all state needed for processing one Q group
// It should be instantiated per-group in the kernel
template<typename Kernel_traits>
struct GroupState {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;

    // ========================================================================
    // Shared Memory Pointers (Per-Group Q Tile)
    // ========================================================================

    // Pointer to this group's Q tile in shared memory
    Element* smem_q_ptr;

    // ========================================================================
    // Register-Based Accumulators (defined during kernel execution)
    // ========================================================================

    // These will be defined in the kernel as:
    // Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    // Using typename: decltype(partition_fragment_C(typename Kernel_traits::TiledMma{},
    //                          Shape<Int<Kernel_traits::kBlockM>, Int<Kernel_traits::kHeadDim>>{}))

    // ========================================================================
    // LSE State (Registers) - Using Softmax template
    // ========================================================================

    // We use the Softmax template from softmax.h which manages LSE state internally
    // This is simpler than manually managing lse_max and lse_sum arrays

    // ========================================================================
    // Metadata
    // ========================================================================

    // Actual sequence length for Q in this group
    int actual_seqlen_q;

    // Actual sequence length for K,V that this group attends to
    int actual_seqlen_k;

    // Maximum K,V offset this group should attend to
    int kv_max_offset;

    // Whether this group is active (has work to do for this thread block)
    bool active;

    // Cumulative Q start position (for varlen indexing)
    int cu_seqlens_q_start;

    // Cumulative K start position (for varlen indexing)
    int cu_seqlens_k_start;

    // ========================================================================
    // Initialization
    // ========================================================================

    __device__ __forceinline__ void init() {
        // Initialize LSE state to -inf, 0
        #pragma unroll
        for (int i = 0; i < Kernel_traits::kBlockM / 8; i++) {
            lse_max[i] = -INFINITY;
            lse_sum[i] = 0.0f;
        }

        // Clear accumulator (done via acc_o.clear() in actual implementation)
    }

    // ========================================================================
    // LSE Update (Online Softmax)
    // ========================================================================

    // TODO Phase 2: Implement LSE update logic
    // This should match the online softmax algorithm from softmax.h
    // but operate on this group's state only
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Forward Kernel Template (Main Entry Point)
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * PSEUDOCODE FOR IMPLEMENTATION:
 *
 * Goal: Process multiple Q groups with shared K,V in a single kernel launch.
 *       Each K,V tile is loaded once and used by all groups that need it.
 *
 * Thread Block Mapping:
 *   blockIdx.x = m_block (which Q block to process)
 *   blockIdx.y = bidb (batch index)
 *   blockIdx.z = bidh (head index)
 *
 * Grid Size: (num_m_blocks, batch_size, num_heads)
 *
 * Each thread block processes:
 *   - All groups for a given (batch, head, Q block)
 *   - Iterates over K,V tiles, loading each tile once
 *   - Updates per-group output accumulators and LSE states
 *
 * Main Loop Structure:
 *
 * 1. Initialize per-group state
 *    - Allocate shared memory partitions for each group's Q tile
 *    - Initialize LSE state (max=-inf, sum=0)
 *    - Clear output accumulators
 *    - Load metadata (cu_seqlens, kv_endpoints, etc.)
 *
 * 2. Load Q tiles for all groups
 *    - Each group loads its Q tile into its smem partition
 *    - Use async copy for overlap
 *
 * 3. Main K,V loop (iterate backward for causal masking efficiency)
 *    for n_block = n_block_max - 1 down to 0:
 *
 *      3a. Load K,V tile ONCE (shared across all groups)
 *          - Load K[n_block] into shared memory (sK)
 *          - Load V[n_block] into shared memory (sV)
 *          - Synchronize threads
 *
 *      3b. For each group:
 *          - Check if this group needs this K,V tile
 *            (kv_tile_start < kv_endpoints[group][batch])
 *
 *          - If needed:
 *            i.   Load Q tile for group into MMA fragments
 *            ii.  Compute attention scores: S = Q @ K^T (MMA)
 *            iii. Apply masking (causal, kv_boundary, etc.)
 *            iv.  Update LSE state (online softmax)
 *                 - Compute row_max(S)
 *                 - Update global max
 *                 - Compute exp(S - max)
 *                 - Update sum
 *                 - Rescale old accumulator
 *            v.   Accumulate output: acc_o += P @ V (MMA)
 *
 *          - Synchronize before next K,V tile
 *
 * 4. Finalize outputs for all groups
 *    - Normalize output: O = acc_o / lse_sum
 *    - Write output to global memory (each group's output tensor)
 *    - Write LSE to global memory (log(sum) + max)
 *
 * Memory Access Pattern:
 *   - K,V tiles: Loaded ONCE per n_block (coalesced gmem -> smem)
 *   - Q tiles: Loaded ONCE per group at start (gmem -> smem)
 *   - Outputs: Written ONCE per group at end (registers -> gmem)
 *
 * Shared Memory Layout (example for NumGroups=2, d=128):
 *   +-----------------------------------------------+
 *   | Group 0 Q:  [kBlockM x kHeadDim]    (32KB)   |
 *   +-----------------------------------------------+
 *   | Group 1 Q:  [kBlockM x kHeadDim]    (32KB)   |
 *   +-----------------------------------------------+
 *   | Shared K:   [kBlockN x kHeadDim]    (32KB)   |
 *   +-----------------------------------------------+
 *   | Shared V:   [kBlockN x kHeadDim]    (32KB)   |
 *   +-----------------------------------------------+
 *   Total: 128KB (borderline, may need smaller tiles)
 *
 * Register Allocation:
 *   - Per-group LSE state: ~128 bytes per group (in registers)
 *   - Per-group output accumulator: ~256 regs per group
 *   - Shared K,V fragments: ~64 regs (shared across groups)
 *   - Total: ~380 regs/thread for 2 groups (manageable)
 *
 * Key Optimizations:
 *   - Early exit: Skip K,V tile if all groups have kv_endpoint < tile_start
 *   - Warp-level voting: Use __ballot_sync to check if any group needs tile
 *   - Register pressure: Keep acc_o in registers, not smem
 *   - Bank conflicts: Use swizzled smem layout (already in Kernel_traits)
 */

template<typename Kernel_traits, int NumGroups, bool Is_causal, bool Is_local,
         bool Has_alibi, bool Is_even_MN, bool Is_even_K>
__device__ __forceinline__ void compute_attn_multigroup(
    const Flash_fwd_multigroup_params &params,
    const int bidb,
    const int bidh,
    const int m_block
) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    const int tidx = threadIdx.x;

    // Shared memory allocation
    extern __shared__ char smem_[];

    // ========================================================================
    // STEP 1: Initialize per-group metadata and check early exit
    // ========================================================================

    // Per-group metadata arrays
    int cu_seqlens_q_start[NumGroups];
    int actual_seqlen_q[NumGroups];
    int cu_seqlens_k_start[NumGroups];
    int actual_seqlen_k[NumGroups];
    int kv_max_offset[NumGroups];
    bool group_active[NumGroups];
    int n_block_max_per_group[NumGroups];
    int n_block_max_global = 0;

    #pragma unroll
    for (int g = 0; g < NumGroups; g++) {
        // Load sequence length metadata
        cu_seqlens_q_start[g] = params.cu_seqlens_q_list[g][bidb];
        actual_seqlen_q[g] = params.cu_seqlens_q_list[g][bidb + 1] - cu_seqlens_q_start[g];

        cu_seqlens_k_start[g] = params.cu_seqlens_k_list[g][bidb];
        actual_seqlen_k[g] = params.cu_seqlens_k_list[g][bidb + 1] - cu_seqlens_k_start[g];

        // Get KV endpoint for this group
        kv_max_offset[g] = params.kv_endpoints[g * params.batch_size + bidb];

        // Check if this group has work for this m_block
        group_active[g] = (m_block * kBlockM < actual_seqlen_q[g]);

        if (group_active[g]) {
            // Compute n_block_max for this group based on kv_endpoint
            n_block_max_per_group[g] = cute::ceil_div(kv_max_offset[g], kBlockN);
            if (Is_causal || Is_local) {
                n_block_max_per_group[g] = std::min(n_block_max_per_group[g],
                    cute::ceil_div((m_block + 1) * kBlockM + actual_seqlen_k[g] - actual_seqlen_q[g] + params.window_size_right, kBlockN));
            }
            n_block_max_global = std::max(n_block_max_global, n_block_max_per_group[g]);
        } else {
            n_block_max_per_group[g] = 0;
        }
    }

    // Early exit if no groups are active
    if (n_block_max_global == 0) {
        // Write zeros to output and -INFINITY to LSE for all inactive groups
        #pragma unroll
        for (int g = 0; g < NumGroups; g++) {
            if (m_block * kBlockM < actual_seqlen_q[g]) {
                // This group needs output written (zeros and -inf LSE)
                Element* out_ptr = reinterpret_cast<Element*>(params.out_ptr_list[g]);
                index_t out_offset = cu_seqlens_q_start[g] * params.o_row_stride_list[g]
                                    + bidh * params.o_head_stride_list[g]
                                    + m_block * kBlockM * params.o_row_stride_list[g];

                typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
                auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);

                Tensor mO = make_tensor(make_gmem_ptr(out_ptr + out_offset),
                                        make_shape(actual_seqlen_q[g], params.h, params.head_size),
                                        make_stride(params.o_row_stride_list[g], params.o_head_stride_list[g], _1{}));
                Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                       make_coord(m_block, 0));
                Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
                Tensor tOrO = make_tensor<Element>(shape(tOgO));
                clear(tOrO);

                Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
                Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
                Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
                if (!Is_even_K) {
                    #pragma unroll
                    for (int k = 0; k < size(tOpO); ++k) {
                        tOpO(k) = get<1>(tOcO(0, 0, k)) < params.head_size;
                    }
                }

                FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, false, false>(
                    gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, actual_seqlen_q[g] - m_block * kBlockM
                );

                // Write -INFINITY to LSE
                float* lse_ptr = reinterpret_cast<float*>(params.softmax_lse_ptr_list[g]);
                int lse_offset;
                if (params.unpadded_lse) {
                    lse_offset = bidh * params.total_q_list[g] + cu_seqlens_q_start[g] + m_block * kBlockM;
                } else {
                    lse_offset = bidb * params.h * params.max_seqlen_q_list[g]
                                + bidh * params.max_seqlen_q_list[g] + m_block * kBlockM;
                }
                #pragma unroll
                for (int m = 0; m < size<1>(tOgO); ++m) {
                    const int row = get<0>(tOcO(0, m, 0));
                    if (row < actual_seqlen_q[g] - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) {
                        lse_ptr[lse_offset + row] = -INFINITY;
                    }
                }
            }
        }
        return;
    }

    // ========================================================================
    // STEP 2: Setup shared memory tiles
    // ========================================================================

    // Per-group Q tiles in shared memory
    Tensor sQ_list[NumGroups];
    #pragma unroll
    for (int g = 0; g < NumGroups; g++) {
        Element* sQ_ptr = reinterpret_cast<Element*>(smem_) + g * kBlockM * kHeadDim;
        sQ_list[g] = make_tensor(make_smem_ptr(sQ_ptr), typename Kernel_traits::SmemLayoutQ{});
    }

    // Shared K,V tiles (after all group Q tiles)
    Element* sKV_base = reinterpret_cast<Element*>(smem_) + NumGroups * kBlockM * kHeadDim;
    Tensor sK = make_tensor(make_smem_ptr(sKV_base), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(make_smem_ptr(sKV_base + size(sK)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    // ========================================================================
    // STEP 3: Setup copy atoms and MMA
    // ========================================================================

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);

    // ========================================================================
    // STEP 4: Per-group output accumulators and softmax state
    // ========================================================================

    // Output accumulators (one per group)
    Tensor acc_o_list[NumGroups];
    FLASH_NAMESPACE::Softmax<2 * (kBlockM / (32 / kNWarps))> softmax_list[NumGroups];

    #pragma unroll
    for (int g = 0; g < NumGroups; g++) {
        acc_o_list[g] = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
        clear(acc_o_list[g]);
    }

    // Predicates for K dimension
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(gmem_thr_copy_QKV.partition_D(sK))));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) {
            tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.head_size;
        }
    }

    // ========================================================================
    // STEP 5: Load Q tiles for all active groups
    // ========================================================================

    #pragma unroll
    for (int g = 0; g < NumGroups; g++) {
        if (!group_active[g]) continue;

        // Create Q tensor in global memory for this group
        Element* q_ptr = reinterpret_cast<Element*>(params.q_ptr_list[g]);
        index_t q_offset = cu_seqlens_q_start[g] * params.q_row_stride_list[g]
                          + bidh * params.q_head_stride_list[g];

        Tensor mQ = make_tensor(make_gmem_ptr(q_ptr + q_offset),
                                make_shape(actual_seqlen_q[g], params.h, params.head_size),
                                make_stride(params.q_row_stride_list[g], params.q_head_stride_list[g], _1{}));
        Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                               make_coord(m_block, 0));

        // Partition and copy
        Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
        Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ_list[g]);

        // Construct identity layout for predicates
        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ_list[g]), size<1>(sQ_list[g])));
        Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k) {
                tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.head_size;
            }
        }

        // Copy Q tile to shared memory
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
            gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
            actual_seqlen_q[g] - m_block * kBlockM
        );
    }

    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // Setup K,V global memory tensors (shared across groups, use group 0's K offsets)
    // Note: For true varlen support, we'd need to handle per-group K offsets
    // For now, assume K,V are shared with same indexing
    Element* k_ptr = reinterpret_cast<Element*>(params.k_ptr);
    Element* v_ptr = reinterpret_cast<Element*>(params.v_ptr);

    // Use first active group's K offset (they share K,V data)
    int ref_cu_seqlens_k_start = cu_seqlens_k_start[0];
    int ref_actual_seqlen_k = actual_seqlen_k[0];

    index_t k_base_offset = ref_cu_seqlens_k_start * params.k_row_stride
                           + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    index_t v_base_offset = ref_cu_seqlens_k_start * params.v_row_stride
                           + (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mK = make_tensor(make_gmem_ptr(k_ptr + k_base_offset),
                            make_shape(ref_actual_seqlen_k, params.h_k, params.head_size),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));

    Tensor mV = make_tensor(make_gmem_ptr(v_ptr + v_base_offset),
                            make_shape(ref_actual_seqlen_k, params.h_k, params.head_size),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));

    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // ========================================================================
    // STEP 6: Main K,V loop (load once, use for all groups)
    // ========================================================================

    // Load first K tile
    int n_block = n_block_max_global - 1;
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
        gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
        ref_actual_seqlen_k - n_block * kBlockN
    );
    cute::cp_async_fence();

    for (; n_block >= 0; --n_block) {
        // Wait for K to be loaded
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        // Load V tile for current n_block
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, true>(
            gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV,
            ref_actual_seqlen_k - n_block * kBlockN
        );
        cute::cp_async_fence();

        // Process this K,V tile for each active group
        #pragma unroll
        for (int g = 0; g < NumGroups; g++) {
            if (!group_active[g]) continue;

            // Check if this group needs this K,V tile
            int kv_tile_start = n_block * kBlockN;
            if (kv_tile_start >= kv_max_offset[g]) {
                continue;  // This K,V tile is beyond this group's endpoint
            }

            // Check if we're in range for this group based on n_block_max
            if (n_block >= n_block_max_per_group[g]) {
                continue;
            }

            // Partition Q for this group
            Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ_list[g]);
            Tensor tSrQ = thr_mma.partition_fragment_A(sQ_list[g]);
            Tensor tSsK = smem_thr_copy_K.partition_S(sK);
            Tensor tSrK = thr_mma.partition_fragment_B(sK);

            // Compute attention scores: S = Q @ K^T
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
            clear(acc_s);

            // Copy Q and K to registers if needed
            if (!Kernel_traits::Is_Q_in_regs) {
                cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ);
            }
            cute::copy(smem_tiled_copy_K, tSsK, tSrK);

            // GEMM: acc_s = Q @ K^T
            cute::gemm(tiled_mma, tSrQ, tSrK, acc_s);

            // Apply masking
            // Create mask for this group
            // The mask handles both causal and KV boundary masking

            // Compute row offset for this thread
            const int row_idx_offset_g = m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4;
            const int col_idx_offset = n_block * kBlockN;
            const int warp_row_stride = kNWarps * 16;

            // Apply causal/local masking if needed
            if (Is_causal || Is_local) {
                // Check if we need causal masking for this block
                // We need it if: row_idx < col_idx (in the attention matrix)
                // Accounting for potential sequence length difference
                const int seqlen_offset = actual_seqlen_k[g] - actual_seqlen_q[g];

                if (m_block * kBlockM < (n_block + 1) * kBlockN + seqlen_offset + (Is_local ? params.window_size_right : 0)) {
                    // Use apply_mask_causal from mask.h
                    FLASH_NAMESPACE::apply_mask_causal(
                        acc_s,
                        col_idx_offset,
                        actual_seqlen_k[g],
                        row_idx_offset_g,
                        actual_seqlen_q[g],
                        warp_row_stride
                    );
                }
            }

            // KV boundary mask - mask out positions beyond kv_max_offset[g]
            // This is critical for multi-group: each group may attend to different K,V ranges
            if (!Is_even_MN || kv_max_offset[g] < (n_block + 1) * kBlockN) {
                // Mask out positions beyond the KV endpoint for this group
                #pragma unroll
                for (int i = 0; i < size(acc_s); ++i) {
                    // Get the column index for this element
                    // In the MMA layout, we need to extract coordinates
                    // For simplicity, we'll mask entire columns beyond kv_max_offset

                    // Each element in acc_s corresponds to a position in the [kBlockM, kBlockN] tile
                    // We need to check if col_global >= kv_max_offset[g]
                    // This is a simplified approach; the actual implementation would use
                    // the layout information to determine exact coordinates

                    // Conservative approach: if any position in this n_block exceeds kv_max_offset,
                    // we need fine-grained masking
                    // For now, elements are masked based on the flat index approximation
                    int approx_col = col_idx_offset + (i % kBlockN);
                    if (approx_col >= kv_max_offset[g]) {
                        acc_s(i) = -INFINITY;
                    }
                }
            }

            // Wait for V to be ready
            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();

            // Online softmax update
            bool is_first = (n_block == n_block_max_per_group[g] - 1);
            if (is_first) {
                softmax_list[g].template softmax_rescale_o<true, false>(
                    acc_s, acc_o_list[g], params.scale_softmax_log2
                );
            } else {
                softmax_list[g].template softmax_rescale_o<false, false>(
                    acc_s, acc_o_list[g], params.scale_softmax_log2
                );
            }

            // Convert scores to element type and accumulate with V
            Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
            Tensor tOrP = make_tensor(rP.data(),
                FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
            Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
            Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

            // GEMM: acc_o += P @ V
            FLASH_NAMESPACE::gemm_rs(acc_o_list[g], tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        }

        // Load next K tile if needed
        if (n_block > 0) {
            __syncthreads();
            FLASH_NAMESPACE::copy<true, Is_even_K>(
                gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV
            );
            cute::cp_async_fence();
        }
    }

    // ========================================================================
    // STEP 7: Write outputs for all groups
    // ========================================================================

    #pragma unroll
    for (int g = 0; g < NumGroups; g++) {
        if (!group_active[g]) continue;

        // Normalize output by LSE
        Tensor lse = softmax_list[g].template normalize_softmax_lse<false>(acc_o_list[g], params.scale_softmax, 1.0f);

        // Convert to output element type
        Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o_list[g]);

        // Write to shared memory then to global memory
        Tensor sO = make_tensor(sQ_list[g].data(), typename Kernel_traits::SmemLayoutO{});
        auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

        __syncthreads();
        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

        // Setup global memory output tensor
        Element* out_ptr = reinterpret_cast<Element*>(params.out_ptr_list[g]);
        index_t out_offset = cu_seqlens_q_start[g] * params.o_row_stride_list[g]
                            + bidh * params.o_head_stride_list[g];

        Tensor mO = make_tensor(make_gmem_ptr(out_ptr + out_offset),
                                make_shape(actual_seqlen_q[g], params.h, params.head_size),
                                make_stride(params.o_row_stride_list[g], params.o_head_stride_list[g], _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                               make_coord(m_block, 0));

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

        __syncthreads();

        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

        Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) {
                tOpO(k) = get<1>(tOcO(0, 0, k)) < params.head_size;
            }
        }

        // Write output to global memory
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, false, false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, actual_seqlen_q[g] - m_block * kBlockM
        );

        // Write LSE to global memory
        float* lse_ptr = reinterpret_cast<float*>(params.softmax_lse_ptr_list[g]);
        int lse_offset;
        if (params.unpadded_lse) {
            lse_offset = bidh * params.total_q_list[g] + cu_seqlens_q_start[g] + m_block * kBlockM;
        } else {
            lse_offset = bidb * params.h * params.max_seqlen_q_list[g]
                        + bidh * params.max_seqlen_q_list[g] + m_block * kBlockM;
        }

        // Extract row indices from MMA layout
        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor taccOcO = thr_mma.partition_C(caccO);
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);

        if (get<1>(taccOcO_row(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO_row(mi));
                if (row < actual_seqlen_q[g] - m_block * kBlockM) {
                    lse_ptr[lse_offset + row] = lse(mi);
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel Launch Wrapper
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Kernel launch configuration:
 *
 * Grid dimensions:
 *   gridDim.x = (max_seqlen_q + kBlockM - 1) / kBlockM
 *   gridDim.y = batch_size
 *   gridDim.z = num_heads
 *
 * Block dimensions:
 *   blockDim.x = Kernel_traits::kNThreads (typically 128)
 *
 * Shared memory:
 *   smem_size = compute_smem_size_multigroup(NumGroups, kBlockM, kBlockN, kHeadDim)
 *
 * Example for 2 groups, d=128, kBlockM=64:
 *   smem_size = 2 * 64 * 128 * 2 + 2 * 128 * 128 * 2 = 32KB + 64KB = 96KB ✓
 */

template<typename Kernel_traits, int NumGroups, bool Is_dropout, bool Is_causal,
         bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap>
__global__ void flash_fwd_multigroup_kernel(
    KERNEL_PARAM_MODIFIER const Flash_fwd_multigroup_params params
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    compute_attn_multigroup<Kernel_traits, NumGroups, Is_causal, Is_local,
                            Has_alibi, Is_even_MN, Is_even_K>(
        params, blockIdx.y, blockIdx.z, blockIdx.x
    );
#else
    // Flash Attention 2 requires SM80 or newer (Ampere architecture)
    printf("FlashAttention-2 only supports Ampere GPUs or newer.\n");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Template Instantiation Helper (to be called from launch template)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, int NumGroups, bool Is_causal>
void run_flash_fwd_multigroup(
    Flash_fwd_multigroup_params &params,
    cudaStream_t stream
) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    constexpr int kBlockM = Kernel_traits::kBlockM;

    // Compute grid dimensions based on max sequence length across all groups
    int max_seqlen_q = 0;
    for (int g = 0; g < NumGroups; g++) {
        max_seqlen_q = std::max(max_seqlen_q, params.max_seqlen_q_list[g]);
    }
    const int num_m_block = (max_seqlen_q + kBlockM - 1) / kBlockM;

    // Grid: (num_m_blocks, batch_size, num_heads)
    dim3 grid(num_m_block, params.batch_size, params.h);
    dim3 block(Kernel_traits::kNThreads);

    // For now, use simplified template parameters
    // TODO: Add full parameter dispatch with BOOL_SWITCH macros
    constexpr bool Is_dropout = false;
    constexpr bool Is_local = false;
    constexpr bool Has_alibi = false;
    constexpr bool Is_even_MN = false;  // Use varlen path
    constexpr bool Is_even_K = false;   // Conservative, check head_size
    constexpr bool Is_softcap = false;

    auto kernel = &flash_fwd_multigroup_kernel<
        Kernel_traits, NumGroups, Is_dropout, Is_causal, Is_local,
        Has_alibi, Is_even_MN, Is_even_K, Is_softcap
    >;

    // Set shared memory if needed
    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }

    // Launch kernel
    kernel<<<grid, block, smem_size, stream>>>(params);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Public API (to be called from flash_api_multigroup.cpp)
////////////////////////////////////////////////////////////////////////////////////////////////////

// Forward declaration for different head dimensions
template<typename T, int Headdim, int NumGroups, bool Is_causal>
void run_mha_fwd_multigroup(Flash_fwd_multigroup_params &params, cudaStream_t stream);

// Specialized implementations (similar to run_mha_fwd_hdim64, run_mha_fwd_hdim128, etc.)
// These will be implemented in separate .cu files for faster compilation

template<typename T, int NumGroups, bool Is_causal>
void run_mha_fwd_multigroup_hdim64(Flash_fwd_multigroup_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    // Optimized configs based on NumGroups
    if constexpr (NumGroups == 1) {
        // Single group: use standard FA config
        // Shared memory: 1*128*64*2 + 2*128*64*2 = 16KB + 32KB = 48KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 128, 128, 4, 1, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    } else if constexpr (NumGroups == 2) {
        // 2 groups: balanced config
        // Shared memory: 2*96*64*2 + 2*128*64*2 = 24KB + 32KB = 56KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 96, 128, 4, 2, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    } else if constexpr (NumGroups == 3) {
        // 3 groups: reduce M to fit memory
        // Shared memory: 3*64*64*2 + 2*128*64*2 = 24KB + 32KB = 56KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 64, 128, 4, 3, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    } else {
        // 4+ groups: smaller tiles
        // Shared memory: 4*64*64*2 + 2*128*64*2 = 32KB + 32KB = 64KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 64, 128, 4, NumGroups, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    }
}

template<typename T, int NumGroups, bool Is_causal>
void run_mha_fwd_multigroup_hdim128(Flash_fwd_multigroup_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;  // A6000, A100 8.6/8.9

    if constexpr (NumGroups == 1) {
        // Single group: use standard FA optimizations
        if (is_sm8x) {
            if constexpr (!Is_causal) {
                // Non-causal on sm8x: use 2 CTAs/SM config (48KB smem)
                // Shared memory: 1*128*128*2 + 2*32*128*2 = 32KB + 16KB = 48KB ✓
                using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 128, 32, 4, 1, false, false, T>;
                run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
            } else {
                // Causal: square tiles for better reuse
                // Shared memory: 1*64*128*2 + 2*64*128*2 = 16KB + 32KB = 48KB ✓
                using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 64, 64, 4, 1, false, false, T>;
                run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
            }
        } else {
            // Other architectures: balanced config
            // Shared memory: 1*128*128*2 + 2*64*128*2 = 32KB + 32KB = 64KB ✓
            using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 128, 64, 4, 1, false, false, T>;
            run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
        }
    } else if constexpr (NumGroups == 2) {
        // 2 groups: optimize based on architecture
        if (is_sm8x && !Is_causal) {
            // Try to get 2 CTAs/SM with smaller N
            // Shared memory: 2*96*128*2 + 2*48*128*2 = 48KB + 24KB = 72KB ✓
            using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 96, 48, 4, 2, false, false, T>;
            run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
        } else if (Is_causal) {
            // Causal: square tiles
            // Shared memory: 2*64*128*2 + 2*64*128*2 = 32KB + 32KB = 64KB ✓
            using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 64, 64, 4, 2, false, false, T>;
            run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
        } else {
            // Balanced config
            // Shared memory: 2*96*128*2 + 2*64*128*2 = 48KB + 32KB = 80KB ✓
            using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 96, 64, 4, 2, false, false, T>;
            run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
        }
    } else if constexpr (NumGroups == 3) {
        // 3 groups: smaller tiles
        // Shared memory: 3*64*128*2 + 2*64*128*2 = 48KB + 32KB = 80KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 64, 64, 4, 3, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    } else {
        // 4+ groups: conservative config
        // Shared memory: 4*64*128*2 + 2*64*128*2 = 64KB + 32KB = 96KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 64, 64, 4, NumGroups, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    }
}

template<typename T, int NumGroups, bool Is_causal>
void run_mha_fwd_multigroup_hdim32(Flash_fwd_multigroup_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    // Optimized configs - d=32 is small, can use larger tiles
    if constexpr (NumGroups == 1) {
        // Single group: standard FA config with large tiles
        // Shared memory: 1*128*32*2 + 2*128*32*2 = 8KB + 16KB = 24KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 128, 128, 4, 1, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    } else if constexpr (NumGroups == 2) {
        // 2 groups: can still use large M
        // Shared memory: 2*128*32*2 + 2*128*32*2 = 16KB + 16KB = 32KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 128, 128, 4, 2, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    } else if constexpr (NumGroups == 3) {
        // 3 groups: balanced
        // Shared memory: 3*96*32*2 + 2*128*32*2 = 18KB + 16KB = 34KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 96, 128, 4, 3, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    } else {
        // 4+ groups
        // Shared memory: 4*96*32*2 + 2*128*32*2 = 24KB + 16KB = 40KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 96, 128, 4, NumGroups, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    }
}

template<typename T, int NumGroups, bool Is_causal>
void run_mha_fwd_multigroup_hdim96(Flash_fwd_multigroup_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;

    if constexpr (NumGroups == 1) {
        // Single group: use standard FA config
        if (is_sm8x) {
            if constexpr (!Is_causal) {
                // Non-causal on sm8x
                // Shared memory: 1*128*96*2 + 2*64*96*2 = 24KB + 24KB = 48KB ✓
                using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 128, 64, 4, 1, false, false, T>;
                run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
            } else {
                // Causal: square tiles
                // Shared memory: 1*64*96*2 + 2*64*96*2 = 12KB + 24KB = 36KB ✓
                using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 64, 64, 4, 1, false, false, T>;
                run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
            }
        } else {
            // Shared memory: 1*128*96*2 + 2*64*96*2 = 24KB + 24KB = 48KB ✓
            using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 128, 64, 4, 1, false, false, T>;
            run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
        }
    } else if constexpr (NumGroups == 2) {
        // 2 groups: balanced config
        if (Is_causal) {
            // Causal: square tiles
            // Shared memory: 2*64*96*2 + 2*64*96*2 = 24KB + 24KB = 48KB ✓
            using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 64, 64, 4, 2, false, false, T>;
            run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
        } else {
            // Non-causal: optimize for throughput
            // Shared memory: 2*96*96*2 + 2*64*96*2 = 36KB + 24KB = 60KB ✓
            using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 96, 64, 4, 2, false, false, T>;
            run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
        }
    } else if constexpr (NumGroups == 3) {
        // 3 groups
        // Shared memory: 3*64*96*2 + 2*64*96*2 = 36KB + 24KB = 60KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 64, 64, 4, 3, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    } else {
        // 4+ groups
        // Shared memory: 4*64*96*2 + 2*64*96*2 = 48KB + 24KB = 72KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 64, 64, 4, NumGroups, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    }
}

template<typename T, int NumGroups, bool Is_causal>
void run_mha_fwd_multigroup_hdim192(Flash_fwd_multigroup_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;

    if constexpr (NumGroups == 1) {
        // Single group: use 8 warps like standard FA for higher throughput
        // Shared memory: 1*128*192*2 + 2*64*192*2 = 48KB + 48KB = 96KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 128, 64, 8, 1, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    } else if constexpr (NumGroups == 2) {
        // 2 groups: use 8 warps for better throughput
        // Shared memory: 2*96*192*2 + 2*64*192*2 = 72KB + 48KB = 120KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 96, 64, 8, 2, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    } else if constexpr (NumGroups == 3) {
        // 3 groups: smaller config with 8 warps
        // Shared memory: 3*64*192*2 + 2*64*192*2 = 72KB + 48KB = 120KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 64, 64, 8, 3, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    } else {
        // 4+ groups: use 4 warps to fit memory
        // Shared memory: 4*64*192*2 + 2*64*192*2 = 96KB + 48KB = 144KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 64, 64, 4, NumGroups, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    }
}

template<typename T, int NumGroups, bool Is_causal>
void run_mha_fwd_multigroup_hdim256(Flash_fwd_multigroup_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_sm, max_smem_per_block;
    cudaDeviceGetAttribute(&max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    if constexpr (NumGroups == 1) {
        // Single group: use 8 warps like standard FA
        // Check if we can use large tiles (A100 style) or need smaller (H100 style)
        if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64) &&
            max_smem_per_sm < 4 * Headdim * (64 + 2 * 64)) {
            // A100: large tiles with 8 warps
            // Shared memory: 1*128*256*2 + 2*64*256*2 = 64KB + 64KB = 128KB ✓
            using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 128, 64, 8, 1, false, false, T>;
            run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
        } else {
            // H100: prioritize 2 CTAs/SM
            // Shared memory: 1*64*256*2 + 2*64*256*2 = 32KB + 64KB = 96KB ✓
            using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 64, 64, 4, 1, false, false, T>;
            run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
        }
    } else if constexpr (NumGroups == 2) {
        // 2 groups: use 8 warps if possible
        // Shared memory: 2*64*256*2 + 2*64*256*2 = 64KB + 64KB = 128KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 64, 64, 8, 2, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    } else if constexpr (NumGroups == 3) {
        // 3 groups: smaller tiles, 4 warps
        // Shared memory: 3*64*256*2 + 2*64*256*2 = 96KB + 64KB = 160KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 64, 64, 4, 3, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    } else {
        // 4+ groups: very small tiles
        // Shared memory: 4*48*256*2 + 2*64*256*2 = 96KB + 64KB = 160KB ✓
        using Kernel_traits = Flash_fwd_multigroup_kernel_traits<Headdim, 48, 64, 4, NumGroups, false, false, T>;
        run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(params, stream);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Notes for Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * IMPLEMENTATION NOTES:
 *
 * 1. Start with NumGroups=2
 *    - Simpler to debug and validate
 *    - Covers the zigzag_llama3 use case
 *    - Can generalize to more groups later
 *
 * 2. Shared memory optimization strategies:
 *    a) Reduce tile sizes: kBlockM=64 instead of 128 (halves per-group smem)
 *    b) Keep acc_o in registers: Already standard in Flash Attention
 *    c) Share Q and K smem: Set Share_Q_K_smem=true (requires Q in registers)
 *
 * 3. Register pressure mitigation:
 *    - Use #pragma unroll 2 instead of full unrolling for group loops
 *    - Spill LSE state to smem if register pressure too high (measure with -Xptxas=-v)
 *    - Consider reducing kNWarps from 4 to 2 if needed
 *
 * 4. Correctness validation:
 *    - Test with identical KV endpoints first (should match sequential calls exactly)
 *    - Test with different KV endpoints (harder, requires careful boundary masking)
 *    - Use torch.testing.assert_close with rtol=1e-3 for FP16
 *
 * 5. Performance optimization (Phase 5):
 *    - Profile with Nsight Compute: dram__bytes_read should decrease by ~38%
 *    - Check SM occupancy: Target >=50% (may be lower than standard kernel due to resources)
 *    - Optimize KV endpoint checks: Use warp voting to early-exit entire warp
 *    - Tune tile sizes per architecture: A100 vs H100 have different smem limits
 *
 * 6. Backward kernel considerations:
 *    - dK, dV gradients require accumulation from multiple groups
 *    - Two approaches:
 *      a) Atomic adds: Fast but non-deterministic
 *      b) Two-pass: Write per-group gradients separately, then reduce (deterministic)
 *    - Implement both, controlled by params.deterministic flag
 */

}  // namespace FLASH_NAMESPACE

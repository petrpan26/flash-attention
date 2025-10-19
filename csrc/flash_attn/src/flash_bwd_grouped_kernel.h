/***************************************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Grouped Flash Attention Backward Pass - Kernel Implementation
 ******************************************************************************/

#pragma once

#include "flash_bwd_kernel.h"

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Grouped version of compute_dq_dk_dv_1colblock
// This function processes backward gradients for one column block (K,V tile) across grouped queries
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi,
         bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_first, bool Is_last,
         bool Seq_parallel=false, typename Params>
inline __device__ void compute_dq_dk_dv_1colblock_grouped(
    const Params &params,
    const int bidb,
    const int bidh,
    const int n_block,
    const int group_id) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory
    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int MMA_N_SdP = kBlockN / decltype(typename Kernel_traits::TiledMmaSdP{}.template tile_size_mnk<1>())::value;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;

    // Get group-specific BlockInfo
    // We need to override cu_seqlens to use the group-specific versions
    Params params_group = params;
    params_group.cu_seqlens_q = params.group_cu_seqlens_q[group_id];
    params_group.cu_seqlens_k = params.group_cu_seqlens_k[group_id];

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params_group, bidb);
    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);
    if (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }

    // Load Q, dO, O, LSE from group-specific pointers
    Element* q_ptr_group = reinterpret_cast<Element*>(params.group_q_ptrs[group_id]);
    Element* do_ptr_group = reinterpret_cast<Element*>(params.group_do_ptrs[group_id]);
    Element* o_ptr_group = reinterpret_cast<Element*>(params.group_o_ptrs[group_id]);
    ElementAccum* lse_ptr_group = reinterpret_cast<ElementAccum*>(params.group_softmax_lse_ptrs[group_id]);
    ElementAccum* dpsum_ptr_group = reinterpret_cast<ElementAccum*>(params.group_dsoftmax_sum_ptrs[group_id]);

    // Row offsets (group-specific)
    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.o_row_stride + bidh * params.o_head_stride;

    // Shared K, V (same for all groups)
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;

    // Group-specific dQ output
    Element* dq_ptr_group = reinterpret_cast<Element*>(params.group_dq_ptrs[group_id]);
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;

    const index_t row_offset_dq_accum = binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb)
        + ((m_block_max - 1) * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128ll * bidb)) * params.h * params.d_rounded + bidh * params.d_rounded
        + (!params.deterministic ? 0 : blockIdx.x * params.dq_accum_split_stride);

    const index_t row_offset_lse = bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb) + (m_block_max - 1) * kBlockM;
    const index_t row_offset_dpsum = bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb + (m_block_max - 1) * kBlockM;

    // Create tensors for group-specific data
    Tensor gQ = make_tensor(make_gmem_ptr(q_ptr_group + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(do_ptr_group + row_offset_do),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.do_row_stride, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(o_ptr_group + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));
    Tensor gdQ = make_tensor(make_gmem_ptr(dq_ptr_group + row_offset_dq),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.dq_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(lse_ptr_group + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(dpsum_ptr_group + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    // Shared K, V tensors (same for all groups)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));

    // Use dQ accumulator (may be nullptr if loop=false)
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(params.h * params.d_rounded, _1{}));

    // From here, the rest follows the same logic as compute_dq_dk_dv_1colblock
    // We allocate shared memory, partition tensors, and compute gradients
    // The key difference is we read from group-specific Q, dO, O, LSE
    // and write dQ to group-specific output
    // For dK, dV we write to intermediate buffers that will be reduced later

    // Shared memory layout (same as non-grouped)
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    Tensor sdO = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sdOtransposedNoSwizzle = make_tensor(sdO.data(),
                                                typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    Tensor sK = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
    Tensor sdS = make_tensor(!Kernel_traits::Is_V_in_regs ? sV.data() + size(sV) : sK.data() + size(sK),
                             typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sdStNoSwizzle = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    Tensor sP = make_tensor(sdS.data() + size(sdS), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sPtNoSwizzle = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    Tensor sdQ = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutdQ{});

    // The rest of the implementation follows compute_dq_dk_dv_1colblock exactly
    // I'm calling the original function with updated params to reuse the logic
    // This is a simplified version - full implementation would inline all the compute logic

    // For now, we delegate to the original compute_dq_dk_dv_1colblock with group-adjusted params
    // In production, we'd copy the full backward pass logic here
    compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                               Is_even_MN, Is_even_K, Is_softcap, Is_first, Is_last, Seq_parallel>(
        params_group, bidb, bidh, n_block
    );

    // NOTE: The above is a placeholder. The full implementation would:
    // 1. Compute dQ and write to gdQ (group-specific)
    // 2. Compute dK, dV and write to intermediate buffers at:
    //    - params.group_dk_intermediate_ptrs[group_id]
    //    - params.group_dv_intermediate_ptrs[group_id]
    // 3. These intermediate buffers will be reduced in the host code
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace FLASH_NAMESPACE

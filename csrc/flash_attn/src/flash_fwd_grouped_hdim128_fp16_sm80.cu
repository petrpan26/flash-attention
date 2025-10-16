/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Grouped flash attention kernel instantiation for fp16, headdim 128, sm80+
 ******************************************************************************/

#include "flash_fwd_launch_template.h"
#include <cuda_runtime.h>

template<>
void run_mha_fwd_grouped_<cutlass::half_t, 128, false>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;

    // Compute total number of M blocks by summing from device array
    int total_num_m_blocks = 0;
    std::vector<int> host_num_m_blocks(params.num_groups);
    cudaMemcpy(host_num_m_blocks.data(), params.group_num_m_blocks, params.num_groups * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < params.num_groups; i++) {
        total_num_m_blocks += host_num_m_blocks[i];
    }

    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            if (is_sm8x) {
                run_flash_fwd_grouped<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, cutlass::half_t>, Is_dropout, false>(params, stream, total_num_m_blocks);
            } else {
                run_flash_fwd_grouped<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, cutlass::half_t>, Is_dropout, false>(params, stream, total_num_m_blocks);
            }
        } else {
            run_flash_fwd_grouped<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, cutlass::half_t>, Is_dropout, false>(params, stream, total_num_m_blocks);
        }
    });
}

template<>
void run_mha_fwd_grouped_<cutlass::half_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;

    // Compute total number of M blocks by summing from device array
    int total_num_m_blocks = 0;
    std::vector<int> host_num_m_blocks(params.num_groups);
    cudaMemcpy(host_num_m_blocks.data(), params.group_num_m_blocks, params.num_groups * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < params.num_groups; i++) {
        total_num_m_blocks += host_num_m_blocks[i];
    }

    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            if (is_sm8x) {
                run_flash_fwd_grouped<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, cutlass::half_t>, Is_dropout, true>(params, stream, total_num_m_blocks);
            } else {
                run_flash_fwd_grouped<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, cutlass::half_t>, Is_dropout, true>(params, stream, total_num_m_blocks);
            }
        } else {
            run_flash_fwd_grouped<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, cutlass::half_t>, Is_dropout, true>(params, stream, total_num_m_blocks);
        }
    });
}


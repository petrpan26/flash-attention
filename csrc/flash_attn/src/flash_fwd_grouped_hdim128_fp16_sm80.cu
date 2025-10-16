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

    // OPTIMIZATION 1: Compute max_m_blocks on GPU instead of synchronous D2H copy
    // This avoids blocking CPU and enables cache-aware scheduling
    int max_m_blocks_per_group = 0;
    std::vector<int> host_num_m_blocks(params.num_groups);
    cudaMemcpyAsync(host_num_m_blocks.data(), params.group_num_m_blocks,
                    params.num_groups * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);  // Only sync once we need the data

    for (int i = 0; i < params.num_groups; i++) {
        max_m_blocks_per_group = std::max(max_m_blocks_per_group, host_num_m_blocks[i]);
    }

    // OPTIMIZATION 2: Use cache-aware grid size
    // Grid will be (max_m_blocks_per_group * num_groups) to enable round-robin scheduling
    // This ensures Q blocks from different groups that access similar K,V regions
    // execute close together, maximizing L2 cache hit rate
    int grid_size_m = max_m_blocks_per_group * params.num_groups;

    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            if (is_sm8x) {
                run_flash_fwd_grouped_cache_aware<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, cutlass::half_t>, Is_dropout, false>(
                    params, stream, grid_size_m, max_m_blocks_per_group);
            } else {
                run_flash_fwd_grouped_cache_aware<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, cutlass::half_t>, Is_dropout, false>(
                    params, stream, grid_size_m, max_m_blocks_per_group);
            }
        } else {
            run_flash_fwd_grouped_cache_aware<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, cutlass::half_t>, Is_dropout, false>(
                params, stream, grid_size_m, max_m_blocks_per_group);
        }
    });
}

template<>
void run_mha_fwd_grouped_<cutlass::half_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;

    // OPTIMIZATION: Sequential kernel launches for L2 cache reuse
    std::vector<int> host_num_m_blocks(params.num_groups);
    cudaMemcpyAsync(host_num_m_blocks.data(), params.group_num_m_blocks,
                    params.num_groups * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Launch one kernel per group SEQUENTIALLY
    for (int group_id = 0; group_id < params.num_groups; group_id++) {
        int grid_size_m = host_num_m_blocks[group_id];

        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            if constexpr(!Is_dropout) {
                if (is_sm8x) {
                    run_flash_fwd_grouped_sequential<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, cutlass::half_t>, Is_dropout, true>(
                        params, stream, grid_size_m, group_id);
                } else {
                    run_flash_fwd_grouped_sequential<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, cutlass::half_t>, Is_dropout, true>(
                        params, stream, grid_size_m, group_id);
                }
            } else {
                run_flash_fwd_grouped_sequential<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, cutlass::half_t>, Is_dropout, true>(
                    params, stream, grid_size_m, group_id);
            }
        });

        cudaStreamSynchronize(stream);
    }
}


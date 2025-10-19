/************************************************************************************* * Copyright (c) 2024, Tri Dao.
 * Grouped flash attention kernel instantiation for fp16, headdim 64, sm80+
 ******************************************************************************/

#include "flash_fwd_launch_template.h"
#include <cuda_runtime.h>

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_grouped_<cutlass::half_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;

    int max_m_blocks_per_group = 0;
    std::vector<int> host_num_m_blocks(params.num_groups);
    cudaMemcpyAsync(host_num_m_blocks.data(), params.group_num_m_blocks,
                    params.num_groups * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < params.num_groups; i++) {
        max_m_blocks_per_group = std::max(max_m_blocks_per_group, host_num_m_blocks[i]);
    }

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;

    // OPTIMIZATION: Use SMEM K,V sharing kernel for exactly 2 groups
    if (params.num_groups == 2) {
        // Grid size is just max_m_blocks (each block processes both groups)
        int grid_size_m = max_m_blocks_per_group;

        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            if constexpr(!Is_dropout) {
                if (is_sm8x) {
                    run_flash_fwd_2groups_smem_share<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, cutlass::half_t>, Is_dropout, false>(
                        params, stream, grid_size_m);
                } else {
                    run_flash_fwd_2groups_smem_share<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, cutlass::half_t>, Is_dropout, false>(
                        params, stream, grid_size_m);
                }
            } else {
                if (is_sm8x) {
                    run_flash_fwd_2groups_smem_share<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, cutlass::half_t>, Is_dropout, false>(
                        params, stream, grid_size_m);
                } else {
                    run_flash_fwd_2groups_smem_share<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, cutlass::half_t>, Is_dropout, false>(
                        params, stream, grid_size_m);
                }
            }
        });
    } else {
        // Use L2 cache-aware round-robin kernel for 3+ groups
        int grid_size_m = max_m_blocks_per_group * params.num_groups;

        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            if constexpr(!Is_dropout) {
                if (is_sm8x) {
                    run_flash_fwd_grouped_cache_aware<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, cutlass::half_t>, Is_dropout, false>(
                        params, stream, grid_size_m);
                } else {
                    run_flash_fwd_grouped_cache_aware<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, cutlass::half_t>, Is_dropout, false>(
                        params, stream, grid_size_m);
                }
            } else {
                if (is_sm8x) {
                    run_flash_fwd_grouped_cache_aware<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, cutlass::half_t>, Is_dropout, false>(
                        params, stream, grid_size_m);
                } else {
                    run_flash_fwd_grouped_cache_aware<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, cutlass::half_t>, Is_dropout, false>(
                        params, stream, grid_size_m);
                }
            }
        });
    }
}

template<>
void run_mha_fwd_grouped_<cutlass::half_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;

    int max_m_blocks_per_group = 0;
    std::vector<int> host_num_m_blocks(params.num_groups);
    cudaMemcpyAsync(host_num_m_blocks.data(), params.group_num_m_blocks,
                    params.num_groups * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < params.num_groups; i++) {
        max_m_blocks_per_group = std::max(max_m_blocks_per_group, host_num_m_blocks[i]);
    }

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;

    // OPTIMIZATION: Use SMEM K,V sharing kernel for exactly 2 groups
    if (params.num_groups == 2) {
        // Grid size is just max_m_blocks (each block processes both groups)
        int grid_size_m = max_m_blocks_per_group;

        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            if constexpr(!Is_dropout) {
                if (is_sm8x) {
                    run_flash_fwd_2groups_smem_share<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, cutlass::half_t>, Is_dropout, true>(
                        params, stream, grid_size_m);
                } else {
                    run_flash_fwd_2groups_smem_share<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, cutlass::half_t>, Is_dropout, true>(
                        params, stream, grid_size_m);
                }
            } else {
                if (is_sm8x) {
                    run_flash_fwd_2groups_smem_share<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, cutlass::half_t>, Is_dropout, true>(
                        params, stream, grid_size_m);
                } else {
                    run_flash_fwd_2groups_smem_share<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, cutlass::half_t>, Is_dropout, true>(
                        params, stream, grid_size_m);
                }
            }
        });
    } else {
        // Use L2 cache-aware round-robin kernel for 3+ groups
        int grid_size_m = max_m_blocks_per_group * params.num_groups;

        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            if constexpr(!Is_dropout) {
                if (is_sm8x) {
                    run_flash_fwd_grouped_cache_aware<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, cutlass::half_t>, Is_dropout, true>(
                        params, stream, grid_size_m);
                } else {
                    run_flash_fwd_grouped_cache_aware<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, cutlass::half_t>, Is_dropout, true>(
                        params, stream, grid_size_m);
                }
            } else {
                if (is_sm8x) {
                    run_flash_fwd_grouped_cache_aware<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, cutlass::half_t>, Is_dropout, true>(
                        params, stream, grid_size_m);
                } else {
                    run_flash_fwd_grouped_cache_aware<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, cutlass::half_t>, Is_dropout, true>(
                        params, stream, grid_size_m);
                }
            }
        });
    }
}

}  // namespace FLASH_NAMESPACE

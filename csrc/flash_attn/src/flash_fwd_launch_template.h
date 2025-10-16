/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "static_switch.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_fwd_kernel.h"

namespace FLASH_NAMESPACE {

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        FLASH_NAMESPACE::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_kernel, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV) {
    #if defined(ARCH_SUPPORTS_FLASH)
        FLASH_NAMESPACE::compute_attn_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_combine_kernel, int kBlockM, int Log_max_splits, bool Is_even_K) {
    static_assert(Log_max_splits >= 1);
    FLASH_NAMESPACE::combine_attn_seqk_parallel<Kernel_traits, kBlockM, Log_max_splits, Is_even_K>(params);
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_grouped_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        FLASH_NAMESPACE::compute_attn_grouped<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    // printf("smem_size = %d\n", smem_size);

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            // Will only return softmax if dropout, to reduce compilation time.
                            // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                            // If return_softmax, set IsEvenMNConst to false to reduce number of templates
                            // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                            // If Is_local, set Is_causal to false
                            auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !Has_alibi && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst && !ReturnSoftmaxConst && !Has_alibi, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                            // auto kernel = &flash_fwd_kernel<Kernel_traits, false, Is_causal, false, false, true, true, false>;
                            // printf("IsEvenMNConst = %d, IsEvenKConst = %d, Is_local = %d, Is_causal = %d, ReturnSoftmaxConst = %d, Is_dropout = %d\n", int(IsEvenMNConst), int(IsEvenKConst), int(Is_local), int(Is_causal), int(ReturnSoftmaxConst), int(Is_dropout));
                            // auto kernel = &flash_fwd_kernel<Kernel_traits, false, Is_causal, false, true, true, false>;
                            if (smem_size >= 48 * 1024) {
                                C10_CUDA_CHECK(cudaFuncSetAttribute(
                                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                            }
                            // int ctas_per_sm;
                            // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                            // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(!Kernel_traits::Is_Q_in_regs, "SplitKV implementation does not support Is_Q_in_regs");
    static_assert(!Kernel_traits::Share_Q_K_smem, "SplitKV implementation does not support Share_Q_K_smem");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    BOOL_SWITCH(params.knew_ptr != nullptr, Append_KV, [&] {
                        ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                            SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                                // If Append_KV, then we must have seqlen_offsets, which means cu_seqlens_k != nullptr.
                                // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                                // If Is_local, set Is_causal to false
                                auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && !Append_KV && IsEvenKConst && !Is_local && !Has_alibi && Kernel_traits::kHeadDim <= 128, IsEvenKConst && !Has_alibi, Is_softcap, Split, Append_KV>;
                                // auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, false, true, Split, Append_KV>;
                                // auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, false, IsEvenKConst>;
                                if (smem_size >= 48 * 1024) {
                                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                                }
                                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                                C10_CUDA_KERNEL_LAUNCH_CHECK();
                            });
                        });
                    });
                });
            });
        });
    });
    if (params.num_splits > 1) {
        // We want kBlockM to be as small as possible for more parallelism.
        // With 128 threads we can load 512 elements at a time, so if headdim is divisible by 128, kBlockM = 4.
        // If headdim is divisible by 64, then we set kBlockM = 8, etc.
        constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 4 : (Kernel_traits::kHeadDim % 64 == 0 ? 8 : 16);
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            if (params.num_splits <= 2) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 1, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 4) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 2, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 8) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 3, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 16) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 4, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 32) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 5, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 64) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 6, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 128) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 7, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
}

template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int kBlockM = 64;  // Fixed for all head dimensions
    // TD [2023-08-28]: nvcc segfaults for headdim 96 with block size 64 x 256,
    // and for headdim 192 with block size 64 x 128.
    constexpr static int kBlockN = Headdim <= 64 ? 256 : (Headdim <= 128 ? 128 : 64);
    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // Using 8 warps is 18% slower for seqlen=2k, 2 warps is 5% slower
            // Using block size (64 x 256) is 27% slower for seqlen=2k
            // Using block size (256 x 64) is 85% slower for seqlen=2k, because of register spilling
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim96(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
        if (is_sm8x) {
            if constexpr(!Is_causal) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        // These two are always slower
        // run_flash_fwd<Flash_fwd_kernel_traits<96, 128, 128, 4, true, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<96, 64, 128, 4, true, T>>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
            // and 128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM.
            if (is_sm8x) {
                if constexpr(!Is_causal) {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                } else {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // Using 8 warps (128 x 128 and 256 x 64) is 28% slower for seqlen=2k
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // 1st ones are good for H100, A100
            // 2nd one is good for A6000 bc we get slightly better occupancy
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, T>>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_sm, max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // printf("max_smem_per_sm = %d, max_smem_per_block = %d\n", max_smem_per_sm, max_smem_per_block);
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // For A100, we want to run with 128 x 64 (128KB smem).
        // For H100 we want to run with 64 x 64 (96KB smem) since then we can get 2 CTAs per SM.
        if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64) && max_smem_per_sm < 4 * Headdim * (64 + 2 * 64)) {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        // 64 KB
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // 96 KB
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

// Grouped flash attention kernel launcher
// Launches a single kernel grid that processes multiple Q groups with shared K,V
// NOTE: total_num_m_blocks must be pre-computed on host before calling this
template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd_grouped(Flash_fwd_params &params, cudaStream_t stream, int total_num_m_blocks) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    // Set up grid with total Q blocks across all groups
    dim3 grid(total_num_m_blocks, params.b, params.h);

    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;

    // For simplicity in Phase 2, we assume:
    // - No local windowing
    // - No alibi
    // - No softcap
    // These can be added later if needed

    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
            // Use the new grouped kernel
            auto kernel = &flash_fwd_grouped_kernel<Kernel_traits, Is_dropout, Is_causal,
                                           false, // Is_local
                                           false, // Has_alibi
                                           false, // IsEvenMN - disabled for simplicity
                                           IsEvenKConst,
                                           false, // Is_softcap
                                           ReturnSoftmaxConst && Is_dropout>;

            if (smem_size >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            }

            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

// OPTIMIZATION: Cache-aware grouped flash attention kernel
// Uses round-robin block scheduling to maximize L2 cache hit rate for K,V sharing
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_grouped_cache_aware_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, int max_m_blocks_per_group) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints

        // OPTIMIZATION 3: Round-robin block indexing for cache locality
        // Instead of processing all blocks from group 0, then group 1, etc.,
        // we interleave: group 0 block 0, group 1 block 0, group 2 block 0, group 0 block 1, ...
        // This ensures Q blocks from different groups that access similar K,V regions
        // execute close together, maximizing L2 cache hit rate

        const int m_block_global = blockIdx.x;
        const int num_groups = params.num_groups;

        // Round-robin indexing: blockIdx maps to (m_block, group_id)
        const int m_block = m_block_global / num_groups;
        const int group_id = m_block_global % num_groups;

        // Early exit if this group doesn't have this many m_blocks
        if (m_block >= params.group_num_m_blocks[group_id]) {
            return;
        }

        // Now process this Q block using the optimized grouped attention logic
        const int bidb = blockIdx.y;
        const int bidh = blockIdx.z;

        FLASH_NAMESPACE::compute_attn_1rowblock_grouped<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(
            params, bidb, bidh, m_block, group_id
        );
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

// Cache-aware grouped flash attention launcher
// Uses round-robin grid layout to ensure K,V tiles are reused across groups via L2 cache
template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd_grouped_cache_aware(Flash_fwd_params &params, cudaStream_t stream, int grid_size_m, int max_m_blocks_per_group) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    // Grid uses round-robin layout: (max_m_blocks_per_group * num_groups, batch, heads)
    // This ensures blocks from different groups that need similar K,V tiles execute close together
    dim3 grid(grid_size_m, params.b, params.h);

    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;

    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
            // Use the cache-aware grouped kernel with round-robin indexing
            auto kernel = &flash_fwd_grouped_cache_aware_kernel<Kernel_traits, Is_dropout, Is_causal,
                                           false, // Is_local
                                           false, // Has_alibi
                                           false, // IsEvenMN - disabled for simplicity
                                           IsEvenKConst,
                                           false, // Is_softcap
                                           ReturnSoftmaxConst && Is_dropout,
                                           max_m_blocks_per_group>;

            if (smem_size >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            }

            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

// Sequential grouped flash attention launcher for L2 cache reuse
// Launches one kernel per group SEQUENTIALLY to enable L2 cache sharing
template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd_grouped_sequential(Flash_fwd_params &params, cudaStream_t stream, int grid_size_m, int group_id) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    // Grid for this single group
    dim3 grid(grid_size_m, params.b, params.h);

    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;

    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
            // Use cache-aware kernel but with sequential launch (only one group at a time)
            auto kernel = &flash_fwd_grouped_cache_aware_kernel<Kernel_traits, Is_dropout, Is_causal,
                                           false, // Is_local
                                           false, // Has_alibi
                                           false, // IsEvenMN
                                           IsEvenKConst,
                                           false, // Is_softcap
                                           ReturnSoftmaxConst && Is_dropout,
                                           1>;  // max_m_blocks_per_group=1 since we're sequential

            if (smem_size >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            }

            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

}  // namespace FLASH_NAMESPACE

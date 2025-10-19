/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace FLASH_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA Hardware Limits (SM80/Ampere and newer)
////////////////////////////////////////////////////////////////////////////////////////////////////

// Maximum shared memory per thread block on SM80 (A100): 164 KB
// Maximum shared memory per thread block on SM86/89 (RTX 3090/4090): 100 KB
// We use conservative 96 KB to support most Ampere+ GPUs
static constexpr int kMaxSmemPerBlock = 96 * 1024;

// Maximum registers per thread block on SM80: 65536
// Maximum registers per thread on SM80: 255
// Practical limit considering occupancy: ~64K registers per block
static constexpr int kMaxRegistersPerBlock = 64 * 1024;

// Register bytes per thread (each register is 4 bytes)
static constexpr int kBytesPerRegister = 4;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Compile-time SMEM calculation for N-group kernel
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kHeadDim, int kBlockM, int kBlockN, typename Element>
constexpr int compute_smem_size_per_group() {
    // Each group needs:
    // - Q tile: kBlockM x kHeadDim (loaded into SMEM, then reused for all K,V tiles)
    // - Intermediate S tile (Q@K^T): kBlockM x kBlockN (can be in registers)
    // - Intermediate P tile (softmax(S)): kBlockM x kBlockN (can be in registers)
    // - Accumulator O: kBlockM x kHeadDim (in registers)

    // Only Q is in SMEM per group (if not shared)
    // But in N-group SMEM sharing, Q is also shared, so per-group SMEM cost is 0!
    return 0;
}

template<int kHeadDim, int kBlockM, int kBlockN, typename Element>
constexpr int compute_smem_size_shared() {
    // Shared across all groups:
    // - Q tile: kBlockM x kHeadDim (with swizzling overhead ~1.125x)
    // - K tile: kBlockN x kHeadDim (with swizzling overhead ~1.125x)
    // - V tile: kBlockN x kHeadDim (with swizzling overhead ~1.125x)

    constexpr int kSwizzleOverhead = 8; // 1/8 extra for swizzling (12.5%)
    constexpr int element_size = sizeof(Element);

    int smem_q = (kBlockM * kHeadDim * element_size * (kSwizzleOverhead + 8)) / 8;
    int smem_k = (kBlockN * kHeadDim * element_size * (kSwizzleOverhead + 8)) / 8;
    int smem_v = (kBlockN * kHeadDim * element_size * (kSwizzleOverhead + 8)) / 8;

    // In SMEM sharing mode, we only need max(Q, K+V) since they don't overlap
    return smem_q > (smem_k + smem_v) ? smem_q : (smem_k + smem_v);
}

template<int kHeadDim, int kBlockM, int kBlockN, typename Element>
constexpr int compute_max_groups_smem() {
    // Calculate maximum groups based on SMEM constraint
    constexpr int smem_shared = compute_smem_size_shared<kHeadDim, kBlockM, kBlockN, Element>();
    constexpr int smem_per_group = compute_smem_size_per_group<kHeadDim, kBlockM, kBlockN, Element>();

    // Since SMEM is fully shared and smem_per_group = 0, we're only limited by registers
    // But we add a sanity check to ensure we don't exceed SMEM
    if (smem_shared >= kMaxSmemPerBlock) {
        return 1; // Can't even fit the shared SMEM
    }

    // SMEM is not the bottleneck for N-group - it's registers!
    // Return a large number to indicate SMEM is not limiting
    return 16;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Compile-time register calculation for N-group kernel
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kHeadDim, int kBlockM, int kBlockN>
constexpr int compute_registers_per_group() {
    // Per-group register usage (using float32 accumulators):
    // 1. Accumulator O: kBlockM x kHeadDim (float32) = kBlockM * kHeadDim * 4 bytes
    // 2. Softmax state: m_i, l_i per row = kBlockM * 2 * 4 bytes
    // 3. Intermediate S: kBlockM x kBlockN (float32) = kBlockM * kBlockN * 4 bytes (during Q@K^T)
    // 4. Intermediate P: kBlockM x kBlockN (Element) = kBlockM * kBlockN * 2 bytes (during P@V)
    //
    // Note: S and P don't coexist, so we take max

    // These are fragment sizes - actual register usage is distributed across threads
    // For a 4-warp kernel (128 threads), each MMA tile is 16x8 per thread group
    // We'll estimate conservatively based on fragment sizes

    constexpr int kNWarps = 4; // Typical configuration
    constexpr int kNThreads = kNWarps * 32;

    // Accumulator O per thread: Each thread holds a portion of kBlockM x kHeadDim
    // Using CuTe TiledMMA with 16x8x16 atom, partition_fragment_C gives ~16 elements per thread
    constexpr int acc_o_elements = (kBlockM * kHeadDim + kNThreads - 1) / kNThreads;
    constexpr int acc_o_regs = (acc_o_elements * 4 + kBytesPerRegister - 1) / kBytesPerRegister;

    // Softmax state (m, l): 2 floats per row, distributed across threads
    constexpr int softmax_elements = (kBlockM * 2 + kNThreads - 1) / kNThreads;
    constexpr int softmax_regs = (softmax_elements * 4 + kBytesPerRegister - 1) / kBytesPerRegister;

    // Intermediate S (Q@K^T): kBlockM x kBlockN elements, distributed
    constexpr int acc_s_elements = (kBlockM * kBlockN + kNThreads - 1) / kNThreads;
    constexpr int acc_s_regs = (acc_s_elements * 4 + kBytesPerRegister - 1) / kBytesPerRegister;

    // Total per group (max of S since S and P don't coexist)
    return acc_o_regs + softmax_regs + acc_s_regs;
}

template<int kHeadDim, int kBlockM, int kBlockN>
constexpr int compute_registers_shared() {
    // Shared register usage (used by all groups):
    // 1. Q fragment in registers: kBlockM x kHeadDim (Element = fp16/bf16)
    // 2. K fragment in registers: kBlockN x kHeadDim (Element)
    // 3. V fragment in registers: kBlockN x kHeadDim (Element)
    //
    // These are loaded from SMEM and reused across groups

    constexpr int kNWarps = 4;
    constexpr int kNThreads = kNWarps * 32;

    // Q fragment per thread (16-bit elements)
    constexpr int q_frag_elements = (kBlockM * kHeadDim + kNThreads - 1) / kNThreads;
    constexpr int q_frag_regs = (q_frag_elements * 2 + kBytesPerRegister - 1) / kBytesPerRegister;

    // K fragment per thread (16-bit elements)
    constexpr int k_frag_elements = (kBlockN * kHeadDim + kNThreads - 1) / kNThreads;
    constexpr int k_frag_regs = (k_frag_elements * 2 + kBytesPerRegister - 1) / kBytesPerRegister;

    // V fragment per thread (16-bit elements)
    constexpr int v_frag_elements = (kBlockN * kHeadDim + kNThreads - 1) / kNThreads;
    constexpr int v_frag_regs = (v_frag_elements * 2 + kBytesPerRegister - 1) / kBytesPerRegister;

    return q_frag_regs + k_frag_regs + v_frag_regs;
}

template<int kHeadDim, int kBlockM, int kBlockN>
constexpr int compute_max_groups_registers() {
    constexpr int regs_shared = compute_registers_shared<kHeadDim, kBlockM, kBlockN>();
    constexpr int regs_per_group = compute_registers_per_group<kHeadDim, kBlockM, kBlockN>();

    constexpr int kNWarps = 4;
    constexpr int kNThreads = kNWarps * 32;
    constexpr int max_regs_per_thread = kMaxRegistersPerBlock / kNThreads;

    // Available registers for groups after shared usage
    int available_for_groups = max_regs_per_thread - regs_shared;
    if (available_for_groups <= 0) return 1;

    // How many groups can we fit?
    int max_groups = available_for_groups / regs_per_group;

    // Ensure at least 1 group
    return max_groups < 1 ? 1 : max_groups;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Combined limit
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kHeadDim, int kBlockM, int kBlockN, typename Element>
constexpr int compute_max_groups() {
    constexpr int max_by_smem = compute_max_groups_smem<kHeadDim, kBlockM, kBlockN, Element>();
    constexpr int max_by_regs = compute_max_groups_registers<kHeadDim, kBlockM, kBlockN>();

    // Take the minimum (most constraining limit)
    return max_by_smem < max_by_regs ? max_by_smem : max_by_regs;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Instantiated configurations for common head dimensions
////////////////////////////////////////////////////////////////////////////////////////////////////

// Forward declarations for explicit instantiation
template<int kHeadDim, int kBlockM, int kBlockN, typename Element>
struct GroupedKernelConfig {
    static constexpr int MaxGroups = compute_max_groups<kHeadDim, kBlockM, kBlockN, Element>();
    static constexpr int MaxGroupsBySmem = compute_max_groups_smem<kHeadDim, kBlockM, kBlockN, Element>();
    static constexpr int MaxGroupsByRegs = compute_max_groups_registers<kHeadDim, kBlockM, kBlockN>();
    static constexpr int SharedSmemSize = compute_smem_size_shared<kHeadDim, kBlockM, kBlockN, Element>();
    static constexpr int PerGroupSmemSize = compute_smem_size_per_group<kHeadDim, kBlockM, kBlockN, Element>();
    static constexpr int SharedRegSize = compute_registers_shared<kHeadDim, kBlockM, kBlockN>();
    static constexpr int PerGroupRegSize = compute_registers_per_group<kHeadDim, kBlockM, kBlockN>();
};

// Standard block sizes
constexpr int kBlockM_default = 64;
constexpr int kBlockN_default = 128;

// Configurations for hdim=32
using Config_hdim32_fp16 = GroupedKernelConfig<32, kBlockM_default, kBlockN_default, cutlass::half_t>;
using Config_hdim32_bf16 = GroupedKernelConfig<32, kBlockM_default, kBlockN_default, cutlass::bfloat16_t>;

// Configurations for hdim=64
using Config_hdim64_fp16 = GroupedKernelConfig<64, kBlockM_default, kBlockN_default, cutlass::half_t>;
using Config_hdim64_bf16 = GroupedKernelConfig<64, kBlockM_default, kBlockN_default, cutlass::bfloat16_t>;

// Configurations for hdim=96
using Config_hdim96_fp16 = GroupedKernelConfig<96, kBlockM_default, kBlockN_default, cutlass::half_t>;
using Config_hdim96_bf16 = GroupedKernelConfig<96, kBlockM_default, kBlockN_default, cutlass::bfloat16_t>;

// Configurations for hdim=128
using Config_hdim128_fp16 = GroupedKernelConfig<128, kBlockM_default, kBlockN_default, cutlass::half_t>;
using Config_hdim128_bf16 = GroupedKernelConfig<128, kBlockM_default, kBlockN_default, cutlass::bfloat16_t>;

// Configurations for hdim=192
using Config_hdim192_fp16 = GroupedKernelConfig<192, kBlockM_default, kBlockN_default, cutlass::half_t>;
using Config_hdim192_bf16 = GroupedKernelConfig<192, kBlockM_default, kBlockN_default, cutlass::bfloat16_t>;

// Configurations for hdim=256
using Config_hdim256_fp16 = GroupedKernelConfig<256, kBlockM_default, kBlockN_default, cutlass::half_t>;
using Config_hdim256_bf16 = GroupedKernelConfig<256, kBlockM_default, kBlockN_default, cutlass::bfloat16_t>;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper to select config at runtime (for debugging/logging)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Element>
inline int get_max_groups_for_config(int headdim, int blockM, int blockN) {
    // This is for debugging - compile-time constants are preferred
    if (headdim == 32 && blockM == 64 && blockN == 128) {
        return std::is_same_v<Element, cutlass::half_t>
            ? Config_hdim32_fp16::MaxGroups
            : Config_hdim32_bf16::MaxGroups;
    } else if (headdim == 64 && blockM == 64 && blockN == 128) {
        return std::is_same_v<Element, cutlass::half_t>
            ? Config_hdim64_fp16::MaxGroups
            : Config_hdim64_bf16::MaxGroups;
    } else if (headdim == 96 && blockM == 64 && blockN == 128) {
        return std::is_same_v<Element, cutlass::half_t>
            ? Config_hdim96_fp16::MaxGroups
            : Config_hdim96_bf16::MaxGroups;
    } else if (headdim == 128 && blockM == 64 && blockN == 128) {
        return std::is_same_v<Element, cutlass::half_t>
            ? Config_hdim128_fp16::MaxGroups
            : Config_hdim128_bf16::MaxGroups;
    } else if (headdim == 192 && blockM == 64 && blockN == 128) {
        return std::is_same_v<Element, cutlass::half_t>
            ? Config_hdim192_fp16::MaxGroups
            : Config_hdim192_bf16::MaxGroups;
    } else if (headdim == 256 && blockM == 64 && blockN == 128) {
        return std::is_same_v<Element, cutlass::half_t>
            ? Config_hdim256_fp16::MaxGroups
            : Config_hdim256_bf16::MaxGroups;
    }
    return 1; // Conservative fallback
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace FLASH_NAMESPACE

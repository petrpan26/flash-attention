# Phase 2: CUDA Kernel Implementation Plan for Multi-Group Varlen Attention

## Document Overview
This document provides a detailed implementation plan for Phase 2 of the multi-group varlen attention optimization. It includes architectural analysis, design decisions, shared memory layouts, and detailed pseudocode for CUDA kernel development.

**Target**: Implement forward CUDA kernel achieving >1.3x speedup by loading K,V tiles once for multiple Q groups.

---

## 1. Existing Flash Attention Kernel Architecture Analysis

### 1.1 Core Data Structures

#### Flash_fwd_params (flash.h)
```cpp
struct Flash_fwd_params : public Qkv_params {
    // Input/Output pointers
    void * __restrict__ q_ptr;
    void * __restrict__ k_ptr;
    void * __restrict__ v_ptr;
    void * __restrict__ o_ptr;
    void * __restrict__ softmax_lse_ptr;

    // Strides (for memory access patterns)
    index_t q_batch_stride, k_batch_stride, v_batch_stride;
    index_t q_row_stride, k_row_stride, v_row_stride;
    index_t q_head_stride, k_head_stride, v_head_stride;
    index_t o_batch_stride, o_row_stride, o_head_stride;

    // Dimensions
    int b, seqlen_q, seqlen_k, d, h, h_k, h_h_k_ratio;
    int seqlen_q_rounded, seqlen_k_rounded, d_rounded;

    // Varlen support
    int * __restrict__ cu_seqlens_q;  // Cumulative sequence lengths for Q
    int * __restrict__ cu_seqlens_k;  // Cumulative sequence lengths for K
    int total_q;                       // Total number of Q tokens across all sequences
    bool unpadded_lse;                 // LSE format flag

    // Softmax parameters
    float scale_softmax;
    float scale_softmax_log2;

    // Other features
    bool is_causal;
    int window_size_left, window_size_right;
    float p_dropout;
    // ... (alibi, rotary, paged KV cache, etc.)
};
```

**Key Insights**:
- Single Q, K, V pointer per call
- Varlen support via `cu_seqlens_q` and `cu_seqlens_k` arrays
- LSE can be in different formats (padded vs unpadded)
- Extensive stride information for flexible memory layouts

#### BlockInfo Template (block_info.h)
```cpp
template<bool Varlen=true>
struct BlockInfo {
    const int sum_s_q;           // Cumulative Q position
    const int sum_s_k;           // Cumulative K position
    const int actual_seqlen_q;   // Actual Q sequence length
    const int actual_seqlen_k;   // Actual K sequence length
    const int leftpad_k;         // Left padding for K

    // Compute Q offset in memory
    index_t q_offset(batch_stride, row_stride, bidb) const;
    // Compute K offset in memory
    index_t k_offset(batch_stride, row_stride, bidb) const;
};
```

**Key Insights**:
- Encapsulates per-batch block information
- Handles variable-length sequences cleanly
- Provides offset calculation helpers

### 1.2 Kernel Launch Configuration

#### Grid Dimensions (flash_fwd_launch_template.h)
```cpp
const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
dim3 grid(num_m_block, params.b, params.h);
```

**Mapping**:
- `blockIdx.x`: Q block index (which M block of Q we're processing)
- `blockIdx.y`: Batch index
- `blockIdx.z`: Head index

**Thread Block Size**: `Kernel_traits::kNThreads` = `kNWarps * 32` (typically 128 threads = 4 warps)

### 1.3 Shared Memory Layout

#### Standard Configuration (kernel_traits.h)
```cpp
// Tile sizes (typical for d=128)
static constexpr int kBlockM = 128;  // Q tile size along M dimension
static constexpr int kBlockN = 128;  // K,V tile size along N dimension
static constexpr int kHeadDim = 128; // Head dimension
static constexpr int kNWarps = 4;    // Number of warps per thread block

// Shared memory tiles
SmemLayoutQ:  Shape<Int<kBlockM>, Int<kHeadDim>>  // Q tile: [128, 128]
SmemLayoutKV: Shape<Int<kBlockN>, Int<kHeadDim>>  // K,V tiles: [128, 128] each

// Total shared memory (Share_Q_K_smem=false)
kSmemQSize = kBlockM * kHeadDim * sizeof(Element)     // e.g., 128*128*2 = 32KB
kSmemKVSize = kBlockN * kHeadDim * 2 * sizeof(Element) // e.g., 128*128*2*2 = 64KB
kSmemSize = kSmemQSize + kSmemKVSize                  // Total: 96KB
```

**Memory Hierarchy**:
```
Shared Memory Layout (Share_Q_K_smem=false):
┌──────────────────────────────────────────────────┐
│ sQ: [kBlockM x kHeadDim]          (32KB for d=128)│
├──────────────────────────────────────────────────┤
│ sK: [kBlockN x kHeadDim]          (32KB for d=128)│
├──────────────────────────────────────────────────┤
│ sV: [kBlockN x kHeadDim]          (32KB for d=128)│
└──────────────────────────────────────────────────┘
Total: 96KB
```

### 1.4 Main Kernel Loop Structure (flash_fwd_kernel.h)

#### High-Level Flow
```cpp
template<typename Kernel_traits, bool Is_causal, ...>
__device__ void compute_attn_1rowblock(params, bidb, bidh, m_block) {
    // 1. Initialize shared memory tiles
    extern __shared__ char smem_[];
    Tensor sQ = make_tensor(...);  // Q in shared memory
    Tensor sK = make_tensor(...);  // K in shared memory
    Tensor sV = make_tensor(...);  // V in shared memory

    // 2. Load Q tile once (stays in smem/regs for entire kernel)
    load_tile(Q[m_block], sQ);

    // 3. Initialize output accumulator and LSE
    Tensor acc_o = partition_fragment_C(...);  // Output accumulator
    clear(acc_o);
    Softmax softmax;  // Maintains LSE state (max, sum)

    // 4. Iterate over K,V tiles (backward for masking efficiency)
    for (n_block = n_block_max - 1; n_block >= n_block_min; --n_block) {
        // 4a. Load K,V tile
        load_tile(K[n_block], sK);
        load_tile(V[n_block], sV);
        __syncthreads();

        // 4b. Compute attention scores: S = Q @ K^T
        Tensor acc_s = gemm(sQ, sK);  // [kBlockM, kBlockN]

        // 4c. Apply masking (causal, local, etc.)
        apply_mask(acc_s, n_block, m_block);

        // 4d. Online softmax update
        // Computes: new_max, new_sum, rescales old acc_o, updates acc_o
        softmax.softmax_rescale_o(acc_s, acc_o, scale);

        // 4e. Accumulate attention output: O += P @ V
        Tensor P = convert_type<Element>(acc_s);  // Softmax output (normalized)
        acc_o += gemm(P, sV);
    }

    // 5. Write output and LSE
    write_output(acc_o, O[m_block]);
    write_lse(softmax.lse, LSE[m_block]);
}
```

#### Online Softmax Algorithm (softmax.h)
The kernel uses online softmax to avoid storing entire attention matrix:

```cpp
// For each new K,V block:
1. Compute S_new = Q @ K_new^T
2. max_new = rowmax(S_new)
3. max_updated = max(max_old, max_new)
4. Rescale old accumulator: acc_o *= exp(max_old - max_updated)
5. P_new = exp(S_new - max_updated)
6. sum_updated = sum_old * exp(max_old - max_updated) + rowsum(P_new)
7. acc_o += P_new @ V_new
8. Update: max_old = max_updated, sum_old = sum_updated

// Final normalization:
O = acc_o / sum_updated
LSE = log(sum_updated) + max_updated  // Log-sum-exp for numerical stability
```

**Key Properties**:
- Constant memory: Only stores (max, sum) state, not full attention matrix
- Numerically stable: Uses log-sum-exp trick
- Enables tiling: Can process K,V in chunks

---

## 2. Multi-Group Kernel Design

### 2.1 Problem Statement

**Current Behavior** (zigzag_llama3):
```python
# Two separate kernel calls
out_0, lse_0 = flash_attn_varlen_forward(q_0, k, v, ..., kv_endpoints_0)  # Loads K[0:256], V[0:256]
out_1, lse_1 = flash_attn_varlen_forward(q_1, k, v, ..., kv_endpoints_1)  # Loads K[0:512], V[0:512]
# Problem: K[0:256] and V[0:256] loaded TWICE (38% redundant memory traffic)
```

**Target Behavior** (multigroup kernel):
```python
# Single kernel call
[out_0, out_1], [lse_0, lse_1] = flash_attn_varlen_multigroup_forward(
    [q_0, q_1], k, v, ..., [kv_endpoints_0, kv_endpoints_1]
)
# Solution: Each K,V tile loaded ONCE, processed for all groups needing it
```

### 2.2 Core Challenge: Per-Group State Management

Each group needs independent:
1. **Q tiles**: Different Q data for each group
2. **Output accumulators**: Separate O accumulation per group
3. **LSE state**: Separate (max, sum) tracking per group
4. **KV boundaries**: Different KV endpoint per group

But all groups share:
1. **K,V tiles**: Same K,V data (with different endpoints)
2. **Thread block**: Single TB processes all groups for a given (batch, head, m_block)

### 2.3 Architectural Decisions

#### Decision 1: Thread Block Mapping
**Choice**: Same grid as standard Flash Attention
```cpp
dim3 grid(num_m_block, params.b, params.h);
// blockIdx.x = m_block (which Q block)
// blockIdx.y = bidb (batch index)
// blockIdx.z = bidh (head index)
```

**Rationale**:
- Each thread block processes all groups for a given (batch, head, Q tile)
- Groups share the same K,V tiles naturally
- Minimal changes to launch configuration

#### Decision 2: Shared Memory Layout
**Choice**: Separate per-group state, shared K,V tiles

```
Shared Memory Layout (2 groups, d=128):
┌────────────────────────────────────────────────────────┐
│ Group 0:                                                │
│   sQ_0: [kBlockM x kHeadDim]           (32KB)          │
│   acc_o_0: [kBlockM x kHeadDim]         (32KB FP32)     │
├────────────────────────────────────────────────────────┤
│ Group 1:                                                │
│   sQ_1: [kBlockM x kHeadDim]           (32KB)          │
│   acc_o_1: [kBlockM x kHeadDim]         (32KB FP32)     │
├────────────────────────────────────────────────────────┤
│ Shared K,V tiles:                                       │
│   sK: [kBlockN x kHeadDim]              (32KB)          │
│   sV: [kBlockN x kHeadDim]              (32KB)          │
└────────────────────────────────────────────────────────┘
Total: 192KB (exceeds typical 100KB limit per SM - see optimization strategies)
```

**Challenge**: Shared memory exceeds typical limits
**Solutions** (Phase 5 optimization):
1. **Smaller tiles**: Use kBlockM=64, kBlockN=64 (reduces to 96KB for 2 groups)
2. **Register spilling**: Keep acc_o in registers, not smem
3. **Two-pass approach**: Process groups sequentially if smem insufficient
4. **Dynamic group count**: Support 2 groups initially, extend to 4+ later

#### Decision 3: LSE Storage
**Choice**: Per-group LSE accumulators in registers

```cpp
struct GroupState {
    float lse_max[kBlockM / (32 / kNWarps)];  // Per-row max values
    float lse_sum[kBlockM / (32 / kNWarps)];  // Per-row sum values
    // Typical: kBlockM=128, kNWarps=4 -> 128/(32/4) = 16 floats per array
};
GroupState group_states[MAX_NUM_GROUPS];
```

**Rationale**:
- Small memory footprint (32 floats per group for kBlockM=128)
- Fast register access
- Matches existing softmax algorithm

#### Decision 4: K,V Endpoint Enforcement
**Choice**: Per-group boundary checks before processing each K,V tile

```cpp
for (int n_block = n_block_max_global - 1; n_block >= 0; --n_block) {
    // Load K,V tile ONCE
    load_kv_tile(n_block);
    __syncthreads();

    for (int g = 0; g < num_groups; g++) {
        // Check if this group needs this K,V tile
        int kv_end = kv_endpoints[g][bidb];  // Per-group, per-batch endpoint
        int kv_tile_end = (n_block + 1) * kBlockN;

        if (n_block * kBlockN >= kv_end) {
            continue;  // This group doesn't need this K,V tile
        }

        // Process tile for this group
        process_group(g, n_block, ...);
    }
}
```

---

## 3. Detailed Shared Memory Requirements

### 3.1 Memory Calculation for 2 Groups (d=128)

#### Per-Group Memory:
```
Q tile (smem):        kBlockM * kHeadDim * sizeof(Element)
                    = 128 * 128 * 2 bytes = 32KB

Output accumulator:  Two options:
  Option A (smem):   kBlockM * kHeadDim * sizeof(float) = 64KB
  Option B (regs):   ~4KB per warp * 4 warps = 16KB (distributed across threads)

LSE state (regs):    2 * (kBlockM / 8) * sizeof(float)
                    = 2 * 16 * 4 = 128 bytes per thread block
```

#### Shared K,V Tiles:
```
K tile:              kBlockN * kHeadDim * sizeof(Element)
                    = 128 * 128 * 2 = 32KB

V tile:              kBlockN * kHeadDim * sizeof(Element)
                    = 128 * 128 * 2 = 32KB
```

#### Total Shared Memory:

**Option A** (acc_o in smem):
```
= 2 groups * (32KB Q + 64KB acc_o) + 32KB K + 32KB V
= 2 * 96KB + 64KB
= 256KB  ❌ EXCEEDS LIMIT (typical max: 100-164KB per SM)
```

**Option B** (acc_o in registers):
```
= 2 groups * 32KB Q + 32KB K + 32KB V
= 64KB + 64KB
= 128KB  ⚠️ HIGH but feasible on A100 (164KB/SM), marginal on H100 (228KB/SM)
```

**Option C** (smaller tiles: kBlockM=64, kBlockN=128):
```
= 2 groups * 16KB Q + 32KB K + 32KB V
= 32KB + 64KB
= 96KB  ✅ SAFE for all architectures
```

### 3.2 Optimization Strategies

#### Strategy 1: Reduce Tile Sizes (Recommended for Phase 2)
```cpp
// For 2 groups, d=128
constexpr int kBlockM = 64;   // Reduced from 128
constexpr int kBlockN = 128;  // Keep standard

// Shared memory: 2*16KB + 32KB + 32KB = 96KB ✓
// Tradeoff: More kernel launches, but safer and simpler
```

#### Strategy 2: Keep acc_o in Registers
```cpp
// Standard Flash Attention already does this
// acc_o is a fragment (register-allocated tensor)
Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

// Only need smem for Q, K, V tiles
// For 2 groups: 2*32KB + 32KB + 32KB = 128KB
```

#### Strategy 3: Sequential Group Processing (Fallback)
```cpp
// If smem insufficient for all groups, process groups one at a time
for (int g = 0; g < num_groups; g++) {
    load_q_tile(group_q_ptrs[g]);

    for (int n_block = ...) {
        load_kv_tile(n_block);  // Still load K,V once per n_block across all groups
        process_attention(g, n_block);
    }

    write_output(group_out_ptrs[g]);
}
// Benefit: Minimal smem (same as standard kernel)
// Cost: K,V tiles loaded multiple times (but fewer than original)
```

### 3.3 Register Pressure Analysis

#### Standard Flash Attention:
```
Per-thread registers (typical):
- Q fragment:     ~32 regs
- K fragment:     ~32 regs
- Acc_s (QK^T):   ~64 regs
- Acc_o (output): ~64 regs
- LSE state:      ~4 regs
- Misc:           ~20 regs
Total: ~216 regs/thread
```

#### Multi-Group (2 groups):
```
Per-thread registers:
- Q fragment (2x):     ~64 regs
- K fragment (shared): ~32 regs
- Acc_s (2x):          ~128 regs
- Acc_o (2x):          ~128 regs
- LSE state (2x):      ~8 regs
- Misc:                ~20 regs
Total: ~380 regs/thread

Max registers per SM: 65536 (A100)
Threads per block: 128
Regs per TB: 128 * 380 = 48640
Max TBs per SM: 65536 / 48640 = 1.35 ✓ (at least 1 TB can run)
```

**Conclusion**: Register pressure is manageable for 2 groups, may limit occupancy slightly.

---

## 4. Kernel Pseudocode

### 4.1 Forward Kernel Main Loop

```cpp
template<typename Kernel_traits, int NumGroups, bool Is_causal>
__global__ void flash_fwd_multigroup_kernel(
    KERNEL_PARAM_MODIFIER const Flash_fwd_multigroup_params params
) {
    // Thread and block indices
    const int tidx = threadIdx.x;
    const int bidb = blockIdx.y;  // Batch index
    const int bidh = blockIdx.z;  // Head index
    const int m_block = blockIdx.x;  // Q block index

    // Shared memory allocation
    extern __shared__ char smem_[];

    // Per-group state structures
    struct GroupState {
        // Shared memory pointers for this group
        Element* sQ_ptr;

        // Register-based accumulators
        float lse_max[kBlockM / 8];  // Per-row max (registers)
        float lse_sum[kBlockM / 8];  // Per-row sum (registers)

        // Fragments (register-allocated)
        decltype(partition_fragment_C(...)) acc_o;  // Output accumulator

        // Metadata
        int actual_seqlen_q;
        int actual_seqlen_k;
        int kv_max_offset;  // Maximum K,V position for this group
        bool active;        // Whether this group has work to do
    };

    GroupState groups[NumGroups];

    // ========================================================================
    // PHASE 1: Initialize per-group state
    // ========================================================================

    #pragma unroll
    for (int g = 0; g < NumGroups; g++) {
        // Setup BlockInfo for this group
        const int cu_seqlens_q_start = params.cu_seqlens_q_list[g][bidb];
        const int cu_seqlens_q_end = params.cu_seqlens_q_list[g][bidb + 1];
        groups[g].actual_seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start;

        const int cu_seqlens_k_start = params.cu_seqlens_k_list[g][bidb];
        const int cu_seqlens_k_end = params.cu_seqlens_k_list[g][bidb + 1];
        groups[g].actual_seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start;

        // Get KV endpoint for this group
        groups[g].kv_max_offset = params.kv_endpoints[g * params.batch_size + bidb];

        // Check if this group has work for this m_block
        groups[g].active = (m_block * kBlockM < groups[g].actual_seqlen_q);

        // Assign shared memory partition
        groups[g].sQ_ptr = reinterpret_cast<Element*>(smem_) + g * kBlockM * kHeadDim;

        // Initialize LSE state
        #pragma unroll
        for (int i = 0; i < kBlockM / 8; i++) {
            groups[g].lse_max[i] = -INFINITY;
            groups[g].lse_sum[i] = 0.0f;
        }

        // Initialize output accumulator
        clear(groups[g].acc_o);
    }

    // ========================================================================
    // PHASE 2: Setup shared K,V tiles (after per-group partitions)
    // ========================================================================

    // Shared K,V tiles (placed after all group Q tiles)
    Element* sK_ptr = reinterpret_cast<Element*>(smem_)
                      + NumGroups * kBlockM * kHeadDim;
    Element* sV_ptr = sK_ptr + kBlockN * kHeadDim;

    Tensor sK = make_tensor(make_smem_ptr(sK_ptr), SmemLayoutKV{});
    Tensor sV = make_tensor(make_smem_ptr(sV_ptr), SmemLayoutKV{});

    // Determine global K,V loop range (maximum across all groups)
    int n_block_max_global = 0;
    #pragma unroll
    for (int g = 0; g < NumGroups; g++) {
        if (groups[g].active) {
            int n_blocks_for_group = (groups[g].kv_max_offset + kBlockN - 1) / kBlockN;
            n_block_max_global = max(n_block_max_global, n_blocks_for_group);
        }
    }

    // ========================================================================
    // PHASE 3: Load Q tiles for all groups
    // ========================================================================

    #pragma unroll
    for (int g = 0; g < NumGroups; g++) {
        if (!groups[g].active) continue;

        // Compute Q global memory pointer for this group
        Element* q_ptr = reinterpret_cast<Element*>(params.q_ptr_list[g]);
        int q_offset = cu_seqlens_q_list[g][bidb] * params.q_row_stride
                       + bidh * params.q_head_stride;

        Tensor gQ = make_tensor(make_gmem_ptr(q_ptr + q_offset), ...);
        Tensor sQ_g = make_tensor(make_smem_ptr(groups[g].sQ_ptr), SmemLayoutQ{});

        // Load Q tile for this group
        copy(gmem_tiled_copy, gQ, sQ_g, ...);
    }

    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // ========================================================================
    // PHASE 4: Main K,V loop (iterate backward for causal masking efficiency)
    // ========================================================================

    for (int n_block = n_block_max_global - 1; n_block >= 0; --n_block) {

        // --------------------------------------------------------------------
        // Step 4a: Load K,V tile ONCE (shared across all groups)
        // --------------------------------------------------------------------

        int kv_tile_start = n_block * kBlockN;
        int kv_tile_end = (n_block + 1) * kBlockN;

        // Load K tile
        Element* k_ptr = reinterpret_cast<Element*>(params.k_ptr);
        int k_offset = kv_tile_start * params.k_row_stride
                       + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        Tensor gK = make_tensor(make_gmem_ptr(k_ptr + k_offset), ...);
        copy(gmem_tiled_copy, gK, sK, ...);

        // Load V tile
        Element* v_ptr = reinterpret_cast<Element*>(params.v_ptr);
        int v_offset = kv_tile_start * params.v_row_stride
                       + (bidh / params.h_h_k_ratio) * params.v_head_stride;
        Tensor gV = make_tensor(make_gmem_ptr(v_ptr + v_offset), ...);
        copy(gmem_tiled_copy, gV, sV, ...);

        cute::cp_async_fence();
        cute::cp_async_wait<0>();
        __syncthreads();

        // --------------------------------------------------------------------
        // Step 4b: Process this K,V tile for each group
        // --------------------------------------------------------------------

        #pragma unroll
        for (int g = 0; g < NumGroups; g++) {
            if (!groups[g].active) continue;

            // Check if this group needs this K,V tile
            if (kv_tile_start >= groups[g].kv_max_offset) {
                continue;  // Skip: this K,V tile is beyond this group's endpoint
            }

            // Load Q tile for this group into MMA fragments
            Tensor sQ_g = make_tensor(make_smem_ptr(groups[g].sQ_ptr), SmemLayoutQ{});
            Tensor tSrQ = partition_fragment_A(tiled_mma, sQ_g);

            // Partition K for MMA
            Tensor tSrK = partition_fragment_B(tiled_mma, sK);

            // -----------------------------------------------------------------
            // Compute attention scores: S = Q @ K^T
            // -----------------------------------------------------------------

            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
            clear(acc_s);
            gemm(tiled_mma, tSrQ, tSrK, acc_s);

            // -----------------------------------------------------------------
            // Apply masking (causal, local, KV boundary)
            // -----------------------------------------------------------------

            // Causal mask
            if (Is_causal) {
                apply_causal_mask(acc_s, m_block, n_block, ...);
            }

            // KV boundary mask (clip to group's actual KV length)
            int kv_valid_end = min(kv_tile_end, groups[g].kv_max_offset);
            int kv_valid_len = kv_valid_end - kv_tile_start;
            apply_kv_boundary_mask(acc_s, kv_valid_len, kBlockN);

            // -----------------------------------------------------------------
            // Online softmax: Update LSE and rescale accumulator
            // -----------------------------------------------------------------

            // Compute row-wise max of new scores
            Tensor max_new = make_fragment_like(groups[g].lse_max);
            reduce_max(acc_s, max_new);

            // Update global max
            Tensor max_prev = make_fragment_like(groups[g].lse_max);
            #pragma unroll
            for (int i = 0; i < size(max_prev); i++) {
                max_prev(i) = groups[g].lse_max[i];
                groups[g].lse_max[i] = max(max_prev(i), max_new(i));
            }

            // Compute exp(S - max_updated)
            scale_apply_exp2(acc_s, groups[g].lse_max, params.scale_softmax_log2);

            // Compute row-wise sum of new probabilities
            Tensor sum_new = make_fragment_like(groups[g].lse_sum);
            reduce_sum(acc_s, sum_new);

            // Rescale old accumulator and sum
            #pragma unroll
            for (int i = 0; i < size(groups[g].lse_sum); i++) {
                float scale_factor = exp2f((max_prev(i) - groups[g].lse_max[i])
                                           * float(M_LOG2E));
                groups[g].lse_sum[i] = groups[g].lse_sum[i] * scale_factor + sum_new(i);

                // Rescale corresponding rows of acc_o
                // (This is implicit in the next GEMM - online softmax property)
            }

            // -----------------------------------------------------------------
            // Accumulate attention output: O += P @ V
            // -----------------------------------------------------------------

            // Convert scores to proper type (FP16/BF16)
            Tensor rP = convert_type<Element>(acc_s);

            // Partition V for MMA
            Tensor tOrV = partition_fragment_B(tiled_mma, sV);

            // Accumulate: acc_o += P @ V
            gemm(tiled_mma, rP, tOrV, groups[g].acc_o);
        }

        __syncthreads();  // Ensure all groups done before loading next K,V tile
    }

    // ========================================================================
    // PHASE 5: Finalize and write outputs
    // ========================================================================

    #pragma unroll
    for (int g = 0; g < NumGroups; g++) {
        if (!groups[g].active) continue;

        // -----------------------------------------------------------------
        // Final normalization: O = acc_o / sum
        // -----------------------------------------------------------------

        #pragma unroll
        for (int i = 0; i < size(groups[g].acc_o); i++) {
            // Determine which row this element belongs to
            int row_idx = get_row_index(i);  // Implementation depends on MMA layout
            groups[g].acc_o(i) /= groups[g].lse_sum[row_idx];
        }

        // -----------------------------------------------------------------
        // Write output to global memory
        // -----------------------------------------------------------------

        Element* out_ptr = reinterpret_cast<Element*>(params.out_ptr_list[g]);
        int out_offset = cu_seqlens_q_list[g][bidb] * params.o_row_stride
                         + bidh * params.o_head_stride
                         + m_block * kBlockM * params.o_row_stride;

        Tensor gO = make_tensor(make_gmem_ptr(out_ptr + out_offset), ...);
        copy(gmem_tiled_copy, groups[g].acc_o, gO, ...);

        // -----------------------------------------------------------------
        // Write LSE to global memory
        // -----------------------------------------------------------------

        float* lse_ptr = reinterpret_cast<float*>(params.softmax_lse_ptr_list[g]);
        int lse_offset;
        if (params.unpadded_lse) {
            // Format: [nheads, total_q]
            lse_offset = bidh * params.total_q
                         + cu_seqlens_q_list[g][bidb]
                         + m_block * kBlockM;
        } else {
            // Format: [batch, nheads, seqlen_q]
            lse_offset = bidb * params.h * params.seqlen_q_rounded
                         + bidh * params.seqlen_q_rounded
                         + m_block * kBlockM;
        }

        #pragma unroll
        for (int i = 0; i < kBlockM / 8; i++) {
            // Compute LSE = log(sum) + max
            float lse_value = logf(groups[g].lse_sum[i]) + groups[g].lse_max[i];
            lse_ptr[lse_offset + i] = lse_value;
        }
    }
}
```

### 4.2 Helper Functions

```cpp
// Apply KV boundary mask to scores
__device__ __forceinline__ void apply_kv_boundary_mask(
    Tensor<float>& scores,     // [kBlockM, kBlockN]
    int valid_kv_len,          // Number of valid KV positions in this tile
    int kBlockN
) {
    #pragma unroll
    for (int i = 0; i < size(scores); i++) {
        int col = get_col_index(i);  // Get column index in [0, kBlockN)
        if (col >= valid_kv_len) {
            scores(i) = -INFINITY;  // Mask out invalid positions
        }
    }
}

// Get row index from flattened MMA fragment index
// (Depends on specific MMA layout - simplified here)
__device__ __forceinline__ int get_row_index(int flat_idx, TiledMMA tiled_mma) {
    // Extract row coordinate from MMA fragment layout
    auto coord = tiled_mma.get_coord(flat_idx);
    return get<0>(coord);  // M dimension
}
```

---

## 5. Backward Kernel Considerations

### 5.1 Gradient Accumulation Strategy

**Challenge**: Multiple groups may attend to the same K,V regions, requiring gradient accumulation for dK and dV.

**Example**:
```
Group 0: attends to K[0:256], V[0:256]
Group 1: attends to K[0:512], V[0:512]
Overlap: K[0:256], V[0:256] receive gradients from BOTH groups
```

**Solution**: Atomic accumulation for overlapping K,V regions

```cpp
// Backward kernel (simplified pseudocode)
for (int g = 0; g < NumGroups; g++) {
    // Compute dK_g, dV_g for this group
    compute_gradients(dQ_g, dK_g, dV_g, ...);

    // Accumulate to global dK, dV using atomics
    for (int kv_idx = 0; kv_idx < kv_endpoint[g]; kv_idx++) {
        atomicAdd(&dK[kv_idx], dK_g[kv_idx]);
        atomicAdd(&dV[kv_idx], dV_g[kv_idx]);
    }
}
```

**Optimization**: Two-pass reduction instead of atomics
```cpp
// Pass 1: Write per-group gradients to separate buffers
dK_group0 = compute_dK(group0);  // Write to separate memory
dK_group1 = compute_dK(group1);  // Write to separate memory

// Pass 2: Reduction kernel (no atomics, coalesced access)
for each kv_position:
    dK[kv_position] = dK_group0[kv_position] + dK_group1[kv_position];
```

### 5.2 Backward Shared Memory Requirements

Similar to forward, but with additional dO, dQ, dK, dV accumulators. Expected to fit within same budget using register allocation strategies.

---

## 6. Potential Challenges and Mitigations

### Challenge 1: Shared Memory Limitations
**Symptom**: Kernel launch fails with "too much shared memory requested"

**Mitigation**:
- Start with smaller tile sizes (kBlockM=64)
- Keep accumulators in registers (already standard)
- Test on target GPU architecture (A100: 164KB/SM, H100: 228KB/SM)
- Fallback to sequential group processing if needed

### Challenge 2: Register Spilling
**Symptom**: Performance degradation due to local memory usage

**Mitigation**:
- Profile with `nvcc --ptxas-options=-v` to check register usage
- Reduce loop unrolling (`#pragma unroll` → `#pragma unroll 2`)
- Limit NumGroups to 2 initially
- Use `-maxrregcount` compiler flag to control occupancy

### Challenge 3: Thread Block Occupancy
**Symptom**: Low SM utilization

**Mitigation**:
- Monitor with Nsight Compute: `Achieved Occupancy` metric
- Balance register/smem usage to allow >=2 TBs per SM
- May need to reduce kNWarps from 4 to 2 for more TBs

### Challenge 4: Correctness Issues with Online Softmax
**Symptom**: Output diverges from reference implementation

**Mitigation**:
- Extensive unit tests comparing to sequential kernel calls
- Test with different KV endpoint configurations
- Verify LSE values match expected log-sum-exp
- Check for numerical stability (use FP32 for LSE accumulators)

### Challenge 5: KV Endpoint Enforcement Overhead
**Symptom**: Performance degrades with many boundary checks

**Mitigation**:
- Optimize branch prediction with likely/unlikely hints
- Precompute per-group validity bitmaps
- Use warp-level voting (`__ballot_sync`) for early exit

---

## 7. Testing and Validation Plan

### 7.1 Correctness Tests

#### Test 1: Two Groups, Identical KV Endpoints
```python
# Should produce identical results to single-group kernel
q0 = torch.randn(100, 8, 128)
q1 = torch.randn(100, 8, 128)
k = torch.randn(200, 8, 128)
v = torch.randn(200, 8, 128)

out_ref0 = flash_attn_forward(q0, k, v, ...)
out_ref1 = flash_attn_forward(q1, k, v, ...)

[out0, out1] = flash_attn_multigroup_forward([q0, q1], k, v, ...)

assert torch.allclose(out0, out_ref0, rtol=1e-3, atol=1e-3)
assert torch.allclose(out1, out_ref1, rtol=1e-3, atol=1e-3)
```

#### Test 2: Two Groups, Different KV Endpoints
```python
# Group 0: KV[0:128], Group 1: KV[0:256]
kv_endpoints = torch.tensor([[128], [256]])

[out0, out1], [lse0, lse1] = flash_attn_multigroup_forward(
    [q0, q1], k, v, kv_endpoints=kv_endpoints, ...
)

# Compare to reference with slicing
out_ref0 = flash_attn_forward(q0, k[:128], v[:128], ...)
out_ref1 = flash_attn_forward(q1, k[:256], v[:256], ...)

assert torch.allclose(out0, out_ref0, rtol=1e-3, atol=1e-3)
assert torch.allclose(out1, out_ref1, rtol=1e-3, atol=1e-3)
```

#### Test 3: Variable Sequence Lengths (Varlen)
```python
# Multiple sequences with different lengths per group
cu_seqlens_q0 = torch.tensor([0, 50, 120, 180])  # 3 sequences
cu_seqlens_q1 = torch.tensor([0, 30, 90, 150])
cu_seqlens_k0 = torch.tensor([0, 80, 160, 240])
cu_seqlens_k1 = torch.tensor([0, 120, 240, 360])

[out0, out1] = flash_attn_varlen_multigroup_forward(
    [q0, q1], k, v,
    cu_seqlens_q_list=[cu_seqlens_q0, cu_seqlens_q1],
    cu_seqlens_k_list=[cu_seqlens_k0, cu_seqlens_k1],
    ...
)
# Verify per-sequence correctness
```

#### Test 4: Causal Masking
```python
# Ensure causal mask is applied correctly per group
[out0, out1] = flash_attn_multigroup_forward(
    [q0, q1], k, v, causal=True, ...
)
# Verify by checking attention doesn't leak to future positions
```

### 7.2 Performance Tests

#### Benchmark 1: K,V Load Reduction
```python
# Profile memory transactions with nvprof/Nsight Compute
# Compare:
# - Baseline: Two separate kernel calls
# - Multigroup: Single kernel call
# Expected: ~38% reduction in K,V loads
```

#### Benchmark 2: Speedup vs. Sequential
```python
# Measure end-to-end time
import time

# Sequential
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    out0 = flash_attn_forward(q0, k[:256], v[:256], ...)
    out1 = flash_attn_forward(q1, k[:512], v[:512], ...)
end.record()
torch.cuda.synchronize()
time_sequential = start.elapsed_time(end) / 100

# Multigroup
start.record()
for _ in range(100):
    [out0, out1] = flash_attn_multigroup_forward([q0, q1], k, v, ...)
end.record()
torch.cuda.synchronize()
time_multigroup = start.elapsed_time(end) / 100

speedup = time_sequential / time_multigroup
print(f"Speedup: {speedup:.2f}x")
assert speedup > 1.3, "Target: >1.3x forward speedup"
```

### 7.3 Profiling Metrics to Monitor

Using Nsight Compute:
```bash
ncu --set full --target-processes all \
    --metrics dram__bytes_read,l2_cache_hit_rate,sm__throughput \
    python test_multigroup.py
```

**Target Metrics**:
- `dram__bytes_read`: Should decrease by ~38% vs. sequential
- `sm__throughput`: Should remain >80%
- `Achieved Occupancy`: Should be >=50%
- `sm__sass_inst_executed_per_cycle`: Should be close to baseline Flash Attention

---

## 8. Implementation Roadmap for Phase 2

### Week 1: Data Structures and Skeleton
- [ ] Create `flash_multigroup.h` with `Flash_fwd_multigroup_params` struct
- [ ] Implement parameter validation and setup functions
- [ ] Create kernel skeleton with empty `flash_fwd_multigroup_kernel` template
- [ ] Setup compilation infrastructure (Makefile/CMake integration)

### Week 2: Single-Group Case (Correctness Baseline)
- [ ] Implement kernel for NumGroups=1 (should match standard kernel)
- [ ] Test correctness against reference implementation
- [ ] Verify shared memory layout and register allocation
- [ ] Profile to ensure no regression vs. standard kernel

### Week 3: Two-Group Implementation
- [ ] Extend to NumGroups=2 with per-group state management
- [ ] Implement KV endpoint enforcement logic
- [ ] Add per-group LSE tracking
- [ ] Write basic correctness tests (identical endpoints)

### Week 4: Optimization and Testing
- [ ] Optimize shared memory layout (minimize bank conflicts)
- [ ] Tune tile sizes for target architecture
- [ ] Implement comprehensive test suite (varlen, causal, boundaries)
- [ ] Profile and compare to target metrics (>1.3x speedup)

---

## 9. Success Criteria

### Phase 2 Completion Checklist:
- [ ] Forward kernel correctly handles 2 groups
- [ ] All correctness tests pass (tolerance: 1e-3 for FP16)
- [ ] Forward-only speedup >1.3x vs. sequential kernels
- [ ] Memory bandwidth reduction >30% (measured via profiler)
- [ ] No numerical stability issues (LSE within expected range)
- [ ] Shared memory usage within limits (<164KB for A100)
- [ ] Code compiles without warnings
- [ ] Documentation complete with usage examples

### Performance Targets:
| Metric | Baseline (Sequential) | Target (Multigroup) | Status |
|--------|----------------------|---------------------|--------|
| K,V memory loads | 4.5× problem size | 3.25× problem size | TBD |
| Forward time (2048 seq) | 1.0× | <0.77× (>1.3x speedup) | TBD |
| Forward efficiency | 82.3% | >88% | TBD |
| SM occupancy | ~75% | >50% | TBD |

---

## 10. References

### Existing Flash Attention Files:
- `/Users/petrpan26/work/flash-attention/csrc/flash_attn/src/flash.h` - Parameter structs
- `/Users/petrpan26/work/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h` - Forward kernel implementation
- `/Users/petrpan26/work/flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h` - Kernel launch logic
- `/Users/petrpan26/work/flash-attention/csrc/flash_attn/src/kernel_traits.h` - Kernel configuration
- `/Users/petrpan26/work/flash-attention/csrc/flash_attn/src/softmax.h` - Online softmax implementation
- `/Users/petrpan26/work/flash-attention/csrc/flash_attn/src/block_info.h` - Varlen sequence handling

### Implementation Plan:
- `/Users/petrpan26/work/flash-attention/MULTIGROUP_IMPLEMENTATION_PLAN.md` - Overall project roadmap

### Papers:
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Dao, 2023)
- Online Softmax: Avoiding Materialization of Attention Matrix (Milakov & Gimelshein, 2018)

---

## Appendices

### Appendix A: Shared Memory Bank Conflict Analysis

For `d=128`, `kBlockM=128`:
- Each row: 128 elements = 256 bytes (FP16)
- Bank width: 4 bytes
- Banks per row: 256 / 4 = 64 banks
- Swizzle pattern: Already applied in `SmemLayoutQ` (3-bit XOR swizzle)

**Conclusion**: Existing swizzle pattern should prevent major bank conflicts. Monitor with Nsight Compute `l1tex__data_bank_conflicts_pipe_lsu` metric.

### Appendix B: Compile Flags for Debugging

```bash
# Verbose register usage
nvcc -Xptxas=-v

# Limit registers per thread (force spilling to test behavior)
nvcc -maxrregcount=128

# Generate line number info for profiler
nvcc -lineinfo

# Full debug symbols
nvcc -G
```

### Appendix C: Nsight Compute Command Reference

```bash
# Profile specific kernel
ncu --kernel-name flash_fwd_multigroup_kernel --launch-skip 0 --launch-count 1 \
    --set full --target-processes all \
    --export report python script.py

# Compare two runs
ncu --import baseline.ncu-rep,multigroup.ncu-rep
```

---

**Document Version**: 1.0
**Last Updated**: Phase 2 Preparation
**Author**: Claude (Assistant)
**Status**: Ready for Implementation

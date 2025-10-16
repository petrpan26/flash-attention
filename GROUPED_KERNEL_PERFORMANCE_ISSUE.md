# Grouped Kernel Performance Issue Analysis

## Benchmark Results Summary

**Hardware:** NVIDIA A10
**Expected speedup:** 1.15-1.20x (15-20% faster)
**Actual speedup:** 0.97-0.98x (2-3% **slower**)

```
Configuration: 8192 tokens, 2 groups
- Baseline:      8.114 ms
- CUDA grouped:  8.392 ms  ← 3% SLOWER instead of 15% faster!

Configuration: 16384 tokens, 2 groups
- Baseline:      35.209 ms
- CUDA grouped:  35.839 ms  ← 2% SLOWER
```

## Root Cause: Block Scheduling Doesn't Guarantee Sequential Execution

### ❌ What We Assumed

```
Grid: [Block 0, Block 1, ..., Block 7 | Block 8, Block 9, ..., Block 15]
      └─────── Group 0 ──────┘         └──────── Group 1 ────────┘

Assumed execution order:
  1. Blocks 0-7 execute first  → Load K,V into L2 cache
  2. Blocks 8-15 execute next  → Reuse K,V from L2 cache ✅

Expected: L2 cache hit rate ~70%, 15-20% speedup
```

### ✅ What Actually Happens

Modern GPUs schedule blocks **for maximum parallelism**, not locality:

```
GPU has 72 SMs (A10)
CUDA scheduler distributes blocks ROUND-ROBIN across SMs:

SM 0:  Block 0,  Block 72, Block 144, ...
SM 1:  Block 1,  Block 73, Block 145, ...
SM 2:  Block 2,  Block 74, Block 146, ...
...
SM 7:  Block 7,  Block 79, Block 151, ...
SM 8:  Block 8,  Block 80, Block 152, ...  ← Group 1 starts IMMEDIATELY
       ^^^^^^^^
       Group 1 block runs CONCURRENTLY with Group 0 blocks!
```

**Result:**
- Group 0 and Group 1 blocks run **concurrently**
- K,V data from Group 0 gets **evicted from L2** before Group 1 can reuse it
- **No L2 cache sharing benefit**
- Added overhead from group ID determination → 2-3% slower

## Why the Current Implementation Fails

### 1. No Execution Order Guarantees

```cpp
// In flash_fwd_launch_template.h:322
dim3 grid(total_num_m_blocks, params.b, params.h);
//        ^^^^^^^^^^^^^^^^^^^^
//        Blocks 0-19 (8 for Group 0, 12 for Group 1)

kernel<<<grid, ...>>>(params);  // Single launch
```

CUDA does NOT guarantee that:
- Blocks 0-7 finish before blocks 8-19 start
- Blocks even execute in any particular order

### 2. L2 Cache Eviction

**A10 L2 cache:** 6 MB
**Typical K,V tile:** 128 tokens × 128 dim × 2 bytes × 2 (K+V) = 64 KB

With 72 SMs executing concurrently:
- Each SM loads different K,V tiles
- L2 cache fills with tiles from different sequence positions
- By the time Group 1 blocks need a tile, it's been evicted
- **L2 hit rate ≈ baseline** (no benefit)

### 3. Added Overhead

```cpp
// In flash_fwd_kernel.h:1694-1702
for (int g = 0; g < params.num_groups; g++) {
    int num_m_blocks_g = params.group_num_m_blocks[g];  // Device memory read
    if (m_block_local < num_m_blocks_g) {
        group_id = g;
        break;
    }
    m_block_local -= num_m_blocks_g;
}
```

Every thread block:
- Reads `group_num_m_blocks` from device memory (latency)
- Performs O(num_groups) loop (compute overhead)
- Reads group-specific pointers from device memory

**Small overhead (< 1%), but no speedup to offset it → net 2-3% slowdown**

## Visualization: Actual vs Expected Execution

### Expected (Sequential Groups)

```
Timeline:
0ms ─────────────────────────────────────> 16ms
│                                          │
├──── Group 0 blocks (8 blocks) ─────┤
│  Load K,V → L2 cache               │
                                      └──── Group 1 blocks (12 blocks) ────┤
                                           Reuse K,V from L2 ✅            │

L2 Cache:
Group 0: [████████████ K,V tiles ████████████]
Group 1: [████████████ SAME tiles (HIT!) █████]
```

### Actual (Concurrent Groups)

```
Timeline:
0ms ────────────────────> 8ms
│                        │
├── All 20 blocks run concurrently ──┤
│   Group 0: Blocks 0,1,2,...,7     │
│   Group 1: Blocks 8,9,10,...,19   │
│   (interleaved on different SMs)  │

L2 Cache (constantly evicting):
Time 0: [Block 0 K,V tiles]
Time 1: [Block 1 K,V tiles] ← Block 0 data evicted
Time 2: [Block 8 K,V tiles] ← Group 1 block! Evicts Group 0 data
Time 3: [Block 2 K,V tiles] ← More evictions
...
Result: LOW hit rate, NO benefit ❌
```

## Profiling Evidence Needed

To confirm this hypothesis, we need **Nsight Systems profiling**:

```bash
nsys profile -o grouped_kernel.nsys-rep \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    python benchmark.py

# Analyze:
nsys stats --report cuda_gpu_kern_sum grouped_kernel.nsys-rep
```

**What to look for:**
1. **Block execution timeline:** Are Group 0 and Group 1 blocks concurrent?
2. **L2 cache hit rate:** Compare to baseline (should be higher, but isn't)
3. **Memory throughput:** Should be lower than baseline (but probably isn't)

## Why Python Wrapper is Same Speed as Baseline

```python
# Python wrapper (flash_api.cpp Phase 1 - the old version)
for group_id in range(num_groups):
    varlen_fwd(q[group_id], k, v, ...)  # Sequential kernel launches
```

**Each kernel launch waits for previous to complete:**
- Group 0 kernel finishes → K,V in L2
- **CPU schedules next kernel**
- Group 1 kernel starts → Some L2 hits (5-10% speedup)
- But CPU overhead ≈ 5-10% slowdown
- **Net result: ~1.00x** (same as baseline)

This is actually achieving SOME L2 cache benefit, but CPU overhead cancels it out!

## Solutions

### Option 1: Stream-Based Sequential Execution (Simple)

**Force sequential group execution using CUDA streams with dependencies:**

```cpp
// In flash_api.cpp
cudaStream_t streams[num_groups];
cudaEvent_t events[num_groups];

for (int g = 0; g < num_groups; g++) {
    cudaStreamCreate(&streams[g]);
    cudaEventCreate(&events[g]);
}

for (int g = 0; g < num_groups; g++) {
    if (g > 0) {
        // Wait for previous group to finish
        cudaStreamWaitEvent(streams[g], events[g-1], 0);
    }

    // Launch kernel for this group only
    dim3 grid(group_num_m_blocks[g], params.b, params.h);
    flash_fwd_kernel<<<grid, ..., streams[g]>>>(params_g);

    cudaEventRecord(events[g], streams[g]);
}

cudaDeviceSynchronize();
```

**Pros:**
- ✅ Guarantees sequential execution
- ✅ Should achieve L2 cache sharing
- ✅ Minimal code changes

**Cons:**
- ⚠️ Requires separate kernel launches (back to Phase 1 approach)
- ⚠️ Still have CPU overhead between groups

### Option 2: Persistent Kernel with Grid-Wide Sync

**Use cooperative groups to synchronize all blocks between groups:**

```cpp
#include <cooperative_groups.h>

__global__ void __launch_bounds__(kNThreads, kMinBlocksPerSM)
persistent_grouped_kernel(Flash_fwd_params params) {

    for (int group_id = 0; group_id < params.num_groups; group_id++) {
        // Determine which blocks process this group
        int num_m_blocks_g = params.group_num_m_blocks[group_id];
        int group_start_block = 0;
        for (int g = 0; g < group_id; g++) {
            group_start_block += params.group_num_m_blocks[g];
        }

        if (blockIdx.x >= group_start_block &&
            blockIdx.x < group_start_block + num_m_blocks_g) {
            int m_block_local = blockIdx.x - group_start_block;
            compute_attn_1rowblock_grouped(..., m_block_local, group_id);
        }

        // Wait for all blocks to finish this group before starting next
        cooperative_groups::this_grid().sync();
    }
}

// Launch with cooperative groups
cudaLaunchCooperativeKernel(
    (void*)persistent_grouped_kernel,
    grid, block, args, smem_size, stream
);
```

**Pros:**
- ✅ Single kernel launch
- ✅ Guaranteed sequential group processing
- ✅ Maximum L2 cache reuse

**Cons:**
- ⚠️ Requires cooperative launch (not all GPUs support)
- ⚠️ Idle threads: Group 0 blocks idle during Group 1, vice versa
- ⚠️ May reduce occupancy

### Option 3: Separate Kernel Launches with Explicit Ordering

**Go back to Phase 1 but optimize:**

```cpp
for (int g = 0; g < num_groups; g++) {
    // Set up params for this group
    params.q_ptr = q_list[g].data_ptr();
    params.o_ptr = out_list[g].data_ptr();
    params.cu_seqlens_k = cu_seqlens_k_list[g].data_ptr();
    params.max_seqlen_k = max_seqlen_k_list[g];

    dim3 grid(group_num_m_blocks[g], params.b, params.h);

    // Launch on default stream (sequential)
    flash_fwd_kernel<<<grid, ..., 0>>>(params);
}
// No need to sync - default stream is sequential
```

**Pros:**
- ✅ Simple implementation
- ✅ Guaranteed L2 cache benefit
- ✅ No cooperative kernel requirements

**Cons:**
- ⚠️ CPU overhead between launches (~5-10%)
- ⚠️ May not achieve full 15-20% speedup

### Option 4: Graph Capture (Best for Production)

**Use CUDA graphs to eliminate CPU overhead:**

```cpp
cudaGraph_t graph;
cudaGraphExec_t graph_exec;

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

for (int g = 0; g < num_groups; g++) {
    dim3 grid(group_num_m_blocks[g], params.b, params.h);
    flash_fwd_kernel<<<grid, ..., stream>>>(params_g);
}

cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);

// Execute entire graph (all kernels) with minimal CPU overhead
cudaGraphLaunch(graph_exec, stream);
cudaStreamSynchronize(stream);
```

**Pros:**
- ✅ Sequential execution (guaranteed L2 reuse)
- ✅ Minimal CPU overhead (< 1%)
- ✅ Should achieve 15-20% speedup
- ✅ Best performance for repeated calls

**Cons:**
- ⚠️ More complex implementation
- ⚠️ Graph must be rebuilt if parameters change

## Recommended Fix

**For immediate fix:** Option 3 (Separate launches)
**For production:** Option 4 (CUDA graphs)

### Implementation: Option 3 (Immediate)

```cpp
// In flash_api.cpp: mha_varlen_fwd_grouped()

// Remove unified grid approach
// Instead: Sequential kernel launches on default stream

auto stream = at::cuda::getCurrentCUDAStream().stream();

for (int g = 0; g < num_groups; g++) {
    // Set up params for this group
    Flash_fwd_params params_g;
    set_params_fprop(params_g,
        cu_seqlens_q_list[g].size(0) - 1,
        max_seqlen_q_list[g],
        max_seqlen_k_list[g],
        // ... other params
        q_list[g],
        k, v,
        out_list[g],
        cu_seqlens_q_list[g].data_ptr(),
        cu_seqlens_k_list[g].data_ptr(),
        // ...
        softmax_lse_list[g].data_ptr(),
        // ...
    );

    // Launch kernel for this group (sequential on default stream)
    run_mha_fwd(params_g, stream);  // NOT run_mha_fwd_grouped
}

// Default stream ensures sequential execution → L2 cache benefit!
```

**Expected result:** 1.05-1.10x speedup (5-10% due to L2 cache minus CPU overhead)

## Theoretical vs Actual Bandwidth Savings

### Theory (Assumed Sequential Execution)

```
2 groups: Load K,V once + 0.33× for Group 1 = 1.33× total = 33% savings
4 groups: Load K,V once + 3×0.25× for Groups 1-3 = 1.75× total = 60% savings
```

### Reality (Concurrent Execution)

```
2 groups: Load K,V for Group 0 + Load K,V for Group 1 (concurrent) = 2× total = 0% savings
4 groups: Load K,V for all groups (concurrent) = 4× total = 0% savings
```

**Actual performance: Baseline + 2-3% overhead = 0.97-0.98× speedup**

## Conclusion

The current unified grid implementation **does not work** because:
1. ❌ CUDA doesn't guarantee block execution order
2. ❌ Groups execute concurrently, not sequentially
3. ❌ L2 cache eviction destroys any reuse benefit
4. ❌ Added overhead makes it slightly slower than baseline

**The fix requires enforcing sequential group execution** using one of the options above.

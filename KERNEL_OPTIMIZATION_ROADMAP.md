# Multi-Group Kernel Optimization Roadmap

## Current Status: Suboptimal Kernel Configurations

### Problem Summary

The multi-group varlen attention kernel currently uses **fixed kernel configurations** for all head dimensions, while standard Flash Attention uses **highly tuned, dimension-specific configurations**. This results in an estimated **10-30% performance degradation** depending on the dimension and GPU architecture.

## Performance Impact Analysis

### Configuration Comparison

| Head Dim | Standard FA Config | Multi-Group Config | Est. Impact |
|----------|-------------------|-------------------|-------------|
| **32** | M=128, N=128, W=4 | M=64, N=128, W=4 | **-5-10%** |
| **64** | M=128, N=128, W=4 | M=64, N=128, W=4 | **-5-10%** |
| **96** | M=128, N=64, W=4 | M=64, N=128, W=4 | **-10-15%** |
| **128** | M=128, N=32, W=4 (sm8x) | M=64, N=128, W=4 | **-15-25%** ⚠️ |
| **192** | M=128, N=64, **W=8** | M=64, N=128, W=4 | **-20-30%** ⚠️ |
| **256** | M=128, N=64, **W=8** (A100) | M=32, N=128, W=4 | **-20-30%** ⚠️ |

*M=kBlockM (Q tile), N=kBlockN (K,V tile), W=kNWarps*

### Why Current Config Was Chosen

The configuration `(M=64, N=128, W=4)` was selected for:

1. ✅ **Simplicity**: Single config for all dimensions
2. ✅ **Memory safety**: Fits within 164KB shared memory limit for NumGroups=2-4
3. ✅ **Fast development**: Easy to implement and debug
4. ❌ **Performance**: Not optimized per dimension/architecture

## Missing Optimizations

### 1. Architecture-Specific Tuning

**Standard FA** detects GPU architecture and adapts:

```cpp
auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
bool is_sm8x = cc_major == 8 && cc_minor > 0;  // A6000, A100 8.6/8.9

if (is_sm8x) {
    // sm8x-specific optimizations
    // - Smaller tile sizes for better occupancy
    // - 2 CTAs/SM on non-causal d=128
}
```

**Impact**:
- d=128 non-causal: Up to **25% faster** with 2 CTAs/SM on A6000/A100
- d=96 causal: **15% faster** with square 64x64 tiles

### 2. Causal vs Non-Causal Optimization

**Standard FA d=128**:
- **Non-causal**: M=128, N=32 (48KB smem → 2 CTAs/SM on sm8x)
- **Causal**: M=64, N=64 (square tiles → better Q/KV reuse)

**Multi-Group**: Always M=64, N=128 regardless

**Impact**:
- Non-causal d=128: Missing **2x occupancy** on sm8x
- Causal d=128: Suboptimal tile shape

### 3. Warp Count Scaling

**Standard FA**:
- d=192, d=256: **8 warps** (256 threads) for higher throughput
- Other dims: **4 warps** (128 threads)

**Multi-Group**: Always **4 warps**

**Impact**:
- d=192, d=256: Up to **30% slower** due to lower parallelism

### 4. GPU-Specific Optimization

**Standard FA d=256**:
```cpp
int max_smem_per_sm, max_smem_per_block;
cudaDeviceGetAttribute(&max_smem_per_sm, ...);
cudaDeviceGetAttribute(&max_smem_per_block, ...);

if (max_smem_per_block >= 2*256*(128 + 2*64) && ...) {
    // A100: M=128, N=64, W=8 (128KB smem)
} else {
    // H100: M=64, N=64, W=4 (96KB smem, 2 CTAs/SM)
}
```

**Multi-Group**: Always M=32, N=128, W=4

**Impact**:
- A100: Missing optimal 128x64 config
- H100: Missing 2 CTAs/SM optimization

### 5. Dropout-Aware Configuration

**Standard FA d=192**:
- **No dropout**: M=128, N=64, W=8 (high throughput)
- **With dropout**: M=64, N=64, W=4 (smaller config, less register pressure)

**Multi-Group**: Doesn't adapt to dropout

**Impact**: Minor (~5%) when dropout is used

## Optimization Roadmap

### Phase 5.1: Basic Per-Dimension Tuning (High Priority)

**Goal**: Match standard FA configurations for each head dimension

**Tasks**:
1. Implement dimension-specific kernel traits:
   ```cpp
   // d=128 example
   template<typename T, int NumGroups, bool Is_causal>
   void run_mha_fwd_multigroup_hdim128(...) {
       if (NumGroups == 2) {
           // Optimized for 2 groups
           using Kernel_traits = Flash_fwd_multigroup_kernel_traits<
               128, 96, 64, 4, 2, false, false, T>;
       } else if (NumGroups == 3) {
           // Different config for 3 groups
           using Kernel_traits = Flash_fwd_multigroup_kernel_traits<
               128, 64, 64, 4, 3, false, false, T>;
       }
       run_flash_fwd_multigroup<Kernel_traits, NumGroups, Is_causal>(...);
   }
   ```

2. Calculate optimal configs considering:
   - Multi-group Q tile overhead: `NumGroups × kBlockM × d × 2 bytes`
   - Shared K,V tiles: `kBlockN × d × 4 bytes` (K+V)
   - Total must fit in 164KB (A100) or 100KB (A6000)

3. Target configs (NumGroups=2):
   - d=32: M=96, N=128, W=4 (closer to 128x128)
   - d=64: M=96, N=128, W=4
   - d=96: M=96, N=96, W=4 (square-ish)
   - d=128: M=96, N=64, W=4 or M=64, N=96, W=4
   - d=192: M=64, N=64, W=**8** ⚠️
   - d=256: M=64, N=64, W=**8** ⚠️

**Expected Gain**: **10-15% average speedup**

**Effort**: 2-3 days (calculate configs, implement, test)

### Phase 5.2: Architecture-Specific Tuning (Medium Priority)

**Goal**: Detect GPU and use architecture-specific configs

**Tasks**:
1. Add GPU capability detection:
   ```cpp
   auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
   bool is_sm8x = cc_major == 8 && cc_minor > 0;
   bool is_sm90 = cc_major >= 9;  // H100
   ```

2. Implement sm8x optimizations:
   - d=128 non-causal: Try M=96, N=32 for 2 CTAs/SM
   - d=96 causal: Try M=64, N=64 square tiles

3. Implement H100 (sm90) optimizations:
   - d=256: Prioritize 2 CTAs/SM over large tiles

**Expected Gain**: **Additional 5-10% on sm8x GPUs**

**Effort**: 1-2 days

### Phase 5.3: Causal vs Non-Causal Optimization (Low Priority)

**Goal**: Use different configs for causal vs non-causal

**Tasks**:
1. Add causal-aware configs:
   ```cpp
   if constexpr (Is_causal) {
       // Square tiles for better Q/KV reuse
   } else {
       // Rectangle tiles for better occupancy
   }
   ```

2. Focus on d=128 where causal matters most

**Expected Gain**: **5-10% for causal workloads**

**Effort**: 1 day

### Phase 5.4: 8-Warp Kernels (High Priority for d=192, d=256)

**Goal**: Implement 8-warp kernels for large dimensions

**Tasks**:
1. Add warp count as template parameter (already done in kernel traits)

2. Implement 8-warp configs for d=192, d=256:
   ```cpp
   // d=192 with 8 warps
   using Kernel_traits = Flash_fwd_multigroup_kernel_traits<
       192, 64, 64, 8, NumGroups, false, false, T>;
   ```

3. Test register usage (may need reduction)

**Expected Gain**: **20-30% for d=192, d=256**

**Effort**: 2-3 days (kernel validation, register optimization)

### Phase 5.5: Dynamic Shared Memory Optimization (Low Priority)

**Goal**: Adapt config based on available shared memory

**Tasks**:
1. Query device capabilities at runtime:
   ```cpp
   cudaDeviceGetAttribute(&max_smem_per_sm, ...);
   cudaDeviceGetAttribute(&max_smem_per_block, ...);
   ```

2. Select largest safe config:
   - High smem GPUs (A100, H100): Use larger tiles
   - Low smem GPUs (RTX 3090): Use smaller tiles

**Expected Gain**: **5-10% on high-end GPUs**

**Effort**: 2-3 days

## Shared Memory Constraints (NumGroups=2)

Must satisfy: `NumGroups × kBlockM × d × 2 + kBlockN × d × 4 ≤ max_smem`

For A100 (164KB max):

| Head Dim | Current | Phase 5.1 Target | Savings |
|----------|---------|------------------|---------|
| 32 | M=64, N=128 (24KB) | M=96, N=128 (28KB) | -4KB |
| 64 | M=64, N=128 (48KB) | M=96, N=128 (56KB) | -8KB |
| 96 | M=64, N=128 (72KB) | M=96, N=96 (73KB) | -1KB |
| 128 | M=64, N=128 (96KB) | M=96, N=64 (88KB) | +8KB |
| 192 | M=64, N=128 (144KB) | M=64, N=64 (96KB) | **+48KB** |
| 256 | M=32, N=128 (96KB) | M=64, N=64 (128KB) | -32KB |

All fit within 164KB limit ✓

## Testing Plan

For each optimization phase:

1. **Correctness**: Run full test suite
   ```bash
   pytest test/test_multigroup_flash_attn.py -v
   ```

2. **Performance**: Benchmark all configs
   ```bash
   python benchmarks/benchmark_multigroup_cuda.py --all-configs
   ```

3. **Memory**: Validate shared memory usage
   ```bash
   nsys profile --stats=true python benchmarks/benchmark_multigroup_cuda.py
   ```

4. **Comparison**: Compare vs standard FA
   ```bash
   python benchmarks/compare_to_standard_fa.py
   ```

## Priority Ranking

| Phase | Priority | Impact | Effort | ROI |
|-------|----------|--------|--------|-----|
| 5.1: Per-Dim Tuning | **HIGH** | +10-15% | 2-3d | ⭐⭐⭐⭐⭐ |
| 5.4: 8-Warp Kernels | **HIGH** | +20-30% (d=192,256) | 2-3d | ⭐⭐⭐⭐⭐ |
| 5.2: Arch-Specific | **MEDIUM** | +5-10% | 1-2d | ⭐⭐⭐⭐ |
| 5.3: Causal Opt | **LOW** | +5-10% (causal) | 1d | ⭐⭐⭐ |
| 5.5: Dynamic Smem | **LOW** | +5-10% | 2-3d | ⭐⭐ |

**Recommendation**: Start with Phase 5.1 and 5.4 for maximum impact.

## Expected Overall Performance Improvement

After all optimizations:

| Scenario | Current | After Phase 5 | Speedup |
|----------|---------|---------------|---------|
| d=128, non-causal, sm8x | 1.3x | **1.5-1.6x** | +15-23% |
| d=192, any | 1.3x | **1.5-1.7x** | +15-30% |
| d=256, A100 | 1.2x | **1.4-1.6x** | +17-33% |
| Average across all dims | 1.3x | **1.45-1.55x** | +12-19% |

**Total Expected Gain**: **+12-19% average performance improvement** over current implementation.

## Backwards Compatibility

All optimizations maintain API compatibility:
- ✅ Same Python API
- ✅ Same function signatures
- ✅ Same output format
- ✅ Transparent to users

Users will automatically benefit from optimizations after recompilation.

## Summary

**Current Implementation**:
- ❌ Fixed config for all dimensions
- ❌ No architecture tuning
- ❌ No causal optimization
- ❌ Always 4 warps

**After Phase 5 Optimizations**:
- ✅ Dimension-specific configs
- ✅ Architecture-aware (sm8x, sm90)
- ✅ Causal vs non-causal optimization
- ✅ 8 warps for d=192, d=256
- ✅ **~12-19% faster on average**

**Recommendation**: Implement Phase 5.1 (per-dimension tuning) and Phase 5.4 (8-warp kernels) first for maximum impact with reasonable effort.

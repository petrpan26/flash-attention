# Multi-Group Varlen Attention API Reference

Quick reference for implementing multi-group varlen attention kernels.

## Problem Statement

**Current Issue**: `zigzag_llama3_flash_attn_varlen.py` calls Flash Attention twice (one per Q group):

```python
# Current: TWO kernel calls
for group_idx in range(2):
    out_group, lse_group = _flash_attn_varlen_forward(
        q_group, k_slice, v_slice, ...
    )
    # K,V slices overlap → redundant loads (~38% wasted bandwidth)
```

**Solution**: Single kernel that processes all groups:

```python
# Optimized: ONE kernel call
out_groups, lse_groups = _flash_attn_varlen_multigroup_forward(
    q_list=[q_group0, q_group1],
    k=k_full, v=v_full,
    kv_endpoints=endpoints,  # Different K,V range per group
    ...
)
# Each K,V tile loaded once, used by all applicable groups
```

---

## Python API

### Forward Function

```python
def _flash_attn_varlen_multigroup_forward(
    q_list: List[torch.Tensor],              # [tokens_g, nheads, d] per group
    k: torch.Tensor,                          # [total_tokens, nheads_k, d]
    v: torch.Tensor,                          # [total_tokens, nheads_k, d]
    cu_seqlens_q_list: List[torch.Tensor],   # [num_seqs+1] per group
    cu_seqlens_k_list: List[torch.Tensor],   # [num_seqs+1] per group
    kv_endpoints: torch.Tensor,               # [num_groups, num_seqs] int32
    max_seqlen_q_list: List[int],
    max_seqlen_k_list: List[int],
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Multi-group varlen forward pass.

    Args:
        q_list: List of Q tensors, one per group
                - Group g: [tokens_in_group_g, nheads, head_dim]
        k, v: Shared K,V tensors (contiguous format)
                - [total_tokens, nheads_k, head_dim]
        cu_seqlens_q_list: Cumulative sequence lengths for each Q group
                - Group g: [num_seqs+1], e.g., [0, 128, 256, 384]
        cu_seqlens_k_list: Cumulative sequence lengths for KV slice per group
                - Group g: [num_seqs+1], specifies which K,V each Q seq uses
        kv_endpoints: Maximum K,V position each group/sequence needs
                - [num_groups, num_seqs], e.g., [[128, 256], [512, 1024]]
                - Group g, seq s attends to k[cu_seqlens_k_list[g][s] : kv_endpoints[g, s]]
        max_seqlen_q_list: [max_seqlen_q_group_0, max_seqlen_q_group_1, ...]
        max_seqlen_k_list: [max_seqlen_k_group_0, max_seqlen_k_group_1, ...]

    Returns:
        out_list: [out_group_0, out_group_1, ...], each [tokens_in_group, nheads, d]
        lse_list: [lse_group_0, lse_group_1, ...], each [nheads, tokens_in_group]
    """
```

### Backward Function

```python
def _flash_attn_varlen_multigroup_backward(
    dout_list: List[torch.Tensor],           # [tokens_g, nheads, d] per group
    q_list: List[torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    out_list: List[torch.Tensor],
    softmax_lse_list: List[torch.Tensor],
    cu_seqlens_q_list: List[torch.Tensor],
    cu_seqlens_k_list: List[torch.Tensor],
    kv_endpoints: torch.Tensor,
    max_seqlen_q_list: List[int],
    max_seqlen_k_list: List[int],
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Multi-group varlen backward pass.

    Returns:
        dq_list: [dq_group_0, dq_group_1, ...], each [tokens_in_group, nheads, d]
        dk: [total_tokens, nheads_k, d] - ACCUMULATED across all groups
        dv: [total_tokens, nheads_k, d] - ACCUMULATED across all groups
    """
```

---

## CUDA Kernel Structure

### Parameter Struct

```cpp
// File: csrc/flash_attn/src/flash_multigroup.h

struct Flash_fwd_multigroup_params {
    // Q data (per-group)
    void **q_ptr_list;                    // [num_groups] pointers
    int *q_row_stride_list;               // [num_groups]
    int *q_head_stride_list;              // [num_groups]

    // K, V data (shared)
    void *k_ptr;
    void *v_ptr;
    int k_row_stride;
    int k_head_stride;
    int v_row_stride;
    int v_head_stride;

    // Output data (per-group)
    void **out_ptr_list;                  // [num_groups] pointers
    int *out_row_stride_list;
    int *out_head_stride_list;

    // LSE data (per-group)
    void **softmax_lse_ptr_list;          // [num_groups] pointers

    // Sequence metadata
    int num_groups;
    int **cu_seqlens_q_list;              // [num_groups][batch_size+1]
    int **cu_seqlens_k_list;              // [num_groups][batch_size+1]
    int *max_seqlen_q_per_group;          // [num_groups]
    int *max_seqlen_k_per_group;          // [num_groups]
    int **kv_endpoints;                   // [num_groups][batch_size]

    // Common parameters
    int num_heads;
    int num_heads_k;
    int head_size;
    float softmax_scale;
    bool is_causal;
    int window_size_left;
    int window_size_right;
    float *alibi_slopes;
};
```

### Kernel Pseudocode

```cpp
// File: csrc/flash_attn/src/flash_fwd_multigroup_kernel.h

template<typename Kernel_traits, bool Is_causal>
__global__ void flash_fwd_multigroup_kernel(Flash_fwd_multigroup_params params) {
    // Allocate shared memory
    __shared__ char smem_k[Br * head_size];  // K tile (shared)
    __shared__ char smem_v[Br * head_size];  // V tile (shared)

    // Per-group state
    struct GroupState {
        char smem_q[Bc * head_size];      // Q tile
        float smem_acc[Bc * head_size];   // Accumulator
        float lse[Bc];                    // LSE
        int kv_max_offset;                // KV endpoint
        bool active;
    };
    GroupState groups[MAX_NUM_GROUPS];

    // Initialize groups
    for (int g = 0; g < params.num_groups; g++) {
        groups[g].kv_max_offset = params.kv_endpoints[g][batch_idx];
        groups[g].active = true;
    }

    // Main loop: iterate over K,V tiles
    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
        // Load K,V tile ONCE (shared across all groups)
        load_kv_tile(smem_k, smem_v, kv_tile_idx);
        __syncthreads();

        // Process each group
        for (int g = 0; g < params.num_groups; g++) {
            if (!groups[g].active) continue;

            // Skip if this K,V tile is beyond group's endpoint
            if (kv_tile_start >= groups[g].kv_max_offset) continue;

            // Load Q tile for this group
            load_q_tile(groups[g].smem_q, g, q_tile_idx);

            // Compute attention: QK^T → softmax → PV
            compute_qk(groups[g].smem_q, smem_k, qk_scores);
            softmax(qk_scores, groups[g].lse);
            matmul_pv(qk_scores, smem_v, groups[g].smem_acc);
        }
    }

    // Write outputs
    for (int g = 0; g < params.num_groups; g++) {
        write_output(groups[g].smem_acc, params.out_ptr_list[g]);
        write_lse(groups[g].lse, params.softmax_lse_ptr_list[g]);
    }
}
```

### Backward Kernel (Gradient Accumulation)

```cpp
template<typename Kernel_traits, bool Is_causal>
__global__ void flash_bwd_multigroup_kernel(Flash_bwd_multigroup_params params) {
    // Similar structure to forward...

    // Compute dQ per group (independent)
    for (int g = 0; g < params.num_groups; g++) {
        compute_dq(dout[g], q[g], k, v, dq[g]);
    }

    // Compute dK, dV with accumulation across groups
    for (int g = 0; g < params.num_groups; g++) {
        compute_dk_dv(dout[g], q[g], k, v, dk_local, dv_local);

        // Atomic add to global dK, dV (for overlapping regions)
        for (int i = tidx; i < tile_size; i += blockDim.x) {
            atomicAdd(&params.dk_ptr[offset + i], dk_local[i]);
            atomicAdd(&params.dv_ptr[offset + i], dv_local[i]);
        }
    }
}
```

**Key Point**: Overlapping K,V regions receive gradients from multiple groups → need atomic accumulation.

---

## Integration with ring-flash-attention

### Modify execute_grouped_attention

```python
# File: ring_flash_attn/zigzag_llama3_flash_attn_varlen.py
# Replace lines 393-525

def execute_grouped_attention(
    chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list,
    kv_buffer, kv_slices, nheads, head_dim, softmax_scale,
    dropout_p, causal, window_size, alibi_slopes, deterministic,
    world_size, cu_seqlens_k
):
    # Step 1: Rearrange K,V from zigzag to contiguous (unchanged)
    kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_k)

    # Step 2: Prepare kv_endpoints tensor
    num_groups = len(chunk_q_list)
    num_seqs = len(cu_seqlens_k) - 1
    kv_endpoints = torch.zeros((num_groups, num_seqs), dtype=torch.int32, device='cuda')

    cu_seqlens_k_list = []
    max_seqlen_q_list = []
    max_seqlen_k_list = []

    for g, (seq_ranges, cu_seqlens_k_slice) in enumerate(kv_slices):
        # Fill kv_endpoints
        for s, (start, end) in enumerate(seq_ranges):
            kv_endpoints[g, s] = end

        # Collect cu_seqlens and max_seqlen
        cu_seqlens_k_list.append(cu_seqlens_k_slice)
        max_seqlen_q_list.append((chunk_cu_seqlens_q_list[g][1:] - chunk_cu_seqlens_q_list[g][:-1]).max().item())
        max_seqlen_k_list.append((cu_seqlens_k_slice[1:] - cu_seqlens_k_slice[:-1]).max().item())

    # Step 3: Call multi-group kernel (SINGLE CALL!)
    try:
        from flash_attn.flash_attn_multigroup_interface import _flash_attn_varlen_multigroup_forward

        out_chunks, lse_chunks = _flash_attn_varlen_multigroup_forward(
            q_list=chunk_q_list,
            k=kv_contiguous[0],
            v=kv_contiguous[1],
            cu_seqlens_q_list=chunk_cu_seqlens_q_list,
            cu_seqlens_k_list=cu_seqlens_k_list,
            kv_endpoints=kv_endpoints,
            max_seqlen_q_list=max_seqlen_q_list,
            max_seqlen_k_list=max_seqlen_k_list,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
        )

    except ImportError:
        # Fallback to original dual-kernel
        print("[WARNING] Multi-group kernel not available")
        return execute_grouped_attention_original(...)

    # Step 4: Scatter outputs back to original positions (unchanged)
    total_q = sum(q.shape[0] for q in chunk_q_list)
    out = torch.zeros((total_q, nheads, head_dim), dtype=out_chunks[0].dtype, device='cuda')
    for out_chunk, indices in zip(out_chunks, chunk_indices_list):
        out[indices] = out_chunk

    # ... (same for LSE)

    return out, lse, chunk_info
```

---

## Example Usage

### Typical Call Pattern (zigzag_llama3)

```python
# Context: After all-gather, we have:
# - Local Q split into 2 groups by chunk index
# - Full K,V from all ranks (contiguous)
# - Group 0 needs K,V prefix, Group 1 needs full K,V

# Prepare inputs
q_group_0 = q[indices_0]  # e.g., [128, 32, 128]
q_group_1 = q[indices_1]  # e.g., [128, 32, 128]

k_full = kv_contiguous[0]  # [1024, 8, 128] - full K from all ranks
v_full = kv_contiguous[1]  # [1024, 8, 128]

cu_seqlens_q_0 = torch.tensor([0, 128], dtype=torch.int32)
cu_seqlens_q_1 = torch.tensor([0, 128], dtype=torch.int32)

cu_seqlens_k_0 = torch.tensor([0, 256], dtype=torch.int32)  # Group 0 needs K,V[0:256]
cu_seqlens_k_1 = torch.tensor([0, 1024], dtype=torch.int32) # Group 1 needs K,V[0:1024]

kv_endpoints = torch.tensor([[256], [1024]], dtype=torch.int32)  # [2 groups, 1 seq]

# Call multi-group kernel
out_list, lse_list = _flash_attn_varlen_multigroup_forward(
    q_list=[q_group_0, q_group_1],
    k=k_full,
    v=v_full,
    cu_seqlens_q_list=[cu_seqlens_q_0, cu_seqlens_q_1],
    cu_seqlens_k_list=[cu_seqlens_k_0, cu_seqlens_k_1],
    kv_endpoints=kv_endpoints,
    max_seqlen_q_list=[128, 128],
    max_seqlen_k_list=[256, 1024],
    causal=True,
)

# Result:
# - out_list[0]: [128, 32, 128] - output for group 0
# - out_list[1]: [128, 32, 128] - output for group 1
# - K,V[0:256] loaded once (used by both groups)
# - K,V[256:1024] loaded once (used only by group 1)
```

---

## Performance Expectations

### Memory Bandwidth Analysis

**Current (dual-kernel)**:
```
Kernel 1 (group 0): Load K,V[0:256]     → 256 tokens
Kernel 2 (group 1): Load K,V[0:1024]    → 1024 tokens
Total K,V loads:                          1280 tokens (256 redundant)
```

**With multi-group kernel**:
```
Kernel 1 (multigroup): Load K,V[0:1024] → 1024 tokens (no redundancy)
Savings: 256 / 1280 = 20% reduction for this example
```

**For world_size=4 (8 chunks)**:
- Current: 4.5× total K,V loaded (due to overlap)
- Optimized: 3.25× total K,V loaded
- **Savings: ~28% reduction in K,V memory traffic**

### Expected Speedup

| Scenario | Speedup |
|----------|---------|
| Forward (memory-bound) | 1.2-1.4x |
| Forward (balanced) | 1.15-1.25x |
| Backward | 1.1-1.3x |
| Overall | 1.17-1.22x |

### Efficiency Improvement

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Forward efficiency | 82.3% | >88% | +5.7% |
| Fwd+Bwd efficiency | 88.9% | >92% | +3.1% |

---

## Implementation Checklist

### Phase 1: Python API + Mock (Week 1-2)

- [ ] Create `flash_attn_multigroup_interface.py`
- [ ] Implement `_flash_attn_varlen_multigroup_forward` (mock)
- [ ] Implement `_flash_attn_varlen_multigroup_backward` (mock)
- [ ] Create `MultiGroupFlashAttnVarlenFunc` AutoGrad wrapper
- [ ] Modify `execute_grouped_attention` with feature flag
- [ ] Write 10+ unit tests
- [ ] Test integration with zigzag_llama3

### Phase 2: Forward CUDA Kernel (Week 3-6)

- [ ] Create `flash_multigroup.h`
- [ ] Implement `flash_fwd_multigroup_kernel.h`
- [ ] Implement `flash_fwd_multigroup_launch.h`
- [ ] Create Python bindings in `flash_api_multigroup.cpp`
- [ ] Write CUDA correctness tests
- [ ] Profile and optimize
- [ ] Verify >1.3x forward speedup

### Phase 3: Backward CUDA Kernel (Week 7-9)

- [ ] Implement `flash_bwd_multigroup_kernel.h`
- [ ] Implement gradient accumulation (atomic adds)
- [ ] Create backward Python bindings
- [ ] Write gradient correctness tests
- [ ] Pass torch.autograd.gradcheck
- [ ] Verify >1.2x backward speedup

### Phase 4: Integration & Testing (Week 10-11)

- [ ] Replace mock with real kernels
- [ ] All zigzag_llama3 tests pass
- [ ] Multi-GPU distributed tests pass
- [ ] Performance benchmarks documented
- [ ] User documentation complete

### Phase 5: Optimization (Week 12-14)

- [ ] Profile with Nsight Compute
- [ ] Optimize shared memory layout
- [ ] Tune thread block dimensions
- [ ] >90% forward efficiency achieved

---

## Quick Commands

### Build

```bash
cd /Users/petrpan26/work/flash-attention
python setup.py install
```

### Test

```bash
cd /Users/petrpan26/work/ring-flash-attention
pytest test/test_multigroup_flash_attn.py -v -s
```

### Benchmark

```bash
python benchmarks/benchmark_multigroup.py
```

### Enable/Disable

```bash
# Use multi-group kernel
export RING_FLASH_ATTN_MULTIGROUP=1

# Fallback to original
export RING_FLASH_ATTN_MULTIGROUP=0
```

### Profile

```bash
nsys profile -o multigroup python script.py
nsys stats multigroup.nsys-rep
```

---

## Troubleshooting

### Common Issues

1. **Import error**: Multi-group kernel not found
   - Check flash-attention installation: `pip list | grep flash-attn`
   - Verify bindings: `python -c "from flash_attn.flash_attn_multigroup_interface import _flash_attn_varlen_multigroup_forward"`

2. **Output mismatch**: Results differ from reference
   - Check kv_endpoints tensor: print and verify values
   - Verify cu_seqlens_q/k lists match expected structure
   - Run with `RING_FLASH_ATTN_DEBUG=1` for verbose logging

3. **Gradient errors**: Backward pass incorrect
   - Verify atomic accumulation is enabled
   - Check for overlapping kv_endpoints (should accumulate)
   - Run `torch.autograd.gradcheck` for detailed diagnosis

4. **Performance not improving**: Speedup < 1.2x
   - Profile with nsys: `nsys profile -o profile python script.py`
   - Check memory bandwidth utilization
   - Verify multi-group kernel is being called (not fallback)

---

## Next Steps

1. **Start with Phase 1**: Implement Python API + mock
2. **Test integration**: Ensure zigzag_llama3 works with mock
3. **Move to Phase 2**: Implement CUDA forward kernel
4. **Iterate**: Profile, optimize, repeat

**Goal**: Reduce K,V redundant loads by ~38%, achieve 17-22% speedup in zigzag_llama3.

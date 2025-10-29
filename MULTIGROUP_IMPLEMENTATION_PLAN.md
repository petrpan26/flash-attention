# Multi-Group Varlen Attention Implementation Plan

## Executive Summary

**Problem**: The `zigzag_llama3_flash_attn_varlen.py` implementation calls Flash Attention kernel twice (once per Q group), causing **~38% redundant K,V memory loads**.

**Solution**: Implement a custom CUDA kernel `_flash_attn_varlen_multigroup_forward` that:
- Loads each K,V tile **once**
- Processes multiple Q groups with different KV endpoints
- Maintains separate LSE accumulators per group
- Expected improvement: **17-22% speedup**, efficiency from 88.9% → 92%+

**Timeline**: 3-3.5 months for experienced CUDA engineer

---

## Quick Start

### Repository Structure

```
flash-attention/               (this repo - new CUDA kernels)
├── csrc/flash_attn/src/
│   ├── flash_multigroup.h                    # NEW: Multigroup param structs
│   ├── flash_fwd_multigroup_kernel.h         # NEW: Forward kernel
│   ├── flash_bwd_multigroup_kernel.h         # NEW: Backward kernel
│   └── flash_api_multigroup.cpp              # NEW: Python bindings
└── flash_attn/
    └── flash_attn_multigroup_interface.py    # NEW: Python API

ring-flash-attention/         (external - integration)
└── ring_flash_attn/
    └── zigzag_llama3_flash_attn_varlen.py    # MODIFY: execute_grouped_attention
```

### Implementation Phases

| Phase | Duration | Description | Deliverables |
|-------|----------|-------------|--------------|
| **1. API Design** | 1-2 weeks | Python API + mock implementation | Working integration with mock |
| **2. Forward Kernel** | 3-4 weeks | CUDA forward kernel | >1.3x forward speedup |
| **3. Backward Kernel** | 2-3 weeks | CUDA backward with gradient accumulation | >1.2x backward speedup |
| **4. Integration** | 1-2 weeks | Full integration + testing | All tests passing |
| **5. Optimization** | 2-3 weeks | Performance tuning | >90% efficiency |

---

## 1. Python API Design

### Core Forward Function

```python
def _flash_attn_varlen_multigroup_forward(
    q_list: List[torch.Tensor],              # List of Q tensors, one per group
    k: torch.Tensor,                          # Shared K [total_tokens, nheads_k, head_dim]
    v: torch.Tensor,                          # Shared V [total_tokens, nheads_k, head_dim]
    cu_seqlens_q_list: List[torch.Tensor],   # cu_seqlens for each Q group
    cu_seqlens_k_list: List[torch.Tensor],   # cu_seqlens for each Q group's KV
    kv_endpoints: torch.Tensor,               # [num_groups, num_seqs] - KV end position per group
    max_seqlen_q_list: List[int],            # Max sequence length per group
    max_seqlen_k_list: List[int],            # Max KV sequence length per group
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Multi-group varlen flash attention forward pass.

    Each K,V tile is loaded once and processed for all applicable groups.

    Returns:
        out_list: List of output tensors, one per group
        lse_list: List of LSE tensors, one per group
    """
```

### Backward Function

```python
def _flash_attn_varlen_multigroup_backward(
    dout_list: List[torch.Tensor],           # Gradients for each group output
    q_list: List[torch.Tensor],              # Q tensors from forward
    k: torch.Tensor,                          # K tensor from forward
    v: torch.Tensor,                          # V tensor from forward
    out_list: List[torch.Tensor],            # Outputs from forward
    softmax_lse_list: List[torch.Tensor],    # LSE from forward
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
    Multi-group varlen flash attention backward pass.

    Returns:
        dq_list: List of Q gradients, one per group
        dk: Shared K gradient (accumulated from all groups)
        dv: Shared V gradient (accumulated from all groups)
    """
```

### Key Design Decisions

1. **List-based API**: Clean mapping to existing zigzag_llama3 structure
2. **Shared K,V**: Single tensor, different endpoints per group
3. **kv_endpoints tensor**: Fine-grained control over K,V access per group
4. **Separate outputs**: List of tensors matches current implementation pattern

---

## 2. CUDA Kernel Structure

### Data Structures (flash_multigroup.h)

```cpp
struct Flash_fwd_multigroup_params {
    // Q pointers - one per group
    void **q_ptr_list;                    // Array of Q pointers [num_groups]

    // Shared K, V pointers
    void *k_ptr;
    void *v_ptr;

    // Output pointers - one per group
    void **out_ptr_list;                  // Array of output pointers [num_groups]

    // LSE pointers - one per group
    void **softmax_lse_ptr_list;          // Array of LSE pointers [num_groups]

    // Sequence metadata
    int num_groups;
    int **cu_seqlens_q_list;              // Array of cu_seqlens_q pointers [num_groups]
    int **cu_seqlens_k_list;              // Array of cu_seqlens_k pointers [num_groups]
    int **kv_endpoints;                   // [num_groups][batch_size] - KV end position

    // Common parameters
    int num_heads, num_heads_k, head_size;
    float softmax_scale;
    bool is_causal;
    // ... other flash attention params
};
```

### Kernel Template (flash_fwd_multigroup_kernel.h)

```cpp
template<typename Kernel_traits, bool Is_causal>
__global__ void flash_fwd_multigroup_kernel(Flash_fwd_multigroup_params params) {
    // Per-group state structures
    struct GroupState {
        char *smem_q;         // Q tile for this group
        char *smem_acc;       // Accumulator for this group
        float *lse;           // LSE accumulator
        int kv_max_offset;    // Maximum K,V position for this group
        bool active;          // Whether this group has remaining Q tokens
    };

    GroupState group_states[MAX_NUM_GROUPS];  // e.g., MAX_NUM_GROUPS = 8

    // Main loop over K,V tiles
    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
        // Load K, V tiles ONCE (shared across all groups)
        load_kv_tile(params.k_ptr, params.v_ptr, kv_tile_idx, ...);
        __syncthreads();

        // Process each group with this K,V tile
        #pragma unroll
        for (int g = 0; g < num_groups; g++) {
            if (!group_states[g].active) continue;

            // Check if this K,V tile is needed by this group
            if (kv_tile_start >= group_states[g].kv_max_offset) {
                continue;  // This group doesn't need this K,V tile
            }

            // Load Q tile for this group
            load_q_tile(params.q_ptr_list[g], group_states[g].smem_q, ...);

            // Compute QK^T, softmax, attention output
            compute_attention_group(group_states[g], kv_tile, ...);

            // Merge LSE
            merge_lse(group_states[g].lse, lse_new, ...);
        }
    }

    // Write outputs for each group
    for (int g = 0; g < num_groups; g++) {
        write_output(params.out_ptr_list[g], group_states[g].smem_acc, ...);
        write_lse(params.softmax_lse_ptr_list[g], group_states[g].lse, ...);
    }
}
```

### Key Implementation Challenges

| Challenge | Solution |
|-----------|----------|
| **Multiple LSE accumulators** | Separate per-group LSE arrays in shared memory |
| **Thread block coordination** | Single TB per (seq, head) processes all groups |
| **KV endpoint enforcement** | Conditional processing based on `kv_endpoints[g][seq]` |
| **Gradient accumulation (backward)** | Atomic adds to dK, dV for overlapping regions |
| **Shared memory allocation** | Careful partitioning: K,V tiles + per-group Q/acc/LSE |

---

## 3. Integration with ring-flash-attention

### Modify execute_grouped_attention

```python
# File: ring_flash_attn/zigzag_llama3_flash_attn_varlen.py

def execute_grouped_attention(
    chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list,
    kv_buffer, kv_slices, nheads, head_dim, softmax_scale,
    dropout_p, causal, window_size, alibi_slopes, deterministic,
    world_size, cu_seqlens_k
):
    # Rearrange K,V from zigzag to contiguous (same as before)
    kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_k)

    # Prepare kv_endpoints tensor
    kv_endpoints = torch.zeros((len(chunk_q_list), num_seqs), dtype=torch.int32, device='cuda')
    for group_idx, (seq_ranges, _) in enumerate(kv_slices):
        for seq_idx, (start, end) in enumerate(seq_ranges):
            kv_endpoints[group_idx, seq_idx] = end

    # Extract cu_seqlens_k and max_seqlen for each group
    cu_seqlens_k_list = [kv_slices[g][1] for g in range(len(chunk_q_list))]
    max_seqlen_q_list = [(cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
                         for cu_seqlens in chunk_cu_seqlens_q_list]
    max_seqlen_k_list = [(cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
                         for cu_seqlens in cu_seqlens_k_list]

    # Call multi-group kernel (SINGLE KERNEL CALL instead of loop!)
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
        # Fallback to original dual-kernel approach
        return execute_grouped_attention_original(...)

    # Scatter outputs back to original positions (same as before)
    total_q = sum(q.shape[0] for q in chunk_q_list)
    out = torch.zeros((total_q, nheads, head_dim), dtype=out_chunks[0].dtype, device='cuda')
    for out_chunk, indices in zip(out_chunks, chunk_indices_list):
        out[indices] = out_chunk

    # ... (same LSE scattering logic)

    return out, lse, chunk_info
```

### Feature Flag for Safe Rollout

```python
# Add at module level
USE_MULTIGROUP_KERNEL = os.environ.get('RING_FLASH_ATTN_MULTIGROUP', '1') == '1'

def execute_grouped_attention(...):
    if USE_MULTIGROUP_KERNEL:
        try:
            return execute_grouped_attention_multikernel(...)
        except Exception as e:
            print(f"[WARNING] Multi-group kernel failed: {e}, falling back")
            return execute_grouped_attention_original(...)
    else:
        return execute_grouped_attention_original(...)
```

---

## 4. Testing Strategy

### Unit Tests

```python
# test/test_multigroup_flash_attn.py

@pytest.mark.parametrize("num_groups", [2, 3, 4])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_multigroup_forward_correctness(num_groups, head_dim):
    """Test multi-group kernel vs sequential single-group calls."""
    # Setup Q groups, K, V
    # Run reference: sequential _flash_attn_varlen_forward calls
    # Run test: _flash_attn_varlen_multigroup_forward
    # Compare outputs and LSE
    torch.testing.assert_close(test_outputs, ref_outputs, rtol=1e-3, atol=1e-3)

def test_overlapping_kv_gradient_accumulation():
    """Test gradients are correctly accumulated for overlapping K,V regions."""
    # Group 0 attends to K,V[0:256]
    # Group 1 attends to K,V[0:512]
    # Verify dk[0:256] == dk_from_group0 + dk_from_group1

def test_multigroup_gradients_autograd():
    """Use torch.autograd.gradcheck for gradient correctness."""
    torch.autograd.gradcheck(MultiGroupFlashAttnVarlenFunc.apply, inputs, ...)
```

### Performance Benchmarks

```python
# benchmarks/benchmark_multigroup.py

def benchmark_multigroup_vs_sequential():
    """Measure speedup of multi-group kernel vs sequential."""
    # Warmup + benchmark multi-group kernel
    # Warmup + benchmark sequential single-group calls
    speedup = sequential_time / multigroup_time
    print(f"Speedup: {speedup:.2f}x")
    assert speedup > 1.2, "Expected at least 1.2x speedup"
```

### Integration Tests

```python
# test/test_zigzag_llama3_multigroup.py

def test_zigzag_llama3_multigroup_equivalence():
    """Test zigzag_llama3 with multi-group kernel matches original."""
    # Run with USE_MULTIGROUP_KERNEL=1
    # Run with USE_MULTIGROUP_KERNEL=0
    # Compare outputs
    torch.testing.assert_close(out_multigroup, out_original, rtol=1e-3, atol=1e-3)
```

---

## 5. Performance Expectations

### Memory Bandwidth Reduction

**Current (dual-kernel)**:
- Group 0: loads K,V[0:X]
- Group 1: loads K,V[0:Y] where Y > X
- **Redundant loads**: K,V[0:X] loaded TWICE

**With multi-group kernel**:
- Each K,V tile loaded ONCE
- **Bandwidth reduction**: ~38% fewer K,V loads

### Speedup Estimates

| Scenario | K,V load % of kernel time | Expected Speedup |
|----------|---------------------------|------------------|
| Compute-bound | 40% | ~15% |
| Balanced | 60% | ~23% |
| Memory-bound | 80% | ~30% |

**Target**: >1.3x forward speedup, >1.2x backward speedup

### Efficiency Improvement

| Metric | Current | Target |
|--------|---------|--------|
| Forward-only efficiency | 82.3% | >88% |
| Forward+backward efficiency | 88.9% | >92% |
| K,V memory transactions | 4.5× total | 3.25× total |

---

## 6. Implementation Roadmap

### Phase 1: API Design + Mock (Weeks 1-2) - START HERE

**Goal**: Define API and create mock implementation for integration testing

**Tasks**:
1. Create `flash_attn_multigroup_interface.py` with function signatures
2. Implement mock that calls sequential kernels
3. Create `MultiGroupFlashAttnVarlenFunc` AutoGrad wrapper
4. Modify `execute_grouped_attention` with feature flag
5. Write unit tests using mock implementation
6. Test integration with zigzag_llama3 end-to-end

**Deliverables**:
- [ ] Python API file with complete function signatures
- [ ] Mock implementation passing all tests
- [ ] Integration working (no speedup yet, but correct)
- [ ] 10+ unit tests covering edge cases

**Success Criteria**: All zigzag_llama3 tests pass with mock multigroup API

### Phase 2: Forward CUDA Kernel (Weeks 3-6)

**Goal**: Implement optimized forward CUDA kernel

**Tasks**:
1. Create `flash_multigroup.h` with param structs
2. Implement `flash_fwd_multigroup_kernel.h`
3. Implement `flash_fwd_launch_template.h`
4. Create Python bindings
5. Write CUDA unit tests
6. Profile and optimize

**Deliverables**:
- [ ] Forward CUDA kernel implementation
- [ ] Python bindings working
- [ ] Correctness tests passing
- [ ] Performance benchmarks showing >1.3x speedup

**Success Criteria**: Forward kernel produces correct results with >1.3x speedup

### Phase 3: Backward CUDA Kernel (Weeks 7-9)

**Goal**: Implement backward with gradient accumulation

**Tasks**:
1. Extend `Flash_bwd_multigroup_params`
2. Implement `flash_bwd_multigroup_kernel.h`
3. Implement gradient accumulation (atomic adds)
4. Create Python bindings
5. Write gradient correctness tests
6. Profile backward performance

**Deliverables**:
- [ ] Backward CUDA kernel implementation
- [ ] Gradient correctness tests passing
- [ ] torch.autograd.gradcheck passing
- [ ] >1.2x backward speedup

**Success Criteria**: Gradients match reference within tolerance, >1.2x speedup

### Phase 4: Integration & Testing (Weeks 10-11)

**Goal**: Full integration and comprehensive testing

**Tasks**:
1. Replace mock with real kernels
2. Run full zigzag_llama3 test suite
3. Multi-GPU distributed tests
4. Performance regression tests
5. Documentation

**Deliverables**:
- [ ] All zigzag_llama3 tests passing
- [ ] Performance benchmarks documented
- [ ] User documentation complete

**Success Criteria**: Production-ready with documented performance improvements

### Phase 5: Optimization (Weeks 12-14)

**Goal**: Squeeze out maximum performance

**Tasks**:
1. Profile with Nsight Compute
2. Optimize shared memory layout
3. Tune thread block dimensions
4. Optimize atomic accumulation if needed
5. Add support for more data types

**Deliverables**:
- [ ] Optimized kernel variants
- [ ] Performance analysis report
- [ ] Tuning guide

**Success Criteria**: >90% forward efficiency, >95% backward efficiency

---

## 7. Quick Reference

### Key Files

| File | Purpose | Status |
|------|---------|--------|
| `flash_attn_multigroup_interface.py` | Python API | NEW |
| `flash_multigroup.h` | Param structs | NEW |
| `flash_fwd_multigroup_kernel.h` | Forward kernel | NEW |
| `flash_bwd_multigroup_kernel.h` | Backward kernel | NEW |
| `flash_api_multigroup.cpp` | Python bindings | NEW |
| `zigzag_llama3_flash_attn_varlen.py` | Integration | MODIFY |

### Build Commands

```bash
# Build flash-attention with multigroup support
cd /Users/petrpan26/work/flash-attention
python setup.py install

# Test
cd /Users/petrpan26/work/ring-flash-attention
pytest test/test_multigroup_flash_attn.py -v
```

### Debugging

```bash
# Enable verbose logging
export RING_FLASH_ATTN_DEBUG=1

# Use original dual-kernel approach
export RING_FLASH_ATTN_MULTIGROUP=0

# Profile with nsys
nsys profile -o multigroup_profile python script.py
```

---

## 8. Success Metrics

### Performance Targets

- [ ] Forward-only efficiency: 82.3% → >88%
- [ ] Forward+backward efficiency: 88.9% → >92%
- [ ] K,V load reduction: ~38%
- [ ] Forward speedup: >1.3x
- [ ] Backward speedup: >1.2x

### Correctness Criteria

- [ ] All unit tests pass (>95% coverage)
- [ ] Gradient correctness within 1e-3 (FP16) or 1e-4 (FP32)
- [ ] torch.autograd.gradcheck passes
- [ ] End-to-end zigzag_llama3 tests pass
- [ ] Multi-GPU distributed tests pass
- [ ] No memory leaks

### Code Quality

- [ ] Code review approved
- [ ] Documentation complete
- [ ] Benchmarks documented
- [ ] Feature flag for safe rollout
- [ ] Backward compatibility maintained

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| CUDA kernel too complex | Start with 2-group case, generalize later |
| Shared memory insufficient | Use smaller tile sizes or multi-pass |
| Atomic contention | Implement two-pass reduction fallback |
| API incompatibility | Maintain backward compatibility with feature flag |

---

## 10. Alternative Approaches (If Needed)

1. **Triton Implementation**: Prototype in Triton first (easier, may be slower)
2. **Kernel Fusion**: Modify Flash Attention kernel directly (harder, cleaner)
3. **Python-Level**: Optimize rearrangement only (easier, less impactful)

**Recommendation**: Proceed with proposed CUDA kernel approach

---

## Contact & Resources

- **Repository**: https://github.com/petrpan26/flash-attention (branch: feature/multigroup-varlen-zigzag)
- **Ring Flash Attention**: /Users/petrpan26/work/ring-flash-attention
- **Documentation**: See `EXPLORATION_KEY_FINDINGS.md` for codebase analysis

**Next Steps**: Start with Phase 1 - Python API + Mock Implementation

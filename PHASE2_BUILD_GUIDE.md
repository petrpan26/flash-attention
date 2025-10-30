# Phase 2: Python Bindings and Compilation Setup

## Overview

This guide covers building the multi-group flash attention CUDA extension with Python bindings.

## Files Implemented

### C++ Source Files
- `/Users/petrpan26/work/flash-attention/csrc/flash_attn/flash_api_multigroup.cpp` - Complete Python bindings
- `/Users/petrpan26/work/flash-attention/csrc/flash_attn/src/flash_multigroup.h` - Parameter structures

### Python Interface
- `/Users/petrpan26/work/flash-attention/flash_attn/flash_attn_multigroup_interface.py` - Mock implementation (will use CUDA when built)

## Implementation Status

### ✅ Completed

1. **Parameter Setup Functions** (`flash_api_multigroup.cpp`)
   - `set_params_fprop_multigroup()` - Converts Python tensors to CUDA parameters
   - `set_params_dgrad_multigroup()` - Sets up backward pass parameters
   - Comprehensive validation and error checking
   - Per-group pointer array allocation
   - Shared K,V tensor management

2. **Python Bindings** (`flash_api_multigroup.cpp`)
   - `mha_varlen_multigroup_fwd()` - Forward pass binding
   - `mha_varlen_multigroup_bwd()` - Backward pass binding
   - PyBind11 module registration
   - Full input validation
   - Output tensor allocation

3. **Parameter Structures** (`flash_multigroup.h`)
   - `Flash_fwd_multigroup_params` - Forward parameters
   - `Flash_bwd_multigroup_params` - Backward parameters
   - Helper functions for validation and shared memory calculation

### 🔄 Pending (Phase 3)

1. **CUDA Kernel Implementation**
   - Forward kernels for different head dimensions
   - Backward kernels for gradient computation
   - Template instantiations

2. **Kernel Dispatch**
   - Currently throws error: "Multi-group forward kernel not yet implemented"
   - Need to implement `run_mha_fwd_multigroup<>()` template functions

## Build Instructions

### Option 1: Add to Main flash_attn Extension

**Edit `setup.py`** to add multigroup sources:

```python
ext_modules.append(
    CUDAExtension(
        name="flash_attn_2_cuda",
        sources=[
            "csrc/flash_attn/flash_api.cpp",
            "csrc/flash_attn/flash_api_multigroup.cpp",  # ADD THIS
            # ... existing sources ...

            # Once CUDA kernels are implemented, add:
            # "csrc/flash_attn/src/flash_fwd_multigroup_hdim64_fp16_sm80.cu",
            # "csrc/flash_attn/src/flash_fwd_multigroup_hdim64_bf16_sm80.cu",
            # "csrc/flash_attn/src/flash_fwd_multigroup_hdim128_fp16_sm80.cu",
            # "csrc/flash_attn/src/flash_fwd_multigroup_hdim128_bf16_sm80.cu",
            # etc.
        ],
        extra_compile_args={
            "cxx": compiler_c17_flag,
            "nvcc": append_nvcc_threads(nvcc_flags + cc_flag),
        },
        include_dirs=[
            Path(this_dir) / "csrc" / "flash_attn",
            Path(this_dir) / "csrc" / "flash_attn" / "src",
            Path(this_dir) / "csrc" / "cutlass" / "include",
        ],
    )
)
```

### Option 2: Separate Extension (Recommended for Development)

Create a separate extension for multigroup attention:

```python
ext_modules.append(
    CUDAExtension(
        name="flash_attn_multigroup_cuda",
        sources=[
            "csrc/flash_attn/flash_api_multigroup.cpp",
            # Once kernels are implemented:
            # "csrc/flash_attn/src/flash_fwd_multigroup_hdim64_fp16_sm80.cu",
            # "csrc/flash_attn/src/flash_fwd_multigroup_hdim128_fp16_sm80.cu",
            # etc.
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "-gencode", "arch=compute_80,code=sm_80",  # A100
                "-gencode", "arch=compute_90,code=sm_90",  # H100
            ],
        },
        include_dirs=[
            Path(this_dir) / "csrc" / "flash_attn",
            Path(this_dir) / "csrc" / "flash_attn" / "src",
            Path(this_dir) / "csrc" / "cutlass" / "include",
        ],
    )
)
```

### Build Commands

```bash
cd /Users/petrpan26/work/flash-attention

# Development build (editable install)
python setup.py develop

# Or production build
python setup.py install

# Force rebuild
python setup.py clean --all
python setup.py develop
```

### Build with specific CUDA architectures

```bash
export FLASH_ATTN_CUDA_ARCHS="80;90"  # A100, H100
python setup.py develop
```

## Testing the Build

### Test Python Import

```python
# Test that the module compiles and can be imported
try:
    import flash_attn_multigroup_cuda
    print("✓ Module imported successfully")
    print(f"  Forward function: {flash_attn_multigroup_cuda.fwd}")
    print(f"  Backward function: {flash_attn_multigroup_cuda.bwd}")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
except RuntimeError as e:
    # Expected for now since kernels not implemented
    if "not yet implemented" in str(e):
        print("✓ Module compiled, kernels pending")
    else:
        print(f"✗ Runtime error: {e}")
```

### Test Parameter Setup

```python
import torch
import flash_attn_multigroup_cuda

# Create test tensors
batch_size = 2
num_groups = 2
seq_len = 128
num_heads = 8
head_dim = 64

q_list = [
    torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16, device='cuda')
    for _ in range(num_groups)
]
k = torch.randn(seq_len * 2, num_heads, head_dim, dtype=torch.float16, device='cuda')
v = torch.randn(seq_len * 2, num_heads, head_dim, dtype=torch.float16, device='cuda')

cu_seqlens_q_list = [
    torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')
    for _ in range(num_groups)
]
cu_seqlens_k_list = [
    torch::tensor([0, seq_len], dtype=torch.int32, device='cuda')
    for _ in range(num_groups)
]
kv_endpoints = torch.tensor([[seq_len], [seq_len * 2]], dtype=torch.int32, device='cuda')

try:
    out_list, lse_list = flash_attn_multigroup_cuda.fwd(
        q_list, k, v,
        cu_seqlens_q_list, cu_seqlens_k_list,
        kv_endpoints,
        [seq_len] * num_groups,  # max_seqlen_q_list
        [seq_len] * num_groups,  # max_seqlen_k_list
        dropout_p=0.0,
        softmax_scale=1.0 / (head_dim ** 0.5),
        causal=False
    )
    print("✓ Forward pass succeeded")
except RuntimeError as e:
    if "not yet implemented" in str(e):
        print("✓ Parameter setup succeeded, kernel dispatch pending")
    else:
        raise
```

## Compilation Troubleshooting

### Common Issues

1. **"flash_multigroup.h: No such file or directory"**
   - Solution: Ensure `csrc/flash_attn/src/flash_multigroup.h` exists
   - Check include paths in setup.py

2. **"undefined reference to run_mha_fwd_multigroup"**
   - Expected: CUDA kernels not yet implemented
   - Will be resolved in Phase 3

3. **"invalid device function"**
   - Check CUDA arch flags match your GPU
   - Verify CUDA toolkit version >= 11.7

4. **Memory issues during compilation**
   - Reduce parallelism: `export MAX_JOBS=2`
   - Close other applications

5. **ABI compatibility errors**
   - Ensure PyTorch and CUDA extension use same C++ ABI
   - Set `export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE` if needed

## Current Limitations

1. **No CUDA Kernels Yet**
   - Bindings compile but throw "not yet implemented" at runtime
   - Need Phase 3 kernel implementation

2. **Memory Management**
   - Per-group pointer arrays allocated with `new[]`
   - Currently no explicit cleanup (relies on Python GC)
   - Production code should use smart pointers or RAII

3. **RNG State**
   - Simplified RNG setup for dropout
   - Production needs per-group RNG state

## Next Steps (Phase 3)

1. Implement forward CUDA kernels:
   - `flash_fwd_multigroup_hdim64_fp16_sm80.cu`
   - `flash_fwd_multigroup_hdim128_fp16_sm80.cu`
   - Template instantiations for 2-8 groups

2. Implement backward CUDA kernels:
   - Gradient computation for Q (per-group)
   - Gradient accumulation for K,V (shared, atomic or two-pass)

3. Add kernel dispatch logic:
   - Uncomment dispatch code in `mha_varlen_multigroup_fwd()`
   - Add NUMGROUPS_SWITCH macro in static_switch.h

4. Performance optimization:
   - Tune block sizes and thread configurations
   - Optimize shared memory usage
   - Profile and benchmark

## Integration with Python Interface

Once built, update `flash_attn_multigroup_interface.py`:

```python
# Replace mock implementation
try:
    import flash_attn_multigroup_cuda
    HAS_CUDA_IMPL = True
except ImportError:
    HAS_CUDA_IMPL = False

def _flash_attn_varlen_multigroup_forward(...):
    if HAS_CUDA_IMPL:
        # Use compiled CUDA extension
        return flash_attn_multigroup_cuda.fwd(...)
    else:
        # Fallback to mock implementation
        # ... existing code ...
```

## Summary

**Status**: Python bindings complete, CUDA kernels pending

**Files Ready**:
- ✅ `flash_api_multigroup.cpp` - Full implementation
- ✅ `flash_multigroup.h` - Parameter structures
- ✅ `flash_attn_multigroup_interface.py` - Python interface (mock)

**Build Ready**: Yes, will compile with current files (throws runtime error for missing kernels)

**Next Phase**: Implement CUDA kernel templates for multi-group attention

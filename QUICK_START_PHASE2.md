# Quick Start: Phase 2 Python Bindings

## TL;DR

Build multi-group flash attention Python bindings in 5 minutes.

## Prerequisites

```bash
# Check CUDA
nvcc --version  # Need ≥ 11.7

# Check PyTorch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Build (3 commands)

```bash
cd /Users/petrpan26/work/flash-attention

# 1. Apply setup.py patch
patch -p1 < setup_multigroup.patch

# 2. Build extension
python setup.py develop

# 3. Test
python test_phase2_bindings.py
```

## Expected Output

```
Test 1: Module Import                     ✓ PASS
Test 2: Forward Parameter Setup           ✓ PASS
Test 3: Input Validation                  ✓ PASS
Test 4: Backward Parameter Setup          ✓ PASS
Test 5: Python Interface (Mock)           ✓ PASS

Results: 5 passed, 0 failed, 0 skipped
```

## Verify Import

```python
import flash_attn_multigroup_cuda
print("✓ Module loaded:", flash_attn_multigroup_cuda)
```

## Try Forward Pass

```python
import torch
import flash_attn_multigroup_cuda

# Setup
q_list = [torch.randn(64, 8, 64, dtype=torch.float16, device='cuda') for _ in range(2)]
k = torch.randn(128, 8, 64, dtype=torch.float16, device='cuda')
v = torch.randn(128, 8, 64, dtype=torch.float16, device='cuda')
cu_seqlens_q_list = [torch.tensor([0, 64], dtype=torch.int32, device='cuda')] * 2
cu_seqlens_k_list = [torch.tensor([0, 128], dtype=torch.int32, device='cuda')] * 2
kv_endpoints = torch.tensor([[64], [128]], dtype=torch.int32, device='cuda')

# Call (will throw "not yet implemented" - expected!)
try:
    out_list, lse_list = flash_attn_multigroup_cuda.fwd(
        q_list, k, v,
        cu_seqlens_q_list, cu_seqlens_k_list, kv_endpoints,
        [64, 64], [128, 128],
        dropout_p=0.0,
        softmax_scale=0.125,
    )
except RuntimeError as e:
    if "not yet implemented" in str(e):
        print("✓ Parameter setup works, kernels pending")
```

## What Works Now

✅ Python bindings compile
✅ Module imports
✅ Parameter conversion
✅ Input validation
✅ Output allocation
✅ Mock Python implementation

## What's Pending

⏳ CUDA kernels (Phase 3)
⏳ Kernel dispatch
⏳ Performance optimization

## Troubleshooting

### Build fails with "nvcc not found"
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

### Import fails
```bash
# Check installation
pip show flash-attn

# Rebuild
python setup.py clean --all
python setup.py develop
```

### Tests fail
```bash
# Check GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Run with verbose
python test_phase2_bindings.py -v
```

## Next Steps

1. **Phase 3**: Implement CUDA kernels
2. **Phase 4**: Add comprehensive tests
3. **Phase 5**: Integrate with zigzag_llama3

## Documentation

- **Build Guide**: `PHASE2_BUILD_GUIDE.md` - Detailed build instructions
- **Summary**: `PHASE2_SUMMARY.md` - Complete implementation details
- **Integration**: `PHASE2_INTEGRATION_STATUS.md` - Integration status
- **Source**: `csrc/flash_attn/flash_api_multigroup.cpp` - Implementation

## Files Added

```
csrc/flash_attn/
  flash_api_multigroup.cpp              529 lines - Python bindings
  src/flash_multigroup.h                353 lines - Parameter structures

flash_attn/
  flash_attn_multigroup_interface.py    308 lines - Python interface

Documentation:
  PHASE2_BUILD_GUIDE.md
  PHASE2_SUMMARY.md
  PHASE2_INTEGRATION_STATUS.md
  QUICK_START_PHASE2.md                 This file

Build:
  setup_multigroup.patch                Patch for setup.py
  test_phase2_bindings.py               Test suite
```

## Help

For issues, check:
1. `PHASE2_BUILD_GUIDE.md` - Troubleshooting section
2. Build log: Look for compilation errors
3. Test output: Shows which component failed

## Status

**Phase 2: COMPLETE ✅**

Ready for Phase 3 CUDA kernel implementation.

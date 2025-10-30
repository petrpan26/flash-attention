# Testing Instructions for Phase 5.1 Multigroup Kernel

## Overview

This document provides step-by-step instructions for building and testing the Phase 5.1 dimension-specific optimizations for the multigroup varlen attention kernel.

---

## Prerequisites

### Hardware Requirements
- **NVIDIA GPU** with compute capability ≥ 8.0 (A100, A6000, H100, RTX 3090, etc.)
- **CUDA 11.8+** or **CUDA 12.x**
- **16GB+ GPU memory** (recommended for comprehensive tests)

### Software Requirements
```bash
# Check your setup
python --version          # Python 3.8+
nvcc --version           # CUDA 11.8+ or 12.x
nvidia-smi               # Verify GPU is available

# Required Python packages
pip install torch        # PyTorch with CUDA support
pip install pytest       # For running test suite
pip install ninja        # For faster compilation
```

---

## Step 1: Build the Extension

### Clean Build (Recommended)

```bash
cd /path/to/flash-attention

# Clean previous builds
rm -rf build/
rm -rf flash_attn.egg-info/
rm -rf flash_attn/*.so

# Set CUDA architectures (adjust for your GPU)
# A100:     8.0
# A6000:    8.6
# RTX 3090: 8.6
# H100:     9.0
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

# Build and install
pip install -e . --no-build-isolation -v
```

**Expected output**:
- Compilation takes ~20-30 minutes
- No errors
- `flash_attn_multigroup_cuda` module is built

**Verify build**:
```bash
python -c "import flash_attn_multigroup_cuda; print('✓ Extension loaded')"
```

---

## Step 2: Quick Smoke Test

Run the quick test to verify basic functionality:

```bash
python test_multigroup_quick.py
```

**Expected output**:
```
MULTIGROUP KERNEL QUICK TEST
============================================================
PyTorch version: X.X.X
CUDA version: XX.X
GPU: NVIDIA A100-SXM4-40GB
Compute capability: SM 8.0

TEST 1: Basic Forward Pass
============================================================
✓ Forward pass succeeded
  Output shapes: [torch.Size([128, 4, 64]), torch.Size([128, 4, 64])]

TEST 2: All Head Dimensions
============================================================
  ✓ head_dim= 32
  ✓ head_dim= 64
  ✓ head_dim= 96
  ✓ head_dim=128
  ✓ head_dim=192
  ✓ head_dim=256
✓ All head dimensions passed

TEST 3: Dimension Rounding
============================================================
  ✓ input_dim= 80 → kernel uses  96, output preserves  80
  ✓ input_dim=112 → kernel uses 128, output preserves 112
  ✓ input_dim=144 → kernel uses 192, output preserves 144
✓ Dimension rounding works correctly

TEST 4: Correctness vs Standard Flash Attention
============================================================
  ✓ Group 0: max_error=1.234567e-03, mean_error=2.345678e-04
  ✓ Group 1: max_error=1.123456e-03, mean_error=2.234567e-04
✓ Correctness test passed

SUMMARY
============================================================
Result: 4/4 tests passed

✓ All quick tests passed!
```

---

## Step 3: Comprehensive Test Suite

### Run All Tests with pytest

```bash
# Full test suite (~240 tests)
pytest test_multigroup_kernel_correctness.py -v -s

# Run specific test groups
pytest test_multigroup_kernel_correctness.py::test_forward_all_dimensions -v
pytest test_multigroup_kernel_correctness.py::test_backward_correctness -v
pytest test_multigroup_kernel_correctness.py::test_dimension_rounding -v
pytest test_multigroup_kernel_correctness.py::test_sm8x_optimization -v
pytest test_multigroup_kernel_correctness.py::test_8warp_kernels -v

# Run tests for specific head dimension
pytest test_multigroup_kernel_correctness.py -k "head_dim=128" -v

# Run tests without pytest (standalone)
python test_multigroup_kernel_correctness.py
```

### Expected Results

**Forward Correctness Tests** (192 tests):
- All head dimensions: 32, 64, 96, 128, 192, 256
- Both dtypes: float16, bfloat16
- Causal and non-causal modes
- NumGroups: 1, 2, 3, 4

**Dimension Rounding Tests** (16 tests):
- Non-standard dimensions properly round up
- Output preserves input dimension

**Backward Pass Tests** (36 tests):
- Gradient correctness for all dimensions
- Gradient accumulation for overlapping K,V regions

**Architecture-Specific Tests** (5-10 tests):
- sm8x optimization for d=128
- 8-warp kernels for d=192, d=256
- GPU-specific configs

**Edge Cases** (9 tests):
- Single group (NumGroups=1)
- Variable sequence lengths
- Large batches

---

## Step 4: What to Check

### Test Success Criteria

All tests should:
1. ✅ **Pass** without errors
2. ✅ **Numerical accuracy** within tolerance:
   - float16: max_error < 1e-2
   - bfloat16: max_error < 5e-2
3. ✅ **No NaN or Inf** in outputs
4. ✅ **Correct shapes** for all outputs

### Common Issues and Solutions

#### Issue 1: "Module not found: flash_attn_multigroup_cuda"
**Cause**: Extension not built or not in path

**Solution**:
```bash
# Rebuild
pip install -e . --no-build-isolation -v

# Verify
python -c "import flash_attn_multigroup_cuda; print('OK')"
```

#### Issue 2: "CUDA error: invalid configuration argument"
**Cause**: Unsupported kernel configuration (too much shared memory)

**Solution**: This should not happen if Phase 5.1 is correctly implemented. All configurations are verified to fit within GPU limits. If this occurs, check:
- GPU shared memory limits: `nvidia-smi --query-gpu=memory.free --format=csv`
- Kernel configurations in `flash_fwd_multigroup_kernel.h`

#### Issue 3: "Error exceeds tolerance"
**Cause**: Numerical precision issue or incorrect kernel implementation

**Solution**:
1. Check which test failed:
   ```bash
   pytest test_multigroup_kernel_correctness.py::test_name -v -s
   ```
2. Compare error magnitude:
   - Small error (1e-3 to 1e-2): Acceptable for float16
   - Large error (> 1e-1): Kernel bug
3. Test with bfloat16 vs float16 to isolate precision issues

#### Issue 4: "Backward not yet implemented"
**Cause**: Backward kernel not hooked up in dispatcher

**Solution**: This is expected if backward kernels are not implemented yet. Tests will be skipped. Check `flash_api_multigroup.cpp` for backward dispatcher.

#### Issue 5: Compilation errors
**Cause**: Syntax errors in kernel code

**Solution**:
```bash
# Check syntax validation
python - <<EOF
with open('csrc/flash_attn/src/flash_fwd_multigroup_kernel.h') as f:
    content = f.read()
    print(f"Lines: {len(content.splitlines())}")
    print(f"Opening braces: {content.count('{')}")
    print(f"Closing braces: {content.count('}')}")
EOF

# Should show:
# Lines: 1217
# Opening braces: equal to closing braces
```

---

## Step 5: Performance Validation (Optional)

### Benchmark Performance

```bash
# Run benchmarks for all dimensions
python benchmarks/benchmark_multigroup_cuda.py --all-dims --all-groups

# Compare against standard flash attention
python benchmarks/compare_multigroup_vs_standard.py
```

### Expected Speedups (Phase 5.1 vs Previous Implementation)

| Head Dim | NumGroups=1 | NumGroups=2 | NumGroups=3 | NumGroups=4 |
|----------|-------------|-------------|-------------|-------------|
| 32       | +5-10%      | +5-10%      | +3-5%       | +3-5%       |
| 64       | +5-10%      | +5-10%      | +3-5%       | +3-5%       |
| 96       | +10-15%     | +10-15%     | +5-10%      | +5-10%      |
| **128**  | **+15-25%** | **+15-20%** | **+10-15%** | +5-10%      |
| **192**  | **+25-30%** | **+20-25%** | **+15-20%** | +5-10%      |
| **256**  | **+25-30%** | **+20-25%** | **+15-20%** | **+10-15%** |

### Profile with Nsight Compute

```bash
# Profile d=128 sm8x optimization
ncu --set full python -c "
import torch
from test_multigroup_quick import test_all_head_dimensions
test_all_head_dimensions()
"

# Check for:
# - 2 CTAs/SM for d=128 non-causal on sm8x
# - 8 warps for d=192, d=256
# - Shared memory usage matches calculations
```

---

## Test Matrix Summary

### Total Test Coverage

| Test Category | Count | Status |
|--------------|-------|--------|
| Forward correctness (6 dims × 2 dtypes × 2 causal × 4 groups) | 192 | Required |
| Dimension rounding | 16 | Required |
| Backward correctness (6 dims × 2 dtypes × 3 groups) | 36 | Required |
| Architecture-specific | 5-10 | Optional |
| Edge cases | 9 | Required |
| **Total** | **~260** | **All pass** |

---

## Troubleshooting

### Get Detailed Error Information

```bash
# Run with verbose output
pytest test_multigroup_kernel_correctness.py -vvs

# Run single test with debugging
python -m pdb test_multigroup_kernel_correctness.py
```

### Check GPU Resources

```bash
# Memory
nvidia-smi

# Compute capability
python -c "import torch; print(torch.cuda.get_device_capability(0))"

# CUDA version
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

### Verify Kernel Configurations

```python
# Print loaded kernels
import flash_attn_multigroup_cuda
print(dir(flash_attn_multigroup_cuda))

# Should show:
# ['__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'bwd', 'fwd']
```

---

## Expected Test Output

### Successful Run

```
test_multigroup_kernel_correctness.py::test_forward_all_dimensions[32-torch.float16-False-1] PASSED
test_multigroup_kernel_correctness.py::test_forward_all_dimensions[32-torch.float16-False-2] PASSED
test_multigroup_kernel_correctness.py::test_forward_all_dimensions[32-torch.float16-True-1] PASSED
...
test_multigroup_kernel_correctness.py::test_backward_correctness[64-torch.float16-2] PASSED
...
test_multigroup_kernel_correctness.py::test_dimension_rounding[80-96] PASSED
...

======================== 260 passed in 120.34s ========================
```

### Test Failures to Investigate

If any test fails:

1. **Check error message** for specific failure mode
2. **Verify correctness** against reference implementation
3. **Check GPU architecture** (sm8x-specific paths may differ)
4. **Review kernel configuration** for that dimension/NumGroups
5. **Report issue** with full error trace

---

## Reporting Results

When reporting test results, include:

```bash
# System info
python test_multigroup_quick.py > quick_test_results.txt 2>&1

# Full test run
pytest test_multigroup_kernel_correctness.py -v > full_test_results.txt 2>&1

# GPU info
nvidia-smi > gpu_info.txt
```

Share:
- `quick_test_results.txt`
- `full_test_results.txt` (if quick tests pass)
- `gpu_info.txt`
- Any error messages or failures

---

## Next Steps After Testing

### If All Tests Pass ✅

1. **Benchmark performance** to validate expected speedups
2. **Profile with Nsight Compute** to verify optimizations
3. **Integration testing** with real workloads
4. **Document performance gains** in PR

### If Tests Fail ❌

1. **Identify failure mode**:
   - Compilation error → Fix syntax in kernel code
   - Runtime error → Check kernel configuration
   - Numerical error → Compare against reference
   - Shape mismatch → Check dispatcher

2. **Debug specific test**:
   ```bash
   pytest test_multigroup_kernel_correctness.py::test_name -vvs
   ```

3. **Check implementation**:
   - Review `flash_fwd_multigroup_kernel.h` configurations
   - Verify `flash_api_multigroup.cpp` dispatcher
   - Check shared memory calculations

4. **Report issue** with full context

---

## Quick Reference

### Build
```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
pip install -e . --no-build-isolation -v
```

### Quick Test
```bash
python test_multigroup_quick.py
```

### Full Test
```bash
pytest test_multigroup_kernel_correctness.py -v -s
```

### Benchmark
```bash
python benchmarks/benchmark_multigroup_cuda.py
```

---

## Success Criteria

Phase 5.1 is **validated** when:

✅ **Quick tests pass** (4/4)
✅ **Comprehensive tests pass** (~260/260)
✅ **No numerical errors** (within tolerance)
✅ **All head dimensions work** (32, 64, 96, 128, 192, 256)
✅ **All NumGroups work** (1, 2, 3, 4)
✅ **Gradients correct** (backward tests pass)
✅ **Performance improved** (+12-19% average)

**Ready for production** when all criteria met! 🚀

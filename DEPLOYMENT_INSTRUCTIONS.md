# Flash Attention Grouped Features - Deployment Instructions

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Deployment Steps](#deployment-steps)
4. [Validation Procedures](#validation-procedures)
5. [Troubleshooting](#troubleshooting)
6. [Rollback Procedure](#rollback-procedure)

---

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with Compute Capability ≥ 8.0 (Ampere or newer)
  - Tested on: A100, A10G, H100
  - Minimum VRAM: 16 GB (24 GB+ recommended for large models)

- **CPU**: x86_64 architecture
  - Recommended: 16+ cores for faster compilation
  - Minimum RAM: 32 GB (64 GB+ recommended)

### Software Requirements
- **CUDA**: 11.8 or 12.x
  - Driver version: ≥ 520.61.05 (for CUDA 12.x) or ≥ 450.80.02 (for CUDA 11.8)

- **Python**: 3.8, 3.9, 3.10, or 3.11

- **PyTorch**: 2.0.0 or newer
  - Must be compiled with CUDA support
  - Verify: `python -c "import torch; print(torch.cuda.is_available())"`

- **Build Tools**:
  - GCC/G++: 9.x or newer
  - CMake: 3.18 or newer
  - ninja-build (recommended for faster builds)

### Python Dependencies
```bash
pip install torch>=2.0.0
pip install numpy
pip install einops
pip install packaging
```

For testing:
```bash
pip install pytest
pip install matplotlib  # For benchmark plots
```

---

## Pre-Deployment Checklist

Before deploying, verify the following:

- [ ] GPU is accessible and supported (SM ≥ 8.0)
- [ ] CUDA toolkit is installed and `nvcc` is in PATH
- [ ] PyTorch with CUDA support is installed
- [ ] Sufficient disk space (≥ 10 GB for compilation artifacts)
- [ ] Sufficient GPU memory for testing (≥ 16 GB)
- [ ] All Python dependencies are installed
- [ ] No conflicting Flash Attention installations
- [ ] Backup of existing environment (if upgrading)

**Check GPU compatibility:**
```bash
nvidia-smi
python -c "import torch; print(f'CUDA Capability: {torch.cuda.get_device_capability()}')"
```

Expected output: `(8, 0)` or higher (e.g., `(8, 6)` for A100, `(9, 0)` for H100)

---

## Deployment Steps

### Step 1: Obtain the Source Code

**Option A: From Git Repository**
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout feature/grouped-flash-attention  # Or the appropriate branch
```

**Option B: From Release Archive**
```bash
# Extract the provided archive
tar -xzf flash-attention-grouped.tar.gz
cd flash-attention
```

### Step 2: Set Environment Variables

```bash
# Force rebuild (important for first-time installation)
export FLASH_ATTENTION_FORCE_BUILD=TRUE

# Optional: Limit parallel jobs to avoid OOM during compilation
export MAX_JOBS=8  # Adjust based on available RAM

# Optional: Use ninja for faster builds
export CMAKE_BUILD_PARALLEL_LEVEL=8
```

### Step 3: Clean Previous Installations

```bash
# Remove any existing Flash Attention installations
pip uninstall flash-attn -y

# Clean build artifacts
python setup.py clean --all
rm -rf build/ dist/ *.egg-info
```

### Step 4: Compile and Install

**Standard Installation:**
```bash
python setup.py install
```

**For Development (editable install):**
```bash
pip install -e .
```

**Expected Compilation Time:**
- With ninja + 8 cores: 10-15 minutes
- Without ninja: 20-30 minutes
- First build is slower; subsequent builds are faster

**Monitor compilation:**
```bash
# Watch for errors in real-time
python setup.py install 2>&1 | tee build.log
```

### Step 5: Verify Installation

```bash
# Check basic import
python -c "import flash_attn; print(f'Version: {flash_attn.__version__}')"

# Check grouped attention functions
python -c "from flash_attn.flash_attn_grouped import _flash_attn_varlen_forward_grouped, _flash_attn_varlen_backward_grouped; print('Grouped attention available')"

# Check available kernels
python -c "import flash_attn_2_cuda; print('CUDA kernels loaded successfully')"
```

---

## Validation Procedures

### Automated Validation

Run the complete validation suite:

```bash
./validate_all.sh
```

This script:
1. Checks GPU availability
2. Verifies CUDA setup
3. Compiles the library
4. Runs all unit tests
5. Runs performance benchmarks (optional)

**Skip benchmarks for faster validation:**
```bash
./validate_all.sh --skip-benchmarks
```

### Manual Validation

**1. Quick Smoke Test:**
```bash
python -c "
import torch
from flash_attn.flash_attn_grouped import _flash_attn_varlen_forward_grouped

# Create simple test inputs
q = torch.randn(128, 8, 64, device='cuda', dtype=torch.float16)
k = torch.randn(256, 8, 64, device='cuda', dtype=torch.float16)
v = torch.randn(256, 8, 64, device='cuda', dtype=torch.float16)

cu_seqlens_q = torch.tensor([0, 128], dtype=torch.int32, device='cuda')
cu_seqlens_k = torch.tensor([0, 256], dtype=torch.int32, device='cuda')

# Run grouped attention with 2 groups
out = _flash_attn_varlen_forward_grouped(
    q_list=[q, q],
    k=k, v=v,
    cu_seqlens_q_list=[cu_seqlens_q, cu_seqlens_q],
    cu_seqlens_k_list=[cu_seqlens_k, cu_seqlens_k],
    max_seqlen_q_list=[128, 128],
    max_seqlen_k_list=[256, 256],
    dropout_p=0.0,
    softmax_scale=0.125,
    causal=False
)
print('Smoke test passed!')
"
```

**2. Run Specific Test Suites:**
```bash
# Test backward pass
pytest tests/test_grouped_backward.py -v

# Test all head dimensions
pytest tests/test_grouped_all_configs.py -v

# Test SMEM optimizations
pytest tests/test_share_q_k_smem.py -v
pytest tests/test_ngroup_smem.py -v
```

**3. Run Integration Tests:**
```bash
pytest tests/test_integration_full.py -v
```

### Performance Validation

Run benchmarks to verify performance meets expectations:

```bash
# N-group SMEM performance
python benchmarks/benchmark_ngroup_smem.py

# Comprehensive benchmark suite
python benchmarks/benchmark_comprehensive.py
```

**Expected Performance Improvements:**
- 2-group SMEM: ~1.5-1.8x faster than separate calls
- 3-group SMEM: ~2.0-2.5x faster than separate calls
- 4-group SMEM: ~2.5-3.0x faster than separate calls

---

## Troubleshooting

### Compilation Issues

**Problem: `nvcc not found`**
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Problem: Compilation runs out of memory**
```bash
# Reduce parallel jobs
export MAX_JOBS=4
python setup.py install
```

**Problem: `undefined reference to` errors**
```bash
# Ensure CUDA toolkit matches PyTorch CUDA version
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
nvcc --version  # Should match major version
```

**Problem: Kernel not found for specific head dimension**
```bash
# Check if all head dimension kernels were compiled
ls csrc/flash_attn/src/flash_fwd_grouped_hdim*.cu

# Expected files: hdim32, hdim64, hdim96, hdim128, hdim192, hdim256
# For both fp16 and bf16
```

### Runtime Issues

**Problem: `CUDA error: invalid configuration argument`**
- Cause: GPU doesn't support required features or kernel launch parameters are invalid
- Solution: Verify GPU compute capability ≥ 8.0
  ```bash
  python -c "import torch; print(torch.cuda.get_device_capability())"
  ```

**Problem: `RuntimeError: CUDA out of memory`**
- Reduce batch size or sequence length in tests
- Clear GPU cache: `torch.cuda.empty_cache()`
- Check GPU memory: `nvidia-smi`

**Problem: Incorrect results (large numerical errors)**
- Verify using FP16 or BF16 (not FP32)
- Check tolerance settings in tests
- For BF16, expect slightly larger errors (~5e-2 vs ~1e-2 for FP16)

**Problem: Import errors**
```python
# Check Python can find the module
import sys
import flash_attn
print(flash_attn.__file__)  # Should point to installed location

# Check CUDA kernels loaded
import flash_attn_2_cuda
print(dir(flash_attn_2_cuda))  # Should include grouped kernel functions
```

### Performance Issues

**Problem: Grouped kernel slower than expected**
- Verify using correct GPU (A100/A10/H100)
- Check GPU is not throttled: `nvidia-smi dmon`
- Ensure GPU is not shared with other processes
- Try larger batch sizes for better hardware utilization

**Problem: Compilation very slow**
```bash
# Use ninja for faster builds
pip install ninja
export CMAKE_BUILD_PARALLEL_LEVEL=8

# Or reduce number of kernels compiled (not recommended for production)
# Edit setup.py to comment out some head dimensions
```

---

## Rollback Procedure

If issues occur, rollback to the stable Flash Attention version:

### Step 1: Uninstall Current Version
```bash
pip uninstall flash-attn -y
```

### Step 2: Clean Build Artifacts
```bash
cd flash-attention
python setup.py clean --all
rm -rf build/ dist/ *.egg-info
```

### Step 3: Checkout Stable Version
```bash
git checkout main  # Or stable branch
# Or: git checkout v2.5.0  # Specific stable release
```

### Step 4: Reinstall Stable Version
```bash
export FLASH_ATTENTION_FORCE_BUILD=TRUE
python setup.py install
```

### Step 5: Verify Rollback
```bash
python -c "import flash_attn; print(f'Version: {flash_attn.__version__}')"

# Should NOT have grouped functions
python -c "from flash_attn.flash_attn_grouped import _flash_attn_varlen_forward_grouped" 2>&1 | grep -q "ModuleNotFoundError" && echo "Rollback successful"
```

### Preserve Logs for Debugging
```bash
# Save all logs before rollback
mkdir -p deployment_logs_$(date +%Y%m%d_%H%M%S)
cp build*.log test*.log benchmark*.log deployment_logs_*/
```

---

## Post-Deployment Verification

After successful deployment:

1. **Run validation suite:**
   ```bash
   ./validate_all.sh
   ```

2. **Check test coverage:**
   ```bash
   pytest tests/test_grouped*.py --cov=flash_attn --cov-report=html
   ```

3. **Benchmark performance:**
   ```bash
   python benchmarks/benchmark_comprehensive.py
   ```

4. **Document deployment:**
   - Record GPU model and driver version
   - Record CUDA and PyTorch versions
   - Save validation logs
   - Note any deviations from standard procedure

5. **Monitor first production runs:**
   - Watch for CUDA errors
   - Monitor GPU memory usage
   - Verify numerical accuracy on real workloads
   - Compare performance with baseline

---

## Support and Resources

**Documentation:**
- `TESTING_CHECKLIST.md` - Comprehensive testing procedures
- `FINAL_STATUS.md` - Feature status and known limitations
- `GPU_SETUP_GUIDE.md` - GPU-specific setup instructions
- `PERFORMANCE_GUIDE.md` - Performance optimization tips

**Logs to Collect for Issues:**
- `build_validation.log` - Compilation log
- `test_*.log` - Test execution logs
- `nvidia-smi` output
- PyTorch and CUDA version info
- GPU compute capability

**Common Questions:**

Q: Can I use this with PyTorch < 2.0?
A: No, PyTorch 2.0+ is required for proper autograd support.

Q: Do I need to recompile when changing PyTorch versions?
A: Yes, always recompile after changing PyTorch or CUDA versions.

Q: Can I use this on V100 (SM 7.0)?
A: No, Ampere (SM 8.0+) is required for the grouped kernels.

Q: How do I know which head dimensions are supported?
A: Check `csrc/flash_attn/src/flash_fwd_grouped_hdim*.cu` for compiled kernels.
   Supported: 32, 64, 96, 128, 192, 256

Q: Can I mix different head dimensions in one grouped call?
A: No, all groups must use the same head dimension.

---

*Last Updated: 2024-10-19*
*Document Version: 1.0*

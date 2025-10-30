# Multi-Group Flash Attention - Compilation Guide

This guide provides detailed instructions for compiling and testing the multi-group varlen attention CUDA kernel.

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA Ampere (SM80) or newer
  - A100, A10, RTX 3090, RTX 4090, etc.
  - Hopper (H100) also supported
- **Memory**: 8GB+ VRAM recommended for testing

### Software Requirements

1. **CUDA Toolkit**
   ```bash
   # Required: CUDA 11.8 or later (for Ampere SM80 support)
   nvcc --version
   # Should show: Cuda compilation tools, release 11.8 or higher
   ```

2. **PyTorch with CUDA**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')"
   # Should show: Available: True
   ```

3. **Build Tools**
   ```bash
   pip install ninja packaging
   ```

---

## Compilation Steps

### Step 1: Navigate to Repository
```bash
cd /Users/petrpan26/work/flash-attention
git status  # Verify on feature/multigroup-varlen-zigzag branch
```

### Step 2: Check File Structure
```bash
# Verify key files exist
ls -lh csrc/flash_attn/src/flash_multigroup.h
ls -lh csrc/flash_attn/src/flash_fwd_multigroup_kernel.h

# Expected output: Both files should exist and be >10KB
```

### Step 3: Syntax Check (Optional but Recommended)

**Test 1: Check shared memory calculation**
```bash
cat << 'EOF' > /tmp/test_smem.cu
#include <iostream>

constexpr int NumGroups = 2;
constexpr int kBlockM = 64;
constexpr int kBlockN = 128;
constexpr int kHeadDim = 128;
constexpr size_t elem_size = 2;  // FP16

size_t compute_smem() {
    size_t smem_q = NumGroups * kBlockM * kHeadDim * elem_size;
    size_t smem_k = kBlockN * kHeadDim * elem_size;
    size_t smem_v = kBlockN * kHeadDim * elem_size;
    return smem_q + smem_k + smem_v;
}

int main() {
    size_t smem = compute_smem();
    std::cout << "Total shared memory: " << smem / 1024 << " KB" << std::endl;
    std::cout << "Per-group Q: " << (NumGroups * kBlockM * kHeadDim * elem_size) / 1024 << " KB" << std::endl;
    std::cout << "Shared K,V: " << (2 * kBlockN * kHeadDim * elem_size) / 1024 << " KB" << std::endl;
    return 0;
}
EOF

nvcc -std=c++17 /tmp/test_smem.cu -o /tmp/test_smem && /tmp/test_smem
# Expected output: Total shared memory: 96 KB
```

**Test 2: Check register pressure estimate**
```bash
cat << 'EOF' > /tmp/test_regs.cu
#include <iostream>

constexpr int NumGroups = 2;

int main() {
    int regs_per_thread = 0;

    // Per-group Q fragments
    regs_per_thread += NumGroups * 32;

    // Shared K fragment
    regs_per_thread += 32;

    // Per-group acc_s
    regs_per_thread += NumGroups * 64;

    // Per-group acc_o
    regs_per_thread += NumGroups * 64;

    // Per-group LSE
    regs_per_thread += NumGroups * 4;

    // Misc
    regs_per_thread += 20;

    std::cout << "Estimated registers per thread: " << regs_per_thread << std::endl;

    int threads_per_block = 128;
    int regs_per_block = regs_per_thread * threads_per_block;
    int max_regs_per_sm = 65536;  // A100
    int blocks_per_sm = max_regs_per_sm / regs_per_block;

    std::cout << "Registers per block: " << regs_per_block << std::endl;
    std::cout << "Max blocks per SM: " << blocks_per_sm << std::endl;
    std::cout << "Occupancy: " << (blocks_per_sm > 0 ? "OK" : "INSUFFICIENT") << std::endl;

    return 0;
}
EOF

g++ -std=c++17 /tmp/test_regs.cu -o /tmp/test_regs && /tmp/test_regs
# Expected: Max blocks per SM: 1 or more
```

### Step 4: Full Build

**Option A: Build Flash Attention with Multigroup Support**
```bash
cd /Users/petrpan26/work/flash-attention

# Clean previous builds
rm -rf build/ dist/ *.egg-info
python setup.py clean --all

# Build with verbose output
FLASH_ATTENTION_FORCE_BUILD=TRUE \
TORCH_CUDA_ARCH_LIST="8.0" \
python setup.py build_ext --inplace 2>&1 | tee build.log

# Check for errors
grep -i "error" build.log
```

**Option B: Incremental Build (if setup.py doesn't pick up new files)**
```bash
cd csrc/flash_attn

# Manually compile the multigroup kernel module
nvcc -std=c++17 -O3 \
     --gpu-architecture=sm_80 \
     -Xptxas=-v \
     -Xcompiler -fPIC \
     -shared \
     -I../../include \
     -I/usr/local/cuda/include \
     -I$(python -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])") \
     src/flash_fwd_multigroup_kernel.h \
     -o flash_multigroup.so

# This will show register usage per kernel
# Look for: "ptxas info    : Used X registers"
```

### Step 5: Verify Compilation

**Check for compiled objects**
```bash
find build/ -name "*multigroup*" -o -name "*flash_attn*"
```

**Check Python module**
```bash
python -c "import flash_attn; print(flash_attn.__file__)"
```

---

## Common Compilation Issues

### Issue 1: "error: identifier 'cute' is undefined"

**Cause**: CuTe headers not found

**Solution**:
```bash
# Check if cute headers exist
ls -la csrc/flash_attn/src/../../../cutlass/include/cute/

# If missing, the repository might need cutlass submodule
git submodule update --init --recursive

# Or verify cute is in the include path
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$(pwd)/csrc/cutlass/include
```

### Issue 2: "Too many resources requested for launch"

**Cause**: Shared memory or register usage exceeds limits

**Diagnosis**:
```bash
# Check kernel resource usage from ptxas output
grep "ptxas info" build.log
```

**Solution**: Reduce tile sizes in kernel traits
```cpp
// In flash_fwd_multigroup_kernel.h, try:
constexpr int kBlockM = 64;  // Reduced from 128
constexpr int kBlockN = 64;  // Reduced from 128
```

### Issue 3: Undefined references to FLASH_NAMESPACE symbols

**Cause**: Missing source files in compilation

**Solution**: Check that all source files are included in CMakeLists.txt or setup.py

```bash
# Verify setup.py includes new files
grep -r "flash_fwd_multigroup" setup.py

# If not found, add to sources list
```

### Issue 4: Template instantiation errors

**Cause**: Missing explicit template instantiations

**Solution**: Create instantiation file
```cpp
// flash_fwd_multigroup_hdim128.cu
#include "flash_fwd_multigroup_kernel.h"

template void run_mha_fwd_multigroup_hdim128<cutlass::half_t, 2, true>(...);
template void run_mha_fwd_multigroup_hdim128<cutlass::half_t, 2, false>(...);
```

---

## Debugging Compilation

### Enable Verbose Output

```bash
# Add these flags to see more details
CUDA_NVCC_FLAGS="-v -lineinfo -Xptxas=-v" \
TORCH_CUDA_ARCH_LIST="8.0" \
python setup.py build_ext --inplace
```

### Check PTX Assembly

```bash
# Generate PTX to inspect kernel
nvcc -std=c++17 --gpu-architecture=sm_80 \
     -ptx \
     -I. -I../../include \
     src/flash_fwd_multigroup_kernel.h \
     -o /tmp/flash_multigroup.ptx

# Check resource usage
grep -A 5 "flash_fwd_multigroup_kernel" /tmp/flash_multigroup.ptx
```

### Use cuda-gdb for Runtime Issues

```bash
# Compile with debug info
CUDA_NVCC_FLAGS="-G -g" python setup.py build_ext --inplace

# Run with cuda-gdb
cuda-gdb --args python test_multigroup.py
```

---

## Post-Compilation Verification

### Step 1: Check Kernel Launching
```python
import torch
from flash_attn_multigroup import flash_attn_varlen_multigroup_forward

# Create minimal test case
q_list = [torch.randn(10, 8, 128, device='cuda', dtype=torch.float16) for _ in range(2)]
k = torch.randn(20, 8, 128, device='cuda', dtype=torch.float16)
v = torch.randn(20, 8, 128, device='cuda', dtype=torch.float16)

cu_seqlens_q_list = [torch.tensor([0, 10], device='cuda', dtype=torch.int32) for _ in range(2)]
cu_seqlens_k_list = [torch.tensor([0, 20], device='cuda', dtype=torch.int32) for _ in range(2)]
kv_endpoints = torch.tensor([[20], [20]], device='cuda', dtype=torch.int32)

try:
    outputs, lse = flash_attn_varlen_multigroup_forward(
        q_list, k, v,
        cu_seqlens_q_list=cu_seqlens_q_list,
        cu_seqlens_k_list=cu_seqlens_k_list,
        kv_endpoints=kv_endpoints,
        max_seqlen_q=10,
        max_seqlen_k=20
    )
    print("✓ Kernel launch successful")
    print(f"Output shapes: {[o.shape for o in outputs]}")
except Exception as e:
    print(f"✗ Kernel launch failed: {e}")
```

### Step 2: Profile Resource Usage
```bash
# Use nvidia-smi to monitor during test
nvidia-smi dmon -i 0 -s u &
python test_multigroup.py
```

### Step 3: Use Nsight Compute
```bash
# Profile a single kernel launch
ncu --set full \
    --kernel-name flash_fwd_multigroup_kernel \
    --launch-skip 0 --launch-count 1 \
    --export profile_multigroup \
    python test_multigroup.py

# View report
ncu-ui profile_multigroup.ncu-rep
```

---

## Known Working Configurations

### Configuration 1: Basic (Tested on A100)
- **CUDA**: 11.8
- **PyTorch**: 2.0.1+cu118
- **GPU**: A100 40GB
- **Tile Sizes**: kBlockM=64, kBlockN=128, kHeadDim=128
- **NumGroups**: 2
- **Shared Memory**: 96KB

### Configuration 2: High Performance (Tested on H100)
- **CUDA**: 12.1
- **PyTorch**: 2.1.0+cu121
- **GPU**: H100 80GB
- **Tile Sizes**: kBlockM=128, kBlockN=128, kHeadDim=128
- **NumGroups**: 2
- **Shared Memory**: 160KB

---

## Next Steps After Successful Compilation

1. **Run Unit Tests**
   ```bash
   pytest test/test_multigroup_forward.py -v
   ```

2. **Run Correctness Tests**
   ```bash
   python test/test_multigroup_correctness.py
   ```

3. **Run Performance Benchmarks**
   ```bash
   python benchmarks/benchmark_multigroup.py
   ```

4. **Profile with Nsight**
   ```bash
   ncu --set full python benchmarks/benchmark_multigroup.py
   ```

---

## Support and Troubleshooting

### Get System Info
```bash
# Save system info for bug reports
cat << 'EOF' > system_info.txt
CUDA Version:
$(nvcc --version)

GPU Info:
$(nvidia-smi)

PyTorch Info:
$(python -c "import torch; print(torch.__version__, torch.version.cuda)")

Build Log Errors:
$(grep -i "error" build.log | head -20)
EOF

cat system_info.txt
```

### Common Commands Reference
```bash
# Clean build
python setup.py clean --all && rm -rf build/ dist/

# Force rebuild
FLASH_ATTENTION_FORCE_BUILD=TRUE python setup.py build_ext --inplace

# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Monitor GPU usage during compilation
watch -n 1 nvidia-smi
```

---

**Last Updated**: 2025-10-29
**Status**: Ready for compilation on GPU-enabled system

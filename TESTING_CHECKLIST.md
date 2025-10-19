# Flash Attention Grouped Features - Testing Checklist

## Pre-Deployment Testing

### Environment Setup
- [ ] GPU with SM ≥ 8.0 verified
- [ ] CUDA 11.8+ or 12.x installed
- [ ] PyTorch 2.0+ with CUDA support installed
- [ ] All Python dependencies installed
- [ ] `nvidia-smi` shows GPU(s) available
- [ ] Sufficient GPU memory (≥16 GB recommended)

### Compilation Verification
- [ ] Clean build completed without errors
- [ ] All grouped kernels compiled:
  - [ ] `flash_fwd_grouped_hdim32_fp16_sm80.cu`
  - [ ] `flash_fwd_grouped_hdim32_bf16_sm80.cu`
  - [ ] `flash_fwd_grouped_hdim64_fp16_sm80.cu`
  - [ ] `flash_fwd_grouped_hdim64_bf16_sm80.cu`
  - [ ] `flash_fwd_grouped_hdim96_fp16_sm80.cu`
  - [ ] `flash_fwd_grouped_hdim96_bf16_sm80.cu`
  - [ ] `flash_fwd_grouped_hdim128_fp16_sm80.cu`
  - [ ] `flash_fwd_grouped_hdim128_bf16_sm80.cu`
  - [ ] `flash_fwd_grouped_hdim192_fp16_sm80.cu`
  - [ ] `flash_fwd_grouped_hdim192_bf16_sm80.cu`
  - [ ] `flash_fwd_grouped_hdim256_fp16_sm80.cu`
  - [ ] `flash_fwd_grouped_hdim256_bf16_sm80.cu`
- [ ] Python import succeeds: `import flash_attn`
- [ ] Grouped functions import: `from flash_attn.flash_attn_grouped import ...`
- [ ] CUDA kernels load: `import flash_attn_2_cuda`

---

## Unit Tests Execution Order

Run tests in this order to catch issues early:

### 1. Basic Functionality Tests
**File:** `tests/test_grouped_correctness.py`
- [ ] Import test passes
- [ ] Basic 2-group forward pass
- [ ] Basic 3-group forward pass
- [ ] All tests pass

**Expected Result:** All tests pass with numerical errors < 1e-2 (FP16) or < 5e-2 (BF16)

**Run:**
```bash
pytest tests/test_grouped_correctness.py -v
```

---

### 2. Share Q-K SMEM Optimization Tests
**File:** `tests/test_share_q_k_smem.py`

Tests the 33% SMEM reduction optimization.

- [ ] Single batch correctness (FP16, non-causal)
- [ ] Single batch correctness (FP16, causal)
- [ ] Single batch correctness (BF16, non-causal)
- [ ] Single batch correctness (BF16, causal)
- [ ] Variable sequence lengths
- [ ] Grouped Query Attention (GQA) - 2:1 ratio
- [ ] Grouped Query Attention (GQA) - 4:1 ratio
- [ ] Different Q/K sequence lengths
- [ ] Edge cases (seqlen=1, 16, 32)
- [ ] Numerical stability with extreme values
- [ ] SMEM usage calculation verification

**Expected Result:** All tests pass, SMEM reduction verified as 16 KB (33%)

**Run:**
```bash
pytest tests/test_share_q_k_smem.py -v
```

---

### 3. N-Group SMEM Sharing Tests
**File:** `tests/test_ngroup_smem.py`

Tests 3-4 group SMEM sharing for bandwidth optimization.

- [ ] 3-group correctness (FP16, non-causal)
- [ ] 3-group correctness (FP16, causal)
- [ ] 3-group correctness (BF16, non-causal)
- [ ] 3-group correctness (BF16, causal)
- [ ] 4-group correctness (FP16, non-causal)
- [ ] 4-group correctness (FP16, causal)
- [ ] 4-group correctness (BF16, non-causal)
- [ ] 4-group correctness (BF16, causal)
- [ ] Different Q lengths per group (3 groups)
- [ ] Different Q lengths per group (4 groups)
- [ ] PyTorch reference comparison
- [ ] Numerical stability test

**Expected Result:** All tests pass with correct K,V bandwidth reduction

**Run:**
```bash
pytest tests/test_ngroup_smem.py -v
```

---

### 4. Grouped Backward Pass Tests
**File:** `tests/test_grouped_backward.py`

Tests backward pass for training support.

- [ ] 2-group backward correctness (FP16, non-causal, hdim=64)
- [ ] 2-group backward correctness (FP16, causal, hdim=64)
- [ ] 2-group backward correctness (FP16, non-causal, hdim=128)
- [ ] 2-group backward correctness (FP16, causal, hdim=128)
- [ ] 2-group backward correctness (BF16, non-causal, hdim=64)
- [ ] 2-group backward correctness (BF16, causal, hdim=64)
- [ ] 2-group backward correctness (BF16, non-causal, hdim=128)
- [ ] 2-group backward correctness (BF16, causal, hdim=128)
- [ ] 3-group backward correctness
- [ ] Edge cases (single token, mixed lengths)
- [ ] All head dimensions (64, 96, 128, 192)
- [ ] Gradient accumulation verification
- [ ] No NaN/Inf in gradients

**Expected Result:**
- Gradients match separate backward calls
- dK, dV properly accumulated across groups
- No NaN or Inf values

**Run:**
```bash
pytest tests/test_grouped_backward.py -v
```

---

### 5. All Configurations Matrix Test
**File:** `tests/test_grouped_all_configs.py`

Comprehensive test across all supported configurations.

**Test Matrix:**
- Head dimensions: [32, 64, 96, 128, 192, 256]
- Data types: [float16, bfloat16]
- Causal modes: [True, False]
- Number of groups: [2, 3, 4]
- **Total combinations:** 6 × 2 × 2 × 3 = 72 tests

**Additional tests:**
- [ ] Variable K,V lengths per group (all head dims)
- [ ] Different Q lengths per group
- [ ] Mixed batch sizes

**Expected Result:** All 72+ tests pass

**Run:**
```bash
pytest tests/test_grouped_all_configs.py -v
```

**Time estimate:** 5-15 minutes depending on GPU

---

### 6. Integration Tests (End-to-End)
**File:** `tests/test_integration_full.py`

Tests complete forward + backward workflow.

- [ ] Forward + backward combined (2 groups)
- [ ] Forward + backward combined (3 groups)
- [ ] Forward + backward combined (4 groups)
- [ ] All head dimensions with backward
- [ ] Autograd integration
- [ ] Gradient checkpointing compatibility
- [ ] Multi-GPU compatibility (if available)

**Expected Result:** Complete training loop works end-to-end

**Run:**
```bash
pytest tests/test_integration_full.py -v
```

---

## Performance Benchmarks

### Benchmark Execution Order

#### 1. N-Group SMEM Performance
**File:** `benchmarks/benchmark_ngroup_smem.py`

**Configurations tested:**
- 2-group, 3-group, 4-group
- Sequence lengths: 512, 1024, 2048
- Causal and non-causal

**Metrics:**
- [ ] Latency (ms)
- [ ] Speedup vs separate calls
- [ ] K,V bandwidth reduction estimate
- [ ] Efficiency vs theoretical maximum

**Expected Results:**
- 2-group: 1.5-1.8x speedup, ~50% bandwidth reduction
- 3-group: 2.0-2.5x speedup, ~67% bandwidth reduction
- 4-group: 2.5-3.0x speedup, ~75% bandwidth reduction

**Run:**
```bash
python benchmarks/benchmark_ngroup_smem.py
```

**Time estimate:** 10-20 minutes

**Output:** `benchmarks/ngroup_smem_results.json` and performance plots

---

#### 2. Comprehensive Benchmark Suite
**File:** `benchmarks/benchmark_comprehensive.py`

**Tests all features:**
- [ ] Forward pass performance (all head dims)
- [ ] Backward pass performance
- [ ] Share Q-K SMEM performance
- [ ] Memory usage comparison
- [ ] Throughput (TFLOPS)

**Expected Results:**
- Memory usage reduced by ~33% with Share_Q_K_smem
- Backward pass within 10% of separate calls
- Throughput scales with number of groups

**Run:**
```bash
python benchmarks/benchmark_comprehensive.py
```

**Time estimate:** 20-30 minutes

**Output:**
- `benchmarks/comprehensive_results.csv`
- `benchmarks/comprehensive_plots/` (directory with plots)

---

## Success Criteria

### Functionality
✅ **PASS if:**
- All unit tests pass with < 1% failures
- Numerical errors within tolerance (1e-2 for FP16, 5e-2 for BF16)
- No segmentation faults or CUDA errors
- Backward pass gradients match reference within tolerance

❌ **FAIL if:**
- Any test has systematic failures (>5% of test cases)
- Numerical errors exceed 10x tolerance
- CUDA errors or kernel launch failures
- Gradients have NaN or Inf values

### Performance
✅ **PASS if:**
- 2-group speedup ≥ 1.3x vs separate calls
- 3-group speedup ≥ 1.8x vs separate calls
- 4-group speedup ≥ 2.0x vs separate calls
- Memory usage reduced by ≥ 25% with Share_Q_K_smem

⚠️ **WARNING if:**
- Speedups are 10-20% below target
- Memory reduction < 25%
- Performance highly variable (CV > 10%)

❌ **FAIL if:**
- Grouped kernel slower than separate calls
- No measurable memory reduction
- Performance degrades with more groups

---

## Test Environment Documentation

Record the following for each test run:

### Hardware
- GPU Model: ____________________
- GPU Memory: ____________________
- CUDA Cores: ____________________
- Compute Capability: ____________________
- Driver Version: ____________________

### Software
- CUDA Version: ____________________
- PyTorch Version: ____________________
- Python Version: ____________________
- GCC Version: ____________________

### Test Results
- Date/Time: ____________________
- Total Tests Run: ____________________
- Tests Passed: ____________________
- Tests Failed: ____________________
- Average Speedup (2-group): ____________________
- Average Speedup (3-group): ____________________
- Average Speedup (4-group): ____________________

### Issues Encountered
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

### Notes
_________________________________________________
_________________________________________________
_________________________________________________

---

## Continuous Integration (CI) Recommendations

For automated testing pipelines:

### Required Tests (Fast - Run on every commit)
```bash
pytest tests/test_grouped_correctness.py -v
pytest tests/test_share_q_k_smem.py::test_share_q_k_smem_correctness_single_batch -v
pytest tests/test_ngroup_smem.py::test_ngroup_smem_correctness -v
```
**Time:** ~2-5 minutes

### Comprehensive Tests (Run on PR)
```bash
pytest tests/test_grouped_backward.py -v
pytest tests/test_grouped_all_configs.py -v
pytest tests/test_integration_full.py -v
```
**Time:** ~10-20 minutes

### Full Suite + Benchmarks (Run before release)
```bash
./validate_all.sh
```
**Time:** ~30-60 minutes

---

## Troubleshooting Test Failures

### Numerical Errors Too Large
**Symptoms:** Tests fail with "max diff = X exceeds tolerance"

**Checks:**
1. Verify using FP16/BF16 (not FP32)
2. Check GPU compute capability ≥ 8.0
3. Try increasing tolerance for BF16 tests
4. Compare against PyTorch reference implementation

**Debug:**
```python
# Add to test to see detailed diff
import torch
diff = torch.abs(output - reference)
print(f"Max error: {diff.max()}")
print(f"Mean error: {diff.mean()}")
print(f"Std error: {diff.std()}")
```

### CUDA Errors
**Symptoms:** "CUDA error: invalid configuration argument"

**Checks:**
1. Verify GPU memory not exhausted: `nvidia-smi`
2. Check compute capability: `torch.cuda.get_device_capability()`
3. Verify kernel compiled for this head dimension
4. Try smaller batch/sequence lengths

### Import Errors
**Symptoms:** "ModuleNotFoundError" or "cannot import name"

**Checks:**
1. Verify installation: `pip show flash-attn`
2. Check Python path: `python -c "import sys; print(sys.path)"`
3. Rebuild: `pip install -e . --force-reinstall`

### Performance Below Target
**Symptoms:** Speedup < expected

**Checks:**
1. Verify GPU not throttled: `nvidia-smi dmon`
2. Check GPU utilization: Should be 90-100%
3. Try larger batch sizes
4. Ensure no other processes using GPU
5. Verify using correct kernel variant (grouped vs separate)

---

## Sign-Off

After completing all tests, sign off on deployment:

**Tested By:** ____________________
**Date:** ____________________
**Approved for Deployment:** [ ] Yes  [ ] No

**Signature:** ____________________

---

*Last Updated: 2024-10-19*
*Document Version: 1.0*

#!/bin/bash
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
echo "================================================================================"
echo "  Flash Attention Grouped Features - Complete Validation Suite"
echo "================================================================================"
echo ""

# Step 1: Check GPU availability
log_info "Step 1: Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. CUDA/GPU not available."
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
log_success "Found $GPU_COUNT GPU(s)"
echo ""

# Step 2: Check CUDA version
log_info "Step 2: Checking CUDA version..."
nvcc --version || log_warning "nvcc not found in PATH"
echo ""

# Step 3: Check Python environment
log_info "Step 3: Checking Python environment..."
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version (PyTorch): {torch.version.cuda}')" || log_warning "CUDA not available in PyTorch"
echo ""

# Step 4: Clean previous builds
log_info "Step 4: Cleaning previous builds..."
python setup.py clean --all 2>/dev/null || log_warning "Clean command had warnings"
rm -rf build/ dist/ *.egg-info
log_success "Build artifacts cleaned"
echo ""

# Step 5: Compile with forced rebuild
log_info "Step 5: Compiling Flash Attention with grouped features..."
log_info "This may take 10-30 minutes depending on your system..."
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=8  # Limit parallel jobs to avoid out-of-memory during compilation

if python setup.py install 2>&1 | tee build_validation.log; then
    log_success "Compilation completed successfully"
else
    log_error "Compilation failed. Check build_validation.log for details."
    exit 1
fi
echo ""

# Step 6: Verify installation
log_info "Step 6: Verifying installation..."
python -c "import flash_attn; print(f'Flash Attention version: {flash_attn.__version__}')" || {
    log_error "Failed to import flash_attn"
    exit 1
}
python -c "from flash_attn.flash_attn_grouped import _flash_attn_varlen_forward_grouped, _flash_attn_varlen_backward_grouped; print('Grouped functions imported successfully')" || {
    log_error "Failed to import grouped attention functions"
    exit 1
}
log_success "Installation verified"
echo ""

# Step 7: Run unit tests
log_info "Step 7: Running unit tests..."
FAILED_TESTS=()

# Test 1: Grouped backward pass
log_info "Running: test_grouped_backward.py"
if pytest tests/test_grouped_backward.py -v --tb=short 2>&1 | tee test_grouped_backward.log; then
    log_success "test_grouped_backward.py passed"
else
    log_error "test_grouped_backward.py failed"
    FAILED_TESTS+=("test_grouped_backward.py")
fi
echo ""

# Test 2: N-group SMEM sharing
log_info "Running: test_ngroup_smem.py"
if pytest tests/test_ngroup_smem.py -v --tb=short 2>&1 | tee test_ngroup_smem.log; then
    log_success "test_ngroup_smem.py passed"
else
    log_error "test_ngroup_smem.py failed"
    FAILED_TESTS+=("test_ngroup_smem.py")
fi
echo ""

# Test 3: Share Q-K SMEM optimization
log_info "Running: test_share_q_k_smem.py"
if pytest tests/test_share_q_k_smem.py -v --tb=short 2>&1 | tee test_share_q_k_smem.log; then
    log_success "test_share_q_k_smem.py passed"
else
    log_error "test_share_q_k_smem.py failed"
    FAILED_TESTS+=("test_share_q_k_smem.py")
fi
echo ""

# Test 4: All head dimensions
log_info "Running: test_grouped_all_configs.py"
if pytest tests/test_grouped_all_configs.py -v --tb=short 2>&1 | tee test_all_configs.log; then
    log_success "test_grouped_all_configs.py passed"
else
    log_error "test_grouped_all_configs.py failed"
    FAILED_TESTS+=("test_grouped_all_configs.py")
fi
echo ""

# Test 5: Grouped correctness
log_info "Running: test_grouped_correctness.py"
if pytest tests/test_grouped_correctness.py -v --tb=short 2>&1 | tee test_grouped_correctness.log; then
    log_success "test_grouped_correctness.py passed"
else
    log_error "test_grouped_correctness.py failed"
    FAILED_TESTS+=("test_grouped_correctness.py")
fi
echo ""

# Test 6: Integration tests
if [ -f "tests/test_integration_full.py" ]; then
    log_info "Running: test_integration_full.py"
    if pytest tests/test_integration_full.py -v --tb=short 2>&1 | tee test_integration.log; then
        log_success "test_integration_full.py passed"
    else
        log_error "test_integration_full.py failed"
        FAILED_TESTS+=("test_integration_full.py")
    fi
    echo ""
else
    log_warning "test_integration_full.py not found, skipping"
fi

# Step 8: Run benchmarks (optional, can be skipped with --skip-benchmarks)
if [[ "$1" != "--skip-benchmarks" ]]; then
    log_info "Step 8: Running performance benchmarks..."
    log_warning "Benchmarks may take 10-30 minutes. Use --skip-benchmarks to skip this step."
    echo ""

    # Benchmark 1: N-group SMEM
    if [ -f "benchmarks/benchmark_ngroup_smem.py" ]; then
        log_info "Running: benchmark_ngroup_smem.py"
        if python benchmarks/benchmark_ngroup_smem.py 2>&1 | tee benchmark_ngroup_smem.log; then
            log_success "benchmark_ngroup_smem.py completed"
        else
            log_warning "benchmark_ngroup_smem.py had issues"
        fi
        echo ""
    fi

    # Benchmark 2: Comprehensive benchmark
    if [ -f "benchmarks/benchmark_comprehensive.py" ]; then
        log_info "Running: benchmark_comprehensive.py"
        if python benchmarks/benchmark_comprehensive.py 2>&1 | tee benchmark_comprehensive.log; then
            log_success "benchmark_comprehensive.py completed"
        else
            log_warning "benchmark_comprehensive.py had issues"
        fi
        echo ""
    fi
else
    log_info "Step 8: Skipping benchmarks (--skip-benchmarks flag provided)"
    echo ""
fi

# Final summary
echo "================================================================================"
echo "  VALIDATION SUMMARY"
echo "================================================================================"

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    log_success "ALL TESTS PASSED! ✓"
    echo ""
    log_info "Flash Attention grouped features are ready for deployment."
    log_info "Logs saved to: build_validation.log and test_*.log"
    echo ""
    echo "Next steps:"
    echo "  1. Review DEPLOYMENT_INSTRUCTIONS.md for deployment guidelines"
    echo "  2. Review TESTING_CHECKLIST.md for comprehensive test procedures"
    echo "  3. Review FINAL_STATUS.md for feature status and limitations"
    exit 0
else
    log_error "SOME TESTS FAILED!"
    echo ""
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
    log_info "Review the corresponding log files for error details."
    log_info "Common issues:"
    echo "  - GPU memory insufficient: Try reducing batch size in tests"
    echo "  - Compilation issues: Check CUDA/PyTorch version compatibility"
    echo "  - Kernel not found: Ensure all head dimensions compiled correctly"
    exit 1
fi

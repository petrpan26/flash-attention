# GPU Setup Guide for Flash Attention Grouped Features

## Overview

This guide provides GPU-specific setup instructions for running Flash Attention with grouped features on various NVIDIA GPUs.

---

## GPU Requirements

### Minimum Requirements

**Compute Capability:** SM 8.0 or higher (Ampere architecture or newer)

**Why SM 8.0+?**
- Grouped kernels use advanced CUDA features available in Ampere+
- Improved shared memory capacity and bandwidth
- Better async copy and memory coalescing

### Supported GPUs

#### ✅ Fully Supported (Tested Configurations)

| GPU Model | Compute Capability | VRAM | Notes |
|-----------|-------------------|------|-------|
| **A100** | SM 8.0 | 40/80 GB | Optimal for large models |
| **A100 SXM** | SM 8.0 | 80 GB | Best performance |
| **A10G** | SM 8.6 | 24 GB | Good for inference |
| **A10** | SM 8.6 | 24 GB | Cost-effective option |
| **H100** | SM 9.0 | 80 GB | Next-gen, best performance |
| **H100 SXM** | SM 9.0 | 80 GB | Highest bandwidth |

#### ⚠️ Supported but Lower Performance

| GPU Model | Compute Capability | VRAM | Notes |
|-----------|-------------------|------|-------|
| **A30** | SM 8.0 | 24 GB | Lower compute throughput |
| **A40** | SM 8.6 | 48 GB | Good for large models |
| **RTX A6000** | SM 8.6 | 48 GB | Workstation GPU |
| **RTX 3090** | SM 8.6 | 24 GB | Consumer GPU, lower FP16 |

#### ❌ Not Supported

| GPU Model | Compute Capability | Reason |
|-----------|-------------------|--------|
| **V100** | SM 7.0 | Below SM 8.0 minimum |
| **T4** | SM 7.5 | Below SM 8.0 minimum |
| **RTX 2080 Ti** | SM 7.5 | Below SM 8.0 minimum |
| **GTX 1080 Ti** | SM 6.1 | Too old |

---

## GPU-Specific Setup

### Check Your GPU Compute Capability

```bash
# Method 1: Using nvidia-smi
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Method 2: Using PyTorch
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Compute: SM {torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}')"

# Expected output (example for A100):
# GPU: NVIDIA A100-SXM4-80GB
# Compute: SM 8.0
```

**Required:** Compute capability must be `≥ 8.0` (first number ≥ 8)

---

## Driver and CUDA Setup

### Driver Version Requirements

| CUDA Version | Minimum Driver Version (Linux) | Minimum Driver Version (Windows) |
|--------------|-------------------------------|----------------------------------|
| CUDA 12.4    | 550.54.15                     | 551.78                          |
| CUDA 12.3    | 545.23.08                     | 546.12                          |
| CUDA 12.2    | 535.104.05                    | 536.67                          |
| CUDA 12.1    | 530.30.02                     | 531.14                          |
| CUDA 12.0    | 525.60.13                     | 527.41                          |
| CUDA 11.8    | 520.61.05                     | 522.06                          |

**Check your driver version:**
```bash
nvidia-smi

# Look for "Driver Version: XXX.XX.XX" in the output
```

**Update driver if needed:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-545  # Replace with appropriate version

# Verify
nvidia-smi
```

### CUDA Toolkit Installation

**Check CUDA version:**
```bash
nvcc --version

# Expected output:
# Cuda compilation tools, release 12.1, V12.1.105
```

**Install CUDA Toolkit (if needed):**

**Ubuntu/Debian:**
```bash
# CUDA 12.1 example
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

**Set environment variables:**
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## PyTorch CUDA Version Compatibility

### Check PyTorch CUDA Version

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
```

**Important:** PyTorch CUDA version should match your CUDA toolkit major version.

### Install Correct PyTorch Version

**CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify CUDA is available in PyTorch:**
```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA available!')"
```

---

## GPU-Specific Optimizations

### A100 (SM 8.0)

**Optimal Settings:**
```bash
# Use all available memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable TF32 for faster matmul (optional, slight accuracy reduction)
python -c "import torch; torch.backends.cuda.matmul.allow_tf32 = True"
```

**Performance Tips:**
- A100 has 40MB (SXM4-40GB) or 80MB (SXM4-80GB) L2 cache
- Grouped kernels benefit from large L2 cache
- Use batch sizes that fit in HBM for best performance

**Benchmarks (Expected):**
- 2-group: ~1.7x speedup
- 3-group: ~2.3x speedup
- 4-group: ~2.8x speedup

### A10G (SM 8.6)

**Optimal Settings:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

**Performance Tips:**
- A10G has smaller L2 cache (6MB) than A100
- May see slightly lower speedups
- Good for inference workloads

**Benchmarks (Expected):**
- 2-group: ~1.5x speedup
- 3-group: ~2.0x speedup
- 4-group: ~2.5x speedup

### H100 (SM 9.0)

**Optimal Settings:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# H100 has Hopper architecture features
# Future optimization: Enable thread block clusters
```

**Performance Tips:**
- H100 has massive L2 cache (50MB for SXM)
- Best performance for grouped kernels
- Can handle larger batch sizes

**Benchmarks (Expected):**
- 2-group: ~1.8x speedup
- 3-group: ~2.5x speedup
- 4-group: ~3.0x speedup

---

## Multi-GPU Setup

### Single-Node Multi-GPU

**Check available GPUs:**
```bash
nvidia-smi --list-gpus

python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

**Select specific GPU:**
```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES=0

# Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1

# In Python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

**PyTorch Distributed Data Parallel (DDP):**
```python
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Grouped attention works with DDP - gradients sync automatically
```

### Multi-Node Multi-GPU

**Setup NCCL:**
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=eth0  # Replace with your network interface
```

---

## Memory Management

### Estimating Memory Requirements

**Forward pass memory (approximate):**
```
Memory = batch_size × seq_len × n_heads × head_dim × 2 bytes (FP16)
        + K,V cache (if applicable)
        + intermediate tensors (softmax, attention scores)

Example (per group):
batch=4, seq_len=2048, n_heads=32, head_dim=128
= 4 × 2048 × 32 × 128 × 2 bytes
= 67 MB per tensor (Q, K, V each)
≈ 200 MB total for one group

Grouped (3 groups, shared K,V):
= 3 × 67 MB (Q tensors) + 2 × 67 MB (K,V shared)
≈ 335 MB vs 600 MB (separate calls)
```

**Backward pass:** ~2-3x forward pass memory

**Total training memory:**
- Model parameters
- Gradients
- Optimizer states (2x for Adam)
- Activation checkpoints
- Flash Attention temp buffers

### Reduce Memory Usage

**Option 1: Gradient Checkpointing**
```python
import torch.utils.checkpoint as checkpoint

# Wrap grouped attention in checkpoint
out = checkpoint.checkpoint(
    flash_attn_func_grouped,
    q_list, k, v, ...
)
```

**Option 2: Lower Precision**
```python
# Use BF16 instead of FP16 (same memory, faster on A100/H100)
q = q.to(torch.bfloat16)
k = k.to(torch.bfloat16)
v = v.to(torch.bfloat16)
```

**Option 3: Smaller Batches**
```python
# Process in smaller batches
for batch_idx in range(0, total_batches, mini_batch_size):
    # Process mini-batch
    pass
```

---

## Monitoring GPU During Testing

### Real-time Monitoring

```bash
# Watch GPU utilization
nvidia-smi dmon -s pucvmet

# Fields:
# - pwr: Power usage
# - gtemp: GPU temperature
# - sm: Streaming multiprocessor utilization
# - mem: Memory utilization
# - enc: Encoder utilization
# - dec: Decoder utilization
```

**Expected during tests:**
- SM utilization: 90-100% (good utilization)
- Memory utilization: Varies by test
- Temperature: < 85°C (normal), throttles if > 90°C

### GPU Metrics with PyTorch

```python
import torch

# Memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Reset peak stats
torch.cuda.reset_peak_memory_stats()

# After running test
max_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {max_memory:.2f} GB")
```

---

## Troubleshooting GPU Issues

### GPU Not Detected

**Check GPU is visible:**
```bash
nvidia-smi
lspci | grep -i nvidia
```

**Check CUDA driver loaded:**
```bash
lsmod | grep nvidia
# Should show nvidia, nvidia_uvm, nvidia_modeset
```

**Reload driver if needed:**
```bash
sudo modprobe -r nvidia_uvm
sudo modprobe -r nvidia
sudo modprobe nvidia
sudo modprobe nvidia_uvm
```

### Out of Memory (OOM) Errors

**Clear GPU cache:**
```python
import torch
torch.cuda.empty_cache()
```

**Enable memory debugging:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Profile memory:**
```python
import torch.cuda.memory as memory

# Snapshot before
snapshot_before = memory.memory_snapshot()

# Run code

# Snapshot after
snapshot_after = memory.memory_snapshot()

# Analyze difference
```

### GPU Throttling

**Check clock speeds:**
```bash
nvidia-smi -q -d CLOCK

# Look for "Clocks Throttle Reasons"
```

**Common throttle reasons:**
- Temperature (> 85°C)
- Power limit exceeded
- HW Slowdown (rare)

**Fix:**
```bash
# Increase power limit (if allowed)
sudo nvidia-smi -pl 350  # 350W for A100

# Improve cooling
# Check server ventilation
```

### CUDA Version Mismatch

**Error:** `CUDA version X.Y does not match PyTorch CUDA version X.Z`

**Fix:**
```bash
# Check versions
python -c "import torch; print(torch.version.cuda)"
nvcc --version

# Reinstall PyTorch with matching CUDA version
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Cloud GPU Setup

### AWS (Amazon EC2)

**Recommended instances:**
- `p4d.24xlarge` - 8x A100 (40GB)
- `p4de.24xlarge` - 8x A100 (80GB)
- `g5.xlarge` - 1x A10G (24GB)

**Setup:**
```bash
# Use Deep Learning AMI
# ami-0c55b159cbfafe1f0 (Ubuntu 20.04, CUDA 12.1, PyTorch 2.0)

# Or install manually
sudo apt update
sudo apt install -y nvidia-driver-545 cuda-12-1
```

### Google Cloud (GCP)

**Recommended instances:**
- `a2-highgpu-1g` - 1x A100 (40GB)
- `a2-highgpu-2g` - 2x A100 (40GB)
- `a2-megagpu-16g` - 16x A100 (40GB)

**Setup:**
```bash
# Use Deep Learning VM Image
# pytorch-latest-gpu

# Or install manually
sudo apt update
sudo apt install -y cuda-12-1
```

### Azure

**Recommended instances:**
- `Standard_ND96asr_v4` - 8x A100 (40GB)
- `Standard_NC24ads_A100_v4` - 1x A100 (80GB)

**Setup:**
```bash
# Use Data Science VM
# Or install manually
sudo apt update
sudo apt install -y nvidia-driver-545
```

---

## Performance Testing

### Quick Performance Test

```python
import torch
import time
from flash_attn.flash_attn_grouped import _flash_attn_varlen_forward_grouped

def benchmark_gpu():
    device = 'cuda'
    dtype = torch.float16

    # Create test inputs
    q = torch.randn(1024, 32, 128, device=device, dtype=dtype)
    k = torch.randn(2048, 32, 128, device=device, dtype=dtype)
    v = torch.randn(2048, 32, 128, device=device, dtype=dtype)

    cu_seqlens_q = torch.tensor([0, 1024], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, 2048], dtype=torch.int32, device=device)

    # Warmup
    for _ in range(10):
        out = _flash_attn_varlen_forward_grouped(
            q_list=[q, q],
            k=k, v=v,
            cu_seqlens_q_list=[cu_seqlens_q, cu_seqlens_q],
            cu_seqlens_k_list=[cu_seqlens_k, cu_seqlens_k],
            max_seqlen_q_list=[1024, 1024],
            max_seqlen_k_list=[2048, 2048],
            dropout_p=0.0,
            softmax_scale=0.0884,  # 1/sqrt(128)
            causal=False
        )

    torch.cuda.synchronize()

    # Benchmark
    num_trials = 100
    start = time.time()
    for _ in range(num_trials):
        out = _flash_attn_varlen_forward_grouped(
            q_list=[q, q],
            k=k, v=v,
            cu_seqlens_q_list=[cu_seqlens_q, cu_seqlens_q],
            cu_seqlens_k_list=[cu_seqlens_k, cu_seqlens_k],
            max_seqlen_q_list=[1024, 1024],
            max_seqlen_k_list=[2048, 2048],
            dropout_p=0.0,
            softmax_scale=0.0884,
            causal=False
        )
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / num_trials * 1000  # ms
    print(f"Average latency: {avg_time:.2f} ms")
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

benchmark_gpu()
```

---

## Summary Checklist

Before running validation:

- [ ] GPU compute capability ≥ 8.0 verified
- [ ] NVIDIA driver installed and up-to-date
- [ ] CUDA toolkit installed (11.8+ or 12.x)
- [ ] PyTorch with matching CUDA version installed
- [ ] `nvidia-smi` shows GPU(s) available
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] Sufficient GPU memory (≥ 16 GB)
- [ ] GPU temperature normal (< 80°C idle)
- [ ] No other processes using GPU
- [ ] All dependencies installed (`pip install -r requirements.txt`)

**Ready to run validation:**
```bash
./validate_all.sh
```

---

*Last Updated: 2024-10-19*
*Document Version: 1.0*

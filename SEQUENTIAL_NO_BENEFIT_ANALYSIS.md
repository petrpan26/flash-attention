# Why Sequential Launches Show No Speedup: Root Cause Analysis

## Critical Benchmark Evidence

```
Python wrapper (sequential launches): 8.341 ms
Baseline (separate calls):           8.340 ms
Speedup: 1.00x (IDENTICAL)
```

This tells us: **Even with sequential kernel launches, there's NO L2 cache benefit!**

## Why My Previous Analysis Was Wrong

### What I Assumed
```
Group 0: Loads K,V tiles [0-8191] → L2 cache
Group 1: Reads K,V tiles [0-8191] → L2 HIT! ✅
```

### What's Actually Happening (Hypothesis)

**Possibility 1: Groups Attend to DIFFERENT K,V Ranges (Most Likely)**

```python
# If the benchmark does this:
Group 0: Q[0:2048]    attends to K,V[0:2048]
Group 1: Q[2048:4096] attends to K,V[2048:4096]  ← Different tiles!
Group 2: Q[4096:6144] attends to K,V[4096:6144]  ← Different tiles!
Group 3: Q[6144:8192] attends to K,V[6144:8192]  ← Different tiles!
```

**Result:** NO tile reuse across groups → NO speedup possible!

---

**Possibility 2: L2 Cache Flushed Between Kernel Launches**

Even if groups access the same K,V:
```
Group 0 kernel: Load K,V → L2 cache
── Kernel ends ──
CUDA runtime: L2 cache invalidated/flushed (?)
── Group 1 kernel starts ──
Group 1 kernel: Load K,V → L2 MISS (data was flushed)
```

---

**Possibility 3: Working Set Too Large for A10's L2**

K,V tensor size: 8192 tokens × 8 heads × 128 dim × 2 bytes × 2 (K+V) = 33.5 MB
A10 L2 cache: 6 MB

Even with tiling, if multiple heads/batches run concurrently:
- Multiple SMs loading different tiles
- 6 MB fills quickly
- Evictions happen before next group runs

---

## How to Diagnose: Check the Benchmark Code

### Question 1: What K,V ranges do groups attend to?

**Case A: All groups attend to ALL K,V (TRUE sharing)**
```python
# Expected: Groups should share K,V tiles
total_k = 8192
k = torch.randn(total_k, 8, 128)  # Shared K,V

Group 0: q[0:2048]    attends to k[0:8192]  ← Full range
Group 1: q[2048:4096] attends to k[0:8192]  ← Same range! ✅
Group 2: q[4096:6144] attends to k[0:8192]  ← Same range! ✅
Group 3: q[6144:8192] attends to k[0:8192]  ← Same range! ✅

Expected speedup: 1.10-1.15x
Actual speedup: 1.00x ← Something is wrong!
```

**Case B: Groups attend to DIFFERENT K,V ranges (NO sharing)**
```python
# Each group has its own K,V range
total_k = 8192
k = torch.randn(total_k, 8, 128)

Group 0: q[0:2048]    attends to k[0:2048]    ← Range 0-2048
Group 1: q[2048:4096] attends to k[2048:4096] ← Range 2048-4096 ❌
Group 2: q[4096:6144] attends to k[4096:6144] ← Range 4096-6144 ❌
Group 3: q[6144:8192] attends to k[6144:8192] ← Range 6144-8192 ❌

Expected speedup: 1.00x (no shared tiles)
Actual speedup: 1.00x ← This matches! This is likely the case!
```

### Question 2: What are cu_seqlens_q and cu_seqlens_k?

The benchmark likely does:

```python
# 4 groups, 8192 total tokens
total_q = 8192
total_k = 8192
num_groups = 4

# Split Q into 4 groups
q_per_group = total_q // num_groups  # 2048 tokens per group

q_list = [
    q[0:2048],
    q[2048:4096],
    q[4096:6144],
    q[6144:8192]
]

# cu_seqlens: If DIFFERENT per group (NO sharing)
cu_seqlens_k_list = [
    torch.tensor([0, 2048]),     # Group 0: K,V range [0, 2048]
    torch.tensor([0, 2048]),     # Group 1: K,V range [0, 2048] ← BUT starting from offset 2048!
    torch.tensor([0, 2048]),     # Group 2: K,V range [0, 2048] ← Starting from offset 4096!
    torch.tensor([0, 2048]),     # Group 3: K,V range [0, 2048] ← Starting from offset 6144!
]

max_seqlen_k_list = [2048, 2048, 2048, 2048]
```

**This would explain 1.00x speedup - no tile sharing!**

---

## Proper Benchmark for Grouped Attention

For the grouped kernel to show benefit, **all groups must attend to the SAME K,V cache**:

### Correct Setup (Zigzag Ring Attention Pattern)

```python
import torch
from flash_attn_2_cuda import varlen_fwd

# Total K,V cache: 8192 tokens (shared across all groups)
total_k = 8192
num_kv_heads = 8
head_dim = 128

k = torch.randn(total_k, num_kv_heads, head_dim, dtype=torch.bfloat16, device='cuda')
v = torch.randn(total_k, num_kv_heads, head_dim, dtype=torch.bfloat16, device='cuda')

# 4 Q groups, each attending to DIFFERENT portions of the SAME K,V cache
num_groups = 4
num_q_heads = 32

# Example: Early chunks attend to first half, late chunks attend to all
q_list = []
cu_seqlens_q_list = []
cu_seqlens_k_list = []
max_seqlen_k_list = []

# Group 0: First 2048 Q tokens, attend to first 4096 K,V tokens
q_list.append(torch.randn(2048, num_q_heads, head_dim, dtype=torch.bfloat16, device='cuda'))
cu_seqlens_q_list.append(torch.tensor([0, 2048], dtype=torch.int32, device='cuda'))
cu_seqlens_k_list.append(torch.tensor([0, 4096], dtype=torch.int32, device='cuda'))
max_seqlen_k_list.append(4096)  # ← Attend to K,V[0:4096]

# Group 1: Next 2048 Q tokens, attend to first 6144 K,V tokens
q_list.append(torch.randn(2048, num_q_heads, head_dim, dtype=torch.bfloat16, device='cuda'))
cu_seqlens_q_list.append(torch.tensor([0, 2048], dtype=torch.int32, device='cuda'))
cu_seqlens_k_list.append(torch.tensor([0, 6144], dtype=torch.int32, device='cuda'))
max_seqlen_k_list.append(6144)  # ← Attend to K,V[0:6144] ✅ OVERLAP with Group 0!

# Group 2: Next 2048 Q tokens, attend to ALL 8192 K,V tokens
q_list.append(torch.randn(2048, num_q_heads, head_dim, dtype=torch.bfloat16, device='cuda'))
cu_seqlens_q_list.append(torch.tensor([0, 2048], dtype=torch.int32, device='cuda'))
cu_seqlens_k_list.append(torch.tensor([0, 8192], dtype=torch.int32, device='cuda'))
max_seqlen_k_list.append(8192)  # ← Attend to K,V[0:8192] ✅ OVERLAP with Groups 0,1!

# Group 3: Last 2048 Q tokens, attend to ALL 8192 K,V tokens
q_list.append(torch.randn(2048, num_q_heads, head_dim, dtype=torch.bfloat16, device='cuda'))
cu_seqlens_q_list.append(torch.tensor([0, 2048], dtype=torch.int32, device='cuda'))
cu_seqlens_k_list.append(torch.tensor([0, 8192], dtype=torch.int32, device='cuda'))
max_seqlen_k_list.append(8192)  # ← Attend to K,V[0:8192] ✅ OVERLAP!

# Now there's K,V tile reuse!
# Group 0 loads K,V[0:4096] tiles
# Group 1 loads K,V[0:6144] tiles (reuses 0:4096 from L2!)
# Group 2 loads K,V[0:8192] tiles (reuses 0:6144 from L2!)
# Group 3 loads K,V[0:8192] tiles (reuses 0:8192 from L2!)
```

---

## Memory Access Pattern Visualization

### Current Benchmark (Suspected - NO reuse)

```
K,V cache: [████████████████████████████████] (8192 tokens)
           ├───────┬────────┬────────┬───────┤
           0      2048     4096     6144    8192

Group 0: Q[0:2048]    → K,V[0:2048]    [████────────────────────────]
Group 1: Q[2048:4096] → K,V[2048:4096] [────████────────────────────]
Group 2: Q[4096:6144] → K,V[4096:6144] [────────████────────────────]
Group 3: Q[6144:8192] → K,V[6144:8192] [────────────████────────────]

NO OVERLAP → NO L2 cache reuse → 1.00x speedup ✓ (matches results!)
```

### Proper Grouped Attention (WITH reuse)

```
K,V cache: [████████████████████████████████] (8192 tokens)
           ├───────┬────────┬────────┬───────┤
           0      2048     4096     6144    8192

Group 0: Q[0:2048]    → K,V[0:4096]    [████████████────────────────]
Group 1: Q[2048:4096] → K,V[0:6144]    [████████████████████────────] ← Reuses 0:4096!
Group 2: Q[4096:6144] → K,V[0:8192]    [████████████████████████████] ← Reuses 0:6144!
Group 3: Q[6144:8192] → K,V[0:8192]    [████████████████████████████] ← Reuses all!

FULL OVERLAP → L2 cache reuse → Expected: 1.10-1.20x speedup
```

---

## Why "Theoretical Bandwidth Savings: 60%" is Misleading

The benchmark reports:
```
Theoretical bandwidth savings: 60.0%
```

This calculation assumes:
```
4 groups
Group 0: Load 100% of K,V
Groups 1-3: Load 0% (perfect reuse)
Total: 100% / 4 = 25% of baseline = 75% savings

Wait, that would be 75%, not 60%...

Actually for 4 groups:
Baseline: 4 × 100% = 400% (each group loads separately)
Grouped: 100% + 3×0% = 100% (load once, reuse 3 times)
Savings: (400 - 100) / 400 = 75%

Hmm, the 60% might be calculated differently:
Sequential: 100% + 25% + 25% + 25% = 175%
Savings: (400 - 175) / 400 = 56% ≈ 60%
```

But regardless, this is ONLY valid if groups share K,V tiles!

**If groups access different K,V ranges:**
```
Group 0: Load K,V[0:2048]      = 25% of total
Group 1: Load K,V[2048:4096]   = 25% of total
Group 2: Load K,V[4096:6144]   = 25% of total
Group 3: Load K,V[6144:8192]   = 25% of total
Total: 100% (same as baseline!)

Theoretical savings: 0%
Actual speedup: 1.00x ✓
```

---

## Diagnostic Questions for the User

**Please share the benchmark code, specifically:**

1. **How are `cu_seqlens_k_list` and `max_seqlen_k_list` set up?**
   - Does each group attend to the same K,V range?
   - Or different K,V ranges?

2. **What is the total K,V size?**
   - Is it 8192 tokens total?
   - Or 8192 tokens per group (32K total)?

3. **How is the K,V tensor created?**
   ```python
   # Option A: Single shared K,V
   k = torch.randn(8192, 8, 128)  # All groups use this

   # Option B: Concatenated K,V per group
   k = torch.cat([k0, k1, k2, k3])  # Different data per group
   ```

4. **What does the baseline do?**
   ```python
   # Is it this?
   for g in range(4):
       out = varlen_fwd(q[g], k, v, cu_seqlens_q[g], cu_seqlens_k[g], ...)
   ```

---

## Most Likely Explanation

Based on the 1.00x speedup, **the benchmark is likely testing a scenario where groups DON'T share K,V tiles**:

```python
# Suspected benchmark setup
# Each group attends to its own K,V range (no overlap)
# → No L2 cache benefit possible
# → Speedup = 1.00x ✓

# This is a VALID use case (e.g., sequence-parallel attention)
# But it's NOT the use case for grouped kernel optimization!
```

---

## Conclusion

The **1.00x speedup for sequential launches** strongly suggests:

1. ✅ **Sequential execution is working** (not concurrent)
2. ✅ **No L2 cache invalidation between kernels**
3. ❌ **Groups are NOT sharing K,V tiles** (accessing different ranges)

The grouped kernel is designed for **zigzag ring attention** where:
- All groups attend to the **same K,V cache**
- Different groups attend to **overlapping portions**
- Example: Group 0 → K,V[0:4K], Group 1 → K,V[0:6K], Group 2 → K,V[0:8K]

If the benchmark tests **sequence-parallel attention** where:
- Each group attends to **disjoint K,V ranges**
- Example: Group 0 → K,V[0:2K], Group 1 → K,V[2K:4K], ...

Then **1.00x speedup is expected and correct** - there's no optimization opportunity!

**Please share the benchmark code so we can verify this hypothesis.**

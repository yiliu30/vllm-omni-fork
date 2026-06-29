# Wan2.2 Sparse Attention: XPU vs CUDA Implementation Gap Analysis

This document compares the XPU sparse attention implementation (`auto_round_kernel`) against the CUDA reference (`SpargeAttn-fork/spas_sage_attn`) and identifies remaining gaps after the routing-fill and geometry alignment fixes.

## Reference Points

- **CUDA**: `/data/model/yiliu7/SpargeAttn-fork/spas_sage_attn/`
  - Entry point: `core.py:spas_sage2_attn_meansim_topk_cuda`
  - Preprocess: `utils.py:get_block_map_meansim_fuse_quant` (fused Triton kernel)
  - Final kernel: `_qattn` C++ custom ops (per-architecture variants, FP8 V)

- **XPU**: `/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/`
  - Entry point: `sparse_attention.py:sparge_sage2_attn_meansim_topk_xpu`
  - Preprocess: `sparge_preprocess_triton.py:_run_triton_xpu_preprocess` (Triton-XPU) falls back to `_sparge_preprocess_topk_torch_impl`
  - Final kernel: `sage_sparse` (SYCL C++, FP16/BF16 V)

## Gap Status Legend

| Icon | Meaning |
|:----:|---------|
| ✅ | Aligned — no known remaining divergence |
| 🔴 | Unresolved — known or suspected divergence |
| ? | Unevaluated — needs direct comparison |

---

## 1. `smooth_k` Mean Subtraction

**Status: ✅ ALIGNED — no gap**

Both implementations identically compute per-head K mean and subtract it **inside** the pool/sim/quant kernel (not from the outer K tensor).

CUDA (`utils.py:184-188`):
```python
if fuse_mean:
    x_mean = tl.load(xm_ptrs)  # loads per-head K mean
    x -= x_mean
pool = tl.sum(x_fp32, axis=0) / BS_       # pool on de-meaned values
scale = tl.max(tl.abs(x_fp32)) / 127.      # quant on de-meaned values
```

XPU (`sparse_attention.py:684-688`):
```python
if mean_subtract is not None:
    x_blocks = x_blocks - mean_values
pooled = x_blocks.sum(dim=-2) / counts     # pool on de-meaned values
max_abs = x_blocks.abs().amax(...) / 127.   # quant on de-meaned values
```

The `# k = k - km` comment in `core.py` only means the outer K tensor reference is not modified — the mean subtraction still happens inside the fused kernel. **No fix needed.**

---

## 2. Preprocess Backend: Triton-XPU vs Torch Fallback

**Status: 🔶 WORTH MONITORING**

XPU has two preprocess backends that dispatch via `_sparge_preprocess_topk_dispatch`:

1. **Triton-XPU** (`sparge_preprocess_triton.py`): Mirrors CUDA's fused Triton kernel. Tried first.
2. **Torch** (`sparse_attention.py`): Pure PyTorch fallback. Used if Triton-XPU fails.

### Dispatch logic
```python
try:
    _ensure_triton_xpu_available(ctx.query, ctx.head_dim)
    result = _run_triton_xpu_preprocess(ctx)
    result["backend"] = "triton_xpu"
except (NotImplementedError, RuntimeError, ValueError, TritonError):
    if backend == "triton_xpu": raise
    _log_fallback_warning_once(error)   # silent fallback
    result = torch_backend()
    result["backend"] = "torch"
```

### Which backend was used
No "Triton-XPU fallback" warning appeared in any run log, so **Triton-XPU was used successfully** in both the topk=0.5 and topk=1 runs.

### Similarity computation precision

| Step | CUDA Triton | XPU Triton | XPU Torch |
|------|------------|------------|-----------|
| Pooling mean | Fused, FP32 accumulate | Fused, FP32 accumulate | `sum / counts`, FP32 |
| Similarity (Gram) | `tl.dot` (FP16) | `tl.dot` (FP16) | `torch.matmul` (FP32) |
| Quant rounding | `+0.5*where` → `to(int8)` | Same as CUDA | `floor/ceil` → `clamp` |
| Quant scale | `max(abs)/127 + 1e-7` | Same as CUDA | Same (1e-7 vs 1e-7) |

**Triton-XPU and CUDA Triton use identical math.** The Torch fallback diverges in similarity precision (FP32 instead of FP16) and quant rounding, but the fallback is not being hit.

---

## 3. K-View Quant-Block Granularity Difference

**Status: 🔴 STRUCTURAL DIFFERENCE — medium impact**

The quant block granularity for K differs between implementations:

### CUDA sm90
- Q: `BLKQ=64` → Q quantized in 64-token blocks
- K: `BLKK=128` → K quantized in 128-token blocks
- Routing: 64 × 128 tile grid (direct, no expansion needed)
- LUT: indexes 128-token K-blocks

### XPU (head_dim=128 after fix)
- Q: `quant_block_size=64` → Q quantized in 64-token blocks
- K: `quant_block_size=64` → K quantized in **64-token** blocks
- K routing: `k_route_block_tokens=128` → re-pooled to 128 tokens for routing
- Routing: tile-level map (64 Q × 128 K) → **expanded** to quant-block map (64 Q × 64 K) via `k_block_to_tile`
- LUT: indexes **64-token** K-blocks (after expansion)

### The pipeline

```
CUDA:
  K[128-token quant] ──→ routing (128-tile grid) ──→ LUT at 128-token ──→ kernel (CTA_K=128)
  Q[64-token quant]   ──→                           ──→               ──→

XPU:
  K[64-token quant]   ──→                          ──→ LUT at 64-token  ──→ kernel (quant_block_size=64)
                     └→ re-pool to 128-token for routing ─┘
                         block_map at tile level
                              ↓
                         expand via k_block_to_tile
                              ↓
                         block_map at quant-block level
```

### Impact on quality
- **Positive**: XPU's 64-token K quant blocks are **finer-grained** than CUDA's 128-token blocks, which should produce **less quantization noise** in K. This is a quality advantage for XPU.
- **Routing equivalence**: XPU's tile-level routing (128-token K blocks) routes at the same granularity as CUDA. The expansion back to 64-token blocks is a lossless transformation (each selected 128-token block maps to 2 × 64-token sub-blocks).
- **Net effect**: The metadata format differs structurally but should be mathematically equivalent for routing. The finer K quant may slightly improve quality.

---

## 4. Value (V) Quantization

**Status: 🔴 MEDIUM-LOW IMPACT**

### CUDA
On sm90 and sage2++ architectures, V is transposed, padded, and quantized to **FP8** with per-head-channel scaling (`scale_fuse_quant_cuda`):
```python
v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn)
v_scale = ...  # per-head-channel FP32 scale
```

### XPU
V is passed directly as **FP16/BF16** without quantization:
```python
sage_sparse(..., value, ...)
```

### Impact
- FP8 V quant saves memory bandwidth at the cost of quantization noise
- XPU not doing this means the final kernel uses higher-precision V
- For quality: XPU's approach is **better** (no V quantization error)
- For perf: XPU may be leaving bandwidth savings on the table

---

## 5. Routing: `topk` Only (No `cdfthreshd`)

**Status: 🔴 LOW IMPACT (for topk-only workloads)**

CUDA supports both routing modes:
- `topk`: Select top-K fraction of blocks
- `cdfthreshd`: Select blocks until CDF threshold is reached

XPU:
```python
if cdfthreshd is not None:
    raise NotImplementedError("cdfthreshd routing is not implemented yet")
```

Not a blocker for topk=0.5 testing, but prevents porting model-zoo hyperparameters that use `cdfthreshd`.

---

## 6. PV Thresholding (`pvthreshd`)

**Status: 🔴 LOW IMPACT (for current workloads)**

CUDA uses `pvthreshd` on sm80/sm86/sm87 architectures to skip low-contribution KV blocks during the PV phase. XPU ignores it. Not relevant for sm90 or Wan2.2.

---

## 7. Final Sparse Kernel

**Status: ? UNEVALUATED — needs direct comparison**

Both implementations consume the same interface (int8 Q/K, LUT, valid_block_num, scale factors) and produce HND-format outputs. However:

| Aspect | CUDA | XPU |
|--------|------|-----|
| Language | CUDA C++ (`_qattn`) | SYCL C++ (`lib.sage_sparse`) |
| V format | FP8 (sm90+) or FP16 | FP16/BF16 only |
| Accumulation | FP32 or FP16 per arch | Unknown |
| K quant block | 128 tokens | 64 tokens |
| LUT stride | 128-token K-blocks | 64-token K-blocks |
| Architecture variants | sm80, sm86, sm87, sm90 | Single implementation |

Without running the same preprocess metadata through both kernels, the numerical equivalence is unconfirmed. The different K quant-block sizes mean the kernels expect different scale/LUT layouts, so a direct A/B test would require adapting the metadata to each kernel's format.

---

## Closed Gaps

### ✅ Routing fill correctness
`_fill_block_map_torch` now tracks `is_new` entries beyond already-forced selections. CUDA's Triton kernel (`triton_fill_block_map_kernel`) always overwrites linearly. Both now produce correct results.

### ✅ Routing tile geometry for head_dim=128
| Parameter | CUDA sm90 | XPU (after fix) |
|-----------|----------:|----------------:|
| Q routing tile | 64 | 64 |
| K routing tile | 128 | 128 |
| Quant block size | 64 (Q) / 128 (K) | 64 (both) |

### ✅ `simthreshd1` default
Normalized from `-1.0` (Wan XPU patch) to `-0.1` (CUDA default), exposed as `SAGE_ATTN_SIMTHRESHD1` env var.

### ✅ `smooth_k` behavior
Both implementations subtract per-head K mean inside the pool/sim/quant kernel. No difference.

---

## Updated Assessment

After correcting the gap analysis, the remaining causes for a quality gap between topk=0.5 and topk=1 are limited:

1. **K quant-block granularity (64 vs 128)** — structural difference, but mathematically equivalent for routing. The finer XPU granularity is actually a quality advantage for quant precision. **Impact: low.**

2. **V quantization (FP8 vs FP16)** — XPU uses higher precision. **Impact: none on quality** (XPU should be at least as good).

3. **Missing `cdfthreshd` / `pvthreshd`** — not used in our config. **Impact: none.**

4. **Final kernel implementation** — different C++ code paths using different LUT/scale layouts. **Impact: unknown**, needs direct comparison.

The quality gap you observed most likely stems from the fact that **topk=1 is dense-equivalent** (all blocks selected, routing irrelevant), while **topk=0.5 activates the routing path** where any combination of small numerical differences in the preprocess can shift which blocks are selected at the margin. These marginal shifts compound across 30 transformer blocks × 40 denoising steps.

## Recommended Next Steps

### Priority 1: Build a metadata diff harness
Save Wan self-attention tensors and run both preprocess paths offline to compare routing metadata at each stage. This is the only way to isolate where the routing diverges.

| Output | CUDA symbol | XPU symbol |
|--------|------------|------------|
| Pooled Q (quant-block) | (inside fused kernel) | `pooled_q` |
| Pooled K (quant-block) | same | `pooled_k` |
| Pooled K (route-block) | `pooled_kblocks` (BLKK=128) | `pooled_k_for_routing` (k_route_block_tokens=128) |
| Similarity mask Q | `sim_qblocks` | `sim_qblocks` |
| Similarity mask K | `sim_kblocks` (BLKK=128) | `sim_k_for_routing` (128-token) |
| Routing scores | `pooled_score` | `pooled_score` |
| Block map (tile) | `final_map` | `final_tile_map` |
| Block map (quant) | `final_map` (no expansion) | `raw_block_map` (expanded) |
| LUT + valid_block_num | `lut`, `valid_block_num` | `lut`, `valid_block_num` |
| Q int8 + scale | `q_int8`, `q_scale` | `q_int8_hnd`, `q_scale` |
| K int8 + scale | `k_int8` (128-token quant), `k_scale` | `k_int8_hnd` (64-token quant), `k_scale` |

### Priority 2: Verify final kernel on identical metadata
Once the preprocess outputs are aligned, feed the same LUT/int8 tensors through both `_qattn` and `sage_sparse` kernels and compare numerical output.

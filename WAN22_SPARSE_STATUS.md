# Wan2.2 Sparse Attention: Status

## 2026-06-29 Cross-Attention Update

### Implemented

- Removed the ARK XPU preprocess guard that rejected `seq_len_q != seq_len_kv` in `/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/sparse_attention.py`.
- Added Wan sparse cross-attention opt-in in `/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wan_sparse_patch.py` via `WAN_SPARSE_ENABLE_CROSS_ATTN=1`, while keeping dense cross-attention as the default behavior.
- Added cross-attention coverage to `/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/test/test_sparge_preprocess_topk_e2e.py` for `Sq=256`, `Skv=128`, `Hq=2`, `Hkv=1`, `D=128`.
- Added a runtime helper opt-in path in `/data/model/yiliu7/vllm-omni/tmp_run_wan_per_role_attention.py` with `--attention-mode sparse_cross`.
- Updated `/data/model/yiliu7/vllm-omni/scripts/run_wan22_sparse_topk0p5_sink.sh` so `ATTENTION_MODE=sparse_cross` is available and `AUTO_ROUND_KERNEL_PATH` points at the local ARK source tree instead of the stale installed package copy.

### Verified

- Public ARK preprocess path now accepts cross-attention shapes and completes with Triton-XPU backend for `NHD` tensors shaped `Q=[1,256,2,128]`, `K/V=[1,128,1,128]`.
- The low-level XPU sparse kernel succeeds for the same cross-attention shape when driven by the generated ARK metadata.
- The updated wrapper test passes end-to-end, including the new unequal `Sq/Skv` case.

### Runtime Caveat Found

- The first reduced Wan smoke run failed because vLLM loaded `auto_round_kernel` from the installed site-packages copy, which still contained the old equal-length guard.
- The stack trace confirmed the failure came from `.venv/lib/python3.13/site-packages/auto_round_kernel/sparse_attention.py`, not from the modified local source tree.
- The script has been patched to use the local ARK source tree on subsequent runs.

## Current Implementation

The active code resides in two locations:

- **vLLM-Omni integration**: `vllm_omni/diffusion/attention/backends/sage_attn.py` (working tree)
- **ARK preprocess + kernel**: `/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/`
- **PyTorch reference kernel**: `vllm_omni/diffusion/attention/backends/sage_sparse_ref.py` (new)

### Implemented fixes

| Fix | File | Status |
|-----|------|--------|
| Ranked top-k fill semantics (track `is_new` entries) | `sparse_attention.py:622-635` | ✅ Verified |
| Wan head_dim=128 routing geometry (q=64, k=128) | `sparse_attention.py:557-570` | ✅ Verified |
| simthreshd1 normalized to -0.1 | `sage_attn.py:53` | ✅ Active |
| ARK import path resolution | `sage_attn.py:20-27` | ✅ Active |
| Per-role attention (self=SAGE_ATTN, cross=TORCH_SDPA) | Config/env | ✅ Active |
| Sparse attention env-var control (topk, sink, smooth_k, simthreshd1) | `sage_attn.py:47-54` | ✅ Active |
| **Q-routing granularity: SYCL sparse Q tile 256/128 → 64 (match CUDA CTA_Q=64)** | `sycl_tla_sdpa.hpp` (both launchers) | ✅ Verified (cos SYCL~ref topk=0.5: 0.71→1.000) |
| **Partial tail-block sim → False (match CUDA NaN-forces-dense)** | `sparge_preprocess_triton.py:111`, `sparse_attention.py:696` | ✅ Verified (tail sim True→False; tail K-col select 0.60→1.00) |
| Removed orphaned duplicate `logger.info` tail that broke import | `sage_attn.py:353` | ✅ Fixed |

## Validation Results

### Passed: Runtime

| Config | Frames | Steps | Time | Result |
|--------|------:|------:|-----:|:------:|
| Dense SDPA, 240x416 | 9 | 2 | 33.1s | ✅ Clean output |
| Sparse SYCL topk=0.5+sink, 240x416 | 9 | 2 | 33.8s | ✅ Clean output |
| Sparse SYCL topk=0.5+sink, 480x832 | 81 | 40 | 1153s | ✅ Clean output |
| Sparse SYCL topk=1.0, 480x832 | 81 | 40 | 1238s | ✅ Clean output |
| Sparse ref kernel topk=0.5+sink, 240x416 | 9 | 2 | 1477s | ⚠️ Noisy output |

### Key Findings

1. **SYCL kernel at topk=1.0 works well** — output quality matches dense SDPA
2. **SYCL kernel at topk=0.5 with sink works well for video** — confirmed acceptable quality
3. **PyTorch ref kernel at topk=1.0 matches SYCL kernel** (cosine sim = 0.999995) — confirms the ref kernel math is correct
4. **PyTorch ref kernel at topk=0.5 diverges from SYCL kernel** (cosine sim = 0.71-0.76) — same preprocess metadata, different output

### RESOLVED: Root cause = Q-routing granularity divergence

**The earlier hypothesis below was wrong.** The SYCL kernel *does* follow the
LUT — it read **one routing row (block-0) for an entire 256-token Q tile**,
discarding 3 of every 4 per-64-token routing rows. CUDA's `CTA_Q=64` reads one
row per 64-token block. At topk=1.0 all 4 rows are identical (cosine 0.999995);
at topk=0.5 they differ, so 3/4 of every tile was routed to the wrong K blocks
(cosine 0.71–0.76).

Full analysis and fix: **[Root Cause: Q-Routing](WAN22_SPARSE_ROOT_CAUSE_Q_ROUTING.md)**.

Fix (implemented): shrink the sparse Q tile to 64 tokens in both launchers
(`sycl_tla_sdpa.hpp`) so `q_blocks_per_tile = 1` and each tile reads its own LUT
row — matching CUDA exactly. SGTileQ stays 16, so MMA math is unchanged.

<details>
<summary>Superseded hypothesis (kept for history)</summary>

The ref kernel strictly follows the LUT, excluding ~46% of blocks... the SYCL
kernel may internally process K in larger tiles / apply implicit smoothing.
**This was incorrect** — the divergence was on the Q axis (tile read one row for
four blocks), not the K axis or V handling.
</details>

### Dump Infrastructure

Added dump capability to `sage_attn.py`:
- `SAGE_ATTN_DUMP_DIR=/path` saves preprocess inputs and outputs per layer
- Combined with `SAGE_ATTN_REF_KERNEL=1` to capture data
- Intended for offline comparison with CUDA preprocess outputs

The dump run failed due to orchestrator init timeout (likely the dump overhead in the init phase).

## Unresolved Items

1. **Routing quality**: XPU routing selects different blocks than CUDA at same topk. Need to dump and compare.
2. **K quant-block granularity**: XPU uses 64-token K quant blocks; CUDA uses 128-token. Different quant noise characteristics.
3. **V handling**: SYCL kernel may handle V loading differently than the ref kernel's explicit V gather.
4. **The dump infrastructure needs refinement**: The current dump approach causes orchestrator init failures. Need a lighter-weight capture strategy.

## Related Documents

- [XPU vs CUDA Gap Analysis](WAN22_SPARSE_XPU_VS_CUDA_GAP_ANALYSIS.md) — Detailed code-level comparison
- [Quality Risk Findings](WAN22_SPARSE_QUALITY_RISK_FINDINGS.md) — Preprocess quality risks

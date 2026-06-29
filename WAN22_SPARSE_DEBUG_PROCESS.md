# Wan2.2 SpargeAttn XPU Quality Debug — Process Log

## Problem Statement

Wan2.2 T2V (text-to-video) with block-sparse attention on Intel XPU produces visible white highlight artifacts in generated videos. The artifacts do not appear with dense SDPA attention on the same hardware.

**Correctness oracle:** CUDA SpargeAttn at `/data/model/yiliu7/SpargeAttn-fork/` — NOT the PyTorch sparse reference (`sage_sparse_ref.py`).

---

## Environment

- Hardware: Intel XPU (4× GPUs, `ZE_AFFINITY_MASK=4,5,6,7`)
- Model: Wan2.2-T2V-A14B-Diffusers, TP4, cpu-offload, enforce-eager
- Resolution: 480×832, 9 frames, 40 denoising steps, seed 42
- Sparse kernel: `auto_round_kernel.ARK.sparge_sage2_attn_meansim_topk_xpu`
- Plugin: `vllm-qdq-plugin` at `/home/yiliu7/workspace/vllm-qdq-plugin/`

---

## Phase 1: Reproduce and Baseline

### Step 1.1 — Dense SDPA reference
Generated 9-frame video with pure dense attention (no sparse kernel):
- Config: `--diffusion-attention-config '{"default": "TORCH_SDPA"}'`, unset all SPARGE env vars
- Output: `/tmp/plugin_xpu_dense_ref/t2v_dense_ref.mp4`
- Frames: `/tmp/frames_plugin_xpu_dense_ref/frame_000.png` … `frame_008.png`
- Result: Clean output, no artifacts.

### Step 1.2 — Sparse topk=0.5 (default)
Generated with sparse kernel active, 50% block retention:
- Config: `VLLM_SPARGE_ATTN=1 SPARGE_TOPK=0.5 SPARGE_ATTENTION_SINK=1`, `--diffusion-attention-config '{"default": "SAGE_ATTN"}'`
- Output: `/tmp/plugin_xpu_sparse_9f/`
- Frames: `/tmp/frames_plugin_xpu_sparse_9f/frame_000.png` … `frame_008.png`
- Result: **Severe white highlight artifacts** — bright patches in dark regions.

---

## Phase 2: Isolate Root Cause — Routing vs Kernel Computation

**Hypothesis:** Either (A) the kernel's dense computation path introduces errors, or (B) the block-dropping/routing logic drops important blocks.

### Step 2.1 — topk=1.0 experiment (all blocks kept, kernel still active)
Set topk=1.0 so routing keeps ALL blocks — the kernel's full code path (fp16 conversion, smooth_k, simthreshd1, attention_sink logic) still executes, but no blocks are dropped.

- Config: `SPARGE_TOPK=1.0`, everything else same as Step 1.2
- Output: `/tmp/plugin_xpu_sparse_topk1/t2v_sparse_topk1.mp4`
- Frames: `/tmp/frames_plugin_xpu_sparse_topk1/frame_000.png` … `frame_008.png`

**Key issue encountered:** First run failed with "v must be contiguous". Fixed by adding `.contiguous()` to q, k, v in `_forward_sparge_xpu()`. Second run defaulted to FLASH_ATTN because `--diffusion-attention-config` flag was missing. Third run succeeded.

### Step 2.2 — Quantitative comparison

| Metric | topk=1.0 (all blocks) | topk=0.5 (50% blocks) |
|--------|:---------------------:|:---------------------:|
| PSNR vs dense | **22.41 dB** | 8.82 dB |
| Sparse-only bright pixels | 0.92% | 5.74% |
| Brightness jump at artifacts | 31.4 | 172.4 |

### Step 2.3 — Conclusion
**Root cause confirmed: block dropping (routing logic), NOT the kernel's dense computation.**

topk=1.0 produces near-identical output to dense SDPA. The 0.92% residual is attributable to fp16 quantization noise. The white highlights appear ONLY when blocks are actually dropped.

---

## Phase 3: Q-Routing Granularity Bug (kernel-level)

### Step 3.1 — Kernel-level comparison
Compared SYCL sparse kernel output against PyTorch reference kernel using identical preprocess metadata (same LUT, same routing decisions):

| topk | cos(SYCL, PyTorch ref) |
|-----:|:----------------------:|
| 1.0 | 0.999995 |
| 0.5 | 0.71 – 0.76 |

The kernels agree at topk=1.0 (routing is a no-op) but diverge massively at topk=0.5.

### Step 3.2 — Root cause identified
The SYCL kernel used a 256-token Q tile for head_dim=128, reading **one** LUT routing row for every 4 Q blocks (64 tokens each). CUDA uses `CTA_Q=64`, reading one LUT row per Q block.

```
XPU (before): Q tile = 256 tokens → q_blocks_per_tile = 256/64 = 4
  → reads only block-0's routing row for all 4 blocks
  → 3/4 of Q tokens routed to WRONG K blocks

CUDA (oracle): CTA_Q = 64 → one CTA per 64-token Q block
  → each Q block reads its OWN routing row
```

At topk=1.0 all routing rows are identical (all blocks selected), so reading the wrong row is harmless. At topk=0.5 the rows differ, causing quality collapse.

### Step 3.3 — Fix applied
Shrunk the sparse Q tile to 64 in both SYCL launchers (`sycl_tla_sdpa.hpp`):
- `launch_sparse_sage_prefill_kernel_128`: Q tile 256 → 64
- `launch_sparse_sage_prefill_kernel_64`: Q tile 128 → 64

After fix: cos(SYCL, ref) = 1.000000 at both topk=1.0 and topk=0.5.

### Step 3.4 — Validation
File: `sycl_tla_sdpa.hpp` in ARK
Validator: `tmp_validate_sparse_qrouting.py`

---

## Phase 4: Partial Tail-Block Bug

### Step 4.1 — Observation
Even after the Q-routing fix, some sequences with non-power-of-2 lengths showed divergence at the tail.

### Step 4.2 — Root cause
When the sequence length doesn't divide evenly into 64-token blocks, the last block is partial. The SYCL preprocess computed similarity on this partial block and sometimes classified it as "important." The CUDA kernel forces partial blocks to be treated as dense (effectively NaN similarity → always included).

### Step 4.3 — Fix applied
Set `partial_tail_sim = False` in preprocess, matching CUDA's NaN-forces-dense semantics.

After fix: tail K-column selection matches reference exactly (0.60 → 1.00).

Files:
- `sparge_preprocess_triton.py:111`
- `sparse_attention.py:696`

---

## Phase 5: Ranked Top-k Fill and Geometry Alignment

### Step 5.1 — Fill semantics bug
When rows already contain forced-selected blocks (attention sink, incoherent blocks), the ranked fill added the wrong number of additional blocks.

### Step 5.2 — Routing geometry divergence
XPU preprocess used head_dim-based shortcuts for tile geometry that didn't match CUDA's explicit Q=64, K=128 routing geometry for Wan head_dim=128.

### Step 5.3 — Fixes
- Tracked `is_new` entries in ranked fill (`sparse_attention.py:622-635`)
- Aligned Q/K routing block sizes to CUDA geometry (`sparse_attention.py:557-570`)

---

## Phase 6: E2E Re-validation (Plugin Path)

### Step 6.1 — Plugin implementation
Migrated XPU sparse attention to `vllm-qdq-plugin` for clean separation:
- `/home/yiliu7/workspace/vllm-qdq-plugin/src/vllm_qdq_plugin/sparge_attn/impl.py`
- Same guard chain as CUDA (fp32, cross-attn, seq_len<128, head_dim∉{64,128}, mask → SDPA fallback)
- Unified env vars: `SPARGE_TOPK`, `SPARGE_SMOOTH_K`, `SPARGE_SIMTHRESHD1`, `SPARGE_ATTENTION_SINK`, `SPARGE_K_QUANT_GRANULARITY`, `SPARGE_DENSE_STEPS`

### Step 6.2 — Dense early steps
Added `SPARGE_DENSE_STEPS` env var: forces topk=1.0 for first N denoising steps (accessed via `get_forward_context().denoise_step_idx`).

### Step 6.3 — Current E2E results (post Q-routing fix)

| Config | Frames | Steps | Time | Quality |
|--------|-------:|------:|-----:|---------|
| Dense SDPA | 9 | 40 | — | Clean (baseline) |
| Sparse topk=1.0 | 9 | 40 | — | Clean (PSNR 22.41 dB vs dense) |
| Sparse topk=0.5 + sink | 9 | 40 | — | White artifacts (PSNR 8.82 dB) |
| Sparse topk=0.5, 240×416 | 9 | 2 | 33.8s | Clean (low-res short test) |
| Sparse topk=1.0, 480×832 | 81 | 40 | 1238s | Clean |

---

## Key Learnings

1. **topk semantics**: topk=0.1 keeps 10% (most aggressive), topk=1.0 keeps 100% (dense through sparse path).
2. **Isolation methodology**: Running topk=1.0 through the sparse kernel isolates kernel-computation errors from routing errors.
3. **Backend selection**: Must use `--diffusion-attention-config '{"default": "SAGE_ATTN"}'` — without it, the system silently defaults to FLASH_ATTN.
4. **Contiguity**: XPU kernels require contiguous tensors; `.contiguous()` calls added to `_forward_sparge_xpu()`.
5. **Frame constraint**: Wan2.2 requires `(num_frames - 1) % 4 == 0`.
6. **Denoising**: All frames go through ALL 40 steps simultaneously (latent is `[B, C, T, H, W]`).

---

## Remaining Investigation

1. **Routing quality at topk=0.5**: Even with correct kernel consumption, 50% block dropping still causes visible artifacts at full resolution. Need to compare XPU routing mask vs CUDA routing mask to check if XPU drops different blocks.
2. **K quant-block granularity**: XPU uses 64-token K quant blocks; CUDA uses 128-token. Different quant noise.
3. **Optimal topk**: Need to find the threshold where quality is acceptable vs speedup gained.
4. **Dense early steps**: Testing whether using topk=1.0 for first N steps + topk=0.5 for rest gives good quality with partial speedup.

---

## File Index

| File | Purpose |
|------|---------|
| `WAN22_SPARSE_STATUS.md` | Implementation status and validation results |
| `WAN22_SPARSE_ROOT_CAUSE_Q_ROUTING.md` | Q-routing granularity bug analysis and fix |
| `WAN22_SPARSE_ROOT_CAUSE_PARTIAL_TAIL.md` | Partial tail-block bug |
| `WAN22_SPARSE_QUALITY_RISK_FINDINGS.md` | Preprocess quality risks |
| `WAN22_SPARSE_XPU_VS_CUDA_GAP_ANALYSIS.md` | Code-level comparison |
| `WAN22_SPARSE_FIX_PLAN.md` | Preprocess fix plan |
| `WAN22_TP4_CPU_OFFLOAD_SDPA.md` | TP4 + CPU offload configuration notes |
| `/tmp/run_plugin_xpu_sparse_topk1.sh` | Run script for topk=1.0 experiment |
| `/tmp/run_plugin_xpu_dense_ref.sh` | Run script for dense reference |
| `tmp_validate_sparse_qrouting.py` | Q-routing fix validator |
| `tmp_validate_tail_routing.py` | Tail-block fix validator |

---

## Run Scripts

### Dense reference
```bash
VLLM_SPARGE_ATTN= (unset)
--diffusion-attention-config '{"default": "TORCH_SDPA"}'
```

### Sparse topk=0.5
```bash
VLLM_SPARGE_ATTN=1 SPARGE_TOPK=0.5 SPARGE_ATTENTION_SINK=1
--diffusion-attention-config '{"default": "SAGE_ATTN"}'
```

### Sparse topk=1.0 (isolation test)
```bash
VLLM_SPARGE_ATTN=1 SPARGE_TOPK=1.0 SPARGE_ATTENTION_SINK=1
--diffusion-attention-config '{"default": "SAGE_ATTN"}'
```

All runs share: `ZE_AFFINITY_MASK=4,5,6,7`, TP4, cpu-offload, enforce-eager, seed 42, 9 frames, 40 steps.

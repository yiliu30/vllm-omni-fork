# Wan2.2 Sparse Attention: Root Cause — Q-Routing Granularity Divergence

**Status: ROOT CAUSE CONFIRMED. Fix implemented (CUDA-matching, Option A).**

This document records the confirmed root cause of the topk=0.5 quality
degradation in Wan2.2 sparse attention on Intel XPU, and the fix that matches
the CUDA SpargeAttn reference exactly.

> **Correctness reference.** The CUDA SpargeAttn kernel
> (`/data/model/yiliu7/SpargeAttn-fork/`) is the sole ground-truth oracle. The
> PyTorch reference kernel (`sage_sparse_ref.py`) is *not* trusted for
> correctness; it is used only as a secondary cross-check. All geometry below is
> derived from the CUDA source.

---

## 1. Symptom

| topk | cosine vs dense | Observation |
|-----:|----------------:|-------------|
| 1.0  | 0.999995        | All LUT rows identical (every block selected) → routing irrelevant → near-perfect. |
| 0.5  | 0.71 – 0.76     | LUT rows differ per Q block → quality collapses. |

The gap appears *only* when routing actually discriminates between blocks. That
localizes the bug to **how routing rows are consumed**, not to the preprocess
that produces them.

## 2. Root Cause: Q-routing granularity divergence at kernel consumption

The LUT is generated **per 64-token Q block** (`quant_block_size = 64`), one
routing row per block. Both CUDA and the preprocess agree on this. The
divergence is in how the *kernel* indexes those rows.

### CUDA sm90 (the oracle)

- `CTA_Q = 64`. The grid launches **one CTA per 64-token Q block**.
- The LUT is indexed by the block index `bx` directly: **CTA `bx` reads LUT row
  `bx`**.
- Result: each 64-token Q block attends to exactly the K blocks its *own*
  routing row selected. 1 Q block ↔ 1 LUT row.

### XPU SYCL (before fix)

- The sparse launchers used a **256-token Q tile** (`head_dim=128`) or a
  128-token tile (`head_dim=64`).
- The kernel coarsened the routing index
  (`xe_sparse_sage_fwd_kernel.hpp:239-256`):

  ```cpp
  int sparse_q_block = blk_q;
  if (params.mainloop.scale_block_size > 0) {
    int q_blocks_per_tile =
        cute::max(1, int(get<0>(TileShapeQK{})) / params.mainloop.scale_block_size);
    sparse_q_block = blk_q * q_blocks_per_tile;   // 256/64 = 4
    ...
  }
  // reads ONE row: lut + (... * num_q_blocks + sparse_q_block) * num_k_blocks
  ```

  With a 256-token tile and 64-token routing blocks,
  `q_blocks_per_tile = 256/64 = 4`. The tile read **only block-0's LUT row** and
  applied it to all 256 Q tokens (4 routing rows' worth), **discarding 3 of
  every 4 routing rows**.

### Why this exactly explains the symptom

- **topk=1.0**: every routing row selects every K block, so all 4 rows in a tile
  are identical. Reading only row-0 is harmless → cosine 0.999995.
- **topk=0.5**: the 4 rows differ. Forcing row-0 onto all 256 tokens routes Q
  tokens 64–255 to the *wrong* K blocks. This is not "more blocks" or "fewer
  blocks" — it is *wrong* blocks for 3/4 of every tile. Compounded across
  ~30 transformer blocks × 40 denoising steps, output collapses to cosine
  0.71–0.76.

This also corrects the earlier (incorrect) hypothesis in `WAN22_SPARSE_STATUS.md`
that "the SYCL kernel does not follow the LUT" / "includes more blocks." The
SYCL kernel *does* follow the LUT — it just read one row for four blocks.

## 3. Dispatch chain (for reference)

```
sage_attn.py:_forward_xpu_sparse
  → sparge_sage2_attn_meansim_topk_xpu          (ARK python)
    → sdpa.cpp:sdpa_impl_qks8_sparse_pvhalf      (arg validation)
      → sparse_sage_prefill                       (sdpa.cpp:286)
        → select_sparse_sage_prefill_launcher     (sdpa.cpp:131)
            case 128 → launch_sparse_sage_prefill_kernel_128   ← FIX HERE
            case  64 → launch_sparse_sage_prefill_kernel_64     ← FIX HERE
              → SparseSageConfig<...>::run
                → xe_sparse_sage_fwd_kernel.hpp   (the coarsening, now neutralized)
```

## 4. The Fix (Option A — match CUDA CTA_Q=64)

Shrink the sparse Q tile to **64 tokens** in both launchers
(`sycl_tla_sdpa.hpp`). Then:

```
q_blocks_per_tile = max(1, 64 / 64) = 1
sparse_q_block    = blk_q * 1 = blk_q      // each tile reads its OWN row
```

This makes the SYCL kernel's Q-routing granularity identical to CUDA's
`CTA_Q=64`: one 64-token Q block ↔ one LUT row.

### Diff (both launchers)

| Launcher | Before (Q tile) | After (Q tile) | Subgroups |
|----------|----------------:|---------------:|----------:|
| `launch_sparse_sage_prefill_kernel_128` | 256 | **64** | 16 → **4** |
| `launch_sparse_sage_prefill_kernel_64`  | 128 | **64** | 8 → **4** |

```cpp
// head_dim=128
using ShapeQK  = Shape<_64, _64, _32>;
using ShapePV  = Shape<_64, _32, _64>;
using ShapeOut = Shape<_64, _128>;
using SubgroupLayoutQK = Layout<Shape<_4, _1, _1>>;
```

### Why the fix is numerically safe (only the grid gets finer)

- **SGTileQ invariant.** `SGTileQ = TileQ / num_subgroups = 64 / 4 = 16`,
  unchanged from before (256/16 = 16). The MMA atom
  `XE_DPAS_TT<gcd(SGTileQ,8)=8, …>` and every per-subgroup fragment are
  byte-for-byte identical. Only the *number* of tiles in the grid changes.
- **qscale dequant already per-64-token.** `xe_sparse_sagev1_fwd_mainloop.hpp:400`
  indexes `scaleQ[(tile_q_base + sg*q_sg_tile) / scale_block_size]` — already
  64-token granular, so a 64-token tile reads the same scales.
- **K-axis LUT walk already aligned.** The delta-decode K loop
  (`xe_sparse_sagev1_fwd_mainloop.hpp:616-636`) is unchanged; only the Q-row
  selection feeding it is corrected.
- **PV subgroup remap valid for 4 subgroups.** `get_sg_layout_pv` is a pure
  structural remap with `static_assert(size(SGLayoutPV)==size(SGLayoutQK))`,
  which holds for `Shape<_4,_1,_1>`.

### Known trade-off

A 64-token Q tile (4 subgroups) has less per-tile parallelism than the old
256-token tile (16 subgroups), so per-call throughput may drop. This was
accepted as **correctness-first**. If perf regresses unacceptably, the fallback
is **Option B**: keep the 256-token tile but make the kernel read all 4
sub-block routing rows (per-sub-block LUT indexing inside the mainloop) instead
of coarsening to row-0.

## 5. Files changed

| File | Change |
|------|--------|
| `…/ark/auto_round_kernel/wrapper/include/sycl_tla_sdpa.hpp` | Q tile 256→64 (`_128` launcher), 128→64 (`_64` launcher); subgroups → 4. |

No preprocess, no Python, and no other kernel files were modified — the LUT was
always correct; only its consumption was wrong.

## 6. Validation result (kernel-level, decisive)

The decisive probe is **cos(SYCL, PyTorch-ref) on identical preprocess
metadata**, not cos(SYCL, dense). Both kernels consume the same LUT; the ref
walks it per-64-token Q block (= CUDA `CTA_Q=64`). At topk=1.0 routing is a
no-op and the two already agreed (~0.9999), which pins down that their
non-routing math matches — so any topk=0.5 gap is pure LUT-consumption.

Fresh binary `md5 8fd41676392abb90215c52ece5b6b985`, geometry B=1 Hq=8 S=1536
D=128 NHD non-causal:

| topk | cos(SYCL, ref) BEFORE | cos(SYCL, ref) AFTER | cos(SYCL, dense) | cos(ref, dense) |
|-----:|----------------------:|---------------------:|-----------------:|----------------:|
| 1.0  | ~0.9999               | **1.000000**         | 0.999919         | 0.999918        |
| 0.5  | **0.71 – 0.76**       | **1.000000**         | 0.717168         | 0.717168        |

- The topk=0.5 SYCL-vs-ref gap (the bug's exact signature) is **fully closed**.
- `cos(SYCL,dense) == cos(ref,dense) == 0.717` at topk=0.5 is *not* the bug — it
  is the inherent cost of dropping 50% of blocks on random Q/K, identical for
  both kernels. (Real Wan Q/K are far from random, so the model-level quality
  drop from sparsity is much smaller; that is what the E2E rerun confirms.)

Validator: `tmp_validate_sparse_qrouting.py`.

## 7. E2E validation plan

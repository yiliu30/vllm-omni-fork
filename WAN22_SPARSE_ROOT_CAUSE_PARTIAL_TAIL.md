# Wan2.2 Sparse Attention: Root Cause #2 — Partial Tail-Block Routing Divergence

**Status: ROOT CAUSE CONFIRMED (both sides proven). Fix implemented (CUDA-matching).**

This is a **second, independent** root cause, separate from the Q-routing
granularity fix ([Root Cause #1](WAN22_SPARSE_ROOT_CAUSE_Q_ROUTING.md)). It
explains the residual **end-of-frames** mismatch versus CUDA that remained after
the Q-routing fix.

> **Correctness reference.** The CUDA SpargeAttn kernel
> (`/data/model/yiliu7/SpargeAttn-fork/`) is the sole ground-truth oracle. The
> divergence below is XPU departing from CUDA; the fix makes XPU match CUDA.

---

## 1. Symptom

After Root Cause #1 was fixed, the Wan2.2 sparse output matched CUDA across most
of the video but **drifted at the last frames**. The earlier kernel-level
validation (cos SYCL~ref = 1.000000) used **S=1536**, which is a clean multiple
of 64 and 256 — so it had **no partial block** and was structurally blind to any
tail-specific bug.

## 2. The geometry that triggers it

Real Wan 480×832×81 latent sequence:

```
latent T=21  H=30  W=52   ->   S = 21*30*52 = 32760
32760 / 64  = 511 full 64-blocks + 56  -> LAST 64-block is PARTIAL (56 valid)
32760 / 128 = 255 full tiles    + 120  -> LAST 128 K-route tile is PARTIAL
```

The flattened (T,H,W) order places that partial block at the **end of the
sequence = the last frames**. Any tail-block routing error shows up there.

## 3. Root Cause: partial-block self-similarity differs (NaN vs guarded)

Both implementations compute a per-block mean self-similarity to decide if a
block is "similar enough" to be routed sparsely. `sim=False` forces a block
**dense** (`final_map[~sim] = 1` selects it for every query). The two differ on
how a **zero-padded partial block** is normalized:

### CUDA oracle (`spas_sage_attn/utils.py:181-197`)

```python
x = tl.load(x_ptrs, mask=xmask)          # padded rows present
...
x = tl.where(xmask, x, 0)                 # padded rows -> 0   (smooth_k path)
x_norm = tl.sqrt(tl.sum(x*x, axis=1, keep_dims=True))   # padded row norm -> 0
x = (x / x_norm).to(tl.float16)           # padded row -> 0/0 = NaN   (NO guard)
grams = tl.dot(x, tl.trans(x))            # NaN propagates
sum_value = tl.sum(grams)                 # -> NaN
cur_sim = (sum_value / (BS_*BS_)) > cur_h1   # NaN > thr == False
```

CUDA has **no zero-norm guard**, so every partial block yields `sim = False` →
**forced dense**.

### XPU (before fix) (`sparge_preprocess_triton.py:107`)

```python
x_norm = tl.where(x_norm > 0, x_norm, 1.0)   # GUARD -> finite
x_normed = x_fp32 / x_norm                   # padded row -> 0, finite
... cur_sim = (sum_value / (bs_eff*bs_eff)) > cur_h1   # REAL value (usually True)
```

XPU's guard makes the partial block `sim = True` → **pruned by topk**.

### Net effect

| | partial tail block | tail K-tile | tail frames |
|---|---|---|---|
| CUDA | sim=False → **dense** | sim=False → **dense** | full attention |
| XPU (before) | sim=True → **pruned** | sim=True → **pruned** | sparsified |

CUDA gives the last frames full dense attention; XPU sparsified them → the
end-of-frames mismatch.

## 4. Evidence (both sides proven)

**XPU, real kernel on hardware** (`tmp_validate_tail_routing.py`, partial-tail
geometry S=1592 → last 64-block has 56 valid tokens):

| metric | before fix | after fix | CUDA target |
|---|---|---|---|
| tail `sim_q` (last block) | **True** | **False** | False |
| tail `sim_k` (last tile) | **True** | **False** | False |
| P(tail K-col selected, all q rows) | **0.600** | **1.000** | 1.000 (forced dense) |
| tail q-row valid_block_num | **11** | **25 (=all)** | all (forced dense) |
| interior q-row valid_block_num | 11.4 | 13.0 | sparse (unchanged) |

**CUDA, line-faithful PyTorch port** (`tmp_prove_cuda_tail_nan.py`, the NaN is
IEEE-deterministic so no CUDA HW is needed):

| block | valid tokens | CUDA sim | XPU sim (before) |
|---|---|---|---|
| FULL | 64 | True | True (**match**) |
| PARTIAL | 56 | **False** | True (**differ**) |
| PARTIAL | 32 | **False** | True (**differ**) |
| PARTIAL | 8 | **False** | True (**differ**) |

Full blocks already matched; only **partial** blocks diverged — exactly the tail.

## 5. The Fix (match CUDA: partial block → sim=False)

Force any block that is not full (`bs_eff < BS`, i.e. `pad_tokens > 0`) to
`sim=False`, reproducing CUDA's NaN-forces-dense outcome without relying on
actual NaN arithmetic.

| File | Change |
|------|--------|
| `…/auto_round_kernel/sparge_preprocess_triton.py` (Triton-XPU runtime path) | `cur_sim = cur_sim & (bs_eff >= BS)` after the sim compare. |
| `…/auto_round_kernel/sparse_attention.py` (torch parity path) | `if pad_tokens: sim_blocks[:, :, -1] = False`. |

No C++ rebuild required — the preprocess is Triton-JIT.

### Why the fix is safe for aligned sequences

When `S` is a multiple of the block size (no partial block), `bs_eff == BS` for
every block and the guard is a no-op. The aligned S=1536 differential validator
still passes at **cos(SYCL, ref) = 1.000000** for both topk=1.0 and topk=0.5 —
no regression to Root Cause #1.

## 6. Relationship to Root Cause #1

These are orthogonal:

- **#1 (Q-routing granularity):** the *kernel* read one LUT row per 256-token
  tile instead of per 64-token block. Affected the whole sequence at topk=0.5.
  Fixed in `sycl_tla_sdpa.hpp`.
- **#2 (partial tail block):** the *preprocess* built a different LUT than CUDA
  for the final partial block. Affected only the sequence tail (last frames).
  Fixed in `sparge_preprocess_triton.py` + `sparse_attention.py`.

#1's validation (S=1536, aligned) could not have caught #2; #2 needs a
non-64-aligned sequence to manifest.

## 7. E2E validation (run 2026-06-24)

Full Wan2.2 sparse topk=0.5 + sink generation (480×832, 81 frames, 40 steps,
seed=42), tail-fix live in the import path. Output:
`/tmp/t2v_480p_sparse_topk0p5_tailfix.mp4` vs the tail-**un**fixed
`/tmp/t2v_480p_sparse_topk0p5_qrouting_fix.mp4` (same prompt/seed/config; the
**only** code delta is the one-line partial-tail fix).

> **Import-path gotcha (fixed before the run).** The launch script sets
> `AUTO_ROUND_KERNEL_PATH` to the venv package. Its compiled `.so` was fresh
> (Q-routing fix, Jun 24 07:56) but its **Python** preprocess files were stale
> (Jun 23) and lacked the tail fix. Synced `sparge_preprocess_triton.py` +
> `sparse_attention.py` into the venv and cleared bytecode so the run actually
> exercised the fix.

| metric | BEFORE (tail-unfixed) | AFTER (tail-fixed) | reading |
|---|---|---|---|
| SSIM before↔after | — | **0.78 mean** | same scene; differences are sampling chaos, not corruption |
| MSE vs dense (mean) | 3271.6 | **3230.9** | AFTER closer to dense |
| MSE vs dense (tail 5) | 4657.2 | **4632.9** | AFTER closer to dense at tail |
| SSIM vs dense (mean) | 0.5731 | **0.5975** | AFTER closer to dense |
| SSIM vs dense (tail 5) | 0.5017 | **0.5274** | AFTER closer to dense at tail |

**What E2E can and cannot show.** topk=0.5 routing feeds a 40-step diffusion
loop; a one-block step-0 routing change amplifies chaotically, so the two videos
differ by SSIM≈0.78 across **all** frames regardless of where the routing change
originated. Consequently **pixel-MSE cannot isolate the tail effect** — the
naive "last frames will be the worst" prediction does **not** survive chaos
(observed: tail/interior MSE ratio 0.78×, worst frame = 60 not 80). The only
clean directional signal is *dense-closeness*, which improves on both metrics,
both overall and at the tail — consistent with the fix pushing the partial
blocks dense the way CUDA does.

**The decisive correctness evidence is therefore the routing-level match in
§4, not this E2E.** The IEEE-faithful CUDA port and the on-HW tail probe test
exactly the quantity the fix changes (the sim flag / LUT) with zero chaos; the
E2E run's role is to confirm (a) the fix runs to completion with no instability,
(b) output stays a coherent same-scene video, (c) it moves *toward* dense — all
satisfied.

**Cleaner output-level test (not yet run).** To confirm the tail effect at the
output level without chaos, compare a **single denoising step** from an
identical latent (before vs after preprocess) — this removes the 40-step
amplification. Deferred; the routing-level proof is already decisive.

Repro: `bash /tmp/run_wan_tailfix.sh`; analysis:
`python tmp_compare_tailfix_frames.py --before … --after … --dense …`.

## 8. Chaos-free output-level confirmation: sparse vs dense SDPA (run 2026-06-24)

The E2E in §7 cannot localize the tail because of diffusion chaos. A **single
attention call** has no such amplification, so its per-query-row error vs a dense
golden reference is a clean, position-resolved signal.

**Golden reference = `torch.nn.functional.scaled_dot_product_attention`** (plain
fp16 full attention). This is *not* the untrusted `sage_sparse_ref` (which has its
own routing); dense SDPA has no routing/LUT/INT8 — it is the exact quantity sparse
approximates, so it is a legitimate golden reference and does not conflict with
"CUDA is the routing oracle."

**Geometry** (`tmp_tail_diff_vs_sdpa.py`): B=1, Hq=16, **S=4216, D=128**. Chosen so
`4216 % 64 = 56` and `4216 % 128 = 120` — bit-identical partial-tail structure to
the real Wan S=32760 (last quant block 56/64, last K-route tile 120/128), but small
enough for dense SDPA. Per-row error is split into **interior** (full blocks) vs
**tail** (the last partial 56-row block).

| config | sel | interior cos | tail cos | interior relerr | tail relerr | tail/int |
|---|---|---|---|---|---|---|
| topk=1.0 (routing no-op) | 1.000 | 0.999919 | 0.999918 | 0.012772 | 0.012836 | **1.01×** |
| topk=0.5 (pure routing) | 0.522 | 0.724835 | 0.999918 | 0.955123 | 0.012836 | **0.01×** |
| topk=0.5 + sink (production) | 0.537 | 0.734814 | 0.999918 | 0.927372 | 0.012836 | **0.01×** |

**Reading:**

- **topk=1.0 ratio = 1.01×** → the tail is no worse than the interior when routing
  is a no-op. This **rules out a routing-independent tail bug** (kernel masking,
  K/V dequant, quant scale on the partial block). The non-routing math is correct
  on the partial tail (cos 0.9999, same as interior).
- **topk=0.5 tail cos = 0.999918, identical to topk=1.0, and tail relerr is
  byte-identical (0.012836) across all three configs** → the last partial Q-block
  is **forced dense** by the fix (sim_q=False), so its rows attend to *all* K and
  match the golden reference regardless of topk. The tail is ~100× *closer* to
  dense than the interior, not worse.
- The interior cos 0.72 at topk=0.5 is the **honest by-design sparsity error** on
  structureless iid data (no similarity → dropping half the blocks is near-random);
  the topk=1.0 baseline of 0.9999 proves it is sparsity, not a defect.

**Conclusion.** At faithful tail geometry, the partial-tail fix makes the tail rows
**forced-dense and bit-faithful to the dense golden reference** — strictly *better*
than the sparse interior. There is **no residual tail-specific kernel/routing/quant
divergence**. The "last-few-frames" impression in the §7 E2E is therefore not a
localizable tail bug; it is consistent with (a) diffusion-sampling chaos (§7) and/or
(b) the **global** K-quant granularity difference (Root Cause #2 follow-up, task
#12) — which is sequence-wide, not tail-specific.

**Caveat.** iid Gaussian inputs make the interior topk=0.5 error large and
unrepresentative of real-video quality; the test's value is purely the
**tail-vs-interior differential** and the **topk=1.0 baseline**, both of which are
decisive and content-independent. Repro: `python tmp_tail_diff_vs_sdpa.py`.

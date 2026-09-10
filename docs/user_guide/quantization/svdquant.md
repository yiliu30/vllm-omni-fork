# SVDQuant W4A4

## Overview

[SVDQuant](https://arxiv.org/abs/2411.05007) combines four-bit weights
and activations with a small low-rank branch that corrects part of the
quantization error. vLLM-Omni consumes an offline-quantized checkpoint;
it does not calibrate the model while loading.

A second configuration derives the four-bit weights and the low-rank branch
from an ordinary BF16 checkpoint during loading, with no calibration data and
no producer tool. See [Online MXFP4](#online-mxfp4) below.

## Checkpoint contract

Place the following entry in the diffusion transformer's `config.json`:

```json
{
  "quantization_config": {
    "quant_method": "svdquant",
    "rank": 32,
    "precision": "nvfp4",
    "act_unsigned": false,
    "modules_to_not_convert": []
  }
}
```

For a quantized linear with input size `K`, output size `N`, and correction
rank `R`, the checkpoint stores:

| Suffix | Shape | dtype |
| --- | --- | --- |
| `qweight` | `(N, K / 2)` | `int8` (two packed FP4 values per byte) |
| `wscales` | `(K / 16, N)` | `float8_e4m3fn` |
| `proj_down` | `(K, R)` | `bfloat16` |
| `proj_up` | `(N, R)` | `bfloat16` |
| `smooth_factor` | `(K,)` | `bfloat16` |
| `wcscales` | `(N,)` | `bfloat16` |
| `wtscale` | `(1,)` | `bfloat16` |

`K` must be divisible by 16 on every tensor-parallel rank. Set `wcscales` to
ones when no per-output correction is needed. Modules listed in
`modules_to_not_convert` keep their checkpoint precision.

## Online MXFP4

`precision="mxfp4"` keeps the same W4A4 plus low-rank shape but quantizes at
load time instead of reading quantized tensors, so any BF16 checkpoint works:

```json
{
  "quantization_config": {
    "quant_method": "svdquant",
    "precision": "mxfp4",
    "is_checkpoint_mxfp4_serialized": false,
    "rank": 32,
    "iterations": 2,
    "modules_to_not_convert": []
  }
}
```

For every quantized linear the loader computes
`W ~= quant_mxfp4(W - L) + (proj_up @ proj_down.T)`, where `L` is the top-`R`
part of the quantization residual `W - dequant(quant(W))`. `iterations` repeats
that step against the branch so far and appends the new factor to `L`, so the
deployed branch is `rank * iterations` wide. `svd_niter` is the power-iteration
count of the truncated SVD.

| Field | Default | Constraint |
| --- | --- | --- |
| `rank` | 32 | `rank * iterations < min(N, K)` for every quantized linear; `rank=0` is online-only and selects branchless MXFP4 as the control |
| `iterations` | 2 | `>= 1`; each extra pass adds `rank` correction columns and one more quantization pass at load |
| `svd_niter` | 2 | `>= 1` |
| `weight_terms` | 1 | 1 or 2; 2 ships a second FP4 tensor holding what the first weight term left behind, at one extra GEMM |
| `act_terms` | 1 | 1 or 2; 2 re-quantizes the activation residual at run time, at one extra GEMM |

### Chained FP4 terms

The XPU quantizer picks one power-of-two (e8m0) scale per group of 32 with
`ceil(log2(amax / 6))`, so a group pays up to twice the resolution it needs and
nothing on the weight side can reach that: a measured oracle that may use a
continuous scale is 27% below `ceil`, while a legal exponent search recovers 4%
and the low-rank branch recovers about the same. Chained terms work around the
format instead of fighting it. With `weight_terms=2` the loader ships a second
FP4 tensor of `W - dequant(term1)`; with `act_terms=2` the layer quantizes
`x - dequant(quant(x))` and runs it through the same GEMM. Two operands times
two terms is up to four W4A4 GEMMs, and each GEMM is priced like the base one.

Relative error of the whole GEMM (real Wan2.2-T2V-A14B linears, activations
drawn from the per-channel profiles a quantized run dumps):

| Terms (weight, activation) | 1, 1 | 2, 1 | 1, 2 | 2, 2 |
| --- | --- | --- | --- | --- |
| Extra W4A4 GEMMs | 0 | 1 | 1 | 3 |
| Relative output error | 0.172 | 0.131 | 0.113 | 0.020 |

Two terms per operand lands below the 0.033 that MXFP8 W8A8 measures, on weights
that stay 4 bits wide. Weight memory roughly doubles at `weight_terms=2` and the
GEMM work follows the term count, so treat this as the accuracy-first setting.

Differences from the serialized NVFP4 path:

- XPU only, and it requires vLLM's XPU MXFP4 W4A4 kernel; loading fails if a
  weight-only MXFP4 backend is selected instead.
- `K` must be divisible by the MX group size 32 rather than 16.
- No `smooth_factor`: without activation statistics a weight-derived smoothing
  scale measures worse than no smoothing, so it stays all-ones and is elided.
- No `wtscale`/`wcscales` and no global alpha; MXFP4 carries no outer scale.
- Roughly 0.2-0.4 s of extra load time per linear. Most of the remaining error
  comes from MXFP4 *activation* quantization, which no weight-side branch can
  remove, so raise `rank` only where measurements justify it: on Wan2.2-T2V-A14B
  a deployed branch rank of 128 measured within 1.2 points of rank 32 end to end.
- Derivation happens inside engine startup. Wan2.2-T2V-A14B (two experts, about
  800 quantized linears) needs roughly six extra minutes, which exceeds the
  600 s default: pass `--init-timeout 1800` to the examples or
  `init_timeout=1800` to `Omni`.

### Accuracy

Relative error against an FP32 matmul of the same BF16 weight (XPU, Wan2.2
shapes, batch 256):

| Weight | Weight-only | W4A4 | W4A4 + rank 32 |
| --- | --- | --- | --- |
| well conditioned | 0.115 | 0.163 | 0.162 |
| outlier channels | 0.171 | 0.204 | 0.194 |

Relative error against an FP32 matmul of the same BF16 weight, by deployed
branch rank (`rank * iterations`), on a 5120x5120 Wan linear with batch 256:

| Branch rank | 0 | 16 | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- |
| well conditioned | 0.1627 | 0.1620 | 0.1613 | 0.1601 | 0.1581 | 0.1543 |
| outlier channels | 0.2045 | 0.1972 | 0.1914 | 0.1821 | 0.1732 | 0.1660 |

About half of the W4A4 error comes from the activation quantizer (weight-only
error is 0.115 well conditioned and 0.171 with outlier channels), and MX
group-32 e8m0 scales already absorb the per-channel outliers that the low-rank
branch is meant to carry, so rank 32 recovers only 1-6% of the total error.
Generating with every transformer linear at W4A4 visibly damages Wan2.2-T2V-A14B
output - a branchless `rank=0` run measures the same, so the damage is inherent to
MXFP4 W4A4 rather than to the low-rank path. Use `rank=0` for that headroom
measurement. A deployable configuration comes from narrowing the quantized set;
calibration-derived smoothing is worth much less here than in the NVFP4 recipe,
because the MX group-32 scales already do the same job (about 5% of activation
error on measured Wan activations, against a 20x gain from exemptions).

### End-to-end measurement and a deployable set

Same-seed PSNR cannot rank two quantizations of a video model: the sampler is
chaotic, so two near-lossless runs of the same checkpoint differ by about 16 dB.
Compare the cache-dit *residual diff* curve instead (the block-residual norm over
the hidden-state norm per step, printed by `--enable-cache-dit-summary`). It is a
property of the network rather than of the frame it happens to land on, and the
run-to-run floor for an unchanged configuration is 0.6-0.8%.

Deviation of that curve from an MXFP8 W8A8 run, Wan2.2-T2V-A14B at 720x1280,
17 frames, 40 steps:

| Configuration | mean deviation | peak memory |
| --- | --- | --- |
| W4A4, rank 32 branch, every linear | 17.9% | 55.2 GiB |
| W4A4, rank 128 branch, every linear | 16.7% | 56.6 GiB |
| W4A4, rank 128, `condition_embedder` + `proj_out` + blocks 0, 1, 38, 39 BF16 | 4.3% | 59.9 GiB |

So extra branch rank is the expensive and ineffective lever, and
`modules_to_not_convert` is the cheap and effective one: about 10% of the
linears kept in BF16 recovered most of the gap for 4 GiB and no extra step time.
Start from a boundary-block exemption list, then widen it only where measurements
say the remaining error lives.

The entry point is `config.json`: the examples' `--quantization` flag does not
accept `svdquant`, and passing a different method there would override the
precision recorded in the checkpoint.

## Runtime support

The compatibility path accepts BF16 inputs and executes an NVFP4 GEMM followed
by the BF16 rank correction. It supports vLLM's FlashInfer, CUTLASS, and FBGEMM
NVFP4 tensor layouts; incompatible forced backends fail during model loading.
SM103 is the currently validated and enabled hardware target. Native fusion of
the NVFP4 GEMM and rank correction is separate from this checkpoint-loading
contract.

# Wan2.2 Sparse Quality Risk Findings

This document isolates the current quality-risk analysis for the Wan2.2 XPU sparse-attention path, separate from the runtime and timing log.

The findings are sorted by estimated risk to end-to-end video quality, highest first.

## Quality Risk Findings

### 1. XPU preprocess routing has a documented correctness bug

Risk: very high

Why this matters:

- if the sparse routing map is wrong, the sparse kernel can be numerically correct and still produce degraded output because it is attending to the wrong blocks

Evidence:

- ARK's own port note documents that the preprocess-generated path can already produce incorrect model generation output in a dense-equivalent regime:
  - `topk=1`
  - `smooth_k=False`
- source: [/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/docs/SPARGE_PREPROCESS_PORT_PLAN.md:65](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/docs/SPARGE_PREPROCESS_PORT_PLAN.md#L65)

Documented suspected root cause:

- forced-selected entries are inserted into the routing map before ranked top-k fill
- `_fill_block_map_torch(...)` then iterates ranks without discounting already-selected entries
- this can waste effective top-k budget on duplicate selections
- source: [/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/docs/SPARGE_PREPROCESS_PORT_PLAN.md:68](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/docs/SPARGE_PREPROCESS_PORT_PLAN.md#L68)

Relevant implementation:

- XPU fill helper: [/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/sparse_attention.py:607](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/sparse_attention.py#L607)

Assessment:

- this is the strongest currently known quality-risk issue
- it directly affects block selection before the sparse kernel runs

### 2. XPU routing tile geometry is not aligned with CUDA `sm90`

Risk: very high

Why this matters:

- routing tile size determines how many tokens share one sparse selection decision
- coarser routing can blur token-level behavior and materially change selected blocks

CUDA `sm90` reference:

- `BLKQ=64`
- `BLKK=128`
- source: [/data/model/yiliu7/SpargeAttn-fork/spas_sage_attn/core.py:126](/data/model/yiliu7/SpargeAttn-fork/spas_sage_attn/core.py#L126)

XPU current behavior for `head_dim=128`:

- `quant_block_size=64`
- `query_tile_tokens=256`
- `blocks_per_qtile=4`
- source: [/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/sparse_attention.py:557](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/sparse_attention.py#L557)
- source: [/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/sparse_attention.py:765](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/sparse_attention.py#L765)

Practical mismatch for Wan `head_dim=128`:

- CUDA `sm90` Q routing tile: `64`
- XPU Q routing tile: `256`

Assessment:

- this is a major structural mismatch
- the XPU preprocess is making one routing decision for 4 query quant blocks where CUDA routes at 1 query block
- this alone can explain quality drift even if both implementations are otherwise correct

### 3. Wan XPU patch hardcodes a more aggressive `simthreshd1`

Risk: high

Why this matters:

- `simthreshd1` changes which Q/K blocks are marked coherent before top-k routing
- different coherence classification changes the sparse mask

CUDA top-k default:

- `simthreshd1=-0.1`
- source: [/data/model/yiliu7/SpargeAttn-fork/spas_sage_attn/core.py:106](/data/model/yiliu7/SpargeAttn-fork/spas_sage_attn/core.py#L106)

Wan XPU patch:

- hardcoded `simthreshd1=-1.0`
- source: [/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wan_sparse_patch.py:212](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wan_sparse_patch.py#L212)

Assessment:

- this is not a minor tuning difference
- it changes preprocess semantics materially and should be normalized before deeper kernel blame

### 4. XPU preprocess/quantization path is not the same implementation as CUDA

Risk: medium-high

Why this matters:

- even if both aim for the same algorithm, separate implementations can diverge in pooling, similarity classification, quantization, and routing fill

CUDA path:

- uses fused route+quant helper `get_block_map_meansim_fuse_quant(...)`
- source: [/data/model/yiliu7/SpargeAttn-fork/spas_sage_attn/utils.py:371](/data/model/yiliu7/SpargeAttn-fork/spas_sage_attn/utils.py#L371)

XPU path:

- uses `_pool_sim_and_quant_torch(...)` plus ARK/Triton preprocess assembly
- source: [/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/sparse_attention.py:632](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/sparse_attention.py#L632)

Assessment:

- this raises risk because the metadata generator is a port, not the same codepath with a different backend
- the largest quality risk is still in metadata mismatch, not necessarily in the final sparse kernel

### 5. XPU only implements the top-k routing slice, not the full CUDA routing family

Risk: medium

Why this matters:

- tuned CUDA settings or model-zoo hyperparameters may not transfer if the routing family differs

CUDA:

- supports both `cdfthreshd` and `topk`
- source: [/data/model/yiliu7/SpargeAttn-fork/spas_sage_attn/utils.py:372](/data/model/yiliu7/SpargeAttn-fork/spas_sage_attn/utils.py#L372)

XPU:

- rejects `cdfthreshd`
- source: [/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/sparse_attention.py:1241](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/sparse_attention.py#L1241)

Assessment:

- this is a feature-gap / portability risk
- it is less likely than the routing-fill and tile-geometry issues to explain the immediate quality drop in the exact top-k runs we tested

### 6. Final sparse kernel may still differ from CUDA, but it is not the leading suspect

Risk: medium

Why this matters:

- the final sparse kernel consumes LUT and quantized tensors
- any mismatch in scaling, dequantization, accumulation, or layout handling could affect quality

Relevant XPU entrypoint:

- `sage_sparse(...)`
- source: [/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/sparse_attention.py:22](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/sparse_attention.py#L22)

Relevant CUDA reference path:

- `qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(...)`
- source: [/data/model/yiliu7/SpargeAttn-fork/spas_sage_attn/core.py:138](/data/model/yiliu7/SpargeAttn-fork/spas_sage_attn/core.py#L138)

Assessment:

- still worth checking
- but current evidence points more strongly to preprocess metadata mismatch than to a single kernel-side numerical bug

## Current Best Hypothesis

Most likely cause of the observed quality drop:

- XPU sparse metadata generation is not aligned with the CUDA reference
- the largest contributors are:
  - known routing-fill correctness bug
  - much coarser XPU Q routing tile for Wan `head_dim=128`
  - hardcoded `simthreshd1=-1.0` in the Wan XPU patch

This means:

- the sparse kernel may be receiving a different and lower-quality routing map
- therefore end-to-end quality can drop even if the final XPU sparse kernel implementation itself is numerically reasonable

## Recommended Validation Order

1. Normalize `simthreshd1` in the Wan XPU patch from `-1.0` to `-0.1`.
2. Fix `_fill_block_map_torch(...)` so the top-k budget counts newly added entries only.
3. Add a metadata diff harness on saved Wan self-attention tensors:
   - compare CUDA vs XPU `block_map`
   - compare CUDA vs XPU `lut`
   - compare CUDA vs XPU `valid_block_num`
4. After metadata is aligned, compare final sparse kernel outputs on identical quantized inputs.

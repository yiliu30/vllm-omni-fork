# Wan2.2 Sparse Preprocess Fix Plan

## Summary

This plan fixes the two highest-risk quality issues in the shared ARK XPU sparse preprocess path:

1. ranked top-k fill semantics are incorrect when rows already contain forced-selected blocks
2. XPU routing tile geometry is not aligned with the CUDA `sm90` reference for Wan `head_dim=128`

The goal is correctness-first parity for sparse metadata generation. The fix applies to the generic ARK preprocess path, not a Wan-only wrapper.

## Implementation Changes

### 1. Fix ranked-fill semantics in both torch and Triton XPU preprocess paths

- Define one semantic contract for ranked fill:
  - start from a row that may already contain forced-selected entries
  - add exactly `num_to_select` new ranked entries beyond the already-selected set
  - preserve the current behavior that `num_to_select == 0` still adds one ranked entry
- Update the torch oracle implementation in:
  - `auto_round_kernel/sparse_attention.py`
  - target helper: `_fill_block_map_torch(...)`
- Update the Triton-XPU fast-path implementation in:
  - `auto_round_kernel/sparge_preprocess_triton.py`
  - target helper: `_fill_block_map_triton(...)`
- Keep forced-selection rules unchanged:
  - incoherent K blocks selected for all rows
  - incoherent Q rows expanded to all K blocks
  - optional attention sink block 0
  - causal filtering after ranked fill
- Treat the torch implementation as the readable oracle and validate Triton against it.

### 2. Align XPU routing geometry with CUDA `sm90` for Wan `head_dim=128`

- Replace the current `head_dim -> query_tile_tokens` shortcut for the relevant prefill path.
- Introduce explicit routing geometry fields in preprocess context:
  - `q_route_block_tokens`
  - `k_route_block_tokens`
  - `quant_block_size`
- For the CUDA-`sm90` parity mode used by Wan `head_dim=128`, use:
  - `q_route_block_tokens = 64`
  - `k_route_block_tokens = 128`
  - `quant_block_size = 64`
- Remove the current oversized Q regrouping behavior from this parity mode:
  - do not route with a `256`-token Q tile for `head_dim=128`
- Add explicit expansion from routing-space selection to quant-block-space selection:
  - one routed K block of `128` expands to two quant blocks of `64`
  - routed Q block of `64` maps one-to-one to Q quant blocks in this mode
- Ensure `raw_block_map`, `lut`, and `valid_block_num` are generated from the expanded quant-block-space mask expected by the XPU sparse kernel contract.

### 3. Keep scope generic, not Wan-only

- Apply the fix in the shared ARK XPU preprocess path used by:
  - `sparge_preprocess_topk(...)`
  - `sparge_sage2_attn_meansim_topk_xpu(...)`
- Do not patch around the issue only in `wan_sparse_patch.py`.
- Wan should inherit the corrected metadata behavior automatically.

## Tests

### Ranked-fill correctness

- Add unit tests for rows with pre-existing forced-selected blocks.
- Validate that final row population equals:
  - existing forced-selected count
  - plus `num_to_select` newly added ranked blocks
- Add `num_to_select == 0` coverage and preserve the current “select one ranked block” behavior.

### Torch vs Triton parity

- Add tests that run both preprocess backends on identical synthetic inputs and compare:
  - `raw_block_map`
  - `lut`
  - `valid_block_num`
- Use the torch implementation as the oracle for Triton parity.

### Routing geometry parity without local CUDA execution

- Add preprocess oracle tests for Wan-relevant `head_dim=128` inputs with:
  - equal `seq_len_q == seq_len_kv`
  - non-causal prefill
  - top-k routing
- Compare XPU-generated metadata against one of:
  - recorded reference metadata produced earlier on a CUDA-capable node
  - hand-constructed oracle cases derived from CUDA `sm90` geometry (`BLKQ=64`, `BLKK=128`)
- Validate:
  - row-level selected blocks
  - `lut`
  - `valid_block_num`

### Dense-equivalent regression

- Add regression coverage for:
  - `topk=1.0`
  - `smooth_k=False`
  - non-causal prefill
- Assert that rows become fully selected in the dense-equivalent case.

### Wan smoke validation

- Capture one real Wan self-attention Q/K snapshot.
- Compare XPU preprocess metadata against recorded reference metadata or expected oracle outputs on that snapshot.
- After metadata alignment, rerun the existing Wan sparse timing and quality workflow:
  - dense baseline
  - sparse `topk=0.1`
  - sparse `topk=0.5` with attention sink

## Assumptions

- correctness is prioritized over preserving the current XPU preprocess performance
- the target parity reference for item 2 is CUDA `sm90` behavior for Wan `head_dim=128`, but validation on this node must not require live CUDA execution
- final sparse kernel changes are out of scope unless corrected metadata exposes a hard kernel contract mismatch

# FLUX AutoRound main-branch fix plan

## Goal
- Create a clean branch from `main`.
- Add the minimal fixes needed for `/media/yiliu7/vllm-project-org/FLUX.1-dev-AutoRound-w4a16` to load through the FLUX text-to-image example.
- Verify with real execution on the example path.

## Scope
- Fix FLUX transformer construction and weight loading so AutoRound / INC checkpoints instantiate matching quantized modules.
- Fix the text-to-image example so omitted sampling flags use model-appropriate defaults instead of generic ones.
- Add focused regression tests for the loader behavior and the example guidance default helper.

## Verification
- Run `py_compile` on touched files.
- Run focused test assertions directly if `pytest` is unavailable.
- Run `examples/offline_inference/text_to_image/text_to_image.py` with the real FLUX AutoRound W4A16 checkpoint and confirm image generation succeeds.

## Root Cause
- `TransformerConfig.from_dict()` correctly built the embedded AutoRound quantization config from `transformer/config.json`.
- `OmniDiffusionConfig.enrich_config()` assigned `self.tf_model_config` directly instead of calling `set_tf_model_config()`, so `self.quantization_config` stayed `None`.
- FLUX then constructed unquantized `weight` parameters inside transformer blocks, and checkpoint `qweight/qzeros/scales` tensors could not satisfy the strict unloaded-weight check.

## Outcome
- `enrich_config()` now uses `set_tf_model_config()` so embedded transformer quantization is propagated into the runtime config.
- The FLUX AutoRound W4A16 example now loads and generates successfully from the clean `main`-based branch.

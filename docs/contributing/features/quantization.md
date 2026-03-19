# Support FP8 Quantization

This section describes how to add FP8 quantization support to a diffusion model. We use the Qwen-Image pipeline as the reference implementation.

---

## Table of Contents

- [Overview](#overview)
- [Step-by-Step Implementation](#step-by-step-implementation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Reference Implementations](#reference-implementations)
- [Summary](#summary)

---

## Overview

### What is FP8 Quantization?

FP8 quantization reduces model memory footprint by storing weights in 8-bit floating point (`float8_e4m3fn`) instead of 16-bit (`bfloat16`/`float16`). This saves ~50% of weight memory.

A diffusion pipeline typically has three quantizable components:

| Component | Layer Types | Quantization Approach |
|-----------|-----------|----------------------|
| **DiT (transformer)** | `nn.Linear` | Replace with vLLM quantized linear layers |
| **Text encoder** | `nn.Linear` | FP8 weight storage via hooks |
| **VAE** | `nn.Conv2d`, `nn.Conv3d` | FP8 weight storage via hooks |

### Architecture

The quantization system has two mechanisms:

**1. vLLM Quantized Linear Layers (for DiT)**

vLLM provides drop-in replacements for `nn.Linear` that perform FP8 W8A8 computation on supported hardware. These are configured via `quant_config` passed to layer constructors.

Key classes:

| Class | Location | Purpose |
|-------|----------|---------|
| `build_quant_config` | `vllm_omni/quantization/__init__.py` | Build quantization config from string/dict/per-component spec |
| `ComponentQuantizationConfig` | `vllm_omni/quantization/component_config.py` | Per-component routing by layer prefix |

**2. FP8 Weight Storage via Hooks (for text encoder and VAE)**

For models loaded via `from_pretrained()` (outside vLLM's weight pipeline), we use forward hooks to store weights in FP8 and dequantize on-the-fly:

```
At rest:     weight.data = fp8_tensor (half memory)
Pre-hook:    weight.data = fp8_weight.to(bf16) * scale
Forward:     Normal computation in BF16
Post-hook:   weight.data = fp8_weight (back to FP8)
```

Key function:

| Function | Location | Purpose |
|----------|----------|---------|
| `apply_fp8_weight_storage` | `vllm_omni/diffusion/models/utils.py` | Apply FP8 hooks to Linear/Conv layers (introduced in [#1412](https://github.com/vllm-project/vllm-omni/pull/1412)) |

---

## Step-by-Step Implementation

### Step 1: Add `quant_config` to the Transformer

Pass `quant_config` to vLLM parallel linear layers in the transformer. This enables FP8 W8A8 for DiT linear layers.

#### 1.1: Accept `quant_config` in Constructor

```python
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

class YourTransformer2DModel(nn.Module):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        quant_config: QuantizationConfig | None = None,
        # ... other params
    ):
        self.quant_config = quant_config
        # Pass to sub-modules
        self.blocks = nn.ModuleList([
            YourTransformerBlock(quant_config=quant_config, ...)
            for _ in range(num_layers)
        ])
```

#### 1.2: Pass `quant_config` to vLLM Linear Layers

Replace `nn.Linear` with vLLM parallel linear layers that accept `quant_config`:

```python
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

class YourAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, quant_config=None, prefix=""):
        # QKV projection
        self.to_qkv = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=hidden_size // num_heads,
            total_num_heads=num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.to_qkv",
        )
        # Output projection
        self.to_out = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.to_out",
        )

class YourFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, quant_config=None, prefix=""):
        self.gate_up_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size * 2,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
```

!!! important
    Always pass the `prefix` parameter. It is used by `ignored_layers` matching to skip sensitive layers.

#### 1.3: Identify Sensitive Layers

Some layers degrade output quality when quantized. Common sensitive layers:

| Layer | Why Sensitive | Typical Models | Quant Methods Affected |
|-------|-------------|---------------|----------------------|
| `img_mlp` | Processes denoising latents with shifting dynamic range | Qwen-Image | FP8, Int8 |
| `feed_forward` | FFN layers in DiT blocks, large dynamic range | Z-Image | Int8 |
| `proj_out` | Final output projection, small quantization errors amplified | Various | FP8, Int8 |
| `lm_head` | Output vocabulary projection, precision-critical | Omni (thinker) | NVFP4 |
| `mlp.gate` | MoE router gates, routing decisions are precision-critical | MoE models | NVFP4 |

**Quantization quality benchmarks** (LPIPS: lower = better, <0.01 imperceptible, >0.1 noticeable):

| Model | Method | All Layers | With `ignored_layers` | Recommendation |
|-------|--------|-----------|----------------------|----------------|
| Qwen-Image-2512 | Int8 | LPIPS 0.0197 | 0.0027 (skip `img_mlp`) | Skip `img_mlp` |
| Z-Image-Turbo | Int8 | LPIPS 0.1597 | 0.0290 (skip `feed_forward`) | Skip `feed_forward` |
| Z-Image-Turbo | FP8 | LPIPS ~0.005 | — | All layers OK |

To identify sensitive layers for a new model:

1. Run inference with all layers quantized
2. Compare output quality with BF16 baseline using LPIPS (see `benchmarks/diffusion/quantization_quality.py`)
3. Selectively disable quantization per layer (via `ignored_layers`) until quality matches
4. Document the recommended `ignored_layers` in the model's supported models table

### Step 2: Wire Up Quantization in the Pipeline

#### 2.1: Create `quant_config` and Pass to Transformer

In the pipeline's `__init__`, extract the vLLM quant config and pass it to the transformer:

```python
from vllm_omni.quantization import build_quant_config

class YourPipeline(nn.Module):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        # ... load scheduler, text_encoder, vae ...

        # Build quant config from OmniDiffusionConfig
        quant_config = build_quant_config(od_config.quantization_config)

        # Pass to transformer
        self.transformer = YourTransformer2DModel(
            od_config=od_config,
            quant_config=quant_config,
            # ... other kwargs
        )
```

#### 2.2: Apply FP8 Weight Storage to Text Encoder and VAE

For models loaded via `from_pretrained()`, apply hook-based FP8 weight storage.

!!! note
    The `apply_fp8_weight_storage` function is provided by `vllm_omni/diffusion/models/utils.py`, introduced in [PR #1412](https://github.com/vllm-project/vllm-omni/pull/1412).

```python
from vllm_omni.diffusion.models.utils import apply_fp8_weight_storage

class YourPipeline(nn.Module):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()

        self.text_encoder = SomeTextEncoder.from_pretrained(...)
        self.vae = SomeVAE.from_pretrained(...).to(self.device)

        # Apply FP8 weight storage to text encoder and VAE
        if (
            od_config.quantization_config is not None
            and getattr(od_config.quantization_config, "quant_method", None) == "fp8"
        ):
            apply_fp8_weight_storage(self.vae)
            apply_fp8_weight_storage(self.text_encoder)

        # ... rest of init
```

`apply_fp8_weight_storage` walks all `nn.Linear`, `nn.Conv2d`, and `nn.Conv3d` modules in the model and applies FP8 hooks to each.

#### 2.3: Fix `load_weights` to Include Pre-loaded Components

Since text encoders and VAEs are loaded via `from_pretrained()` (not through vLLM's weight pipeline), their parameters won't appear in the `loaded_weights` set. Mark them explicitly:

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    loader = AutoWeightsLoader(self)
    loaded_weights = loader.load_weights(weights)
    # Mark pre-loaded component weights as loaded
    loaded_weights |= {f"vae.{name}" for name, _ in self.vae.named_parameters()}
    loaded_weights |= {f"text_encoder.{name}" for name, _ in self.text_encoder.named_parameters()}
    return loaded_weights
```

### Step 3: Add to Supported Models Documentation

Update the FP8 supported models table in `docs/user_guide/diffusion/quantization/fp8.md`:

```markdown
| Your-Model | `org/Your-Model` | ✅ | ✅ | ✅ | `sensitive_layer` |
```

---

## Testing

### Basic Verification

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# Test with FP8 enabled
omni = Omni(model="your-model", quantization="fp8")

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

### What to Verify

1. **Model loads without errors** — Check logs for FP8 initialization messages
2. **Output quality** — Compare FP8 outputs with BF16 baseline (should be visually identical)
3. **Memory usage** — Verify memory reduction vs BF16 (expect ~50% savings on quantized components)
4. **Sensitive layers** — If quality degrades, test with `ignored_layers` to identify which layers need BF16

### Example: Testing with Ignored Layers

```python
# If quality degrades, try skipping sensitive layers
omni = Omni(
    model="your-model",
    quantization_config={
        "method": "fp8",
        "ignored_layers": ["img_mlp"],  # Keep image MLPs in BF16
    },
)
```

---

## Troubleshooting

### Issue: Output Quality Degradation

**Symptoms:** Generated images have visible artifacts, color shifts, or blurriness compared to BF16 baseline.

**Causes & Solutions:**

- **Sensitive layers being quantized:**

    **Solution:** Identify and skip sensitive layers:
    ```python
    omni = Omni(
        model="your-model",
        quantization_config={
            "method": "fp8",
            "ignored_layers": ["img_mlp"],
        },
    )
    ```

- **All layers sensitive (rare):**

    **Solution:** The model may not be suitable for FP8 quantization. Consider weight-only quantization or other methods.

### Issue: Missing Weights Warning

**Symptoms:** Warnings about unloaded weights for `vae.*` or `text_encoder.*` parameters.

**Solution:** Ensure `load_weights` marks pre-loaded component weights. See [Step 2.3](#23-fix-load_weights-to-include-pre-loaded-components).

### Issue: CUDA Out of Memory

**Symptoms:** OOM errors during model loading despite FP8 being enabled.

**Causes & Solutions:**

- **Text encoder/VAE loaded in BF16 before FP8 conversion:**

    FP8 weight storage is applied after `from_pretrained()`, so peak load memory is still BF16. This is a one-time cost during initialization.

---

## Reference Implementations

| Model | Pipeline File | Transformer File | Notes |
|-------|-------------|-----------------|-------|
| **Z-Image** | `models/z_image/pipeline_z_image.py` | `models/z_image/z_image_transformer.py` | DiT + text encoder FP8 |
| **Qwen-Image** | `models/qwen_image/pipeline_qwen_image.py` | `models/qwen_image/qwen_image_transformer.py` | DiT + text encoder + VAE FP8 |

All files are under `vllm_omni/diffusion/`.

---

## Summary

Adding FP8 quantization support to a new model:

1. ✅ **Add `quant_config` to transformer** — Accept and pass to vLLM parallel linear layers with proper `prefix`
2. ✅ **Identify sensitive layers** — Profile which layers need `ignored_layers` for quality
3. ✅ **Apply FP8 weight storage** — Call `apply_fp8_weight_storage()` on text encoder and VAE in pipeline `__init__`
4. ✅ **Fix `load_weights`** — Mark pre-loaded component weights as loaded
5. ✅ **Update docs** — Add model to the supported models table in `docs/user_guide/diffusion/quantization/fp8.md`
6. ✅ **Test** — Verify quality, memory savings, and sensitive layer behavior

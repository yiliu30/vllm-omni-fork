# Quantization

vLLM-OMNI provides a unified quantization framework that supports all 35+ quantization methods from vLLM, with extensions for multi-stage models (diffusion, omni).

## Quick Start

### Single Method

```python
from vllm_omni.quantization import build_quant_config

# String shorthand (dynamic activation quantization)
config = build_quant_config("fp8")

# Dict with parameters (static activation quantization)
config = build_quant_config({"method": "fp8", "activation_scheme": "static"})
```

### Per-Component Quantization

Multi-stage models like Bagel or Qwen3-Omni have multiple components (transformer, VAE, talker, etc.) that benefit from different quantization strategies:

```python
# Quantize transformer with FP8, leave VAE unquantized
config = build_quant_config({
    "transformer": {"method": "fp8"},
    "vae": None,
})

# Qwen3-Omni: different configs for visual, language, and talker
config = build_quant_config({
    "visual": None,
    "language_model": {"method": "fp8", "activation_scheme": "dynamic"},
    "talker": {"method": "fp8"},
})
```

Routing uses longest-prefix match on layer names, so `"transformer"` matches `transformer.blocks.0.attn.to_q`, etc.

### With OmniDiffusionConfig

```python
from vllm_omni.diffusion.data import OmniDiffusionConfig

# String
config = OmniDiffusionConfig(model="black-forest-labs/FLUX.1-dev", quantization="fp8")

# Per-component dict
config = OmniDiffusionConfig(
    model="ByteDance-Seed/BAGEL-7B-MoT",
    quantization_config={
        "language_model": {"method": "fp8"},
        "vae": None,
    },
)
```

## Supported Methods

All methods from vLLM's `QUANTIZATION_METHODS` registry are supported, including:

| Method | Description | Min GPU Capability |
|--------|-------------|-------------------|
| `fp8` | FP8 weight + activation quantization | SM 89 (Ada) |
| `awq` | Activation-aware Weight Quantization (INT4) | SM 75 |
| `gptq` | GPTQ (INT4/INT8) | SM 75 |
| `bitsandbytes` | BitsAndBytes (INT8/NF4) | SM 75 |
| `gguf` | GGUF format (uses dequant+GEMM for N-D tensors) | SM 60 |
| `compressed-tensors` | CompressedTensors format | Varies |
| `modelopt` | NVIDIA ModelOpt (INT4, FP8, NVFP4, MXFP4) | Varies |

Run `from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS` to see the full list.

## Dynamic vs Static Quantization

- **Dynamic** (`activation_scheme="dynamic"`): Activations are quantized on-the-fly during inference. No calibration needed. This is the default for FP8.
- **Static** (`activation_scheme="static"`): Uses pre-calibrated activation scales stored in the checkpoint. Requires a calibrated model (e.g., from `llm-compressor` or NVIDIA ModelOpt).

```python
# Dynamic (default, no calibration needed)
config = build_quant_config("fp8")

# Static (requires calibrated checkpoint)
config = build_quant_config({"method": "fp8", "activation_scheme": "static"})
```

## GGUF

For GGUF models, the framework uses a dequant+GEMM path instead of the fused kernel path (which expects 2D inputs), making it compatible with N-D diffusion tensors:

```python
config = build_quant_config({
    "method": "gguf",
    "gguf_model": "/path/to/model.gguf",
})
```

## Migration Guide

### Before (v0.14.x)

```python
# Old diffusion-specific API
from vllm_omni.diffusion.quantization import get_diffusion_quant_config
config = get_diffusion_quant_config("fp8", activation_scheme="static")

# Old GGUF wrapper
from vllm_omni.diffusion.quantization.gguf_utils import DiffusionGGUFConfig
config = DiffusionGGUFConfig(gguf_model="model.gguf")
```

### After (v0.16.0+)

```python
# Unified API — delegates to vLLM's registry
from vllm_omni.quantization import build_quant_config
config = build_quant_config({"method": "fp8", "activation_scheme": "static"})

# GGUF through the same API
config = build_quant_config({"method": "gguf", "gguf_model": "model.gguf"})

# Per-component (new capability)
config = build_quant_config({
    "transformer": {"method": "fp8"},
    "vae": None,
})
```

## Architecture

```
build_quant_config(spec)
    ├── str  → _build_single(method) → vLLM registry / _OVERRIDES
    ├── dict with "method" → _build_single(method, **kwargs)
    ├── per-component dict → ComponentQuantizationConfig
    │       routes get_quant_method() by longest-prefix match
    ├── QuantizationConfig → passthrough
    └── None / "none" → None
```

- **`_OVERRIDES`**: Registry for methods needing OMNI-specific behavior (currently only GGUF for N-D tensor support).
- **`ComponentQuantizationConfig`**: Wraps multiple `QuantizationConfig` instances, routing `get_quant_method()` calls to the correct config based on the layer's prefix in the model tree.
- **`validate_quant_config()`**: Checks GPU capability, dtype compatibility, and component prefix validity. Called automatically in `OmniDiffusionConfig.__post_init__()`.

# Quantization for Diffusion and Omni Models

vLLM-OMNI provides a unified quantization framework that delegates to vLLM's quantization registry (35+ methods) with extensions for multi-stage models.

## Quick Start

### Single Method

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# String shorthand (dynamic activation quantization)
omni = Omni(model="<your-model>", quantization="fp8")

# Dict with parameters (static activation quantization)
omni = Omni(
    model="<your-model>",
    quantization_config={"method": "fp8", "activation_scheme": "static"},
)
```

### Per-Component Quantization

Multi-stage models like Bagel or Qwen3-Omni have components (transformer, VAE, talker, etc.) that benefit from different quantization:

```python
# Quantize transformer with FP8, leave VAE unquantized
omni = Omni(
    model="ByteDance-Seed/BAGEL-7B-MoT",
    quantization_config={
        "language_model": {"method": "fp8"},
        "vae": None,
    },
)

# Qwen3-Omni: different configs per component
omni = Omni(
    model="Qwen/Qwen3-Omni",
    quantization_config={
        "visual": None,
        "language_model": {"method": "fp8", "activation_scheme": "dynamic"},
        "talker": {"method": "fp8"},
    },
)
```

Routing uses longest-prefix match on layer names, so `"transformer"` matches `transformer.blocks.0.attn.to_q`, etc.

## Supported Methods

| Method | Guide | Description | Min GPU |
|--------|-------|-------------|---------|
| FP8 | [FP8](fp8.md) | FP8 W8A8, dynamic or static | SM 89 (Ada) |
| Int8 | [Int8](int8.md) | Int8 W8A8 | SM 89 (Ada) / Ascend NPU |
| GGUF | [GGUF](gguf.md) | GGUF format, dequant+GEMM for N-D tensors | SM 60 |
| AWQ | — | Activation-aware Weight Quantization (INT4) | SM 75 |
| GPTQ | — | GPTQ (INT4/INT8) | SM 75 |
| BitsAndBytes | — | BitsAndBytes (INT8/NF4) | SM 75 |
| ModelOpt | — | NVIDIA ModelOpt (INT4, FP8, NVFP4, MXFP4) | Varies |

All methods from vLLM's `QUANTIZATION_METHODS` registry are automatically available. Run `from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS` to see the full list.

## Dynamic vs Static Quantization

- **Dynamic** (`activation_scheme="dynamic"`): Activations quantized on-the-fly. No calibration needed. Default for FP8.
- **Static** (`activation_scheme="static"`): Uses pre-calibrated activation scales from the checkpoint. Requires a model calibrated with `llm-compressor` or NVIDIA ModelOpt.

```python
# Dynamic (default)
omni = Omni(model="<your-model>", quantization="fp8")

# Static (requires calibrated checkpoint)
omni = Omni(
    model="<your-model>",
    quantization_config={"method": "fp8", "activation_scheme": "static"},
)
```

## Device Compatibility for FP8

| GPU Generation | Example GPUs | FP8 Mode |
|---------------|-------------------|----------|
| Ada/Hopper (SM 89+) | RTX 4090, H100, H200 | Full W8A8 with native hardware |

Kernel selection is automatic.

## Device Compatibility for Int8

| Device Type | Generation | Example | Int8 Mode |
|-------------|---------------|-------------------|----------|
| NVIDIA GPU | Ada/Hopper (SM 89+) | RTX 4090, H100, H200 | Full W8A8 with native hardware |
| Ascend NPU | Atlas A2/Atlas A3 | Atlas 800T A2/Atlas 900 A3 | Full W8A8 with native hardware |

## Python API

The `build_quant_config()` factory accepts multiple input formats:

```python
from vllm_omni.quantization import build_quant_config

# String
config = build_quant_config("fp8")

# Dict with parameters
config = build_quant_config({"method": "fp8", "activation_scheme": "static"})

# Per-component dict
config = build_quant_config({
    "transformer": {"method": "fp8"},
    "vae": None,
})

# None / "none"
config = build_quant_config(None)  # returns None
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

## Migration Guide

### Before (v0.14.x)

```python
# Old diffusion-specific API (removed)
from vllm_omni.diffusion.quantization import get_diffusion_quant_config
config = get_diffusion_quant_config("fp8", activation_scheme="static")
```

### After (v0.16.0+)

```python
# Unified API — delegates to vLLM's registry
from vllm_omni.quantization import build_quant_config
config = build_quant_config({"method": "fp8", "activation_scheme": "static"})

# Per-component (new)
config = build_quant_config({
    "transformer": {"method": "fp8"},
    "vae": None,
})
```

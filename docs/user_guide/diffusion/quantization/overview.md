# Quantization for Diffusion Models

vLLM-Omni supports quantization of diffusion model components to reduce memory usage and accelerate inference. This includes DiT, text encoders, and VAEs.

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

### Pre-quantized LLM Checkpoints (Multi-stage Models)

For multi-stage models like Qwen3-Omni, the unified quantization framework auto-detects
pre-quantized checkpoints via `quantization_config` in the HF config. Supported formats:

| Format | `quant_algo` | Hardware | Example |
|--------|-------------|----------|---------|
| ModelOpt FP8 | `FP8` | Ada/Hopper (SM 89+) | `asdazd/Qwen3-Omni-30B-A3B-Instruct_modelopt_FP8` |
| ModelOpt NVFP4 | `NVFP4` | Blackwell (SM 100+) | NVFP4 quantized checkpoint |

Quantization is automatically scoped to the thinker's `language_model` — audio encoder,
vision encoder, talker, and code2wav remain in BF16.

## Quantization Scope

When `--quantization fp8` is enabled, the following components are quantized:

| Component | What Gets Quantized | Mechanism |
|-----------|-------------------|-----------|
| **DiT (transformer)** | `nn.Linear` layers | vLLM W8A8 FP8 compute (Ada/Hopper) or weight-only (older GPUs) |
| **Text encoder** | `nn.Linear` layers | FP8 weight storage, BF16 compute |
| **VAE** | `nn.Conv2d`, `nn.Conv3d` layers | FP8 weight storage, BF16 compute |

!!! note
    Not all models support all three components. See the [FP8 supported models table](fp8.md#supported-models) for per-model details.

## Device Compatibility for FP8

| Device | Example Hardware | FP8 | NVFP4 |
|--------|-----------------|-----|-------|
| Blackwell GPU (SM 100+) | RTX 5090, B100, B200 | Yes | Yes (native FP4 HW) |
| Ada/Hopper GPU (SM 89+) | RTX 4090, H100, H200 | Yes (W8A8 native) | No |
| Turing/Ampere GPU (SM 75-86) | RTX 3090, A100 | Yes (weight-only Marlin) | No |
| Ascend NPU | Atlas 800T A2 (910B) | Not yet supported | No |

Kernel selection is automatic on CUDA GPUs.

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

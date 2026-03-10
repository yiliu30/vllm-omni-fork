# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified quantization framework for vLLM-OMNI.

Supports ALL 35+ vLLM quantization methods, per-component quantization
for multi-stage models, and all platforms (CUDA, ROCm, CPU, XPU, NPU).

Quick start:
    from vllm_omni.quantization import build_quant_config

    # Any vLLM method works
    config = build_quant_config("fp8")
    config = build_quant_config("awq")
    config = build_quant_config("bitsandbytes")

    # Per-component for multi-stage models
    config = build_quant_config({
        "transformer": {"method": "fp8"},
        "vae": None,
    })

    # Pass to vLLM layers
    linear = QKVParallelLinear(..., quant_config=config)
"""

from .component_config import ComponentQuantizationConfig
from .defaults import COMPONENT_SKIP_DEFAULTS, get_default_skip_patterns
from .factory import SUPPORTED_QUANTIZATION_METHODS, build_quant_config
from .validation import validate_quant_config

# DiffusionGGUFConfig and DiffusionGGUFLinearMethod are NOT imported here
# to avoid pulling in vllm's GGUF → fused_moe → w8a8_utils → pynvml at
# module load time (crashes in subprocess environments without CUDA).
# Import them directly from .gguf_config when needed.

__all__ = [
    "build_quant_config",
    "ComponentQuantizationConfig",
    "validate_quant_config",
    "get_default_skip_patterns",
    "COMPONENT_SKIP_DEFAULTS",
    "SUPPORTED_QUANTIZATION_METHODS",
]

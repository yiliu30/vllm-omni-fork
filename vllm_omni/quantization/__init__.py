# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified quantization framework for vLLM-OMNI.

Delegates to vLLM's quantization registry (35+ methods, all platforms).
Adds per-component quantization for multi-stage models.

    from vllm_omni.quantization import build_quant_config

    config = build_quant_config("fp8")
    config = build_quant_config({"transformer": {"method": "fp8"}, "vae": None})
"""

from .component_config import ComponentQuantizationConfig
from .defaults import COMPONENT_SKIP_DEFAULTS, get_default_skip_patterns
from .factory import SUPPORTED_QUANTIZATION_METHODS, build_quant_config
from .validation import validate_quant_config

# DiffusionGGUFConfig is NOT imported here to avoid pulling in
# GGUF -> fused_moe -> pynvml at module load time.

__all__ = [
    "build_quant_config",
    "ComponentQuantizationConfig",
    "validate_quant_config",
    "get_default_skip_patterns",
    "COMPONENT_SKIP_DEFAULTS",
    "SUPPORTED_QUANTIZATION_METHODS",
]

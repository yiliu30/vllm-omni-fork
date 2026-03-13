# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DEPRECATED: Use vllm_omni.quantization instead.

This shim will be removed in v0.3.0.
"""

import warnings

warnings.warn(
    "vllm_omni.diffusion.quantization is deprecated. "
    "Use vllm_omni.quantization instead. This shim will be removed in v0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

from vllm_omni.quantization import (  # noqa: F401, E402
    SUPPORTED_QUANTIZATION_METHODS,
    ComponentQuantizationConfig,
    build_quant_config,
)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation utilities for quantization configs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )


logger = init_logger(__name__)


def validate_quant_config(
    config: QuantizationConfig | None,
    model: torch.nn.Module | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> list[str]:
    """Validate a quantization config and return a list of warnings.

    Checks:
    - GPU compute capability meets minimum requirement
    - Activation dtype is supported
    - Component prefixes match model structure (if model provided)

    Args:
        config: The quantization config to validate
        model: Optional model to validate prefix matching against
        dtype: The model dtype to validate against

    Returns:
        List of warning messages (empty if all checks pass)
    """
    if config is None:
        return []

    warnings: list[str] = []

    # Check GPU capability
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        min_cap = config.get_min_capability()
        device_cap = capability[0] * 10 + capability[1]
        if device_cap < min_cap:
            warnings.append(
                f"GPU compute capability {capability} is below minimum "
                f"required {min_cap} for {config.get_name()} quantization"
            )

    # Check dtype support
    supported_dtypes = config.get_supported_act_dtypes()
    if supported_dtypes and dtype not in supported_dtypes:
        warnings.append(
            f"Model dtype {dtype} is not in supported activation dtypes "
            f"{supported_dtypes} for {config.get_name()} quantization"
        )

    # Validate component prefixes against model structure
    if model is not None:
        _validate_component_prefixes(config, model, warnings)

    return warnings


def _validate_component_prefixes(
    config: QuantizationConfig,
    model: torch.nn.Module,
    warnings: list[str],
) -> None:
    """Check that component prefixes match actual model module names."""
    from .component_config import ComponentQuantizationConfig

    if not isinstance(config, ComponentQuantizationConfig):
        return

    model_prefixes = {name.split(".")[0] for name, _ in model.named_modules() if "." in name}

    for comp_prefix in config.component_configs:
        top_level = comp_prefix.split(".")[0]
        if top_level not in model_prefixes:
            warnings.append(
                f"Component prefix '{comp_prefix}' does not match any "
                f"top-level module. Available: {sorted(model_prefixes)}"
            )

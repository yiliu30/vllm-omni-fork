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

    from .component_config import ComponentQuantizationConfig

logger = init_logger(__name__)


def validate_quant_config(
    config: QuantizationConfig | None,
    model: torch.nn.Module | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Validate a quantization config, logging warnings directly."""
    if config is None:
        return

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        min_cap = config.get_min_capability()
        device_cap = capability[0] * 10 + capability[1]
        if device_cap < min_cap:
            logger.warning(
                "GPU compute capability %s is below minimum required %s for %s quantization",
                capability,
                min_cap,
                config.get_name(),
            )

    supported_dtypes = config.get_supported_act_dtypes()
    if supported_dtypes and dtype not in supported_dtypes:
        logger.warning(
            "Model dtype %s is not in supported activation dtypes %s for %s quantization",
            dtype,
            supported_dtypes,
            config.get_name(),
        )

    if model is not None:
        from .component_config import ComponentQuantizationConfig

        if isinstance(config, ComponentQuantizationConfig):
            _validate_component_prefixes(config, model)


def _validate_component_prefixes(
    config: ComponentQuantizationConfig,
    model: torch.nn.Module,
) -> None:
    """Check that component prefixes match actual model module names.

    Note: vLLM may remap quantization prefixes vs model prefixes
    (e.g. via WeightsMapper). This validation checks top-level module
    names only, which may not catch all remapping mismatches.
    """
    model_prefixes = {name.split(".")[0] for name, _ in model.named_modules() if "." in name}

    for comp_prefix in config.component_configs:
        top_level = comp_prefix.split(".")[0]
        if top_level not in model_prefixes:
            logger.warning(
                "Component prefix '%s' does not match any top-level module. "
                "Available: %s. Note: vLLM may remap prefixes via WeightsMapper.",
                comp_prefix,
                sorted(model_prefixes),
            )

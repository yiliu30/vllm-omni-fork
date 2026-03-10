# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-component quantization routing for multi-stage models.

ComponentQuantizationConfig is a QuantizationConfig that routes
get_quant_method() calls to different underlying configs based on the
layer prefix. This enables per-component quantization in multi-stage
models (e.g., transformer at FP8, VAE unquantized).

Prefix matching uses longest-prefix-match semantics:
    {"transformer": fp8_config, "vae": None}
    prefix="transformer.blocks.0.attn.to_q" -> fp8_config
    prefix="vae.encoder.conv_in"             -> None (skip)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )


class ComponentQuantizationConfig(QuantizationConfig):
    """Routes quantization to different configs by layer prefix.

    Args:
        component_configs: Mapping of prefix -> QuantizationConfig.
            Use None as value to skip quantization for that component.
        default_config: Config for prefixes that don't match any component.
            If None, unmatched layers are not quantized.
    """

    def __init__(
        self,
        component_configs: dict[str, QuantizationConfig | None],
        default_config: QuantizationConfig | None = None,
    ) -> None:
        self._components = component_configs
        self._default = default_config
        # Pre-sort by prefix length (longest first) for efficient matching
        self._sorted_prefixes = sorted(self._components.keys(), key=len, reverse=True)

    def _resolve(self, prefix: str) -> QuantizationConfig | None:
        """Find the config for a given layer prefix (longest-prefix match)."""
        for comp_prefix in self._sorted_prefixes:
            if prefix.startswith(comp_prefix):
                return self._components[comp_prefix]
        return self._default

    def get_name(self) -> str:
        return "component"

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        config = self._resolve(prefix)
        if config is None:
            return None
        return config.get_quant_method(layer, prefix)

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Defer to individual component configs at runtime
        return 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ComponentQuantizationConfig:
        raise NotImplementedError("ComponentQuantizationConfig should be built via build_quant_config()")

    def get_config_filenames(self) -> list[str]:
        return []

    @property
    def component_configs(self) -> dict[str, QuantizationConfig | None]:
        return self._components

    @property
    def default_config(self) -> QuantizationConfig | None:
        return self._default

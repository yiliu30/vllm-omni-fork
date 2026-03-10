# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default quantization skip patterns for common model families.

These patterns identify layers that should NOT be quantized, typically:
- Normalization layers (already lightweight)
- Embedding layers (accuracy-sensitive)
- Final projection / output layers

Usage:
    from vllm_omni.quantization.defaults import get_default_skip_patterns

    skip = get_default_skip_patterns("diffusion")
    config = build_quant_config("fp8", ignored_layers=skip)
"""

from __future__ import annotations

# Default skip patterns by model family.
# These are substring patterns matched against layer names.
COMPONENT_SKIP_DEFAULTS: dict[str, list[str]] = {
    # Diffusion transformers: skip normalization and embedding layers
    "diffusion": [
        "norm",
        "layer_norm",
        "group_norm",
        "time_embed",
        "label_emb",
        "pos_embed",
    ],
    # Audio models: skip codec and embedding layers
    "audio": [
        "norm",
        "embed",
        "codec",
    ],
    # Generic: minimal skip set
    "generic": [
        "norm",
    ],
}


def get_default_skip_patterns(family: str = "generic") -> list[str]:
    """Get default skip patterns for a model family.

    Args:
        family: Model family name ("diffusion", "audio", "generic")

    Returns:
        List of layer name patterns to skip during quantization
    """
    return list(COMPONENT_SKIP_DEFAULTS.get(family, COMPONENT_SKIP_DEFAULTS["generic"]))

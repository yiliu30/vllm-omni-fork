# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default quantization skip patterns for common model families."""

from __future__ import annotations

COMPONENT_SKIP_DEFAULTS: dict[str, list[str]] = {
    "diffusion": [
        "norm",
        "layer_norm",
        "group_norm",
        "time_embed",
        "label_emb",
        "pos_embed",
    ],
    "audio": [
        "norm",
        "embed",
        "codec",
    ],
    "generic": [
        "norm",
    ],
}


def get_default_skip_patterns(family: str = "generic") -> list[str]:
    """Get default skip patterns for a model family."""
    return list(COMPONENT_SKIP_DEFAULTS.get(family, COMPONENT_SKIP_DEFAULTS["generic"]))

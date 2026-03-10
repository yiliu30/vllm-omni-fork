# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory for building quantization configs.

build_quant_config() is the single entry point for all quantization
in vLLM-OMNI.  It delegates to vLLM's quantization registry, which
supports ALL methods on ALL platforms.  The framework is completely
method-agnostic, model-agnostic, and platform-agnostic.

The only extension point is _OVERRIDES: a registry for methods that
need a different QuantizationConfig subclass in the OMNI context
(e.g. GGUF uses dequant+GEMM for N-D diffusion tensors instead of
the fused 2D kernel path).  Most methods need NO override.

Examples:
    config = build_quant_config("fp8")
    config = build_quant_config("awq")
    config = build_quant_config("bitsandbytes")
    config = build_quant_config("gptq_marlin")

    config = build_quant_config({"method": "fp8", "activation_scheme": "dynamic"})

    config = build_quant_config({
        "transformer": {"method": "fp8"},
        "vae": None,
    })
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    get_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

from .component_config import ComponentQuantizationConfig

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Override registry.  Only register a method here if the OMNI context
# requires a DIFFERENT QuantizationConfig subclass than what vLLM provides.
# Most methods need NO entry here.
# ---------------------------------------------------------------------------


def _build_gguf(**kw: Any) -> QuantizationConfig:
    """Lazy import to avoid pulling in CUDA/pynvml at module load time."""
    from .gguf_config import DiffusionGGUFConfig

    return DiffusionGGUFConfig(**kw)


_OVERRIDES: dict[str, Callable[..., QuantizationConfig]] = {
    # GGUF: diffusion tensors are N-D, so we need dequant+GEMM instead of
    # the fused 2D kernel path used by LLMs.
    "gguf": _build_gguf,
}

# All supported methods = everything vLLM supports
SUPPORTED_QUANTIZATION_METHODS: list[str] = list(dict.fromkeys(QUANTIZATION_METHODS + list(_OVERRIDES.keys())))


# ---------------------------------------------------------------------------
# Generic instantiation — works for ANY vLLM quantization method
# ---------------------------------------------------------------------------


def _build_single(method: str, **kwargs: Any) -> QuantizationConfig:
    """Build a single quantization config by method name.

    Resolution order:
    1. _OVERRIDES (if OMNI needs a different config subclass)
    2. vLLM's get_quantization_config() → try kwargs → try from_config()
    """
    method = method.lower()

    # 1. OMNI-specific override
    if method in _OVERRIDES:
        return _OVERRIDES[method](**kwargs)

    # 2. vLLM's registry
    if method not in QUANTIZATION_METHODS:
        raise ValueError(f"Unknown quantization method: {method!r}. Supported: {SUPPORTED_QUANTIZATION_METHODS}")

    config_cls = get_quantization_config(method)

    # Try direct construction with kwargs
    if kwargs:
        try:
            return config_cls(**kwargs)
        except TypeError:
            pass
        # Fall back to from_config() (HF checkpoint pattern)
        try:
            return config_cls.from_config(kwargs)
        except (TypeError, KeyError, ValueError):
            sig = inspect.signature(config_cls.__init__)
            raise TypeError(
                f"Cannot instantiate {config_cls.__name__} with kwargs {kwargs}. Expected signature: {sig}"
            ) from None

    # No kwargs — try no-arg construction, fall back to from_config({})
    try:
        return config_cls()
    except TypeError:
        pass
    try:
        return config_cls.from_config({})
    except (TypeError, KeyError, ValueError):
        sig = inspect.signature(config_cls.__init__)
        raise TypeError(
            f"Cannot instantiate {config_cls.__name__} without arguments. "
            f"Expected signature: {sig}. "
            f"Provide constructor kwargs via dict config."
        ) from None


# ---------------------------------------------------------------------------
# Per-component routing
# ---------------------------------------------------------------------------


def _is_per_component_dict(spec: dict[str, Any]) -> bool:
    """Check if a dict describes per-component quantization."""
    if "method" in spec:
        return False
    return all(isinstance(v, (dict, str, type(None))) for v in spec.values())


def _build_component_config(
    spec: dict[str, Any],
) -> ComponentQuantizationConfig:
    """Build ComponentQuantizationConfig from a per-component dict."""
    component_configs: dict[str, QuantizationConfig | None] = {}
    default_config: QuantizationConfig | None = None

    for prefix, value in spec.items():
        if value is None:
            config = None
        elif isinstance(value, str):
            config = _build_single(value)
        elif isinstance(value, dict):
            method = value.pop("method", None)
            if method is None:
                raise ValueError(f"Component '{prefix}' config dict must have a 'method' key")
            config = _build_single(method, **value)
        else:
            raise TypeError(f"Component '{prefix}' config must be str, dict, or None, got {type(value).__name__}")

        if prefix == "default":
            default_config = config
        else:
            component_configs[prefix] = config

    logger.info(
        "Building per-component quantization: %s",
        {k: (v.get_name() if v else None) for k, v in component_configs.items()},
    )
    return ComponentQuantizationConfig(component_configs, default_config)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_quant_config(
    spec: str | dict[str, Any] | QuantizationConfig | None,
    **kwargs: Any,
) -> QuantizationConfig | None:
    """Build a quantization config from a flexible specification.

    Method-agnostic: supports ALL vLLM quantization methods (35+).
    Model-agnostic: ComponentQuantizationConfig routes by prefix.
    Platform-agnostic: kernel selection handled by vLLM internally.

    Args:
        spec: One of:
            - None or "none": No quantization
            - str: Any vLLM method name ("fp8", "awq", "bitsandbytes", ...)
            - dict with "method" key: Single method with params
            - dict without "method" key: Per-component config
            - QuantizationConfig instance: Passthrough
        **kwargs: Extra params merged with dict spec
    """
    if spec is None:
        return None

    if isinstance(spec, QuantizationConfig):
        return spec

    if isinstance(spec, str):
        if spec.lower() == "none":
            return None
        logger.info("Building quantization config: %s", spec)
        return _build_single(spec, **kwargs)

    if isinstance(spec, Mapping):
        spec = dict(spec)

        if _is_per_component_dict(spec):
            return _build_component_config(spec)

        method = spec.pop("method", None)
        if method is None:
            raise ValueError(
                "Dict quantization config must have a 'method' key or "
                "be a per-component config with component prefixes as keys."
            )
        merged = {**spec, **kwargs}
        logger.info("Building quantization config: %s", method)
        return _build_single(method, **merged)

    raise TypeError(f"quantization config must be str, dict, QuantizationConfig, or None, got {type(spec).__name__}")

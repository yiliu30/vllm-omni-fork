# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the unified quantization framework."""

import pytest
import torch

# ---------------------------------------------------------------------------
# build_quant_config — string specs
# ---------------------------------------------------------------------------


def test_build_quant_config_fp8():
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config("fp8")
    assert config is not None
    assert config.get_name() == "fp8"
    assert config.activation_scheme == "dynamic"


def test_build_quant_config_none():
    from vllm_omni.quantization import build_quant_config

    assert build_quant_config(None) is None
    assert build_quant_config("none") is None


def test_build_quant_config_invalid():
    from vllm_omni.quantization import build_quant_config

    with pytest.raises(ValueError, match="Unknown quantization method"):
        build_quant_config("invalid_method")


# ---------------------------------------------------------------------------
# build_quant_config — dict specs
# ---------------------------------------------------------------------------


def test_build_quant_config_dict():
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config({"method": "fp8", "activation_scheme": "static"})
    assert config is not None
    assert config.get_name() == "fp8"
    assert config.activation_scheme == "static"


def test_build_quant_config_dict_not_mutated():
    from vllm_omni.quantization import build_quant_config

    original = {"method": "fp8", "activation_scheme": "static"}
    copy = original.copy()
    build_quant_config(original)
    assert original == copy


# ---------------------------------------------------------------------------
# build_quant_config — per-component specs
# ---------------------------------------------------------------------------


def test_build_quant_config_per_component():
    from vllm_omni.quantization import ComponentQuantizationConfig, build_quant_config

    config = build_quant_config(
        {
            "transformer": {"method": "fp8"},
            "vae": None,
        }
    )
    assert isinstance(config, ComponentQuantizationConfig)
    assert config.component_configs["transformer"].get_name() == "fp8"
    assert config.component_configs["vae"] is None


def test_build_quant_config_per_component_string():
    from vllm_omni.quantization import ComponentQuantizationConfig, build_quant_config

    config = build_quant_config({"transformer": "fp8", "vae": None})
    assert isinstance(config, ComponentQuantizationConfig)
    assert config.component_configs["transformer"].get_name() == "fp8"


# ---------------------------------------------------------------------------
# build_quant_config — passthrough
# ---------------------------------------------------------------------------


def test_build_quant_config_passthrough():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    from vllm_omni.quantization import build_quant_config

    fp8 = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    assert build_quant_config(fp8) is fp8


# ---------------------------------------------------------------------------
# ComponentQuantizationConfig
# ---------------------------------------------------------------------------


def test_component_config_routing():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    from vllm_omni.quantization import ComponentQuantizationConfig

    fp8 = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    config = ComponentQuantizationConfig(
        component_configs={"transformer": fp8, "vae": None},
    )

    assert config.get_name() == "component"
    assert config._resolve("transformer.blocks.0.attn") is fp8
    assert config._resolve("vae.encoder.conv_in") is None
    assert config._resolve("unknown.layer") is None


def test_component_config_with_default():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    from vllm_omni.quantization import ComponentQuantizationConfig

    fp8 = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    config = ComponentQuantizationConfig(
        component_configs={"vae": None},
        default_config=fp8,
    )

    assert config._resolve("transformer.blocks.0") is fp8
    assert config._resolve("vae.encoder") is None


# ---------------------------------------------------------------------------
# GGUF config
# ---------------------------------------------------------------------------


def test_gguf_config():
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.gguf_config import DiffusionGGUFConfig

    config = build_quant_config(
        {
            "method": "gguf",
            "gguf_model": "path/to/model.gguf",
        }
    )
    assert isinstance(config, DiffusionGGUFConfig)
    assert config.gguf_model == "path/to/model.gguf"
    assert config.get_name() == "gguf"


# ---------------------------------------------------------------------------
# OmniDiffusionConfig integration
# ---------------------------------------------------------------------------


def test_integration_string():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(model="test", quantization="fp8")
    assert config.quantization_config is not None
    assert config.quantization_config.get_name() == "fp8"


def test_integration_dict():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(
        model="test",
        quantization_config={"method": "fp8", "activation_scheme": "static"},
    )
    assert config.quantization_config is not None
    assert config.quantization_config.get_name() == "fp8"
    assert config.quantization_config.activation_scheme == "static"


def test_integration_no_quant():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(model="test")
    assert config.quantization_config is None


# ---------------------------------------------------------------------------
# Supported methods
# ---------------------------------------------------------------------------


def test_supported_methods_includes_vllm():
    from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS

    # Must include core vLLM methods
    for method in ["fp8", "gguf", "awq", "gptq", "bitsandbytes"]:
        assert method in SUPPORTED_QUANTIZATION_METHODS, f"{method} missing"


def test_supported_methods_count():
    from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS

    # vLLM has 35+ methods
    assert len(SUPPORTED_QUANTIZATION_METHODS) >= 30


# ---------------------------------------------------------------------------
# Per-component quantization integration test
# ---------------------------------------------------------------------------


def test_per_component_routing_with_model_layers():
    """Verify per-component quantization routes correctly to model layers.

    Builds a ComponentQuantizationConfig with FP8 for 'transformer' and
    None for 'vae', then verifies get_quant_method() returns the correct
    quantization method (or None) based on layer prefix.
    """
    from vllm.model_executor.layers.linear import LinearBase

    from vllm_omni.quantization import ComponentQuantizationConfig, build_quant_config

    # Build per-component config: transformer at FP8, vae unquantized
    config = build_quant_config(
        {
            "transformer": {"method": "fp8"},
            "vae": None,
        }
    )
    assert isinstance(config, ComponentQuantizationConfig)

    # Simulate model layers with different prefixes
    transformer_layer = LinearBase(128, 128)
    vae_layer = LinearBase(128, 128)

    # Transformer layers should get a quantization method
    transformer_method = config.get_quant_method(transformer_layer, "transformer.blocks.0.attn.to_q")
    assert transformer_method is not None, "Transformer layer should be quantized with FP8"

    # VAE layers should NOT get a quantization method
    vae_method = config.get_quant_method(vae_layer, "vae.encoder.conv_in")
    assert vae_method is None, "VAE layer should not be quantized"

    # Unknown prefix should also NOT get a quantization method (no default)
    unknown_method = config.get_quant_method(transformer_layer, "unknown.layer.0.weight")
    assert unknown_method is None, "Unknown prefix should not be quantized"


def test_per_component_routing_with_default():
    """Verify default config applies to unmatched prefixes."""
    from vllm.model_executor.layers.linear import LinearBase

    from vllm_omni.quantization import ComponentQuantizationConfig, build_quant_config

    config = build_quant_config(
        {
            "vae": None,
            "default": "fp8",
        }
    )
    assert isinstance(config, ComponentQuantizationConfig)

    layer = LinearBase(128, 128)

    # VAE should be unquantized (explicit None)
    assert config.get_quant_method(layer, "vae.decoder.conv") is None

    # Everything else should get FP8 from the default
    method = config.get_quant_method(layer, "transformer.blocks.0.attn")
    assert method is not None, "Default FP8 should apply to unmatched prefix"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_quant_config():
    from vllm_omni.quantization import build_quant_config, validate_quant_config

    config = build_quant_config("fp8")
    warnings = validate_quant_config(config, dtype=torch.bfloat16)
    # FP8 supports bfloat16, so no dtype warnings expected
    dtype_warnings = [w for w in warnings if "dtype" in w.lower()]
    assert len(dtype_warnings) == 0


def test_validate_quant_config_none():
    from vllm_omni.quantization import validate_quant_config

    # Validating None should return empty warnings
    assert validate_quant_config(None) == []

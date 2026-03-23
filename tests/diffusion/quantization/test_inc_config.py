# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for INC (AutoRound) quantization config via the unified factory."""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_inc_config_creation():
    """Test that INC config can be created via auto-round method name."""
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config("auto-round", bits=4, group_size=128)
    assert config is not None


def test_inc_config_get_name():
    """Test that get_name() returns 'inc' (delegated from vLLM INCConfig)."""
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config("auto-round", bits=4, group_size=128)
    assert config.get_name() == "inc"


def test_inc_config_delegates_to_vllm():
    """Test that auto-round resolves to vLLM INCConfig correctly."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from vllm_omni.quantization import build_quant_config

    config = build_quant_config("auto-round", bits=4, group_size=128)
    assert isinstance(config, INCConfig)
    assert config.get_name() == INCConfig.get_name()


def test_inc_vllm_config_extraction():
    """Test that build_quant_config returns a valid INCConfig instance."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from vllm_omni.quantization import build_quant_config

    config = build_quant_config("auto-round", bits=4, group_size=128)
    assert config is not None
    assert isinstance(config, INCConfig)
    assert config.weight_bits == 4
    assert config.group_size == 128


def test_inc_config_with_custom_params():
    """Test INC config with custom parameters passed through to INCConfig."""
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config(
        "auto-round",
        bits=4,
        group_size=64,
        sym=False,
        packing_format="auto_round:auto_gptq",
    )
    assert config is not None
    assert config.weight_bits == 4
    assert config.group_size == 64
    assert config.sym is False
    assert config.packing_format == "auto_round:auto_gptq"


def test_auto_round_in_supported_methods():
    """Test that 'auto-round' is listed in supported quantization methods."""
    from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS

    assert "auto-round" in SUPPORTED_QUANTIZATION_METHODS


def test_inc_config_kwargs_passthrough():
    """Test that extra kwargs (backend, data_type) are forwarded to INCConfig."""
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config(
        "auto-round",
        bits=4,
        group_size=128,
        backend="auto",
        data_type="int",
    )
    assert config is not None
    assert config.backend == "auto"
    assert config.data_type == "int"


def test_inc_bits_kwarg_normalization():
    """Test that 'bits' is normalized to 'weight_bits' for INC."""
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config("inc", bits=4, group_size=128)
    assert config.weight_bits == 4
    assert config.group_size == 128

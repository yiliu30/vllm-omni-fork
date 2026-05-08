# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.models.flux.flux_transformer import (
    FluxTransformer2DModel,
    _resolve_flux_precision_sensitive_quant_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeQuantConfig:
    def __init__(self, name: str):
        self._name = name

    def get_name(self) -> str:
        return self._name


class _RecordingParam:
    def __init__(self):
        self.calls: list[tuple[torch.Tensor, str | None]] = []

    def weight_loader(self, _param, loaded_weight, shard_id=None):
        self.calls.append((loaded_weight, shard_id))


class _FakeFluxTransformer:
    load_weights = FluxTransformer2DModel.load_weights

    def __init__(self):
        self._params = {
            "transformer_blocks.0.attn.add_kv_proj.qweight": _RecordingParam(),
            "transformer_blocks.0.attn.to_out.qweight": _RecordingParam(),
            "single_transformer_blocks.0.attn.to_qkv.qweight": _RecordingParam(),
        }

    def named_parameters(self):
        return self._params.items()

    def named_buffers(self):
        return ()


def test_flux_precision_sensitive_quant_config_keeps_autoround():
    quant_config = _FakeQuantConfig("inc")

    assert _resolve_flux_precision_sensitive_quant_config(quant_config) is quant_config


def test_flux_precision_sensitive_quant_config_skips_fp8():
    assert _resolve_flux_precision_sensitive_quant_config(_FakeQuantConfig("fp8")) is None


def test_flux_transformer_load_weights_maps_and_skips_missing_quantized_keys():
    transformer = _FakeFluxTransformer()

    loaded = transformer.load_weights(
        [
            ("transformer_blocks.0.attn.add_q_proj.qweight", torch.ones(1)),
            ("transformer_blocks.0.attn.add_v_proj.qzeros", torch.zeros(1)),
            ("transformer_blocks.0.attn.to_out.0.qweight", torch.ones(1)),
            ("single_transformer_blocks.0.attn.to_q.qweight", torch.ones(1)),
            ("unexpected.weight", torch.ones(1)),
        ]
    )

    assert transformer._params["transformer_blocks.0.attn.add_kv_proj.qweight"].calls[0][1] == "q"
    assert transformer._params["transformer_blocks.0.attn.to_out.qweight"].calls[0][1] is None
    assert transformer._params["single_transformer_blocks.0.attn.to_qkv.qweight"].calls[0][1] == "q"

    assert "transformer_blocks.0.attn.add_q_proj.qweight" in loaded
    assert "transformer_blocks.0.attn.add_kv_proj.qweight" in loaded
    assert "transformer_blocks.0.attn.to_out.0.qweight" in loaded
    assert "transformer_blocks.0.attn.to_out.qweight" in loaded
    assert "single_transformer_blocks.0.attn.to_q.qweight" in loaded
    assert "single_transformer_blocks.0.attn.to_qkv.qweight" in loaded

    assert "transformer_blocks.0.attn.add_v_proj.qzeros" not in loaded
    assert "unexpected.weight" not in loaded

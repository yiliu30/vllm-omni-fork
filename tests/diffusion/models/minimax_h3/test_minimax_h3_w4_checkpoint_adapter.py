# SPDX-License-Identifier: Apache-2.0
"""Focused regression tests for the MiniMax-H3 W4A16 checkpoint contract."""

from types import SimpleNamespace

import torch

from vllm_omni.diffusion.model_loader.checkpoint_adapters.minimax_h3 import (
    MiniMaxH3W4CheckpointAdapter,
    _swap_mlp_fc1_quantized_output,
)


class MiniMaxH3Pipeline:
    def __init__(self):
        parameter = lambda tensor: torch.nn.Parameter(tensor, requires_grad=False)
        self._parameters_for_test = {
            "transformer.blocks.0.attn.qkv_proj.qweight": parameter(torch.empty(1, 768, dtype=torch.int32)),
            "transformer.blocks.0.attn.qkv_proj.qzeros": parameter(torch.empty(1, 96, dtype=torch.int32)),
            "transformer.blocks.0.attn.qkv_proj.scales": parameter(torch.empty(1, 768, dtype=torch.float16)),
            "transformer.blocks.0.mlp.fc1.qweight": parameter(torch.empty(1, 8, dtype=torch.int32)),
            "transformer.blocks.0.mlp.fc1.qzeros": parameter(torch.empty(1, 8, dtype=torch.int32)),
            "transformer.blocks.0.mlp.fc1.scales": parameter(torch.empty(1, 8, dtype=torch.float16)),
            "transformer.video_patch_proj.weight": parameter(torch.empty(8, 4)),
        }

    def named_parameters(self):
        return self._parameters_for_test.items()


def _quant_config():
    return SimpleNamespace(
        get_name=lambda: "inc",
        weight_bits=4,
        data_type="int",
        packing_format="auto_round:auto_gptq",
    )


def test_w4_adapter_fuses_packed_qkv_and_maps_dense_projection():
    model = MiniMaxH3Pipeline()
    source = SimpleNamespace(subfolder="transformer", prefix="transformer.")
    adapter = MiniMaxH3W4CheckpointAdapter(model, source)
    weights = []
    for suffix in ("qweight", "qzeros", "scales"):
        for part, value in zip("qkv", (1, 2, 3)):
            if suffix == "qweight":
                tensor = torch.full((1, 256), value, dtype=torch.int32)
            elif suffix == "qzeros":
                tensor = torch.full((1, 32), 0x77777777, dtype=torch.int32)
            else:
                tensor = torch.ones((1, 256), dtype=torch.float16)
            weights.append((f"transformer.transformer_blocks.0.attn.to_{part}.{suffix}", tensor))
    weights.extend(
        [
            ("transformer.proj_in.qweight", torch.full((1, 8), 0x11111111, dtype=torch.int32)),
            ("transformer.proj_in.qzeros", torch.full((1, 1), 0x77777777, dtype=torch.int32)),
            ("transformer.proj_in.scales", torch.ones((1, 8), dtype=torch.float16)),
        ]
    )

    loaded = dict(adapter.adapt(weights))
    assert loaded["transformer.blocks.0.attn.qkv_proj.qweight"].shape == (1, 768)
    assert loaded["transformer.blocks.0.attn.qkv_proj.qzeros"].shape == (1, 96)
    assert loaded["transformer.blocks.0.attn.qkv_proj.scales"].shape == (1, 768)
    assert loaded["transformer.video_patch_proj.weight"].shape == (8, 4)
    assert loaded["transformer.blocks.0.attn.qkv_proj.qweight"][0].tolist() == [1] * 256 + [2] * 256 + [3] * 256


def test_w4_adapter_swaps_diffusers_fc1_output_axis():
    model = MiniMaxH3Pipeline()
    source = SimpleNamespace(subfolder="transformer", prefix="transformer.")
    adapter = MiniMaxH3W4CheckpointAdapter(model, source)
    values = torch.arange(8).reshape(1, 8)
    loaded = dict(
        adapter.adapt(
            [
                ("transformer.transformer_blocks.0.ff.net.0.proj.qweight", values),
                ("transformer.transformer_blocks.0.ff.net.0.proj.qzeros", values),
                ("transformer.transformer_blocks.0.ff.net.0.proj.scales", values),
            ]
        )
    )
    expected = torch.tensor([[4, 5, 6, 7, 0, 1, 2, 3]])
    assert torch.equal(loaded["transformer.blocks.0.mlp.fc1.qweight"], expected)
    assert torch.equal(_swap_mlp_fc1_quantized_output(values), expected)


def test_rope_frequency_is_derived_and_nonpersistent():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3Rope

    rope = MiniMaxH3Rope(inv_freq_len=16, rope_theta=10000.0)
    expected = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
    assert torch.allclose(rope.inv_freq, expected)
    assert "inv_freq" not in rope.state_dict()

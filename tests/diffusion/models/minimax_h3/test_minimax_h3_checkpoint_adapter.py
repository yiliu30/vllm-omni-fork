# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AutoRound MiniMax-H3 checkpoint adaptation and derived state."""

from types import SimpleNamespace

import torch

from vllm_omni.diffusion.model_loader.checkpoint_adapters.minimax_h3 import (
    MiniMaxH3DenseCheckpointAdapter,
    MiniMaxH3MXFP8CheckpointAdapter,
    MiniMaxH3W4CheckpointAdapter,
    _swap_mlp_fc1_halves,
    _swap_mlp_fc1_quantized_output,
)


class MiniMaxH3Pipeline:
    def __init__(self):
        def parameter(tensor):
            return torch.nn.Parameter(tensor, requires_grad=False)

        self._parameters_for_test = {
            "transformer.blocks.0.attn.qkv_proj.qweight": parameter(torch.empty(1, 768, dtype=torch.int32)),
            "transformer.blocks.0.attn.qkv_proj.qzeros": parameter(torch.empty(1, 96, dtype=torch.int32)),
            "transformer.blocks.0.attn.qkv_proj.scales": parameter(torch.empty(1, 768, dtype=torch.float16)),
            "transformer.video_patch_proj.weight": parameter(torch.empty(8, 4)),
        }

    def named_parameters(self):
        return self._parameters_for_test.items()

    def named_buffers(self):
        return {}.items()


def _quant_config():
    return SimpleNamespace(
        get_name=lambda: "inc",
        weight_bits=4,
        data_type="int",
        packing_format="auto_round:auto_gptq",
    )


def test_w4_adapter_fuses_packed_qkv_and_dequantizes_dense_projection():
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

    qweight = loaded["transformer.blocks.0.attn.qkv_proj.qweight"]
    assert qweight[0].tolist() == [1] * 256 + [2] * 256 + [3] * 256


class MiniMaxH3MXFP8Pipeline:
    def __init__(self):
        def parameter(tensor):
            return torch.nn.Parameter(tensor, requires_grad=False)

        self._parameters_for_test = {
            "transformer.blocks.0.attn.qkv_proj.weight": parameter(torch.empty(12, 4)),
            "transformer.blocks.0.attn.qkv_proj.weight_scale": parameter(torch.empty(12, 1, dtype=torch.uint8)),
            "transformer.blocks.0.mlp.fc1.weight": parameter(torch.empty(8, 4)),
            "transformer.blocks.0.mlp.fc1.weight_scale": parameter(torch.empty(8, 1, dtype=torch.uint8)),
            "transformer.video_patch_proj.weight": parameter(torch.empty(4, 4)),
        }

    def named_parameters(self):
        return self._parameters_for_test.items()


def test_mxfp8_adapter_fuses_qkv_and_dequantizes_dense_projection():
    model = MiniMaxH3MXFP8Pipeline()
    source = SimpleNamespace(subfolder="transformer", prefix="transformer.")
    adapter = MiniMaxH3MXFP8CheckpointAdapter(model, source)
    weights = []
    for part, value in zip("qkv", (1.0, 2.0, 3.0)):
        weights.extend(
            [
                (f"transformer.transformer_blocks.0.attn.to_{part}.weight", torch.full((4, 4), value)),
                (
                    f"transformer.transformer_blocks.0.attn.to_{part}.weight_scale",
                    torch.full((4, 1), 127, dtype=torch.uint8),
                ),
            ]
        )
    weights.extend(
        [
            ("transformer.transformer_blocks.0.ff.net.0.proj.weight", torch.cat((torch.ones(4, 4), torch.full((4, 4), 2.0)))),
            (
                "transformer.transformer_blocks.0.ff.net.0.proj.weight_scale",
                torch.cat((torch.full((4, 1), 126, dtype=torch.uint8), torch.full((4, 1), 127, dtype=torch.uint8))),
            ),
            ("transformer.proj_in.weight", torch.full((4, 4), 2.0, dtype=torch.float8_e4m3fn)),
            ("transformer.proj_in.weight_scale", torch.full((4, 1), 127, dtype=torch.uint8)),
        ]
    )
    loaded = dict(adapter.adapt(weights))
    assert loaded["transformer.blocks.0.attn.qkv_proj.to_q.weight"].shape == (4, 4)
    assert loaded["transformer.blocks.0.attn.qkv_proj.to_k.weight"].shape == (4, 4)
    assert loaded["transformer.blocks.0.attn.qkv_proj.to_v.weight"].shape == (4, 4)
    assert loaded["transformer.blocks.0.attn.qkv_proj.to_q.weight"].eq(1).all()
    assert loaded["transformer.blocks.0.attn.qkv_proj.to_k.weight"].eq(2).all()
    assert loaded["transformer.blocks.0.attn.qkv_proj.to_v.weight"].eq(3).all()
    assert torch.allclose(
        loaded["transformer.video_patch_proj.weight"],
        torch.full((4, 4), 2.0, dtype=torch.float32),
    )
    assert torch.allclose(loaded["transformer.blocks.0.mlp.fc1.diffusers_weight"][:4], torch.ones((4, 4)))
    assert torch.allclose(loaded["transformer.blocks.0.mlp.fc1.diffusers_weight"][4:], torch.full((4, 4), 2.0))
    assert loaded["transformer.blocks.0.mlp.fc1.diffusers_weight_scale"][:4].eq(126).all()
    assert loaded["transformer.blocks.0.mlp.fc1.diffusers_weight_scale"][4:].eq(127).all()


class MiniMaxH3DensePipeline:
    def __init__(self):
        def parameter(tensor):
            return torch.nn.Parameter(tensor, requires_grad=False)

        self._parameters_for_test = {
            "transformer.blocks.0.attn.qkv_proj.weight": parameter(torch.empty(12, 4)),
            "transformer.blocks.0.attn.qkv_proj.bias": parameter(torch.empty(12)),
            "transformer.blocks.0.attn.q_norm.weight": parameter(torch.empty(4)),
            "transformer.blocks.0.mlp.fc1.weight": parameter(torch.empty(8, 4)),
            "transformer.blocks.0.mlp.fc2.weight": parameter(torch.empty(4, 4)),
            "transformer.condition_proj.weight": parameter(torch.empty(4, 4)),
            "transformer.video_patch_proj.weight": parameter(torch.empty(4, 4)),
        }

    def named_parameters(self):
        return self._parameters_for_test.items()

    def named_buffers(self):
        return {}.items()


def test_dense_adapter_maps_diffusers_names_and_marks_fc1_order():
    model = MiniMaxH3DensePipeline()
    source = SimpleNamespace(subfolder="transformer", prefix="transformer.")
    adapter = MiniMaxH3DenseCheckpointAdapter(model, source)
    weights = [
        ("transformer.transformer_blocks.0.attn.to_q.weight", torch.full((4, 4), 1.0)),
        ("transformer.transformer_blocks.0.attn.to_k.bias", torch.full((4,), 2.0)),
        ("transformer.transformer_blocks.0.attn.norm_q.weight", torch.full((4,), 3.0)),
        ("transformer.transformer_blocks.0.ff.net.0.proj.weight", torch.arange(32).reshape(8, 4).float()),
        ("transformer.transformer_blocks.0.ff.net.2.weight", torch.full((4, 4), 4.0)),
        ("transformer.context_embedder.weight", torch.full((4, 4), 5.0)),
        ("transformer.proj_in.weight", torch.full((4, 4), 6.0)),
    ]

    loaded = dict(adapter.adapt(weights))

    assert loaded["transformer.blocks.0.attn.qkv_proj.to_q.weight"].eq(1).all()
    assert loaded["transformer.blocks.0.attn.qkv_proj.to_k.bias"].eq(2).all()
    assert loaded["transformer.blocks.0.attn.q_norm.weight"].eq(3).all()
    assert "transformer.blocks.0.mlp.fc1.diffusers_weight" in loaded
    assert loaded["transformer.blocks.0.mlp.fc2.weight"].eq(4).all()
    assert loaded["transformer.condition_proj.weight"].eq(5).all()
    assert loaded["transformer.video_patch_proj.weight"].eq(6).all()


def test_dense_adapter_preserves_native_fc1_order():
    model = MiniMaxH3DensePipeline()
    source = SimpleNamespace(subfolder="transformer", prefix="transformer.")
    adapter = MiniMaxH3DenseCheckpointAdapter(model, source)

    fc1 = torch.arange(32).reshape(8, 4).float()
    loaded = dict(adapter.adapt([("transformer.blocks.0.mlp.fc1.weight", fc1)]))

    assert "transformer.blocks.0.mlp.fc1.weight" in loaded
    assert "transformer.blocks.0.mlp.fc1.diffusers_weight" not in loaded
    assert torch.equal(loaded["transformer.blocks.0.mlp.fc1.weight"], fc1)


def test_diffusers_fc1_rows_are_swapped_at_native_boundary():
    diffusers_fc1 = torch.arange(16).reshape(8, 2)
    native_fc1 = _swap_mlp_fc1_halves(diffusers_fc1)
    assert torch.equal(native_fc1, torch.cat((diffusers_fc1[4:], diffusers_fc1[:4])))


def test_w4_adapter_swaps_packed_diffusers_fc1_outputs():
    model = MiniMaxH3Pipeline()
    model._parameters_for_test.update(
        {
            "transformer.blocks.0.mlp.fc1.qweight": torch.nn.Parameter(
                torch.empty(1, 8, dtype=torch.int32), requires_grad=False
            ),
            "transformer.blocks.0.mlp.fc1.qzeros": torch.nn.Parameter(
                torch.empty(1, 8, dtype=torch.int32), requires_grad=False
            ),
            "transformer.blocks.0.mlp.fc1.scales": torch.nn.Parameter(
                torch.empty(1, 8), requires_grad=False
            ),
        }
    )
    source = SimpleNamespace(subfolder="transformer", prefix="transformer.")
    adapter = MiniMaxH3W4CheckpointAdapter(model, source)

    loaded = dict(
        adapter.adapt(
            [
                ("transformer.transformer_blocks.0.ff.net.0.proj.qweight", torch.arange(8).reshape(1, 8)),
                ("transformer.transformer_blocks.0.ff.net.0.proj.qzeros", torch.arange(8).reshape(1, 8)),
                ("transformer.transformer_blocks.0.ff.net.0.proj.scales", torch.arange(8).reshape(1, 8)),
            ]
        )
    )

    expected = torch.tensor([[4, 5, 6, 7, 0, 1, 2, 3]])
    assert torch.equal(loaded["transformer.blocks.0.mlp.fc1.qweight"], expected)
    assert torch.equal(loaded["transformer.blocks.0.mlp.fc1.qzeros"], expected)
    assert torch.equal(loaded["transformer.blocks.0.mlp.fc1.scales"], expected)
    assert torch.equal(_swap_mlp_fc1_quantized_output(torch.arange(8).reshape(1, 8)), expected)


def test_rope_is_initialized_without_checkpoint_buffer():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3Rope

    rope = MiniMaxH3Rope(inv_freq_len=16, rope_theta=10000.0)
    expected = 1.0 / (
        10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32)
    )

    assert torch.allclose(rope.inv_freq, expected)
    assert "inv_freq" not in rope.state_dict()

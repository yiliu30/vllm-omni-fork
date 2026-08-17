# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint adaptation for AutoRound's Diffusers MiniMax-H3 W4A16 export."""

from __future__ import annotations

import re
from collections.abc import Generator, Iterable

import torch
from torch import nn

_QKV_RE = re.compile(
    r"^(?P<prefix>transformer\.(?:transformer_blocks|token_refiner\.refiner_blocks)\."
    r"(?P<index>\d+)\.attn)\.to_(?P<part>[qkv])\.(?P<suffix>weight|qweight|qzeros|scales)$"
)
_QUANT_RE = re.compile(r"^(?P<base>.+)\.(?P<suffix>qweight|qzeros|scales)$")
_MXFP8_QKV_RE = re.compile(
    r"^(?P<prefix>transformer\.(?:transformer_blocks|token_refiner\.refiner_blocks)\."
    r"(?P<index>\d+)\.attn)\.to_(?P<part>[qkv])\.(?P<suffix>weight|weight_scale)$"
)
_DENSE_QKV_RE = re.compile(
    r"^(?P<prefix>transformer\.(?:transformer_blocks|token_refiner\.refiner_blocks)\."
    r"(?P<index>\d+)\.attn)\.to_(?P<part>[qkv])\.(?P<suffix>weight|bias)$"
)
_PACK_FACTOR = 8  # AutoRound W4A16 GPTQ packing factor.


def _native_name(name: str) -> str:
    """Map one non-fused Diffusers H3 parameter to Omni's native name."""
    replacements = (
        ("transformer.transformer_blocks.", "transformer.blocks."),
        ("transformer.token_refiner.refiner_blocks.", "transformer.token_refiner.blocks."),
        (".attn.norm_q.", ".attn.q_norm."),
        (".attn.norm_k.", ".attn.k_norm."),
        (".attn.to_out.0.", ".attn.out_proj."),
        (".ff.net.0.proj.", ".mlp.fc1."),
        (".ff.net.2.", ".mlp.fc2."),
        ("transformer.proj_in.", "transformer.video_patch_proj."),
        ("transformer.audio_proj_in.", "transformer.audio_patch_proj."),
        ("transformer.context_embedder.", "transformer.condition_proj."),
        ("transformer.time_embedder.linear_1.", "transformer.time_embedder.proj_in."),
        ("transformer.time_embedder.linear_2.", "transformer.time_embedder.proj_out."),
        ("transformer.proj_out.", "transformer.final_layer.video_out."),
        ("transformer.audio_proj_out.", "transformer.final_layer.audio_out."),
        ("transformer.norm_out.norm.", "transformer.final_layer.norm."),
        ("transformer.norm_out.linear.", "transformer.final_layer.adaln_proj.linear."),
    )
    for old, new in replacements:
        if name.startswith(old) or (old.startswith(".") and old in name):
            name = name.replace(old, new, 1)
    return name


def _unpack_nibbles(tensor: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Unpack uint4 values from int32 words along ``dim``."""
    tensor = tensor.to(torch.int64)
    shifts = torch.arange(_PACK_FACTOR, device=tensor.device, dtype=torch.int64) * 4
    values = [(tensor >> int(shift)) & 0xF for shift in shifts]
    return torch.stack(values, dim=dim + 1).flatten(dim, dim + 1)


def _pack_nibbles(tensor: torch.Tensor, *, dim: int) -> torch.Tensor:
    if tensor.shape[dim] % _PACK_FACTOR:
        raise ValueError(f"W4 tensor dimension {tensor.shape[dim]} is not divisible by {_PACK_FACTOR}")
    moved = tensor.movedim(dim, -1)
    groups = moved.reshape(*moved.shape[:-1], -1, _PACK_FACTOR)
    shifts = torch.arange(_PACK_FACTOR, device=tensor.device, dtype=torch.int64) * 4
    packed = ((groups.to(torch.int64) & 0xF) << shifts).sum(dim=-1)
    return packed.movedim(-1, dim).to(torch.int32).contiguous()


def _fuse_qkv_packed(parts: list[torch.Tensor], suffix: str) -> torch.Tensor:
    if suffix in {"qweight", "scales"}:
        # Packed weights bypass MiniMaxH3DiTModel.load_weights' dense QKV
        # reorder branch and therefore must already be in runtime order.
        return torch.cat(parts, dim=1).contiguous()
    if suffix == "qzeros":
        unpacked = [_unpack_nibbles(part, dim=1) for part in parts]
        return _pack_nibbles(torch.cat(unpacked, dim=1), dim=1)
    raise ValueError(f"Unsupported packed QKV suffix {suffix!r}")


def _fuse_qkv_dense(parts: list[torch.Tensor], *, head_dim: int = 1) -> torch.Tensor:
    # Native H3's loader converts grouped [q0,k0,v0,...] rows to the standard
    # QKV layout. Diffusers stores each projection as contiguous Q/K/V rows.
    if any(part.shape[0] % head_dim for part in parts):
        raise ValueError(
            f"QKV output dimension must be divisible by head_dim={head_dim}: "
            f"{[tuple(part.shape) for part in parts]}"
        )
    heads = parts[0].shape[0] // head_dim
    reshaped = [part.reshape(heads, head_dim, part.shape[1]) for part in parts]
    return torch.stack(reshaped, dim=1).reshape(-1, parts[0].shape[1]).contiguous()


def _dequantize_gptq(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Return a dense [input, output] matrix from AutoRound GPTQ tensors."""
    codes = _unpack_nibbles(qweight, dim=0)
    zeros = _unpack_nibbles(qzeros, dim=1)
    group_size = codes.shape[0] // scales.shape[0]
    if group_size <= 0 or group_size * scales.shape[0] < codes.shape[0]:
        raise ValueError(
            f"Cannot infer W4 group size from qweight={tuple(qweight.shape)} and scales={tuple(scales.shape)}"
        )
    scales_expanded = scales.to(torch.float32).repeat_interleave(group_size, dim=0)[: codes.shape[0]]
    zeros_expanded = zeros.repeat_interleave(group_size, dim=0)[: codes.shape[0]]
    # AutoRound's GPTQ exporter stores qzeros as (zero_point - 1) & 0xF.
    zeros_expanded = (zeros_expanded + 1) & 0xF
    return (codes.to(torch.float32) - zeros_expanded.to(torch.float32)) * scales_expanded


def _dequantize_mxfp8(weight: torch.Tensor, weight_scale: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize AutoRound's llm-compressor MXFP8 [out, in] representation."""
    scale = torch.pow(2.0, weight_scale.to(torch.int16) - 127).repeat_interleave(32, dim=1)
    return (weight.to(torch.bfloat16) * scale[:, : weight.shape[1]]).to(target_dtype).contiguous()


def _swap_mlp_fc1_halves(tensor: torch.Tensor) -> torch.Tensor:
    """Convert Diffusers [up, gate] rows to native H3 [gate, up] rows."""
    if tensor.shape[0] % 2:
        raise ValueError(f"MLP fc1 output dimension must be even, got {tuple(tensor.shape)}")
    first, second = tensor.chunk(2, dim=0)
    return torch.cat((second, first), dim=0).contiguous()


def _swap_mlp_fc1_quantized_output(tensor: torch.Tensor) -> torch.Tensor:
    """Convert packed Diffusers [up, gate] outputs to native [gate, up].

    AutoRound's GPTQ tensors keep the output-feature axis as dimension 1 for
    qweight, qzeros, and scales.  MiniMax H3's MergedColumnParallelLinear
    expects gate followed by up on that axis.  Its 14,336-feature halves are
    aligned to the GPTQ packing factor, so exchanging the packed columns also
    exchanges the underlying 4-bit output features without unpacking.
    """
    if tensor.ndim < 2 or tensor.shape[1] % 2:
        raise ValueError(
            "MiniMax H3 quantized fc1 output axis must split evenly, got "
            f"{tuple(tensor.shape)}"
        )
    first, second = tensor.chunk(2, dim=1)
    return torch.cat((second, first), dim=1).contiguous()


class MiniMaxH3MXFP8CheckpointAdapter:
    """Translate AutoRound MXFP8 Diffusers tensors to native H3 tensors."""

    def __init__(self, model: nn.Module, source: object):
        self._parameters = dict(model.named_parameters())
        self._loadable = set(self._parameters)

    @classmethod
    def is_compatible(
        cls,
        model: nn.Module,
        source: object,
        quant_config: object | None,
        use_safetensors: bool,
    ) -> bool:
        return (
            use_safetensors
            and model.__class__.__name__ == "MiniMaxH3Pipeline"
            and getattr(source, "subfolder", None) == "transformer"
            and getattr(quant_config, "get_name", lambda: None)() == "inc"
            and getattr(quant_config, "weight_bits", None) == 8
            and getattr(quant_config, "data_type", None) == "mx_fp"
            and "llm_compressor" in getattr(quant_config, "packing_format", "")
        )

    @staticmethod
    def _qkv_target_base(prefix: str) -> str:
        if prefix.startswith("transformer.transformer_blocks"):
            return prefix.replace("transformer.transformer_blocks", "transformer.blocks", 1) + ".qkv_proj"
        return (
            prefix.replace("transformer.token_refiner.refiner_blocks", "transformer.token_refiner.blocks", 1)
            + ".qkv_proj"
        )

    def _target_for(self, name: str) -> str:
        target = _native_name(name)
        if target in self._loadable:
            return target
        raise KeyError(f"MiniMax-H3 MXFP8 adapter mapped {name!r} to missing target {target!r}")

    def adapt(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Generator[tuple[str, torch.Tensor], None, None]:
        dense_pending: dict[str, dict[str, torch.Tensor]] = {}

        def flush_dense(target_base: str) -> Generator[tuple[str, torch.Tensor], None, None]:
            values = dense_pending.get(target_base, {})
            if set(values) != {"weight", "weight_scale"}:
                return
            target_name = f"{target_base}.weight"
            if target_name not in self._loadable:
                raise KeyError(f"MiniMax-H3 MXFP8 adapter mapped to missing target {target_name!r}")
            yield target_name, _dequantize_mxfp8(
                values["weight"], values["weight_scale"], self._parameters[target_name].dtype
            )
            dense_pending.pop(target_base, None)

        for name, tensor in weights:
            qkv_match = _MXFP8_QKV_RE.match(name)
            if qkv_match:
                target_base = self._qkv_target_base(qkv_match.group("prefix"))
                suffix = qkv_match.group("suffix")
                yield f"{target_base}.to_{qkv_match.group('part')}.{suffix}", tensor
                continue

            if name.endswith((".weight", ".weight_scale")):
                suffix = "weight" if name.endswith(".weight") else "weight_scale"
                source_base = name[: -len("." + suffix)]
                target_base = _native_name(source_base + ".weight")[: -len(".weight")]
                if target_base.endswith(".mlp.fc1"):
                    target_suffix = "diffusers_weight" if suffix == "weight" else "diffusers_weight_scale"
                    yield f"{target_base}.{target_suffix}", tensor
                    continue
                # AutoRound leaves patch, timestep, conditioning, and output
                # projections dense while still serializing the MXFP8 scale
                # pair.  Convert those pairs back to a single dense weight;
                # feeding ``weight_scale`` to an unquantized layer would leave
                # the model with an unexpected parameter and an uninitialized
                # scale in the quantized path.
                if f"{target_base}.weight" in self._loadable and f"{target_base}.weight_scale" not in self._loadable:
                    dense_pending.setdefault(target_base, {})[suffix] = tensor
                    yield from flush_dense(target_base)
                    continue
                target_name = f"{target_base}.{suffix}"
                if target_name not in self._loadable:
                    raise KeyError(f"MiniMax-H3 MXFP8 adapter mapped {name!r} to missing target {target_name!r}")
                yield target_name, tensor
                continue

            yield self._target_for(name), tensor

        incomplete = {key: sorted(value) for key, value in dense_pending.items() if value}
        if incomplete:
            raise ValueError(f"Incomplete MiniMax-H3 MXFP8 groups: {incomplete}")


class MiniMaxH3DenseCheckpointAdapter:
    """Translate dense Diffusers MiniMax-H3 tensors to native Omni names.

    This adapter intentionally has no quantization behavior.  It is used for
    BF16 checkpoints produced by dequantizing an AutoRound MXFP8 checkpoint so
    that Diffusers/Omni model parity can be tested independently of kernels.
    """

    def __init__(self, model: nn.Module, source: object):
        self._loadable = set(dict(model.named_parameters())) | set(dict(model.named_buffers()))

    @classmethod
    def is_compatible(
        cls,
        model: nn.Module,
        source: object,
        quant_config: object | None,
        use_safetensors: bool,
    ) -> bool:
        return (
            use_safetensors
            and model.__class__.__name__ == "MiniMaxH3Pipeline"
            and getattr(source, "subfolder", None) == "transformer"
            and quant_config is None
        )

    @staticmethod
    def _qkv_target(name: str, part: str) -> str:
        prefix = name.rsplit(".to_", 1)[0]
        if name.startswith("transformer.transformer_blocks"):
            prefix = prefix.replace("transformer.transformer_blocks", "transformer.blocks", 1)
        else:
            prefix = prefix.replace("transformer.token_refiner.refiner_blocks", "transformer.token_refiner.blocks", 1)
        return f"{prefix}.qkv_proj.to_{part}"

    def _target_for(self, name: str) -> str:
        target = _native_name(name)
        if target in self._loadable:
            return target
        raise KeyError(f"MiniMax-H3 dense adapter mapped {name!r} to missing target {target!r}")

    def adapt(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Generator[tuple[str, torch.Tensor], None, None]:
        for name, tensor in weights:
            qkv_match = _DENSE_QKV_RE.match(name)
            if qkv_match:
                yield f"{self._qkv_target(name, qkv_match.group('part'))}.{qkv_match.group('suffix')}", tensor
                continue

            target = _native_name(name)
            is_diffusers_fc1 = name.startswith(
                (
                    "transformer.transformer_blocks.",
                    "transformer.token_refiner.refiner_blocks.",
                )
            )
            # Diffusers SwiGLU stores [up, gate]; MiniMax-H3DiTModel.load_weights
            # swaps the two output shards when this marker is present. Native
            # FL2VA checkpoints already store [gate, up], so preserve their
            # original ``blocks.*.mlp.fc1`` names and ordering.
            if is_diffusers_fc1 and target.endswith(".mlp.fc1.weight"):
                target = target.removesuffix(".weight") + ".diffusers_weight"
            elif is_diffusers_fc1 and target.endswith(".mlp.fc1.bias"):
                target = target.removesuffix(".bias") + ".diffusers_bias"

            if target not in self._loadable:
                # These logical names are consumed by MiniMaxH3DiTModel.load_weights
                # and are not parameters themselves.
                if target.endswith((".mlp.fc1.diffusers_weight", ".mlp.fc1.diffusers_bias")):
                    yield target, tensor
                    continue
                raise KeyError(f"MiniMax-H3 dense adapter mapped {name!r} to missing target {target!r}")
            yield target, tensor


class MiniMaxH3W4CheckpointAdapter:
    """Translate AutoRound Diffusers W4A16 tensors to native H3 tensors.

    Native H3 uses fused QKV/MLP modules. Quantized QKV tensors are fused in
    packed space; top-level projection tensors are dequantized because the
    native H3 implementation intentionally keeps those projections dense.
    """

    def __init__(self, model: nn.Module, source: object):
        self._parameters = dict(model.named_parameters())
        self._loadable = set(self._parameters)

    @classmethod
    def is_compatible(
        cls,
        model: nn.Module,
        source: object,
        quant_config: object | None,
        use_safetensors: bool,
    ) -> bool:
        return (
            use_safetensors
            and model.__class__.__name__ == "MiniMaxH3Pipeline"
            and getattr(source, "subfolder", None) == "transformer"
            and getattr(quant_config, "get_name", lambda: None)() == "inc"
            and getattr(quant_config, "weight_bits", None) == 4
            and getattr(quant_config, "data_type", None) == "int"
            and "gptq" in getattr(quant_config, "packing_format", "")
        )

    def _target_for(self, name: str) -> str:
        target = _native_name(name)
        if target in self._loadable:
            return target
        raise KeyError(f"MiniMax-H3 W4 adapter mapped {name!r} to missing target {target!r}")

    def _flush_quantized(
        self,
        target_base: str,
        pending: dict[tuple[str, str], dict[str, torch.Tensor]],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        target_q_names = {suffix: f"{target_base}.{suffix}" for suffix in ("qweight", "qzeros", "scales")}
        if target_q_names["qweight"] in self._loadable:
            for suffix, target_name in target_q_names.items():
                values = pending.get((target_base, suffix))
                if values is not None and set(values) == {"q", "k", "v"}:
                    yield target_name, _fuse_qkv_packed([values[p] for p in ("q", "k", "v")], suffix)
                    pending.pop((target_base, suffix), None)
            return

        target_name = f"{target_base}.weight"
        if target_name not in self._loadable:
            return
        values = [pending.get((target_base, suffix), {}) for suffix in ("qweight", "qzeros", "scales")]
        if not all(set(item) == {"q", "k", "v"} for item in values):
            return
        fused = _dequantize_gptq(
            _fuse_qkv_packed([values[0][p] for p in ("q", "k", "v")], "qweight"),
            _fuse_qkv_packed([values[1][p] for p in ("q", "k", "v")], "qzeros"),
            _fuse_qkv_packed([values[2][p] for p in ("q", "k", "v")], "scales"),
        )
        target_shape = self._parameters[target_name].shape
        fused = fused[: target_shape[1], :].transpose(0, 1).to(dtype=self._parameters[target_name].dtype)
        for suffix in ("qweight", "qzeros", "scales"):
            pending.pop((target_base, suffix), None)
        yield target_name, fused.contiguous()

    def adapt(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Generator[tuple[str, torch.Tensor], None, None]:
        pending: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

        for name, tensor in weights:
            match = _QKV_RE.match(name)
            if match:
                prefix = match.group("prefix")
                if prefix.startswith("transformer.transformer_blocks"):
                    target_base = prefix.replace("transformer.transformer_blocks", "transformer.blocks", 1)
                else:
                    target_base = prefix.replace(
                        "transformer.token_refiner.refiner_blocks", "transformer.token_refiner.blocks", 1
                    )
                target_base += ".qkv_proj"
                suffix = match.group("suffix")
                pending.setdefault((target_base, suffix), {})[match.group("part")] = tensor
                yield from self._flush_quantized(target_base, pending)
                if suffix == "weight" and set(pending.get((target_base, suffix), {})) == {"q", "k", "v"}:
                    values = pending.pop((target_base, suffix))
                    yield f"{target_base}.weight", _fuse_qkv_dense([values[p] for p in ("q", "k", "v")])
                continue

            quant_match = _QUANT_RE.match(name)
            if quant_match:
                source_base = quant_match.group("base")
                suffix = quant_match.group("suffix")
                target_base = _native_name(source_base + ".weight")[: -len(".weight")]
                if target_base.endswith(".mlp.fc1"):
                    # Diffusers exports SwiGLU as [up, gate], whereas native
                    # H3's fused fc1 uses [gate, up].  The dense loader uses
                    # a logical marker to make this swap; W4A16 needs the
                    # equivalent reorder in packed GPTQ output space.
                    tensor = _swap_mlp_fc1_quantized_output(tensor)
                pending.setdefault((target_base, suffix), {})["value"] = tensor
                values = {
                    item: pending.get((target_base, item), {}).get("value")
                    for item in ("qweight", "qzeros", "scales")
                }
                target_q_name = f"{target_base}.qweight"
                if target_q_name in self._loadable:
                    yield f"{target_base}.{suffix}", tensor
                    pending.pop((target_base, suffix), None)
                elif f"{target_base}.weight" in self._loadable and all(value is not None for value in values.values()):
                    dense = _dequantize_gptq(values["qweight"], values["qzeros"], values["scales"])
                    target = self._parameters[f"{target_base}.weight"]
                    dense = dense[: target.shape[1], :].transpose(0, 1).to(dtype=target.dtype).contiguous()
                    yield f"{target_base}.weight", dense
                    for item in ("qweight", "qzeros", "scales"):
                        pending.pop((target_base, item), None)
                continue

            yield self._target_for(name), tensor

        incomplete = {key: sorted(value) for key, value in pending.items() if value}
        if incomplete:
            raise ValueError(f"Incomplete MiniMax-H3 W4 quantized groups: {incomplete}")

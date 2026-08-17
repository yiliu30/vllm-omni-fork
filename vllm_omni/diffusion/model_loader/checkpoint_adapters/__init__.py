# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from torch import nn

from .minimax_h3 import MiniMaxH3W4CheckpointAdapter
from .modelopt import (
    ModelOptFp8CheckpointAdapter,
    ModelOptMixedPrecisionCheckpointAdapter,
    ModelOptNvFp4CheckpointAdapter,
)


def get_checkpoint_adapter(
    model: nn.Module,
    source: object,
    quant_config: object | None,
    use_safetensors: bool,
) -> MiniMaxH3W4CheckpointAdapter | ModelOptFp8CheckpointAdapter | ModelOptNvFp4CheckpointAdapter | ModelOptMixedPrecisionCheckpointAdapter | None:
    if MiniMaxH3W4CheckpointAdapter.is_compatible(model, source, quant_config, use_safetensors):
        return MiniMaxH3W4CheckpointAdapter(model, source)
    if ModelOptFp8CheckpointAdapter.is_compatible(source, quant_config, use_safetensors):
        return ModelOptFp8CheckpointAdapter(model, source)
    if ModelOptNvFp4CheckpointAdapter.is_compatible(source, quant_config, use_safetensors):
        return ModelOptNvFp4CheckpointAdapter(model, source)
    if ModelOptMixedPrecisionCheckpointAdapter.is_compatible(source, quant_config, use_safetensors):
        return ModelOptMixedPrecisionCheckpointAdapter(model, source)
    return None


__all__ = [
    "ModelOptFp8CheckpointAdapter",
    "ModelOptMixedPrecisionCheckpointAdapter",
    "ModelOptNvFp4CheckpointAdapter",
    "MiniMaxH3W4CheckpointAdapter",
    "get_checkpoint_adapter",
]

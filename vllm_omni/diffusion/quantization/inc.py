# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPTQ quantization config for diffusion transformers."""

from typing import Any

from vllm.model_executor.layers.quantization.inc import INCConfig

from .base import DiffusionQuantizationConfig


class DiffusionINCConfig(DiffusionQuantizationConfig):
    """INC quantization config optimized for diffusion transformers.

    INC (Integer Neural Compression) provides 4-bit weight quantization with 16-bit
    activations (w4a16) for significant memory reduction with minimal quality loss.

    This implementation supports:
        - 4-bit weight quantization (w4a16) - recommended for diffusion models
        - Multiple group sizes for fine-grained quantization
        - Compatibility with GPTQ-quantized checkpoints from AutoGPTQ

    Device Compatibility:
        - Requires CUDA compute capability 6.0+
        - Optimized for modern GPUs with tensor cores

    Args:
        weight_bits: Number of bits for weight quantization (4 recommended)
        group_size: Group size for quantization. Smaller = higher quality but more memory.
            Common values: 128, 64, 32, or -1 for per-channel
        desc_act: Whether to use descending order for activation quantization
        lm_head_quantized: Whether to quantize the LM head (output projection)
        ignored_layers: List of layer name patterns to skip quantization
    """

    # Tight coupling with vLLM's GPTQMarlinConfig - delegates get_name() and get_min_capability()
    quant_config_cls = INCConfig

    def __init__(
        self,
        bits: int,
        group_size: int,
        sym: bool = True,
        packing_format: str = "auto_round:auto_gptq",
        block_name_to_quantize: str | list[str] | None = None,
        extra_config: dict[str, Any] | None = None,
        data_type: str = "int",
        backend: str = "auto",
        **kwargs,
    ):
        # Create underlying vLLM GPTQ config
        vllm_inc_quant_config = INCConfig(
            weight_bits=bits,
            group_size=group_size,
            sym=sym,
            packing_format=packing_format,
            block_name_to_quantize=block_name_to_quantize,
            extra_config=extra_config,
            data_type=data_type,
            backend=backend,
        )
        self._vllm_config = vllm_inc_quant_config
        # self._vllm_config.modules_in_block_to_quantize = None

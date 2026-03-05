# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPTQ quantization config for diffusion transformers."""

from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig

from .base import DiffusionQuantizationConfig


class DiffusionGPTQMarlinConfig(DiffusionQuantizationConfig):
    """GPTQ-Marlin quantization config optimized for diffusion transformers.

    GPTQ-Marlin (GPT-Quantization Marlin) provides 4-bit weight quantization with 16-bit
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
    quant_config_cls = GPTQMarlinConfig

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
        lm_head_quantized: bool = False,
        dynamic: dict[str, dict[str, int | bool]] | None = None,
        ignored_layers: list[str] | None = None,
        **kwargs,
    ):
        self.bits = bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.dynamic = dynamic or {}
        self.ignored_layers = ignored_layers or []

        # Validate parameters
        if bits not in [2, 3, 4, 8]:
            raise ValueError(f"Unsupported bits: {bits}. Supported: [2, 3, 4, 8]")

        if group_size not in [-1, 32, 64, 128, 256, 512, 1024]:
            raise ValueError(f"Unsupported group_size: {group_size}. Supported: [-1, 32, 64, 128, 256, 512, 1024]")

        # Create underlying vLLM GPTQ config
        quant_args_marlin = GPTQMarlinConfig(
            weight_bits=bits,
            group_size=group_size,
            is_sym=True,
            lm_head_quantized=False,
            desc_act=False,
            dynamic={},
            full_config={},
        )
        self._vllm_config = quant_args_marlin
        # self._vllm_config = GPTQMarlinConfig.from_config(quant_args_marlin)
        self._vllm_config.modules_in_block_to_quantize = None
    
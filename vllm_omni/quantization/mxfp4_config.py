# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A4 MXFP4 (Microscaling FP4) online/offline quantization for diffusion transformers.

True OCP Microscaling FP4 format:
  - Weight: uint8 packed (2 × FP4 E2M1 per byte), shape [N, K/2]
  - Scale: uint8 (E8M0 exponent-only), per-group with block_size=32, shape [N, K/32]
  - No global scale (unlike NVFP4)

CUDA Kernel Backends (auto-selected via init_mxfp4_linear_kernel()):
  - FlashInfer (SM100+ Blackwell): true W4A4 GEMM
  - Marlin (SM80+ Ampere/Hopper): W4A16 weight-only GEMM

Architecture:

  CUDAMxfp4LinearMethod          – offline: pre-quantized MXFP4 checkpoint
    CUDAMxfp4OnlineLinearMethod   – online: _LazyWeightMixin + BF16 → MXFP4 at load time
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import (
    _copy_missing_attrs,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.model_loader.weight_utils import (
    initialize_single_dummy_weight,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter

from .mxfp8_config import _LazyWeightMixin

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

_MXFP4_GROUP_SIZE = 32


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class DiffusionMXFP4Config(QuantizationConfig):
    """W4A4 MXFP4 quantization config for diffusion transformers.

    Supports both online (BF16 checkpoint → quantize at load time) and offline
    (pre-quantized MXFP4 checkpoint) modes.

    MX (microscaling) format: groups of 32 K-dimension elements share one
    E8M0 exponent scale, matching MXFP4 as defined in the OCP MX spec.
    """

    def __init__(
        self,
        is_checkpoint_mxfp4_serialized: bool = False,
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        self.ignored_layers = ignored_layers or []
        self._kernel = None

    @property
    def kernel(self):
        """Lazily initialize and cache the MXFP4 GEMM kernel."""
        if self._kernel is None:
            from vllm.model_executor.kernels.linear import (
                init_mxfp4_linear_kernel,
            )

            self._kernel = init_mxfp4_linear_kernel()
        return self._kernel

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        if self.ignored_layers:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(
                self.ignored_layers
            )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionMXFP4Config:
        is_serialized = cls.get_from_keys_or(
            config, ["is_checkpoint_mxfp4_serialized"], False
        )
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        return cls(
            is_checkpoint_mxfp4_serialized=is_serialized,
            ignored_layers=ignored_layers,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if self.is_checkpoint_mxfp4_serialized:
                return CUDAMxfp4LinearMethod(self)
            return CUDAMxfp4OnlineLinearMethod(self)
        return None


# ---------------------------------------------------------------------------
# Weight quantization: BF16 → MXFP4
# ---------------------------------------------------------------------------


def _quantize_weight_mxfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16/FP16 weight tensor to MXFP4 format.

    Uses compressed-tensors' MX scale generation and FP4 E2M1 quantization
    to produce the standard kernel-agnostic format.

    Steps:
      1. Compute per-group (block_size=32) max absolute values
      2. Generate E8M0 scales via power-of-2 rounding
      3. Quantize: divide by scale, round to nearest FP4 E2M1 value
      4. Pack two FP4 nibbles per uint8 byte

    Returns:
        (weight_packed, weight_scale):
          weight_packed: [N, K/2] uint8 (two E2M1 nibbles per byte)
          weight_scale:  [N, K/32] uint8 (E8M0 shared exponents)
    """
    from compressed_tensors.compressors.mx_utils import decompress_mx_scale
    from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8
    from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA
    from compressed_tensors.quantization.utils.mxfp_utils import (
        generate_mx_scales,
    )

    N, K = weight.shape
    assert K % _MXFP4_GROUP_SIZE == 0, (
        f"K={K} must be divisible by {_MXFP4_GROUP_SIZE}"
    )
    num_groups = K // _MXFP4_GROUP_SIZE

    # 1. Per-group max absolute value
    weight_groups = weight.reshape(N, num_groups, _MXFP4_GROUP_SIZE)
    amax = weight_groups.abs().amax(dim=-1)  # [N, num_groups]

    # 2. E8M0 scales (uint8 biased exponents, bias=127)
    weight_scale = generate_mx_scales(amax.float(), num_bits=4).to(torch.uint8)

    # 3. Quantize: scale → float, divide, round to FP4 E2M1
    scale_float = decompress_mx_scale(weight_scale)  # [N, num_groups] bfloat16
    scale_expanded = scale_float.unsqueeze(-1).expand_as(weight_groups)
    scaled = weight_groups.float() / scale_expanded.float()
    quantized = FP4_E2M1_DATA.cast_to_fp4(
        scaled.clamp(FP4_E2M1_DATA.min, FP4_E2M1_DATA.max)
    )

    # 4. Pack pairs of FP4 values into uint8
    weight_packed = pack_fp4_to_uint8(quantized.reshape(N, K))

    return weight_packed, weight_scale


# ---------------------------------------------------------------------------
# Offline method (pre-quantized MXFP4 checkpoint)
# ---------------------------------------------------------------------------


class CUDAMxfp4LinearMethod(LinearMethodBase):
    """CUDA MXFP4 offline linear method for pre-quantized checkpoints.

    Delegates weight processing and GEMM to the kernel selected by
    init_mxfp4_linear_kernel() (FlashInfer on SM100+, Marlin on SM80+).
    """

    def __init__(self, quant_config: DiffusionMXFP4Config) -> None:
        self.quant_config = quant_config
        self.kernel = quant_config.kernel

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.params_dtype = params_dtype

        # Packed FP4 weights (2 values per byte)
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // 2,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

        # Per-group E8M0 scales
        num_groups = (
            input_size_per_partition + _MXFP4_GROUP_SIZE - 1
        ) // _MXFP4_GROUP_SIZE
        layer.register_parameter(
            "weight_scale",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    num_groups,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(
            layer, "_already_called_process_weights_after_loading", False
        ):
            return
        self.kernel.process_weights_after_loading(layer)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


# ---------------------------------------------------------------------------
# Online method (BF16 checkpoint → MXFP4 at load time)
# ---------------------------------------------------------------------------


class CUDAMxfp4OnlineLinearMethod(_LazyWeightMixin, CUDAMxfp4LinearMethod):
    """CUDA MXFP4 online linear method.

    MRO: CUDAMxfp4OnlineLinearMethod → _LazyWeightMixin → CUDAMxfp4LinearMethod
         → LinearMethodBase

      create_weights   : _LazyWeightMixin      (meta device + patched loader)
      process_weights  : CUDAMxfp4OnlineLinearMethod  (BF16 → MXFP4 + kernel)
      apply            : CUDAMxfp4LinearMethod  (kernel.apply_weights)
    """

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(
            layer, "_already_called_process_weights_after_loading", False
        ):
            return

        # Materialize weight if still on meta device (dummy-weight init path).
        if layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(
                    layer.weight, device=layer._load_device
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        # Ensure params_dtype is set for Marlin kernel compatibility
        if not hasattr(layer, "params_dtype"):
            layer.params_dtype = layer.orig_dtype

        # Quantize BF16/FP16 weight → MXFP4
        weight_packed, weight_scale = _quantize_weight_mxfp4(
            layer.weight.data
        )

        # Replace weight with packed uint8
        replace_parameter(layer, "weight", weight_packed)
        # Register weight_scale parameter
        layer.weight_scale = torch.nn.Parameter(
            weight_scale, requires_grad=False
        )

        # Delegate to kernel for kernel-specific transforms (swizzle, repack)
        self.kernel.process_weights_after_loading(layer)
        layer._already_called_process_weights_after_loading = True

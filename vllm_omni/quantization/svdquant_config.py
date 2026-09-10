# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serialized SVDQuant NVFP4 support for diffusion transformers.

The checkpoint stores NVFP4 weights plus a rank-R correction for each
quantized linear. The four-bit GEMM uses vLLM's existing NVFP4 kernel
registry, while the rank correction uses ordinary BF16 matrix multiplication.
Native SVDQuant fusion is a separate optimization and is not required to load
or run the checkpoint.

An online MXFP4 path that derives everything from BF16 checkpoints, with no
calibration data and no serialized tensors, lives in ``svdquant_online.py``.
Select it with ``precision="mxfp4"`` plus
``is_checkpoint_mxfp4_serialized=false``.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Parameter
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    CutlassNvFp4LinearKernel,
    FbgemmNvFp4LinearKernel,
    FlashInferB12xNvFp4LinearKernel,
    FlashInferCudnnNvFp4LinearKernel,
    FlashInferCuteDslNvFp4LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
    FlashInferTrtllmNvFp4LinearKernel,
    NvFp4LinearKernel,
    init_nvfp4_linear_kernel,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

logger = init_logger(__name__)

_SUPPORTED_CAPABILITIES = {(10, 3)}
_COMPATIBLE_NVFP4_KERNELS = (
    CutlassNvFp4LinearKernel,
    FbgemmNvFp4LinearKernel,
    FlashInferB12xNvFp4LinearKernel,
    FlashInferCudnnNvFp4LinearKernel,
    FlashInferCuteDslNvFp4LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
    FlashInferTrtllmNvFp4LinearKernel,
)


def _supports_capability(capability: DeviceCapability | None) -> bool:
    return (
        capability is not None
        and (
            capability.major,
            capability.minor,
        )
        in _SUPPORTED_CAPABILITIES
    )


def _assert_supported() -> None:
    if not current_platform.is_cuda():
        raise RuntimeError("SVDQuant NVFP4 requires a CUDA device")
    capability = current_platform.get_device_capability()
    if not _supports_capability(capability):
        device = current_platform.device_name
        sm = capability.to_int() if capability is not None else "unknown"
        raise RuntimeError(f"SVDQuant NVFP4 is validated on SM103 only; got {device!r} (SM{sm})")


@functools.cache
def _nvfp4_kernel() -> NvFp4LinearKernel:
    kernel = init_nvfp4_linear_kernel()
    if not isinstance(kernel, _COMPATIBLE_NVFP4_KERNELS):
        raise RuntimeError(
            "The selected vLLM NVFP4 backend is incompatible with the "
            f"SVDQuant checkpoint layout: {type(kernel).__name__}. Use one of "
            "the FlashInfer, CUTLASS, or FBGEMM NVFP4 backends."
        )
    return kernel


class DiffusionSVDQuantConfig(QuantizationConfig):
    """Configuration for W4A4 SVDQuant plus a low-rank correction.

    ``precision="nvfp4"`` loads serialized checkpoints. ``precision="mxfp4"``
    with ``is_checkpoint_mxfp4_serialized=False`` instead quantizes BF16
    checkpoints at load time. ``rank=0`` is allowed on that online path only and
    selects branchless MXFP4, which is the control a rank sweep is read against.

    The online path can also carry each operand in more than one chained FP4 term:
    ``weight_terms=2`` ships a second FP4 tensor holding what the first weight term
    left behind, and ``act_terms=2`` re-quantizes the activation residual at run
    time. Every combination costs one extra W4A4 GEMM per added term and buys back
    most of the error the single-term group-32 e8m0 scales cannot represent.
    """

    def __init__(
        self,
        rank: int = 32,
        precision: str = "nvfp4",
        act_unsigned: bool = False,
        modules_to_not_convert: list[str] | None = None,
        is_checkpoint_mxfp4_serialized: bool = True,
        iterations: int = 2,
        svd_niter: int = 2,
        weight_terms: int = 1,
        act_terms: int = 1,
    ) -> None:
        super().__init__()
        if rank < 0:
            raise ValueError(f"SVDQuant rank must be zero or positive, got {rank}")
        if rank == 0 and precision == "nvfp4":
            raise ValueError(
                "A serialized NVFP4 SVDQuant checkpoint has to carry proj_up/proj_down, so the "
                "branchless rank=0 control is available on the online MXFP4 path only."
            )
        if precision not in ("nvfp4", "mxfp4"):
            raise ValueError(
                "SVDQuant supports precision='nvfp4' (serialized checkpoints) and "
                f"'mxfp4' (online quantization); got {precision!r}"
            )
        if precision == "nvfp4" and not is_checkpoint_mxfp4_serialized:
            raise ValueError(
                "NVFP4 SVDQuant requires a serialized checkpoint; vLLM has no dynamic "
                "NVFP4 quantization. Use precision='mxfp4' to quantize at load time."
            )
        if precision == "mxfp4" and is_checkpoint_mxfp4_serialized:
            raise ValueError(
                "MXFP4 SVDQuant is online-only: no serialized MXFP4 producer exists. "
                "Set is_checkpoint_mxfp4_serialized=false."
            )
        if iterations < 1:
            raise ValueError(f"SVDQuant iterations must be >= 1, got {iterations}")
        if svd_niter < 1:
            raise ValueError(f"SVDQuant svd_niter must be >= 1, got {svd_niter}")
        if act_unsigned:
            raise ValueError("SVDQuant does not support unsigned activations")
        for name, terms in (("weight_terms", weight_terms), ("act_terms", act_terms)):
            if terms not in (1, 2):
                raise ValueError(f"SVDQuant {name} must be 1 or 2, got {terms}")
        if (weight_terms > 1 or act_terms > 1) and precision == "nvfp4":
            raise ValueError(
                "Chained FP4 terms are an online-quantization feature: a serialized "
                "checkpoint would have to ship the residual tensors as well."
            )
        self.rank = rank
        self.precision = precision
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        self.iterations = iterations
        self.svd_niter = svd_niter
        self.weight_terms = weight_terms
        self.act_terms = act_terms
        self.modules_to_not_convert = modules_to_not_convert or []
        # Online-quantization load-time statistics.
        self.online_derive_seconds = 0.0
        self.online_derived_layers = 0

    def __repr__(self) -> str:
        return f"DiffusionSVDQuantConfig(rank={self.rank}, precision={self.precision!r})"

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "svdquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # NVFP4 is narrowed to SM103 by _assert_supported(); MXFP4 runs on XPU,
        # where a 103 floor would wrongly reject the device.
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionSVDQuantConfig:
        return cls(
            rank=config.get("rank", 32),
            precision=config.get("precision", "nvfp4"),
            act_unsigned=config.get("act_unsigned", False),
            modules_to_not_convert=config.get("modules_to_not_convert"),
            is_checkpoint_mxfp4_serialized=config.get("is_checkpoint_mxfp4_serialized", True),
            iterations=config.get("iterations", 2),
            svd_niter=config.get("svd_niter", 2),
            weight_terms=config.get("weight_terms", 1),
            act_terms=config.get("act_terms", 1),
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(
            prefix,
            self.modules_to_not_convert,
            self.packed_modules_mapping,
            # vLLM 0.28: skip_with_substr was replaced by match_mode
            # ("substring" preserves the v0.27 skip_with_substr=True
            # behavior verbatim).
            match_mode="substring",
        ):
            return UnquantizedLinearMethod()
        if self.precision == "mxfp4" and not self.is_checkpoint_mxfp4_serialized:
            from vllm_omni.quantization.svdquant_online import SVDQuantOnlineLinearMethod

            return SVDQuantOnlineLinearMethod(self)
        return DiffusionSVDQuantLinearMethod(self)


class DiffusionSVDQuantLinearMethod(LinearMethodBase):
    """Load and execute a serialized SVDQuant linear layer."""

    def __init__(self, quant_config: DiffusionSVDQuantConfig) -> None:
        _assert_supported()
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del params_dtype
        weight_loader = extra_weight_attrs.pop(
            "weight_loader",
            default_weight_loader,
        )
        if input_size_per_partition % 16 != 0:
            raise ValueError(
                "SVDQuant NVFP4 requires each input partition to be divisible "
                f"by the block size 16; got {input_size_per_partition}"
            )
        output_size_per_partition = sum(output_partition_sizes)
        rank = self.quant_config.rank

        qweight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "weight_loader": weight_loader,
            },
        )

        wscales = Parameter(
            torch.empty(
                input_size_per_partition // 16,
                output_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            wscales,
            {
                "input_dim": 0,
                "output_dim": 1,
                "weight_loader": weight_loader,
            },
        )

        proj_down = Parameter(
            torch.empty(
                input_size_per_partition,
                rank,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            proj_down,
            {
                "input_dim": 0,
                "weight_loader": weight_loader,
            },
        )

        proj_up = Parameter(
            torch.empty(
                output_size_per_partition,
                rank,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            proj_up,
            {
                "output_dim": 0,
                "weight_loader": weight_loader,
            },
        )

        smooth_factor = Parameter(
            torch.empty(
                input_size_per_partition,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            smooth_factor,
            {
                "input_dim": 0,
                "weight_loader": weight_loader,
            },
        )

        wcscales = Parameter(
            torch.ones(
                output_size_per_partition,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            wcscales,
            {
                "output_dim": 0,
                "weight_loader": weight_loader,
            },
        )

        wtscale = Parameter(
            torch.ones(1, dtype=torch.bfloat16),
            requires_grad=False,
        )
        set_weight_attrs(wtscale, {"weight_loader": default_weight_loader})

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("wscales", wscales)
        layer.register_parameter("proj_down", proj_down)
        layer.register_parameter("proj_up", proj_up)
        layer.register_parameter("smooth_factor", smooth_factor)
        layer.register_parameter("wcscales", wcscales)
        layer.register_parameter("wtscale", wtscale)

        del input_size, output_size
        layer.output_size_per_partition = output_size_per_partition

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Adapt the canonical row-major checkpoint to vLLM's NVFP4 ABI."""
        qweight = layer.qweight
        wscales = layer.wscales
        del layer.qweight
        del layer.wscales
        layer.register_parameter(
            "weight",
            Parameter(
                qweight.detach().view(torch.uint8),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            Parameter(
                wscales.detach().transpose(0, 1).contiguous(),
                requires_grad=False,
            ),
        )

        layer.register_parameter(
            "input_global_scale_inv",
            Parameter(
                torch.ones(
                    1,
                    dtype=torch.float32,
                    device=layer.weight.device,
                ),
                requires_grad=False,
            ),
        )

        wtscale = layer.wtscale.detach().to(dtype=torch.float32)
        del layer.wtscale
        layer.register_parameter(
            "alpha",
            Parameter(wtscale, requires_grad=False),
        )

        channel_scale = layer.wcscales.detach()
        del layer.wcscales
        if torch.all(channel_scale == 1).item():
            layer.output_channel_scale = None
        else:
            layer.register_parameter(
                "output_channel_scale",
                Parameter(channel_scale, requires_grad=False),
            )

        _nvfp4_kernel().process_weights_after_loading(layer)
        logger.info_once(
            "SVDQuant NVFP4 is using vLLM's compatibility path; the rank correction is not fused into the GEMM."
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the base NVFP4 GEMM plus the BF16 rank correction."""
        if x.dtype != torch.bfloat16:
            raise ValueError(f"SVDQuant NVFP4 requires BF16 activations; got {x.dtype}")

        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]).contiguous()

        # The residual branch consumes the original activation. Only the
        # four-bit base GEMM consumes the smoothed activation.
        smoothed = x_2d / layer.smooth_factor
        out = _nvfp4_kernel().apply_weights(
            layer=layer,
            x=smoothed,
            bias=None,
        )

        channel_scale = getattr(layer, "output_channel_scale", None)
        if channel_scale is not None:
            # Fused QKV can store independent Q/K/V outer scales. Apply them
            # explicitly until a vector-alpha GEMM epilogue is available.
            out.mul_(channel_scale)

        correction_input = torch.mm(x_2d, layer.proj_down)
        out = torch.addmm(
            out,
            correction_input,
            layer.proj_up.transpose(0, 1),
        )
        if bias is not None:
            out.add_(bias)
        return out.reshape(
            *original_shape[:-1],
            layer.output_size_per_partition,
        )


__all__ = [
    "DiffusionSVDQuantConfig",
    "DiffusionSVDQuantLinearMethod",
]

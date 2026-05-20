# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone MXFP4 linear kernel selection for vllm-omni.

Ported from vllm.model_executor.kernels.linear (vLLM >=0.20) so that
vllm-omni does not depend on the ``vllm.model_executor.kernels`` package
existing at import time. The actual CUDA/XPU ops are still lazy-imported
from vllm at *runtime* (e.g. marlin_utils_fp4, flashinfer helpers).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import PlatformEnum

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Base class + config
# ---------------------------------------------------------------------------

_MXFP4_GROUP_SIZE = 32


@dataclass
class MxFp4LinearLayerConfig:
    """Configuration for an MXFP4 linear layer."""

    pass


class MxFp4LinearKernel(ABC):
    """ABC for MXFP4 quantized linear kernels."""

    def __init__(self, config: MxFp4LinearLayerConfig) -> None:
        assert self.can_implement(config)[0]
        assert self.is_supported()[0]
        self.config = config

    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(
        cls, config: MxFp4LinearLayerConfig
    ) -> tuple[bool, str | None]:
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Marlin backend  (SM80+ — W4A16 weight-only GEMM)
# ---------------------------------------------------------------------------


class MarlinMxFp4LinearKernel(MxFp4LinearKernel):

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            is_fp4_marlin_supported,
        )

        if is_fp4_marlin_supported():
            return True, None
        return False, "Marlin FP4 not available"

    @classmethod
    def can_implement(
        cls, c: MxFp4LinearLayerConfig
    ) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            prepare_fp4_layer_for_marlin,
        )

        prepare_fp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
        )

        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=None,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )


# ---------------------------------------------------------------------------
# FlashInfer / CUTLASS backend  (SM100+ Blackwell — W4A4)
# ---------------------------------------------------------------------------

@torch.library.custom_op(
    "vllm::flashinfer_mm_fp4",
    mutates_args=[],
    device_types="cuda",
)
def flashinfer_mm_fp4(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    g_scale: torch.Tensor,
    dtype: torch.dtype,
    use_8x4_sf_layout: bool,
    backend: str,
    block_size: int = 16,
    use_nvfp4: bool = True,
) -> torch.Tensor:
    from flashinfer import mm_fp4 as flashinfer_mm_fp4_

    return flashinfer_mm_fp4_(
        A,
        B,
        A_scale,
        B_scale,
        g_scale,
        dtype,
        block_size=block_size,
        use_8x4_sf_layout=use_8x4_sf_layout,
        use_nvfp4=use_nvfp4,
        backend=backend,
    )

@torch.library.register_fake(
    "vllm::flashinfer_mm_fp4",
)
def flashinfer_mm_fp4_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    g_scale: torch.Tensor,
    dtype: torch.dtype,
    use_8x4_sf_layout: bool,
    backend: str,
    block_size: int = 16,
    use_nvfp4: bool = True,
) -> torch.Tensor:
    return torch.empty(A.shape[0], B.shape[1], dtype=dtype, device=A.device)


def flashinfer_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor | None,
    out_dtype: torch.dtype,
    backend: str,
    block_size: int = 16,
    use_nvfp4: bool = True,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert block_scale_a.ndim == 2 and block_scale_b.ndim == 2
    assert a.stride(-1) == 1 and b.stride(-1) == 1
    assert a.shape[1] == b.shape[1]

    if alpha is None:
        alpha = torch.ones(1, dtype=torch.float32, device=a.device)

    if backend in ("cutlass", "cudnn"):
        block_scale_a = block_scale_a.view(torch.uint8)
        block_scale_b = block_scale_b.view(torch.uint8)

    use_8x4_sf_layout = True if backend == "trtllm" and a.shape[0] <= 32 else False  # noqa: SIM210

    return flashinfer_mm_fp4(
        a,
        b.t(),
        block_scale_a,
        block_scale_b.t(),
        alpha,
        out_dtype,
        use_8x4_sf_layout=use_8x4_sf_layout,
        backend=backend,
        block_size=block_size,
        use_nvfp4=use_nvfp4,
    )



@torch.library.custom_op(
    "vllm::flashinfer_mxfp4_quantize",
    mutates_args=[],
    device_types="cuda",
)
def flashinfer_mxfp4_quantize(
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer import mxfp4_quantize as _mxfp4_quantize

    return _mxfp4_quantize(a)

@torch.library.register_fake("vllm::flashinfer_mxfp4_quantize")
def flashinfer_mxfp4_quantize_fake(
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = a.shape
    sf_vec_size = 32
    from vllm.utils.math_utils import cdiv
    padded_m = cdiv(m, 128) * 128
    sf_cols = cdiv(k // sf_vec_size, 4) * 4
    return (
        torch.empty(m, k // 2, dtype=torch.uint8, device=a.device),
        torch.empty(padded_m, sf_cols, dtype=torch.uint8, device=a.device),
    )

class FlashInferMxFp4LinearKernel(MxFp4LinearKernel):

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        from vllm.utils.flashinfer import has_flashinfer_cutedsl

        if (
            current_platform.has_device_capability(100)
            and has_flashinfer_cutedsl()
        ):
            return True, None
        return False, "FlashInfer + >=sm_100 (Blackwell) required"

    @classmethod
    def can_implement(
        cls, config: MxFp4LinearLayerConfig
    ) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from torch.nn.parameter import Parameter

        from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
            swizzle_mxfp4_scales,
        )

        N, scale_K = layer.weight_scale.shape
        K = scale_K * _MXFP4_GROUP_SIZE

        padded_N = ((N + 127) // 128) * 128
        layer.weight_scale = Parameter(
            swizzle_mxfp4_scales(
                layer.weight_scale.data, N, K
            ).reshape(padded_N, -1),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # from vllm.utils.flashinfer import (
            # flashinfer_mxfp4_quantize,
            # flashinfer_scaled_fp4_mm,
        # )
        # from flashinfer import mxfp4_quantize as flashinfer_mxfp4_quantize
        # from flashinfer import mxfp4_quantize as flashinfer_mxfp4_quantize
        out_shape = x.shape[:-1] + (layer.output_size_per_partition,)
        x_2d = x.reshape(-1, x.shape[-1])

        x_fp4, x_scale = flashinfer_mxfp4_quantize(x_2d)
        out = flashinfer_scaled_fp4_mm(
            x_fp4,
            layer.weight,
            x_scale,
            layer.weight_scale,
            alpha=None,
            out_dtype=x.dtype,
            backend="auto",
            block_size=_MXFP4_GROUP_SIZE,
            use_nvfp4=False,
        )

        if bias is not None:
            out = out + bias
        return out.view(out_shape)


# ---------------------------------------------------------------------------
# XPU backend
# ---------------------------------------------------------------------------


class XPUMxFp4LinearKernel(MxFp4LinearKernel):

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUMxFp4 only supported on XPU"
        return True, None

    @classmethod
    def can_implement(
        cls, c: MxFp4LinearLayerConfig
    ) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.utils import replace_parameter

        weight = layer.weight.view(torch.float4_e2m1fn_x2)
        replace_parameter(layer, "weight", weight.data.t())

        weight_scale = layer.weight_scale.view(torch.float8_e8m0fnu)
        weight_scale = weight_scale.t().contiguous()
        replace_parameter(layer, "weight_scale", weight_scale.data)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            xpu_mxfp4_quantize as quant_mxfp4,
        )

        out_dtype = x.dtype
        x_fp4, x_blockscale = quant_mxfp4(x)
        return torch.ops._xpu_C.fp4_gemm(
            x_fp4,
            layer.weight,
            x_blockscale,
            layer.weight_scale,
            out_dtype,
            bias,
        )


# ---------------------------------------------------------------------------
# Kernel selector  (dispatch by platform + env vars)
# ---------------------------------------------------------------------------

_POSSIBLE_MXFP4_KERNELS: dict[
    PlatformEnum, list[type[MxFp4LinearKernel]]
] = {
    PlatformEnum.CUDA: [
        FlashInferMxFp4LinearKernel,
        MarlinMxFp4LinearKernel,
    ],
    PlatformEnum.XPU: [
        XPUMxFp4LinearKernel,
    ],
}


def _get_disabled_kernels() -> list[str]:
    raw = os.environ.get("VLLM_DISABLED_KERNELS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _get_mxfp4_use_marlin() -> bool | None:
    raw = os.environ.get("VLLM_MXFP4_USE_MARLIN")
    if raw is None:
        return None
    return raw.lower() in ("1", "true", "yes")


def init_mxfp4_linear_kernel() -> MxFp4LinearKernel:
    """Select and instantiate the best MXFP4 linear kernel."""
    force_marlin = _get_mxfp4_use_marlin()
    disabled_kernels = _get_disabled_kernels()

    if force_marlin:
        is_supported, reason = MarlinMxFp4LinearKernel.is_supported()
        if not is_supported:
            raise ValueError(
                f"Forced MXFP4 kernel MarlinMxFp4LinearKernel is not "
                f"supported: {reason}"
            )
        logger.info_once(
            "Using MarlinMxFp4LinearKernel for MXFP4 GEMM (forced)"
        )
        return MarlinMxFp4LinearKernel(MxFp4LinearLayerConfig())

    platform = current_platform._enum
    possible = _POSSIBLE_MXFP4_KERNELS.get(platform, [])

    failure_reasons: list[str] = []
    for kernel_cls in possible:
        if kernel_cls.__name__ in disabled_kernels:
            failure_reasons.append(
                f"  {kernel_cls.__name__} disabled by environment variable"
            )
            continue

        is_supported, reason = kernel_cls.is_supported()
        if not is_supported:
            failure_reasons.append(f"{kernel_cls.__name__}: {reason}")
            continue

        logger.info_once("Using %s for MXFP4 GEMM", kernel_cls.__name__)
        return kernel_cls(MxFp4LinearLayerConfig())

    raise ValueError(
        "Failed to find a kernel that can implement the "
        "MXFP4 linear layer. Reasons: \n" + "\n".join(failure_reasons)
    )

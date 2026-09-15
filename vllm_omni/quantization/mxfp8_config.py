# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""W8A8 MXFP8 (Microscaling FP8) online/offline quantization for diffusion transformers.

Architecture:

  MXFPLinearMethodBase            – platform-agnostic skeleton; defines apply() (reshape only),
                                     _apply_inner() (default single-scale dispatch), and two
                                     abstract ops (_quantize_activation, _quant_matmul).
    NPUMxfp8LinearMethod          – NPU offline: create_weights for pre-quantized checkpoint,
                                     process_weights normalization, and NPU MXFP8 ops.
      NPUMxfp8OnlineLinearMethod  – NPU online: _LazyWeightMixin for create_weights,
                                     overrides process_weights to quantize BF16 → FP8.

XPU builds on vLLM's kernels directly rather than on its quantization *methods*:
``XPUMxFp8LinearKernel`` is the stable interface, while the INC / online linear
method wrappers around it are not. MoE is left to vLLM for now.

  VllmMxfp8OfflineLinearMethod  – offline AutoRound MXFP8 checkpoints
  VllmMxfp8OnlineLinearMethod   – BF16 checkpoint, quantized at load time
  VllmMxfp8OnlineMoEMethod      – vLLM Mxfp8OnlineMoEMethod, re-exported unchanged

Extending to a new platform (e.g. CUDA MXFP8 once the ops are available):

    class CUDAMxfp8LinearMethod(MXFPLinearMethodBase):
        def create_weights(self, ...): ...
        def process_weights_after_loading(self, layer): ...
        def _quantize_activation(self, x): ...   # CUDA FP8 quant op
        def _quant_matmul(self, ...): ...         # CUDA FP8 GEMM

    class CUDAMxfp8OnlineLinearMethod(_LazyWeightMixin, CUDAMxfp8LinearMethod):
        def process_weights_after_loading(self, layer): ...  # BF16 → FP8

Extending to a new MX precision with a different calling convention (e.g. dual-scale FP4):

    class NPUMxfp4DualScaleLinearMethod(MXFPLinearMethodBase):
        def _apply_inner(self, layer, x, bias, ori_dtype): ...  # override for 3-tuple path
        def _quantize_activation(self, x, smooth_scale): ...    # returns 3-tuple
        def _quant_matmul(self, x_q, l0, l1, layer, ...): ...  # dual-level GEMM
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
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
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization._copy_missing_attrs import (
    copy_missing_attrs as _copy_missing_attrs,
)

# ---------------------------------------------------------------------------
# Compat shims for symbols vLLM moved between releases
# ---------------------------------------------------------------------------

try:
    from vllm.model_executor.model_loader.reload.meta import CopyCounter
except ImportError:
    from vllm.model_executor.layers.quantization.fp8 import (  # type: ignore[attr-defined,no-redef]
        CopyNumelCounter as CopyCounter,
    )


def copy_missing_attrs(old: torch.Tensor, new: torch.Tensor) -> None:
    new_attrs = set(dir(new))
    set_weight_attrs(new, {attr: getattr(old, attr) for attr in dir(old) if attr not in new_attrs})


if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class DiffusionMXFP8Config(QuantizationConfig):
    """W8A8 MXFP8 quantization config for diffusion transformers.

    Supports both online (BF16 checkpoint → quantize at load time) and offline
    (pre-quantized MXFP8 checkpoint) modes, mirroring DiffusionInt8Config.

    MX (microscaling) format: groups of 32 K-dimension elements share one
    float8_e8m0fnu exponent scale, matching MXFP8 as defined in the OCP MX spec.
    """

    def __init__(
        self,
        is_checkpoint_mxfp8_serialized: bool = False,
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.is_checkpoint_mxfp8_serialized = is_checkpoint_mxfp8_serialized
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp8"

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
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionMXFP8Config:
        is_serialized = cls.get_from_keys_or(config, ["is_checkpoint_mxfp8_serialized"], False)
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(
            is_checkpoint_mxfp8_serialized=is_serialized,
            ignored_layers=ignored_layers,
        )

    _XPU_OFFLINE_UNSUPPORTED = (
        "Native MXFP8 offline mode is not supported on XPU. "
        "Use AutoRound MXFP8 checkpoints (quant_method='auto-round', data_type='mx_fp') instead, "
        "or use online MXFP8 mode without is_checkpoint_mxfp8_serialized."
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
            if current_omni_platform.is_npu():
                if self.is_checkpoint_mxfp8_serialized:
                    return NPUMxfp8LinearMethod(self)
                return NPUMxfp8OnlineLinearMethod(self)
            if current_omni_platform.is_xpu():
                if self.is_checkpoint_mxfp8_serialized:
                    raise NotImplementedError(self._XPU_OFFLINE_UNSUPPORTED)
                return VllmMxfp8OnlineLinearMethod()
            raise NotImplementedError(
                "DiffusionMXFP8Config (W8A8 MXFP8) is currently only supported "
                "on NPU (Ascend) and XPU (Intel) platforms."
            )
        if isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if current_omni_platform.is_xpu():
                if self.is_checkpoint_mxfp8_serialized:
                    raise NotImplementedError(self._XPU_OFFLINE_UNSUPPORTED)
                return VllmMxfp8OnlineMoEMethod(layer=layer)
            # NPU has no MXFP8 MoE op yet; leave the experts in BF16 rather
            # than failing the whole model load.
            return None
        return None


# ---------------------------------------------------------------------------
# _LazyWeightMixin — shared by all online methods
# ---------------------------------------------------------------------------


class _LazyWeightMixin:
    """Weight registered on meta device, materialised just-in-time on first load chunk.

    Platform-agnostic; shared by all online MXFP methods.
    Imported by mxfp4_config.py to avoid duplication.
    """

    uses_meta_device: bool = True

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
        layer.weight_block_size = None

        def patched_weight_loader(param, loaded_weight, *args, **kwargs):
            if not hasattr(layer, "_loaded_numel"):
                layer._loaded_numel = 0
                weight = ModelWeightParameter(
                    data=torch.empty_like(layer.weight, device=layer._load_device),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=patched_weight_loader,
                )
                copy_missing_attrs(layer.weight, weight)
                layer.register_parameter("weight", weight)
                del layer._load_device

            param = layer.weight
            counter = CopyCounter()
            with counter:
                res = weight_loader(param, loaded_weight, *args, **kwargs)
            layer._loaded_numel += counter.copied_numel

            if layer._loaded_numel == layer.weight.numel():
                self.process_weights_after_loading(layer)
                layer._already_called_process_weights_after_loading = True
            return res

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=patched_weight_loader,
        )
        layer._load_device = torch.get_default_device()
        layer.register_parameter("weight", weight)


# ---------------------------------------------------------------------------
# Abstract base — platform-agnostic apply skeleton
# ---------------------------------------------------------------------------


class MXFPLinearMethodBase(LinearMethodBase, ABC):
    """Platform-agnostic MXFP linear method base.

    Defines the apply() skeleton (flatten → _apply_inner → reshape) and three
    hooks that subclasses implement:

      _quantize_activation(x)                              → tuple  (arity depends on variant)
      _quant_matmul(x_q, x_scale, layer, bias, ori_dtype)  → Tensor (single-scale default)
      _apply_inner(layer, x, bias, ori_dtype)               → Tensor (override for dual-scale)

    Extension guide:
      Single-scale (MXFP8, MXFP4): implement _quantize_activation + _quant_matmul only.
      Dual-scale (MXFP4 DualScale): override _apply_inner for a different calling convention.
      Subclasses must NOT override apply() — reshape logic lives here exclusively.
    """

    @abstractmethod
    def _quantize_activation(self, x: torch.Tensor) -> tuple:
        """Quantize 2-D activation. Return arity must match what _apply_inner expects."""

    @abstractmethod
    def _quant_matmul(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Fused MXFP quantized GEMM. Weight and scale accessed from layer."""

    def _apply_inner(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Default single-scale inner loop: 2-tuple quantize → matmul.

        Override this (not apply()) when a different quantize/matmul convention
        is needed, e.g. dual-scale variants that return a 3-tuple from
        _quantize_activation and pass extra tensors to _quant_matmul.
        """
        x_q, x_scale = self._quantize_activation(x)
        return self._quant_matmul(x_q, x_scale, layer, bias, ori_dtype)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Shared apply skeleton: reshape → _apply_inner → unreshape.

        Do NOT override this method. Override _apply_inner() instead.
        """
        ori_shape = x.shape
        ori_dtype = x.dtype
        x = x.reshape(-1, ori_shape[-1])
        output = self._apply_inner(layer, x, bias, ori_dtype)
        return output.reshape(*ori_shape[:-1], -1)


# ---------------------------------------------------------------------------
# NPU MXFP8 offline method (pre-quantized checkpoint)
# ---------------------------------------------------------------------------


class NPUMxfp8LinearMethod(MXFPLinearMethodBase):
    """NPU W8A8 MXFP8 offline linear method for pre-quantized checkpoints.

    Weight canonical layout after process_weights_after_loading:
      weight      : (K, N)              float8_e4m3fn   – pre-transposed for GEMM
      weight_scale: (K_groups/2, N, 2)  float8_e8m0fnu  – reshaped + pre-transposed

    NPUMxfp8OnlineLinearMethod normalises to the same layout so apply() is shared.
    """

    def __init__(self, quant_config: DiffusionMXFP8Config) -> None:
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

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
        """Register weight and per-group MX scale for a pre-quantized checkpoint."""
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition, input_size_per_partition, dtype=params_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

        # Scale stored as uint8 in safetensors (float8_e8m0fnu is same bit width).
        # Using uint8 avoids a lossy float32 round-trip when loading the checkpoint.
        num_groups = (input_size_per_partition + 31) // 32
        layer.register_parameter(
            "weight_scale",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition, num_groups, dtype=torch.uint8),
                input_dim=None,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        """Cast checkpoint weight to FP8 and normalise to canonical GEMM layout."""
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        import torch_npu

        # Weight: BF16 → float8_e4m3fn via npu_dtype_cast, then transpose (N,K) → (K,N).
        w = layer.weight
        if w.dtype != torch_npu.float8_e4m3fn:
            w = torch_npu.npu_dtype_cast(w.npu(), torch_npu.float8_e4m3fn)
        w = w.transpose(0, 1).contiguous()

        # Scale: checkpoint stores uint8 bytes that ARE float8_e8m0fnu bits.
        # Only convert if neither uint8 nor the target NPU dtype already.
        # Pad K_groups to even so the (K_groups/2, N, 2) reshape is always valid.
        s = layer.weight_scale.data
        if s.dtype not in (torch.uint8, torch_npu.float8_e8m0fnu):
            s = s.to(torch_npu.float8_e8m0fnu)
        N, K_groups = s.shape
        if K_groups % 2 == 1:
            s = torch.cat([s, torch.zeros(N, 1, dtype=s.dtype, device=s.device)], dim=1)
            K_groups += 1
        s = s.reshape(N, K_groups // 2, 2).transpose(0, 1).contiguous()

        replace_parameter(layer, "weight", w)
        replace_parameter(layer, "weight_scale", s)
        layer._already_called_process_weights_after_loading = True

    # --- NPU MXFP8 ops — shared with online path via inheritance ---

    def _quantize_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        return torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)

    def _quant_matmul(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        import torch_npu

        # NPU npu_quant_matmul requires bias in float32.
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)
        return torch_npu.npu_quant_matmul(
            x_q,
            layer.weight,  # (K, N) float8_e4m3fn
            layer.weight_scale,  # (S/2, N, 2) float8_e8m0fnu
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=x_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=ori_dtype,
            group_sizes=[1, 1, 32],
        )


# ---------------------------------------------------------------------------
# NPU MXFP8 online method (BF16 checkpoint → quantize at load time)
# ---------------------------------------------------------------------------


class NPUMxfp8OnlineLinearMethod(_LazyWeightMixin, NPUMxfp8LinearMethod):
    """NPU W8A8 MXFP8 online linear method.

    MRO: NPUMxfp8OnlineLinearMethod → _LazyWeightMixin → NPUMxfp8LinearMethod
         → MXFPLinearMethodBase → LinearMethodBase

      create_weights   : _LazyWeightMixin      (meta device + patched loader)
      process_weights  : NPUMxfp8OnlineLinearMethod  (BF16 → FP8 + normalize)
      apply / ops      : NPUMxfp8LinearMethod / MXFPLinearMethodBase  (shared)
    """

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        import torch_npu

        # Materialise weight if still on meta device (dummy-weight init path).
        if layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        # NPU: quantize BF16/FP16 (N, K) → FP8 (N, K) + MX scale (N, S).
        weight_fp8, weight_scale_raw = torch_npu.npu_dynamic_mx_quant(layer.weight, dst_type=torch_npu.float8_e4m3fn)

        # Normalize to canonical layout shared with offline path.
        weight_scale = weight_scale_raw.reshape(weight_scale_raw.shape[0], -1, 2).transpose(0, 1).contiguous()
        weight_fp8 = weight_fp8.transpose(0, 1).contiguous()

        replace_parameter(layer, "weight", weight_fp8)
        replace_parameter(layer, "weight_scale", weight_scale)
        layer._already_called_process_weights_after_loading = True


# ---------------------------------------------------------------------------
# XPU methods — built directly on XPUMxFp8LinearKernel
#
# The kernel is the stable contract: process_weights_after_loading() owns the
# [N, K//32] -> [K//32, N] e8m0 scale relayout and apply_weights() owns the
# oneDNN fp8_gemm call. vLLM's INC / online *method* layers are not depended on.
# ---------------------------------------------------------------------------

try:
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.mxfp8.xpu import XPUMxFp8LinearKernel
    from vllm.model_executor.layers.quantization.online.mxfp8 import (
        Mxfp8OnlineMoEMethod as VllmMxfp8OnlineMoEMethod,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        MXFP8_BLOCK_SIZE,
        MXFP8_SCALE_DTYPE,
        MXFP8_VALUE_DTYPE,
        mxfp8_e4m3_quantize,
    )
    from vllm.model_executor.parameter import GroupQuantScaleParameter
except ImportError as e:
    raise ImportError(
        "vLLM MXFP8 support is not available. "
        "MXFP8 quantization requires vLLM with XPUMxFp8LinearKernel support. "
        "Please upgrade vLLM or use a compatible build."
    ) from e


def _repack_weight_ab(layer: Module) -> None:
    """Repack an MXFP8 weight so the GEMM's B operand is physically contiguous.

    ``fp8_gemm`` consumes B as ``[K, N]``. By default that is a transposed *view*
    of an ``[N, K]`` buffer, so B is K-contiguous -- oneDNN's "ba" layout. Making
    it N-contiguous ("ab") is worth 2.5x on every linear in a
    ``WanTransformerBlock`` at the 720p shape: measured at 75600 tokens, the four
    shapes go 199.1 -> 79.6 ms in total, which turns MXFP8 from 1.34x slower than
    bf16 into 1.86x faster. The repack is a one-off load-time copy; the GEMM path
    is untouched.

    Done here rather than via vLLM's ``VLLM_XPU_FORCE_AB_LAYOUT_WEIGHT`` because
    that env is global -- it also repacks unquantized bf16 weights -- and it only
    exists in newer vLLM. This keeps the change scoped to MXFP8 and works against
    either vLLM.

    Which tensor is B differs by vLLM version, and the weight's own shape cannot
    settle it -- for a square linear (Wan's 5120x5120 attention projections) both
    orientations are ``[5120, 5120]``. The *scale* is unambiguous, because
    ``K//32`` is never equal to ``N`` for any real layer:

      * scale ``[N, K//32]``  -> newer vLLM: ``weight`` is ``[N, K]`` and
        ``apply_weights`` transposes, so B is ``weight.t()`` and we want
        ``weight.stride(0) == 1`` (via ``.t().contiguous().t()``)
      * scale ``[K//32, N]``  -> older vLLM: ``weight`` was already transposed to
        ``[K, N]`` and is passed straight through, so B is ``weight`` and we just
        want it contiguous

    Both forms are written as an idempotent "is B already contiguous?" check, so
    running this after a vLLM that already repacked (its own
    ``VLLM_XPU_FORCE_AB_LAYOUT_WEIGHT``) is a no-op rather than an undo.

    Set ``VLLM_XPU_FORCE_AB_LAYOUT_WEIGHT=0`` to skip this. The only reason to do
    so is bitwise reproducibility: oneDNN's ab-weights matmul is not
    reproducible run to run. For diffusion that is usually moot, since the
    pipeline already has an ~18 dB PSNR nondeterminism floor between identical
    seeded runs, but it matters when bisecting numerics.
    """
    if os.environ.get("VLLM_XPU_FORCE_AB_LAYOUT_WEIGHT", "1") == "0":
        return
    weight = getattr(layer, "weight", None)
    scale = getattr(layer, "weight_scale", None)
    if weight is None or scale is None or weight.dim() != 2 or scale.dim() != 2:
        return

    k = getattr(layer, "input_size_per_partition", None)
    n = getattr(layer, "output_size_per_partition", None)
    if k is None or n is None or k % MXFP8_BLOCK_SIZE != 0:
        return
    k_blocks = k // MXFP8_BLOCK_SIZE
    if k_blocks == n:
        return  # degenerate: scale orientation would be ambiguous too

    scale_shape = tuple(scale.shape)
    if scale_shape == (n, k_blocks):
        # B is weight.t(); B contiguous <=> weight.stride(0) == 1.
        if weight.data.stride(0) != 1:
            layer.weight.data = weight.data.t().contiguous().t()
    elif scale_shape == (k_blocks, n):
        # B is weight itself.
        if not weight.data.is_contiguous():
            layer.weight.data = weight.data.contiguous()
    else:
        # Fused/sharded layouts we do not recognise: leave alone rather than
        # guess, since a wrong repack silently corrupts the GEMM.
        logger.warning_once(
            "MXFP8 ab-layout repack skipped: weight_scale shape %s matches "
            "neither (N,K//32)=%s nor (K//32,N)=%s",
            scale_shape,
            (n, k_blocks),
            (k_blocks, n),
        )


class VllmMxfp8OfflineLinearMethod(LinearMethodBase):
    """Offline MXFP8 linear method for AutoRound checkpoints (data_type="mx_fp").

    Weights arrive already quantized: FP8-E4M3 values plus one uint8 e8m0 scale
    per 32 input elements. Layout normalization and the GEMM are the kernel's.
    """

    def __init__(self) -> None:
        self.kernel = XPUMxFp8LinearKernel(Mxfp8LinearLayerConfig())

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
        if input_size_per_partition % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"MXFP8 requires input_size_per_partition ({input_size_per_partition}) "
                f"to be divisible by {MXFP8_BLOCK_SIZE}."
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=MXFP8_VALUE_DTYPE,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            GroupQuantScaleParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // MXFP8_BLOCK_SIZE,
                    dtype=MXFP8_SCALE_DTYPE,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        # Hands the [N, K//32] e8m0 scale to the kernel for its [K//32, N]
        # relayout, which apply_weights() then recovers with a free .t().
        self.kernel.process_weights_after_loading(layer)
        _repack_weight_ab(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() <= 2:
            return self.kernel.apply_weights(layer, x, bias)
        ori_shape = x.shape
        output = self.kernel.apply_weights(layer, x.reshape(-1, ori_shape[-1]), bias)
        return output.reshape(*ori_shape[:-1], -1)


class VllmMxfp8OnlineLinearMethod(_LazyWeightMixin, VllmMxfp8OfflineLinearMethod):
    """Online MXFP8 linear method: BF16 checkpoint, quantized at load time.

    create_weights comes from :class:`_LazyWeightMixin` (meta device, weights
    materialized just-in-time); apply() is shared with the offline method.
    """

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        weight_fp8, weight_scale = mxfp8_e4m3_quantize(layer.weight.data.contiguous())
        replace_parameter(layer, "weight", weight_fp8)
        replace_parameter(layer, "weight_scale", weight_scale)

        self.kernel.process_weights_after_loading(layer)
        _repack_weight_ab(layer)
        layer._already_called_process_weights_after_loading = True

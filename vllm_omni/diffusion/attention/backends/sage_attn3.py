# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import os

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

sageattn_v1_v2 = None

if current_omni_platform.is_xpu():
    try:
        import deepklox

        logger.warning_once(
            "SAGE_ATTN_3 is using DeepKloX from %s",
            getattr(deepklox, "__file__", "<unknown>"),
        )
    except ImportError:
        deepklox = None
    try:
        from deepklox.sageattn_interface import sageattn_v3_hybrid
    except ImportError:
        sageattn_v3_hybrid = None
        logger.warning(
            "DeepKLOX SageAttention V3 Hybrid is not available on XPU; "
            "SAGE_ATTN_3 will fall back to flash attention. Install `deepklox` "
            "with SageAttention V3 support."
        )
    try:
        from deepklox.sageattn_interface import sageattn as sageattn_v1_v2
    except ImportError:
        sageattn_v1_v2 = None
else:
    try:
        from sageattn3 import sageattn3_blackwell  # noqa: F401
    except ImportError:
        logger.warning(
            "SageAttention3Backend is not available. Install `sageattn3` from "
            "https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell"
        )
        raise ImportError

_SAGE_ATTN3_FORCE_SDPA_BLOCKS = frozenset(
    int(block) for block in os.environ.get("SAGE_ATTN_FORCE_SDPA_BLOCKS", "").split(",") if block.strip()
)
_SAGE_ATTN3_REPORT_FALLBACKS = os.environ.get("SAGE_ATTN_REPORT_FALLBACKS", "0") == "1"
_SAGE_ATTN3_DEBUG_CHECK_FINITE = os.environ.get("SAGE_ATTN_DEBUG_CHECK_FINITE", "0") == "1"
_SAGE_ATTN_PRECISION = os.environ.get("SAGE_ATTN_PRECISION", "").strip().lower()
# Which implementation serves the calls the V3 kernel declines: flash (the default),
# SDPA, or a DeepKloX SageAttention V1/V2 kernel. The paths differ on every forced
# block, every cross-attention call and every contract rejection.
#
# Deliberately fatal on an unrecognised value rather than defaulting to flash: the whole
# point of the knob is attributing a timing to one of the two paths, and a typo that
# silently produced flash numbers under an sdpa label would be worse than not starting.
_SAGE_ATTN3_FALLBACK = os.environ.get("SAGE_ATTN_FALLBACK", "flash").strip().lower() or "flash"
_SAGE_ATTN3_FALLBACKS = ("flash", "sdpa", "sage_v1", "sage_v2")
if _SAGE_ATTN3_FALLBACK not in _SAGE_ATTN3_FALLBACKS:
    raise ValueError(
        f"SAGE_ATTN_FALLBACK must be one of {', '.join(_SAGE_ATTN3_FALLBACKS)}, got {_SAGE_ATTN3_FALLBACK!r}"
    )
_SAGE_ATTN3_FORCE_FALLBACK = (
    os.environ.get("SAGE_ATTN_FORCE_FALLBACK", _SAGE_ATTN3_FALLBACK).strip().lower() or _SAGE_ATTN3_FALLBACK
)
if _SAGE_ATTN3_FORCE_FALLBACK not in _SAGE_ATTN3_FALLBACKS:
    raise ValueError(
        "SAGE_ATTN_FORCE_FALLBACK must be one of "
        f"{', '.join(_SAGE_ATTN3_FALLBACKS)}, got {_SAGE_ATTN3_FORCE_FALLBACK!r}"
    )

if current_omni_platform.is_xpu():
    if _SAGE_ATTN3_REPORT_FALLBACKS:
        logger.info("SageAttention3 XPU fallback reporting is enabled.")
    logger.info(
        "SageAttention3 XPU fallback implementation: %s (SAGE_ATTN_FALLBACK)",
        _SAGE_ATTN3_FALLBACK,
    )
    if _SAGE_ATTN3_FORCE_SDPA_BLOCKS:
        # The env var keeps its published name, but the path it selects is whatever
        # SAGE_ATTN_FALLBACK picks -- flash by default, because SDPA at these sequence
        # lengths costs 5.3x the sage kernel per call.
        logger.info(
            "SageAttention3 XPU forcing the %s fallback for transformer blocks: %s",
            _SAGE_ATTN3_FORCE_FALLBACK,
            sorted(_SAGE_ATTN3_FORCE_SDPA_BLOCKS),
        )

_V3_HYBRID_HEAD_SIZE = 128
_V3_HYBRID_BATCH = 1

_sage_attn3_call_counts = {
    # Why a call did not reach the V3 kernel.
    "sage": 0,
    "forced_sdpa": 0,
    "cross_sdpa": 0,
    "sdpa_fallback": 0,
    "nonfinite_sdpa": 0,
    # Which reference implementation served those calls. The four keys above keep
    # their historical names -- they answer "which branch declined the kernel", not
    # "what ran" -- so existing log parsers keep working now that the reference path
    # is flash attention rather than torch SDPA.
    "fallback_flash": 0,
    "fallback_sdpa": 0,
    "fallback_sage_v1": 0,
    "fallback_sage_v2": 0,
}


def get_sage_attn3_call_counts() -> dict[str, int]:
    """Snapshot of the SAGE_ATTN_3 XPU dispatch counters."""
    return dict(_sage_attn3_call_counts)


def reset_sage_attn3_call_counts() -> None:
    for key in _sage_attn3_call_counts:
        _sage_attn3_call_counts[key] = 0


def _sage_attn3_record(kind: str) -> None:
    if _SAGE_ATTN3_REPORT_FALLBACKS:
        _sage_attn3_call_counts[kind] += 1


def _sage_attn3_report_call_counts() -> None:
    if _SAGE_ATTN3_REPORT_FALLBACKS:
        print(
            "Sage attention V3 call counts: "
            f"sage={_sage_attn3_call_counts['sage']}, "
            f"forced_sdpa={_sage_attn3_call_counts['forced_sdpa']}, "
            f"cross_sdpa={_sage_attn3_call_counts['cross_sdpa']}, "
            f"sdpa_fallback={_sage_attn3_call_counts['sdpa_fallback']}, "
            f"nonfinite_sdpa={_sage_attn3_call_counts['nonfinite_sdpa']}, "
            f"fallback_flash={_sage_attn3_call_counts['fallback_flash']}, "
            f"fallback_sdpa={_sage_attn3_call_counts['fallback_sdpa']}, "
            f"fallback_sage_v1={_sage_attn3_call_counts['fallback_sage_v1']}, "
            f"fallback_sage_v2={_sage_attn3_call_counts['fallback_sage_v2']}"
        )


atexit.register(_sage_attn3_report_call_counts)


def _try_extract_layer_index(prefix: str) -> int | None:
    if not prefix:
        return None
    try:
        from vllm.model_executor.models.utils import extract_layer_index

        return extract_layer_index(prefix)
    except (AssertionError, ValueError, ImportError):
        return None


_xpu_v3_hybrid_available = current_omni_platform.is_xpu() and sageattn_v3_hybrid is not None
_xpu_warned_once: dict[str, bool] = {}


def _xpu_v3_hybrid_kernel_ready() -> bool:
    """Report whether the platform-selected DeepKLOX kernel is available."""
    return _xpu_v3_hybrid_available


def _xpu_v3_hybrid_unsupported(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "cri" in message or "xe3p" in message or "without the sageattention v3 hybrid" in message


def _xpu_warn_once(key: str, message: str) -> None:
    if not _xpu_warned_once.get(key):
        logger.warning(message)
        _xpu_warned_once[key] = True


if not current_omni_platform.is_xpu():
    # Wrapping sageattn3_blackwell as a torch.library custom op keeps it opaque to
    # torch.compile. Otherwise Dynamo graph-breaks on the raw pybind11 kernel and
    # Inductor fails scheduling with KeyError: 'op5'. The hasattr guard keeps this
    # idempotent across test re-imports that pop the module from sys.modules.
    if not hasattr(torch.ops.vllm_omni, "sageattn3_blackwell"):

        @torch.library.custom_op("vllm_omni::sageattn3_blackwell", mutates_args=())
        def _sageattn3_blackwell_op(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            is_causal: bool,
        ) -> torch.Tensor:
            from sageattn3 import sageattn3_blackwell as _kernel

            return _kernel(query, key, value, is_causal=is_causal)

        @_sageattn3_blackwell_op.register_fake
        def _(query, key, value, is_causal):
            return torch.empty_like(query)

    _sageattn3_blackwell_op = torch.ops.vllm_omni.sageattn3_blackwell


def _xpu_v3_hybrid_kernel_call(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Run the DeepKLOX Sage V3 Hybrid kernel using NHD BF16 tensors."""
    assert sageattn_v3_hybrid is not None
    return sageattn_v3_hybrid(query, key, value, tensor_layout="NHD")


def _xpu_sage_fallback_kernel_call(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    softmax_scale: float,
    mode: str,
) -> torch.Tensor:
    if sageattn_v1_v2 is None:
        raise RuntimeError(f"SAGE_ATTN_FALLBACK={mode} requires DeepKloX built with SageAttention V1/V2 support.")
    precision = _xpu_sage_fallback_precision(mode)
    return sageattn_v1_v2(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        tensor_layout="NHD",
        is_causal=causal,
        sm_scale=softmax_scale,
        precision=precision,
    )


def _xpu_sage_fallback_precision(mode: str) -> str:
    defaults = {"sage_v1": "v1fp8", "sage_v2": "mxfp8"}
    supported = {
        "sage_v1": {"int8", "v1", "sagev1", "v1fp8", "v1fp8_fp8"},
        "sage_v2": {"auto", "fp8", "mxfp8"},
    }
    if mode not in defaults:
        raise ValueError(f"Unsupported Sage fallback mode: {mode!r}")

    precision = _SAGE_ATTN_PRECISION or defaults[mode]
    if precision not in supported[mode]:
        raise ValueError(
            f"SAGE_ATTN_PRECISION={precision!r} is incompatible with "
            f"SAGE_ATTN_FALLBACK={mode}; expected one of "
            f"{', '.join(sorted(supported[mode]))}"
        )
    return precision


def _xpu_fallback_mode(kind: str) -> str:
    if kind == "forced_sdpa":
        return _SAGE_ATTN3_FORCE_FALLBACK
    return _SAGE_ATTN3_FALLBACK


def _make_xpu_flash_fallback(**impl_kwargs) -> AttentionImpl | None:
    """Build a FlashAttentionImpl to serve the calls the V3 kernel cannot take.

    Flash attention rather than torch SDPA, because at these sequence lengths SDPA is
    not a cheap safety net. Measured on CRI at Wan2.2 720p/81f self-attention
    (S=75600, H=40, D=128, bf16): the V3 kernel is 0.252s per call and torch SDPA is
    1.326s, 5.3x more. With SAGE_ATTN_FORCE_SDPA_BLOCKS=33,34,38,39 that put four
    blocks at 5.3s against the other thirty-six at 9.1s combined -- a "quality
    exception" for 10% of the blocks costing 37% of the pass. Flash also consumes the
    NHD layout the model already has, so it skips the three contiguous transposes in
    and one out that the SDPA path needs on ~774 MB tensors.

    Delegating to FlashAttentionImpl rather than calling flash_attn_varlen_func here
    keeps one implementation of the varlen plumbing, the attention-mask handling and
    the FP8 K/V path, so the forced blocks compute what DIFFUSION_ATTENTION_BACKEND=
    FLASH_ATTN would have computed for the same layer.

    Returns None when this is not XPU, when a non-flash fallback was requested, or
    when no flash kernel is installed -- in which case the caller selects the
    configured alternative.
    """
    if not current_omni_platform.is_xpu():
        return None
    # One gate for the whole knob: fallback() already treats None as "use SDPA", and
    # both outcomes are counted separately (fallback_flash / fallback_sdpa), so the log
    # proves which path a timing came from without trusting the env var was read.
    if "flash" not in (_SAGE_ATTN3_FALLBACK, _SAGE_ATTN3_FORCE_FALLBACK):
        return None
    try:
        from vllm_omni.diffusion.attention.backends.utils.fa import HAS_FLASH_ATTN

        if not HAS_FLASH_ATTN:
            return None
        from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
    except ImportError as error:
        _xpu_warn_once(
            "flash_import",
            "SageAttention3 XPU could not import the flash-attention fallback "
            f"({error}); non-kernel calls will use torch SDPA instead.",
        )
        return None
    return FlashAttentionImpl(**impl_kwargs)


class SageAttention3Backend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        if current_omni_platform.is_xpu():
            return [_V3_HYBRID_HEAD_SIZE]
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN_3"

    @staticmethod
    def get_impl_cls() -> type["SageAttention3Impl"]:
        return SageAttention3Impl


class SageAttention3Impl(AttentionImpl):
    _warned_gqa_fallback_global: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)
        self.layer_idx = _try_extract_layer_index(prefix)
        requested_fallbacks = {_SAGE_ATTN3_FALLBACK, _SAGE_ATTN3_FORCE_FALLBACK}
        if any(mode.startswith("sage_") for mode in requested_fallbacks) and sageattn_v1_v2 is None:
            raise RuntimeError("The configured Sage fallback requires DeepKloX built with SageAttention V1/V2 support.")
        # Reference path for V3-declined calls when flash is configured.
        self._flash_fallback = _make_xpu_flash_fallback(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            **extra_impl_args,
        )

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        if key.shape[1] != query.shape[1]:
            if query.shape[1] % key.shape[1] != 0:
                raise ValueError(
                    "GQA/MQA requires query heads to be a multiple of KV heads, "
                    f"got q_heads={query.shape[1]} and kv_heads={key.shape[1]}"
                )
            if not type(self)._warned_gqa_fallback_global:
                logger.warning("SageAttention3 does not support GQA/MQA (Hq != Hkv); falling back to torch SDPA.")
                type(self)._warned_gqa_fallback_global = True
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=self.causal,
                dropout_p=self.dropout,
                scale=self.softmax_scale,
                enable_gqa=True,
            )
        else:
            output = _sageattn3_blackwell_op(query, key, value, self.causal)

        return output.transpose(1, 2).contiguous()

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        global _xpu_v3_hybrid_available

        orig_dtype = query.dtype
        batch, seq_q, heads_q, head_dim = query.shape
        heads_kv = key.shape[2]
        seq_kv = key.shape[1]

        if orig_dtype != torch.bfloat16:
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)

        def sdpa_fallback() -> torch.Tensor:
            query_hnd = query.transpose(1, 2).contiguous()
            key_hnd = key.transpose(1, 2).contiguous()
            value_hnd = value.transpose(1, 2).contiguous()
            output = F.scaled_dot_product_attention(
                query_hnd,
                key_hnd,
                value_hnd,
                is_causal=self.causal,
                dropout_p=self.dropout,
                scale=self.softmax_scale,
                enable_gqa=heads_q != heads_kv,
            )
            return output.transpose(1, 2).contiguous().to(orig_dtype)

        def fallback(kind: str) -> torch.Tensor:
            """Serve a call the V3 kernel declined with the configured backend.

            ``kind`` records why V3 was skipped; the fallback counters record which
            implementation actually ran.
            """
            _sage_attn3_record(kind)
            fallback_mode = _xpu_fallback_mode(kind)
            if self.dropout:
                _sage_attn3_record("fallback_sdpa")
                return sdpa_fallback()
            if fallback_mode.startswith("sage_"):
                output = _xpu_sage_fallback_kernel_call(
                    query,
                    key,
                    value,
                    causal=self.causal,
                    softmax_scale=self.softmax_scale,
                    mode=fallback_mode,
                )
                _sage_attn3_record(f"fallback_{fallback_mode}")
                return output.to(orig_dtype)
            if fallback_mode == "flash" and self._flash_fallback is not None:
                try:
                    output = self._flash_fallback.forward_xpu(query, key, value, attn_metadata)
                    _sage_attn3_record("fallback_flash")
                    return output.to(orig_dtype)
                except (ImportError, RuntimeError, ValueError) as error:
                    # A capability or shape rejection is a property of this layer, not
                    # of this call, so stop retrying instead of paying the exception
                    # once per block per step.
                    self._flash_fallback = None
                    _xpu_warn_once(
                        "flash_fallback",
                        "SageAttention3 XPU flash-attention fallback failed "
                        f"({type(error).__name__}: {error}); using torch SDPA instead.",
                    )
            _sage_attn3_record("fallback_sdpa")
            return sdpa_fallback()

        if seq_q != seq_kv:
            return fallback("cross_sdpa")

        if self.layer_idx is not None and self.layer_idx in _SAGE_ATTN3_FORCE_SDPA_BLOCKS:
            return fallback("forced_sdpa")

        if batch != _V3_HYBRID_BATCH or heads_q != heads_kv or head_dim != _V3_HYBRID_HEAD_SIZE or self.causal:
            _xpu_warn_once(
                "contract",
                "SageAttention3 XPU kernel requires batch=1, Hq==Hkv, "
                f"head_dim={_V3_HYBRID_HEAD_SIZE} and non-causal attention; "
                "falling back to flash attention.",
            )
            return fallback("sdpa_fallback")

        if not _xpu_v3_hybrid_kernel_ready():
            _xpu_warn_once(
                "missing",
                "SageAttention3 XPU kernel (deepklox) is not importable; "
                "install deepklox with the Sage V3 Hybrid kernel enabled. "
                "Falling back to flash attention.",
            )
            return fallback("sdpa_fallback")

        try:
            output = _xpu_v3_hybrid_kernel_call(query, key, value)
        except RuntimeError as error:
            if _xpu_v3_hybrid_unsupported(error):
                _xpu_v3_hybrid_available = False
                _xpu_warn_once(
                    "unsupported",
                    "DeepKLOX Sage V3 Hybrid requires CRI (Xe3P); falling back to flash attention.",
                )
                return fallback("sdpa_fallback")
            raise

        if _SAGE_ATTN3_DEBUG_CHECK_FINITE and not torch.isfinite(output).all():
            _xpu_warn_once(
                "nonfinite",
                "SageAttention3 XPU kernel produced non-finite output; re-running with flash attention for this call.",
            )
            return fallback("nonfinite_sdpa")

        _sage_attn3_record("sage")
        return output.to(orig_dtype)

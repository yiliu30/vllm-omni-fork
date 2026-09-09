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

try:
    from sageattn3 import sageattn3_blackwell  # noqa: F401
    _SAGEATTN3_BLACKWELL_AVAILABLE = True
except ImportError:
    _SAGEATTN3_BLACKWELL_AVAILABLE = False
    logger.warning(
        "SageAttention3 Blackwell kernel not available (required for "
        "SAGE_ATTN_3 on CUDA); SAGE_ATTN_3 on CUDA will fall back to torch "
        "SDPA. Install `sageattn3` from "
        "https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell"
    )

# ---------------------------------------------------------------------------
# XPU: Sage V3 Hybrid via deepklox.
#
# The env knobs below are read here on purpose (module-local) rather than in
# the config layer; moving them to the config surface is a follow-up.
# ---------------------------------------------------------------------------
_SAGE_ATTN3_FORCE_SDPA_BLOCKS = frozenset(
    int(block)
    for block in os.environ.get("SAGE_ATTN_FORCE_SDPA_BLOCKS", "").split(",")
    if block.strip()
)
_SAGE_ATTN3_REPORT_FALLBACKS = os.environ.get("SAGE_ATTN_REPORT_FALLBACKS", "0") == "1"
_SAGE_ATTN3_DEBUG_CHECK_FINITE = os.environ.get("SAGE_ATTN_DEBUG_CHECK_FINITE", "0") == "1"

# deepklox Sage V3 Hybrid kernel contract.
_V3_HYBRID_HEAD_SIZE = 128
_V3_HYBRID_BATCH = 1

_sage_attn3_call_counts = {
    "sage": 0,
    "forced_sdpa": 0,
    "cross_sdpa": 0,
    "sdpa_fallback": 0,
    "nonfinite_sdpa": 0,
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
            f"nonfinite_sdpa={_sage_attn3_call_counts['nonfinite_sdpa']}"
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


_xpu_v3_hybrid_available: bool | None = None
_xpu_warned_once: dict[str, bool] = {}


def _xpu_v3_hybrid_kernel_ready() -> bool:
    """Lazily probe deepklox. CRI/Xe3P support is confirmed on the first kernel call."""
    global _xpu_v3_hybrid_available

    if _xpu_v3_hybrid_available is None:
        try:
            from deepklox.sageattn_interface import sageattn_v3_hybrid  # noqa: F401

            _xpu_v3_hybrid_available = True
        except ImportError:
            _xpu_v3_hybrid_available = False
    return _xpu_v3_hybrid_available


def _xpu_v3_hybrid_unsupported(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "cri" in message or "xe3p" in message or "without the sageattention v3 hybrid" in message


def _xpu_warn_once(key: str, message: str) -> None:
    if not _xpu_warned_once.get(key):
        logger.warning(message)
        _xpu_warned_once[key] = True


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

# The XPU Sage V3 Hybrid kernel (deepklox) is called directly rather than
# through a torch.library custom op: the raw pybind11 call graph-breaks
# cleanly under torch.compile on XPU, while the custom-op path adds per-call
# dispatch overhead there. The import is deferred to first call.


def _xpu_v3_hybrid_kernel_call(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Run the deepklox Sage V3 Hybrid kernel (HND, BF16, CRI only)."""
    from deepklox.sageattn_interface import sageattn_v3_hybrid

    return sageattn_v3_hybrid(query, key, value)


class SageAttention3Backend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        if current_omni_platform.is_xpu():
            # The deepklox Sage V3 Hybrid kernel requires head_dim=128.
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
        query_hnd = query.transpose(1, 2).contiguous()
        key_hnd = key.transpose(1, 2).contiguous()
        value_hnd = value.transpose(1, 2).contiguous()
        batch, heads_q, seq_q, head_dim = query_hnd.shape
        heads_kv = key_hnd.shape[1]
        seq_kv = key_hnd.shape[2]

        # The validated mk-2 behavior runs all attention — the quantized
        # kernel and the SDPA fallbacks — in BF16, then casts back. Do the
        # cast before any branch so kernel and SDPA paths stay identical.
        if orig_dtype != torch.bfloat16:
            query_hnd = query_hnd.to(torch.bfloat16)
            key_hnd = key_hnd.to(torch.bfloat16)
            value_hnd = value_hnd.to(torch.bfloat16)

        def sdpa_fallback() -> torch.Tensor:
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

        # Cross-attention (e.g. text conditioning) has mismatched sequence
        # lengths; the V3 Hybrid kernel is self-attention only, so SDPA by design.
        if seq_q != seq_kv:
            _sage_attn3_record("cross_sdpa")
            return sdpa_fallback()

        # First-class selective fallback: per-layer forced SDPA.
        if self.layer_idx is not None and self.layer_idx in _SAGE_ATTN3_FORCE_SDPA_BLOCKS:
            _sage_attn3_record("forced_sdpa")
            return sdpa_fallback()

        # Kernel contract: batch=1, Hq == Hkv, head_dim == 128, non-causal.
        if (
            batch != _V3_HYBRID_BATCH
            or heads_q != heads_kv
            or head_dim != _V3_HYBRID_HEAD_SIZE
            or self.causal
        ):
            _xpu_warn_once(
                "contract",
                "SageAttention3 XPU kernel requires batch=1, Hq==Hkv, "
                f"head_dim={_V3_HYBRID_HEAD_SIZE} and non-causal attention; "
                "falling back to torch SDPA.",
            )
            _sage_attn3_record("sdpa_fallback")
            return sdpa_fallback()

        if not _xpu_v3_hybrid_kernel_ready():
            _xpu_warn_once(
                "missing",
                "SageAttention3 XPU kernel (deepklox) is not importable; "
                "install the deepklox package with the Sage V3 Hybrid kernel "
                "enabled. Falling back to torch SDPA.",
            )
            _sage_attn3_record("sdpa_fallback")
            return sdpa_fallback()

        try:
            output = _xpu_v3_hybrid_kernel_call(query_hnd, key_hnd, value_hnd)
        except RuntimeError as error:
            if _xpu_v3_hybrid_unsupported(error):
                _xpu_v3_hybrid_available = False
                _xpu_warn_once(
                    "unsupported",
                    "deepklox Sage V3 Hybrid kernel requires CRI (Xe3P); "
                    "falling back to torch SDPA.",
                )
                _sage_attn3_record("sdpa_fallback")
                return sdpa_fallback()
            raise

        if _SAGE_ATTN3_DEBUG_CHECK_FINITE and not torch.isfinite(output).all():
            _xpu_warn_once(
                "nonfinite",
                "SageAttention3 XPU kernel produced non-finite output; "
                "re-running with torch SDPA for this call.",
            )
            _sage_attn3_record("nonfinite_sdpa")
            return sdpa_fallback()

        _sage_attn3_record("sage")
        return output.transpose(1, 2).contiguous().to(orig_dtype)

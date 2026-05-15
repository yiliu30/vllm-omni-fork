# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)

# ---------- sage3 standalone (preferred) ----------
_USE_SAGE3 = False
_sage3_fn = None

_SAGE3_PATH = os.environ.get(
    "SAGE3_STANDALONE_PATH",
    "/home/yiliu7/workspace/sage-attention-inner/standalone",
)

try:
    import sys
    if _SAGE3_PATH not in sys.path:
        sys.path.insert(0, _SAGE3_PATH)
    from sage3 import sageattn3_standalone
    _sage3_fn = sageattn3_standalone
    _USE_SAGE3 = True
    logger.info("SageAttention: using sage3 standalone backend from %s", _SAGE3_PATH)
except Exception as e:
    logger.info("sage3 standalone not available (%s), falling back to sageattention v2", e)

# ---------- sageattention v2 fallback ----------
_sageattn_v2 = None
if not _USE_SAGE3:
    try:
        from sageattention import sageattn
        _sageattn_v2 = sageattn
    except ImportError:
        logger.warning(
            "Neither sage3 standalone nor sageattention v2 is available. "
            "Install one of them to use SageAttentionBackend."
        )
        raise ImportError


class SageAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl


class SageAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale

        # sage3 config from env or backend_kwargs
        self._sage3_config = os.environ.get("SAGE3_QUANT_FORMAT", "mxfp4")
        self._sage3_acc_dtype = os.environ.get("SAGE3_ACC_DTYPE", "fp32")
        if backend_kwargs:
            self._sage3_config = backend_kwargs.pop("sage3_config", self._sage3_config)
            self._sage3_acc_dtype = backend_kwargs.pop("sage3_acc_dtype", self._sage3_acc_dtype)
            if backend_kwargs:
                logger.warning("SageAttentionImpl ignoring backend_kwargs: %s",
                               list(backend_kwargs.keys()))

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        # Input layout: NHD = [B, N, H, D]
        if _USE_SAGE3:
            return self._forward_sage3(query, key, value)
        else:
            return _sageattn_v2(
                query, key, value,
                tensor_layout="NHD",
                is_causal=self.causal,
                sm_scale=self.softmax_scale,
            )

    def _forward_sage3(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        # sage3 expects HND = [B, H, N, D], input is NHD = [B, N, H, D]
        q = query.transpose(1, 2)   # [B, H, N, D]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # sage3 requires Q/K/V to have the same seq length.
        # For cross-attention (different N), fall back to torch SDPA.
        if q.shape[2] != k.shape[2]:
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=self.causal,
                scale=self.softmax_scale,
            )
            return out.transpose(1, 2)  # back to NHD

        out = _sage3_fn(
            q, k, v,
            config=self._sage3_config,
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
            acc_dtype=self._sage3_acc_dtype,
        )
        return out.transpose(1, 2)  # back to NHD

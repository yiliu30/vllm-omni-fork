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

# ---------- GPU shared memory detection ----------
# sage3 kernel with fp32 acc needs ~192KB shmem (3 × 128 × 128 × 4 bytes).
# bf16_both_dot reduces to ~96KB but causes accuracy loss (black images in diffusion).
# On GPUs with insufficient shmem for fp32, we fall back to torch SDPA.
_SAGE3_SHMEM_THRESHOLD = 192 * 1024  # minimum shmem for fp32 sage3

def _gpu_has_enough_shmem() -> bool:
    """Check if GPU has enough shared memory for sage3 fp32 kernel."""
    try:
        shmem = torch.cuda.get_device_properties(0).shared_memory_per_multiprocessor
        return shmem >= _SAGE3_SHMEM_THRESHOLD
    except Exception:
        return True  # optimistic default


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

        # Determine whether sage3 kernel can run on this GPU
        self._use_sage3_kernel = False
        if _USE_SAGE3:
            if self._sage3_acc_dtype == "fp32" and not _gpu_has_enough_shmem():
                try:
                    shmem = torch.cuda.get_device_properties(0).shared_memory_per_multiprocessor
                except Exception:
                    shmem = 0
                logger.warning(
                    "sage3: GPU shmem (%dKB) < %dKB needed for fp32 kernel. "
                    "Falling back to torch SDPA. Use a GPU with >= 192KB shmem "
                    "(e.g. H100) or set SAGE3_ACC_DTYPE to force bf16 (may lose accuracy).",
                    shmem // 1024, _SAGE3_SHMEM_THRESHOLD // 1024,
                )
            else:
                self._use_sage3_kernel = True

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        # Input layout: NHD = [B, N, H, D]
        if self._use_sage3_kernel:
            return self._forward_sage3(query, key, value)
        elif _sageattn_v2 is not None:
            return _sageattn_v2(
                query, key, value,
                tensor_layout="NHD",
                is_causal=self.causal,
                sm_scale=self.softmax_scale,
            )
        else:
            return self._forward_sdpa(query, key, value)

    def _forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback to torch scaled_dot_product_attention (always correct)."""
        # Convert NHD → HND for SDPA
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal,
            scale=self.softmax_scale,
        )
        return out.transpose(1, 2)  # back to NHD

    def _forward_sage3(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        # sage3 expects HND = [B, H, N, D], input is NHD = [B, N, H, D]
        q = query.transpose(1, 2).contiguous()   # [B, H, N, D]
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        # Cross-attention (different Q/K seq lengths) — sage3 can't handle
        if q.shape[2] != k.shape[2]:
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=self.causal,
                scale=self.softmax_scale,
            )
            return out.transpose(1, 2)

        out = _sage3_fn(
            q, k, v,
            config=self._sage3_config,
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
            acc_dtype=self._sage3_acc_dtype,
        )
        return out.transpose(1, 2)  # back to NHD

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from pathlib import Path

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_AUTO_ROUND_KERNEL_PATH = Path(
    os.environ.get(
        "AUTO_ROUND_KERNEL_PATH",
        "/home/yiliu7/workspace/auto-round/auto_round_extension/ark",
    )
)
if _AUTO_ROUND_KERNEL_PATH.exists() and str(_AUTO_ROUND_KERNEL_PATH) not in sys.path:
    sys.path.insert(0, str(_AUTO_ROUND_KERNEL_PATH))

# Omni uses NHD layout on the attention path; keep the ARK adapter fixed to it.
_SAGE_ATTN_TENSOR_LAYOUT = "NHD"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


_SAGE_ATTN_XPU_BACKEND = os.environ.get("SAGE_ATTN_XPU_BACKEND", "sparse").strip().lower()
assert _SAGE_ATTN_XPU_BACKEND in ("sparse", "sagev1"), (
    f"SAGE_ATTN_XPU_BACKEND must be 'sparse' or 'sagev1', got '{_SAGE_ATTN_XPU_BACKEND}'"
)
# Keep topk>=1.0 on the sparse path by default. Dense-equivalent sparse routing is
# useful for debug and differential analysis, even when it selects every block.
_SAGE_ATTN_ALLOW_TOPK_EQ_1 = _env_bool("SAGE_ATTN_ALLOW_TOPK_EQ_1", True)
_SAGE_ATTN_REF_KERNEL = _env_bool("SAGE_ATTN_REF_KERNEL", False)
_SAGE_ATTN_DUMP_DIR = os.environ.get("SAGE_ATTN_DUMP_DIR", "").strip()
_SAGE_ATTN_SPARSE_TOPK = _env_float("SAGE_ATTN_TOPK", 0.5)
_SAGE_ATTN_SPARSE_SMOOTH_K = _env_bool("SAGE_ATTN_SMOOTH_K", True)
_SAGE_ATTN_SPARSE_SIMTHRESHD1 = _env_float("SAGE_ATTN_SIMTHRESHD1", -0.1)
_SAGE_ATTN_SPARSE_ATTENTION_SINK = _env_bool("SAGE_ATTN_ATTENTION_SINK", False)
_SAGE_ATTN_K_QUANT_GRANULARITY = _env_int("SAGE_ATTN_K_QUANT_GRANULARITY", 64)
_SAGE_ATTN_DEBUG_CALLS = _env_bool("SAGE_ATTN_DEBUG_CALLS", False)
_SAGE_ATTN_DEBUG_CALLS_FILE = os.environ.get("SAGE_ATTN_DEBUG_CALLS_FILE", "").strip()

if current_omni_platform.is_xpu():
    try:
        import inspect

        try:
            from auto_round_kernel import ARK

            _ark = ARK()
        except ImportError:
            import auto_round_kernel as _ark  # type: ignore[no-redef]

        xpu_sageattn_v1 = _ark.sagev1
        xpu_sageattn_sparse = getattr(_ark, "sparge_sage2_attn_meansim_topk_xpu", None)
        _sagev1_params = inspect.signature(xpu_sageattn_v1).parameters
        _sagev1_has_tensor_layout = "tensor_layout" in _sagev1_params
        _sagev1_scale_param = "sm_scale" if "sm_scale" in _sagev1_params else "scale"
        _sparse_has_tensor_layout = False
        if xpu_sageattn_sparse is not None:
            _sparse_params = inspect.signature(xpu_sageattn_sparse).parameters
            _sparse_has_tensor_layout = "tensor_layout" in _sparse_params
    except ImportError:
        logger.warning(
            "XPU SageAttention (auto_round_kernel.ARK.sagev1) is not available. "
            "Install auto-round-lib for XPU sage attention support."
        )
        xpu_sageattn_v1 = None
        xpu_sageattn_sparse = None
        _sagev1_has_tensor_layout = False
        _sagev1_scale_param = "scale"
        _sparse_has_tensor_layout = False
else:
    try:
        from sageattention import sageattn
    except ImportError:
        logger.warning(
            "SageAttentionBackend is not available. You may install sage-attention"
            " by pip install git+https://github.com/thu-ml/SageAttention.git"
        )
        raise ImportError

_dump_counter = 0
_debug_call_counter = 0


def _dump_preprocess_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
    scale: float,
    topk: float,
    smooth_k: bool,
    simthreshd1: float,
    attention_sink: bool,
    layout: str,
) -> None:
    """Dump preprocess inputs for offline CUDA comparison."""
    global _dump_counter
    dump_dir = Path(_SAGE_ATTN_DUMP_DIR)
    dump_dir.mkdir(parents=True, exist_ok=True)
    idx = _dump_counter
    _dump_counter += 1
    base = dump_dir / f"sparse_dump_{idx:04d}"

    torch.save(query.cpu().float(), f"{base}_query.pt")
    torch.save(key.cpu().float(), f"{base}_key.pt")
    torch.save(value.cpu().float(), f"{base}_value.pt")

    import json
    params = {
        "is_causal": is_causal,
        "scale": scale,
        "topk": topk,
        "smooth_k": smooth_k,
        "simthreshd1": simthreshd1,
        "attention_sink": attention_sink,
        "tensor_layout": layout,
        "quant_block_size": 64,
        "B": query.shape[0],
        "Sq": query.shape[1] if layout.upper() == "NHD" else query.shape[2],
        "Hq": query.shape[2] if layout.upper() == "NHD" else query.shape[1],
        "Skv": key.shape[1] if layout.upper() == "NHD" else key.shape[2],
        "Hkv": key.shape[2] if layout.upper() == "NHD" else key.shape[1],
        "D": query.shape[-1],
    }
    with open(f"{base}_params.json", "w") as f:
        json.dump(params, f, indent=2)
    logger.info("SPARSE_DUMP: saved preprocess inputs to %s_*", base)


def _dump_preprocess_outputs(metadata: dict) -> None:
    """Dump preprocess outputs for offline CUDA comparison."""
    global _dump_counter
    dump_dir = Path(_SAGE_ATTN_DUMP_DIR)
    idx = _dump_counter - 1
    base = dump_dir / f"sparse_dump_{idx:04d}"

    for key in ("query_i8", "key_i8", "qscale", "kscale",
                "lut", "valid_block_num",
                "block_map", "raw_block_map", "tile_block_map",
                "sim_qblocks", "sim_kblocks"):
        val = metadata.get(key)
        if val is not None:
            torch.save(val.cpu(), f"{base}_{key}.pt")
    logger.info("SPARSE_DUMP: saved preprocess outputs to %s_*", base)


def _log_sparse_call(stage: str, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> int | None:
    global _debug_call_counter
    if not _SAGE_ATTN_DEBUG_CALLS:
        return None
    call_id = _debug_call_counter
    _debug_call_counter += 1
    logger.info(
        "SAGE_DEBUG call=%d stage=%s q=%s k=%s v=%s dtype=%s device=%s",
        call_id,
        stage,
        tuple(query.shape),
        tuple(key.shape),
        tuple(value.shape),
        query.dtype,
        query.device,
    )
    if _SAGE_ATTN_DEBUG_CALLS_FILE:
        with open(_SAGE_ATTN_DEBUG_CALLS_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"call={call_id} stage={stage} q={tuple(query.shape)} "
                f"k={tuple(key.shape)} v={tuple(value.shape)} "
                f"dtype={query.dtype} device={query.device}\n"
            )
            f.flush()
    return call_id


# TODO add sage3 attention backend


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
        if backend_kwargs:
            logger.warning("SageAttentionImpl ignoring backend_kwargs: %s", list(backend_kwargs.keys()))

    def _forward_xpu_sagev1(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        if xpu_sageattn_v1 is None:
            raise ImportError("XPU SageAttention requires auto-round-lib. Install with: pip install auto-round-lib")
        orig_dtype = query.dtype
        q = query.to(torch.float16) if orig_dtype != torch.float16 else query
        k = key.to(torch.float16) if orig_dtype != torch.float16 else key
        v = value.to(torch.float16) if orig_dtype != torch.float16 else value

        if _sagev1_has_tensor_layout:
            output = xpu_sageattn_v1(
                q,
                k,
                v,
                tensor_layout=_SAGE_ATTN_TENSOR_LAYOUT,
                is_causal=self.causal,
                **{_sagev1_scale_param: self.softmax_scale},
            )
        else:
            output = xpu_sageattn_v1(
                q,
                k,
                v,
                is_causal=self.causal,
                **{_sagev1_scale_param: self.softmax_scale},
            )

        if orig_dtype != torch.float16:
            output = output.to(orig_dtype)
        return output

    def _forward_xpu_sparse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        if xpu_sageattn_sparse is None:
            logger.warning("Sparse XPU SageAttention is unavailable; falling back to sagev1")
            return self._forward_xpu_sagev1(query, key, value, attn_metadata)

        orig_dtype = query.dtype
        q = query
        k = key
        v = value

        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.float16)
        if k.dtype != q.dtype:
            k = k.to(q.dtype)
        if v.dtype != q.dtype:
            v = v.to(q.dtype)

        sparse_kwargs = {
            "is_causal": self.causal,
            "scale": self.softmax_scale,
            "topk": _SAGE_ATTN_SPARSE_TOPK,
            "smooth_k": _SAGE_ATTN_SPARSE_SMOOTH_K,
            "simthreshd1": _SAGE_ATTN_SPARSE_SIMTHRESHD1,
            "attention_sink": _SAGE_ATTN_SPARSE_ATTENTION_SINK,
            "k_quant_granularity": _SAGE_ATTN_K_QUANT_GRANULARITY,
        }
        if sparse_kwargs["topk"] >= 1.0 and not _SAGE_ATTN_ALLOW_TOPK_EQ_1:
            logger.info("SAGE_ATTN topk>=1 disabled for sparse debug path; falling back to sagev1")
            return self._forward_xpu_sagev1(query, key, value, attn_metadata)
        if sparse_kwargs["topk"] >= 1.0:
            logger.info("SAGE_ATTN topk>=1 stays on sparse path for debug")
        call_id = _log_sparse_call("start", q, k, v)

        if _SAGE_ATTN_REF_KERNEL:
            output = self._forward_xpu_sparse_ref(q, k, v, **sparse_kwargs)
        elif _sparse_has_tensor_layout:
            output = xpu_sageattn_sparse(q, k, v, tensor_layout=_SAGE_ATTN_TENSOR_LAYOUT, **sparse_kwargs)
        else:
            output = xpu_sageattn_sparse(q, k, v, **sparse_kwargs)

        if _SAGE_ATTN_DEBUG_CALLS:
            logger.info("SAGE_DEBUG call=%d stage=finish out=%s", call_id, tuple(output.shape))
            if _SAGE_ATTN_DEBUG_CALLS_FILE:
                with open(_SAGE_ATTN_DEBUG_CALLS_FILE, "a", encoding="utf-8") as f:
                    f.write(f"call={call_id} stage=finish out={tuple(output.shape)}\n")
                    f.flush()

        if orig_dtype != torch.float16:
            output = output.to(orig_dtype)
        return output

    def _forward_xpu_sparse_ref(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
        scale: float,
        topk: float,
        smooth_k: bool,
        simthreshd1: float,
        attention_sink: bool,
        k_quant_granularity: int = 64,
    ) -> torch.Tensor:
        """Run the same preprocess then use PyTorch reference kernel instead of SYCL sage_sparse.

        This isolates whether quality issues come from the preprocess (routing)
        or the final kernel execution.  Environment: SAGE_ATTN_REF_KERNEL=1.
        """
        from auto_round_kernel import sparge_preprocess_topk
        from vllm_omni.diffusion.attention.backends.sage_sparse_ref import sage_sparse_ref

        layout = _SAGE_ATTN_TENSOR_LAYOUT

        # --- Optional dump: save preprocess inputs for CUDA comparison ---
        if _SAGE_ATTN_DUMP_DIR:
            _dump_preprocess_inputs(
                query, key, value,
                is_causal=is_causal, scale=scale,
                topk=topk, smooth_k=smooth_k,
                simthreshd1=simthreshd1,
                attention_sink=attention_sink,
                layout=layout,
            )

        # 1. Preprocess (same Triton-XPU path as the SYCL kernel)
        metadata = sparge_preprocess_topk(
            query,
            key,
            is_causal=is_causal,
            smooth_k=smooth_k,
            simthreshd1=simthreshd1,
            topk=topk,
            attention_sink=attention_sink,
            quant_block_size=64,
            tensor_layout=layout,
            k_quant_granularity=k_quant_granularity,
        )

        # --- Optional dump: save preprocess outputs ---
        if _SAGE_ATTN_DUMP_DIR:
            _dump_preprocess_outputs(metadata)

        # 2. Reference kernel
        output = sage_sparse_ref(
            metadata["query_i8"],
            metadata["key_i8"],
            value,
            metadata["lut"],
            metadata["valid_block_num"],
            is_causal=is_causal,
            scale=scale,
            quant_block_size=metadata["quant_block_size"],
            qscale=metadata["qscale"],
            kscale=metadata["kscale"],
            tensor_layout=layout,
        )

        logger.info(
            "sparse ref kernel used: %d/%d blocks selected (%.1f%%), backend=%s",
            metadata["stats"]["total_selected"],
            metadata["stats"]["total_candidates"],
            metadata["stats"]["selected_ratio"] * 100,
            metadata["backend"],
        )
        return output

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        output = sageattn(
            query,
            key,
            value,
            tensor_layout="NHD",
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
        )
        return output

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        if _SAGE_ATTN_XPU_BACKEND == "sagev1":
            return self._forward_xpu_sagev1(query, key, value, attn_metadata)
        return self._forward_xpu_sparse(query, key, value, attn_metadata)

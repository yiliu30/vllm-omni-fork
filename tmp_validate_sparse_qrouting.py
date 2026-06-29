#!/usr/bin/env python
"""Differential validation of the Q-routing fix: SYCL sage_sparse vs PyTorch ref.

Both consume IDENTICAL preprocess metadata (same LUT / int8 Q,K / scales). The
PyTorch ref walks the LUT per-64-token Q block (one row per block), which is
CUDA's CTA_Q=64 behavior. The SYCL kernel BEFORE the fix read one LUT row per
256-token tile (1-of-4 rows), so it diverged from the ref at topk=0.5.

Why ref-vs-SYCL is a valid differential probe even though the ref is not the
absolute oracle: at topk=1.0 routing is a no-op (all blocks selected) and the
two already agree to ~0.9999, which empirically pins down that their non-routing
math (int8 matmul, dequant, softmax) matches. Any topk=0.5 divergence is THEN
attributable to LUT-row consumption alone — exactly what the fix changes.

Expected BEFORE fix: topk=1.0 cos ~0.9999 ; topk=0.5 cos ~0.71-0.76
Expected AFTER  fix: topk=0.5 cos rises toward the topk=1.0 agreement (~0.99).
"""
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ARK_ROOT = Path("/home/yiliu7/workspace/auto-round/auto_round_extension/ark")
VLLM_ROOT = Path("/data/model/yiliu7/vllm-omni")
for p in (str(ARK_ROOT), str(VLLM_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import auto_round_kernel as ark
from vllm_omni.diffusion.attention.backends.sage_sparse_ref import sage_sparse_ref


def ensure_sparse_binding() -> None:
    if getattr(ark, "xpu_lib", None) is not None and hasattr(ark.xpu_lib, "sage_sparse"):
        return
    cands = sorted((ARK_ROOT / "xbuild").glob("auto_round_kernel_xpu*.so"))
    if not cands:
        cands = sorted((ARK_ROOT / "auto_round_kernel" / "xbuild").glob("auto_round_kernel_xpu*.so"))
    if not cands:
        raise RuntimeError("no built XPU extension found")
    ext = cands[-1]
    spec = importlib.util.spec_from_file_location("auto_round_kernel_xpu", ext)
    module = importlib.util.module_from_spec(spec)
    sys.modules["auto_round_kernel_xpu"] = module
    spec.loader.exec_module(module)
    ark.xpu_lib = module
    print(f"[validate] loaded SYCL ext: {ext}")
    import hashlib
    print(f"[validate] ext md5: {hashlib.md5(ext.read_bytes()).hexdigest()}")


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0))


def main() -> None:
    ensure_sparse_binding()
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU required"
    device = torch.device("xpu")
    dtype = torch.float16

    B, Hq, Hkv, S, D = 1, 8, 8, 1536, 128
    scale = 1.0 / math.sqrt(D)
    is_causal = False
    layout = "NHD"

    torch.manual_seed(20260624)
    q = torch.randn(B, S, Hq, D, dtype=dtype, device=device)
    k = torch.randn(B, S, Hkv, D, dtype=dtype, device=device)
    v = torch.randn(B, S, Hkv, D, dtype=dtype, device=device)

    # dense reference (HND for SDPA), for context only
    dense = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        dropout_p=0.0, is_causal=is_causal, scale=scale,
    ).permute(0, 2, 1, 3).contiguous()  # NHD

    print(f"[validate] geometry B={B} Hq={Hq} S={S} D={D} layout={layout} causal={is_causal}")
    print(f"[validate] {'topk':>6} | {'sel':>5} | {'SYCL~ref':>9} | {'SYCL~dense':>10} | {'ref~dense':>9}")
    res = {}
    for topk in (1.0, 0.5):
        meta = ark.sparge_preprocess_topk(
            q, k, is_causal=is_causal, smooth_k=True, simthreshd1=-0.1,
            topk=topk, attention_sink=False, quant_block_size=64, tensor_layout=layout,
        )
        sr = float(meta.get("stats", {}).get("selected_ratio", float("nan")))

        # SYCL kernel on the metadata
        sycl = ark.sage_sparse(
            meta["query_i8"], meta["key_i8"], v,
            meta["lut"], meta["valid_block_num"],
            is_causal=is_causal, scale=scale,
            quant_block_size=meta["quant_block_size"],
            qscale=meta["qscale"], kscale=meta["kscale"],
            tensor_layout=layout,
        )
        torch.xpu.synchronize()

        # PyTorch ref on the SAME metadata (per-64-block LUT walk = CUDA CTA_Q=64)
        ref = sage_sparse_ref(
            meta["query_i8"], meta["key_i8"], v,
            meta["lut"], meta["valid_block_num"],
            is_causal=is_causal, scale=scale,
            quant_block_size=meta["quant_block_size"],
            qscale=meta["qscale"], kscale=meta["kscale"],
            tensor_layout=layout,
        )
        torch.xpu.synchronize()

        c_sr = cos(sycl, ref)
        c_sd = cos(sycl, dense)
        c_rd = cos(ref, dense)
        res[topk] = c_sr
        print(f"[validate] {topk:6.2f} | {sr:5.3f} | {c_sr:9.6f} | {c_sd:10.6f} | {c_rd:9.6f}")

    print()
    print(f"[validate] DECISIVE METRIC = cos(SYCL, ref) on identical metadata:")
    print(f"[validate]   topk=1.0 : {res[1.0]:.6f}  (routing no-op baseline)")
    print(f"[validate]   topk=0.5 : {res[0.5]:.6f}  (was 0.71-0.76 before fix)")
    if res[0.5] >= 0.99:
        print("[validate] PASS: SYCL now matches ref per-64-block LUT consumption (>=0.99)")
    elif res[0.5] >= 0.95:
        print("[validate] LIKELY PASS: large recovery toward the topk=1.0 baseline (>=0.95)")
    elif res[0.5] > 0.85:
        print("[validate] PARTIAL: improved but residual gap remains")
    else:
        print("[validate] FAIL: SYCL still diverges from ref at topk=0.5")


if __name__ == "__main__":
    main()

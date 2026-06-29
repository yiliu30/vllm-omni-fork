#!/usr/bin/env python
"""Chaos-free tail differential: XPU sparse vs torch SDPA (dense golden reference).

WHY dense SDPA is the right golden reference here:
  * It is plain fp16 full attention — no routing, no LUT, no INT8, nothing that
    can be "geometrically wrong". It is the exact quantity sparse approximates.
  * This is NOT the sage_sparse_ref kernel (which has its own routing and is NOT
    trusted). It is torch.nn.functional.scaled_dot_product_attention.

WHAT this isolates that the E2E run cannot:
  A single attention call, so there is NO 40-step diffusion amplification. The
  per-query-row error vs dense is therefore a clean, position-resolved signal.

GEOMETRY (faithful to real Wan S=32760):
  Real: 32760 % 64 = 56 (last quant block 56/64),  32760 % 128 = 120 (last
        K-route tile 120/128).
  Here: S=4216 -> 4216 % 64 = 56,  4216 % 128 = 120.  Same partial-tail
        structure, small enough that even math-backend dense SDPA fits.

THE TWO READS:
  (1) topk=1.0 (routing no-op: ALL blocks selected). Sparse should ~= dense up to
      uniform INT8 quant noise. If the LAST-BLOCK rows diverge MORE than interior
      rows here, the residual tail error is a KERNEL/QUANT/MASK bug (routing-
      independent).
  (2) topk=0.5 (production). The tail-region error profile vs dense localizes the
      residual ROUTING effect after the partial-tail sim fix.

Pass/fail is by the interior-vs-tail RATIO, which cancels the uniform quant noise,
NOT by absolute closeness to dense (sparse is meant to differ at topk<1).
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
    print(f"[tail-diff] loaded SYCL ext: {ext}")


def per_row_cos_to_dense(out_nhd: torch.Tensor, dense_nhd: torch.Tensor) -> torch.Tensor:
    """Return [S] per-query-row cosine similarity (flatten heads*D per row)."""
    B, S, H, D = out_nhd.shape
    a = out_nhd.reshape(S, H * D).float()
    b = dense_nhd.reshape(S, H * D).float()
    return F.cosine_similarity(a, b, dim=1)  # [S]


def per_row_relerr(out_nhd: torch.Tensor, dense_nhd: torch.Tensor) -> torch.Tensor:
    B, S, H, D = out_nhd.shape
    a = out_nhd.reshape(S, H * D).float()
    b = dense_nhd.reshape(S, H * D).float()
    return (a - b).norm(dim=1) / (b.norm(dim=1) + 1e-8)  # [S]


def main() -> None:
    ensure_sparse_binding()
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU required"
    device = torch.device("xpu")
    dtype = torch.float16
    layout = "NHD"

    B, Hq, Hkv, D = 1, 16, 16, 128
    S = 4216  # 4216 % 64 = 56 (last quant blk), 4216 % 128 = 120 (last K-tile)  -- mirrors S=32760
    QB = 64
    num_blocks = (S + QB - 1) // QB
    tail_valid = S - (num_blocks - 1) * QB  # 56
    scale = 1.0 / math.sqrt(D)
    is_causal = False

    print(f"[tail-diff] geometry B={B} Hq={Hq} S={S} D={D} layout={layout} causal={is_causal}")
    print(f"[tail-diff] num quant blocks={num_blocks}  last block valid={tail_valid}/{QB} "
          f"(real Wan: 56/64);  S%128={S % 128} (real: 120)")

    torch.manual_seed(20260624)
    q = torch.randn(B, S, Hq, D, dtype=dtype, device=device)
    k = torch.randn(B, S, Hkv, D, dtype=dtype, device=device)
    v = torch.randn(B, S, Hkv, D, dtype=dtype, device=device)

    # ---- GOLDEN: dense fp16 full attention ----
    dense = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        dropout_p=0.0, is_causal=is_causal, scale=scale,
    ).permute(0, 2, 1, 3).contiguous()  # NHD
    torch.xpu.synchronize()

    interior_slice = slice(0, (num_blocks - 1) * QB)   # all full blocks
    tail_slice = slice((num_blocks - 1) * QB, S)       # the partial last block (last 56 rows)

    configs = [
        ("topk=1.0 sink=off  [routing no-op]", 1.0, False),
        ("topk=0.5 sink=off  [pure routing]", 0.5, False),
        ("topk=0.5 sink=on   [production]", 0.5, True),
    ]

    print(f"\n[tail-diff] {'config':<34} | {'sel':>5} | {'interior cos':>12} | {'tail cos':>9} | "
          f"{'interior relerr':>15} | {'tail relerr':>11} | tail/int")
    print("[tail-diff] " + "-" * 116)

    for label, topk, sink in configs:
        meta = ark.sparge_preprocess_topk(
            q, k, is_causal=is_causal, smooth_k=True, simthreshd1=-0.1,
            topk=topk, attention_sink=sink, quant_block_size=QB, tensor_layout=layout,
        )
        sr = float(meta.get("stats", {}).get("selected_ratio", float("nan")))
        out = ark.sage_sparse(
            meta["query_i8"], meta["key_i8"], v,
            meta["lut"], meta["valid_block_num"],
            is_causal=is_causal, scale=scale,
            quant_block_size=meta["quant_block_size"],
            qscale=meta["qscale"], kscale=meta["kscale"],
            tensor_layout=layout,
        )
        torch.xpu.synchronize()

        cos_rows = per_row_cos_to_dense(out, dense)      # [S]
        err_rows = per_row_relerr(out, dense)            # [S]
        ci = float(cos_rows[interior_slice].mean())
        ct = float(cos_rows[tail_slice].mean())
        ei = float(err_rows[interior_slice].mean())
        et = float(err_rows[tail_slice].mean())
        ratio = et / max(ei, 1e-9)
        print(f"[tail-diff] {label:<34} | {sr:5.3f} | {ci:12.6f} | {ct:9.6f} | "
              f"{ei:15.6f} | {et:11.6f} | {ratio:6.2f}x")

    print()
    print("[tail-diff] READING:")
    print("[tail-diff]  * topk=1.0 row: routing is a no-op (all blocks selected), so interior and")
    print("[tail-diff]    tail should both be ~quant-noise and the tail/int ratio ~1.0. A ratio >>1")
    print("[tail-diff]    here = a routing-INDEPENDENT tail bug (kernel masking / quant scale / V).")
    print("[tail-diff]  * topk=0.5 rows: sparse deliberately differs from dense; compare the tail/int")
    print("[tail-diff]    ratio to the topk=1.0 baseline. Ratio close to the topk=1.0 baseline => the")
    print("[tail-diff]    tail is no worse than interior (residual is uniform sparsity error, not a")
    print("[tail-diff]    tail-specific divergence). Ratio markedly higher => residual tail problem.")


if __name__ == "__main__":
    main()

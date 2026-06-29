#!/usr/bin/env python
"""Decisive probe for the END-OF-FRAMES mismatch: partial tail-block routing.

Real Wan S=32760 has a PARTIAL last 64-block (56 valid tokens) and a partial
last 128 K-routing tile (56 valid). The earlier S=1536 validation was 64/128
aligned -> NO partial block -> blind to any tail-specific divergence.

CUDA oracle (SpargeAttn-fork/spas_sage_attn/utils.py:194-198) computes block
self-similarity WITHOUT a zero-norm guard:  x = x / x_norm  with padded rows = 0
-> 0/0 = NaN -> sum_value = NaN -> cur_sim = (NaN > thr) = False.
=> partial tail blocks are ALWAYS sim=False in CUDA.

XPU triton (sparge_preprocess_triton.py:107) guards: x_norm = where(norm>0,norm,1)
=> partial tail blocks get a REAL sim value (usually True).

In routing, sim=False FORCES a block dense (final_map[~sim]=1). So:
  CUDA  -> tail Q-block + tail K-tile forced DENSE
  XPU   -> tail blocks PRUNED by topk

This script runs the actual XPU preprocess at a partial-tail geometry and reports
the sim flag of the tail blocks + whether the tail differs from the interior.
If XPU marks the tail sim=True (prunable) it confirms the divergence vs CUDA.
"""
import math
import sys
from pathlib import Path

import torch

ARK_ROOT = Path("/home/yiliu7/workspace/auto-round/auto_round_extension/ark")
for p in (str(ARK_ROOT),):
    if p not in sys.path:
        sys.path.insert(0, p)

import auto_round_kernel as ark


def main() -> None:
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU required"
    device = torch.device("xpu")
    dtype = torch.float16
    layout = "NHD"

    # Partial-tail geometry: 24 full 64-blocks + 56 -> last block 56 valid.
    # 1592/128 = 12 full 128-tiles + 56 -> last K routing tile 56 valid (partial).
    B, Hq, Hkv, D = 1, 8, 8, 128
    S = 24 * 64 + 56  # 1592
    qbs = 64
    num_k_blocks = (S + qbs - 1) // qbs
    print(f"[tail] geometry B={B} Hq={Hq} S={S} D={D} layout={layout}")
    print(f"[tail] num_64_blocks={num_k_blocks} last_block_valid={S - (num_k_blocks-1)*qbs}")

    torch.manual_seed(20260624)
    q = torch.randn(B, S, Hq, D, dtype=dtype, device=device)
    k = torch.randn(B, S, Hkv, D, dtype=dtype, device=device)

    for topk in (0.5,):
        meta = ark.sparge_preprocess_topk(
            q, k, is_causal=False, smooth_k=True, simthreshd1=-0.1,
            topk=topk, attention_sink=False, quant_block_size=qbs, tensor_layout=layout,
        )
        sim_q = meta["sim_qblocks"]  # [B, Hq, num_q_tiles]
        sim_k = meta["sim_kblocks"]  # [B, Hkv, num_k_route_tiles]
        block_map = meta["block_map"]  # [B, Hq, num_q_blocks, num_k_blocks] bool
        lut = meta["lut"]
        vbn = meta["valid_block_num"]  # [B, Hq, num_q_blocks]

        print(f"\n[tail] topk={topk}")
        print(f"[tail] sim_q shape={tuple(sim_q.shape)} sim_k shape={tuple(sim_k.shape)}")
        print(f"[tail] block_map shape={tuple(block_map.shape)} vbn shape={tuple(vbn.shape)}")

        # Tail vs interior sim flags (head 0, batch 0)
        sq = sim_q[0, 0]
        sk = sim_k[0, 0]
        print(f"[tail] sim_q  interior all-True? {bool(sq[:-1].all())}  TAIL(last)={bool(sq[-1])}")
        print(f"[tail] sim_k  interior all-True? {bool(sk[:-1].all())}  TAIL(last)={bool(sk[-1])}")

        # valid_block_num for interior q rows vs the TAIL q row
        v0 = vbn[0, 0]
        nqb = v0.numel()
        interior_mean = float(v0[:-1].float().mean())
        tail_val = int(v0[-1])
        print(f"[tail] valid_block_num: interior_mean={interior_mean:.1f}  TAIL_qrow={tail_val}  (num_k_blocks={num_k_blocks})")

        # Does the tail K column get force-selected for interior q rows? (CUDA would force it.)
        tail_kcol_selected_frac = float(block_map[0, 0, :, -1].float().mean())
        interior_kcol_selected_frac = float(block_map[0, 0, :, :-1].float().mean())
        print(f"[tail] P(tail K-col selected over all q rows) = {tail_kcol_selected_frac:.3f}")
        print(f"[tail] P(interior K-col selected)            = {interior_kcol_selected_frac:.3f}")

        # Verdict
        xpu_prunes_tail_q = not bool(sq[-1]) is False and bool(sq[-1])  # sim True => prunable
        print()
        if bool(sq[-1]) or bool(sk[-1]):
            print("[tail] XPU marks a TAIL block sim=True (prunable). CUDA would mark it")
            print("[tail] sim=False (NaN) -> forced DENSE. => DIVERGENCE at the tail. CONFIRMED.")
        else:
            print("[tail] XPU also marks tail sim=False -> matches CUDA; look elsewhere.")


if __name__ == "__main__":
    main()

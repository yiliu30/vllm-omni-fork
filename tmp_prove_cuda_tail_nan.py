#!/usr/bin/env python
"""Prove the CUDA side of the tail divergence WITHOUT CUDA hardware.

The CUDA K-path (SpargeAttn-fork/spas_sage_attn/utils.py, smooth_k=True) is
deterministic for the partial block:

    x -= x_mean
    x = tl.where(xmask, x, 0)          # padded rows -> 0
    x_norm = sqrt(sum(x*x, axis=D))    # padded row norm -> 0
    x = x / x_norm                     # padded row -> 0/0 = NaN   (NO guard)
    grams = x @ x.T                     # rows/cols with NaN -> NaN
    sum_value = sum(grams)              # -> NaN
    cur_sim = (sum_value/(BS_*BS_)) > thr   # NaN > thr -> False

XPU triton (sparge_preprocess_triton.py:107) instead guards:
    x_norm = where(x_norm>0, x_norm, 1.0)   # padded row -> 0, finite
    -> grams finite -> sum finite -> cur_sim = REAL value (usually True)

This reproduces both in plain torch on CPU and shows the tail-block sim flag
differs: CUDA=False (forced dense), XPU=True (prunable).
"""
import torch


def cuda_block_sim(x_block: torch.Tensor, xmask: torch.Tensor, thr: float) -> bool:
    """Faithful port of CUDA triton sim for ONE block. x_block [BS, D], xmask [BS]."""
    BS = x_block.shape[0]
    valid = int(xmask.sum())
    x = x_block.clone()
    x[~xmask] = 0.0  # CUDA: tl.where(xmask, x, 0) on the smooth_k path
    xf = x.to(torch.float32)
    x_norm = torch.sqrt((xf * xf).sum(dim=1, keepdim=True))  # [BS,1], padded -> 0
    xn = (xf / x_norm)  # padded rows -> 0/0 = NaN  (NO guard, exactly like CUDA)
    grams = xn @ xn.t()
    sum_value = grams.sum()
    return bool((sum_value / (valid * valid)) > thr)


def xpu_block_sim(x_block: torch.Tensor, xmask: torch.Tensor, thr: float) -> bool:
    """Faithful port of XPU triton sim (guarded norm)."""
    BS = x_block.shape[0]
    valid = int(xmask.sum())
    x = x_block.clone()
    x[~xmask] = 0.0  # XPU: other=0.0 on load
    xf = x.to(torch.float32)
    x_norm = torch.sqrt((xf * xf).sum(dim=1, keepdim=True))
    x_norm = torch.where(x_norm > 0, x_norm, torch.ones_like(x_norm))  # GUARD
    xn = (xf / x_norm).to(torch.float16).to(torch.float32)
    grams = xn @ xn.t()
    sum_value = grams.sum()
    return bool((sum_value / (valid * valid)) > thr)


def main() -> None:
    torch.manual_seed(0)
    BS, D = 64, 128
    thr = -0.1  # simthreshd1 default

    print(f"{'case':<22} | {'valid':>5} | {'CUDA sim':>8} | {'XPU sim':>7} | differ?")
    for valid in (64, 56, 32, 8):  # 64=full, others=partial (last block)
        xmask = torch.zeros(BS, dtype=torch.bool)
        xmask[:valid] = True
        # subtract a mean to emulate smooth_k (km); content irrelevant to the NaN
        x = torch.randn(BS, D, dtype=torch.float16)
        km = x[:valid].mean(0, keepdim=True)
        xb = (x - km).to(torch.float16)
        c = cuda_block_sim(xb, xmask, thr)
        u = xpu_block_sim(xb, xmask, thr)
        tag = "FULL" if valid == BS else "PARTIAL(tail)"
        print(f"{tag:<22} | {valid:5d} | {str(c):>8} | {str(u):>7} | {'<<< DIFFER' if c != u else 'same'}")

    print()
    print("Interpretation:")
    print("  CUDA forces every PARTIAL (tail) block to sim=False -> dense fallback")
    print("  (final_map[~sim]=1 selects it for all queries).")
    print("  XPU's guard gives the tail block sim=True -> it is pruned by topk.")
    print("  => For S not divisible by 64/128 (e.g. Wan S=32760), the LAST block")
    print("     (= last frames) is routed densely by CUDA but sparsely by XPU.")


if __name__ == "__main__":
    main()

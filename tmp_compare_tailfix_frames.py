#!/usr/bin/env python
"""Per-frame comparison for the partial-tail-block fix E2E validation.

Compares two 81-frame Wan2.2 sparse outputs produced by the SAME prompt/seed/config,
differing only by the one-line partial-tail fix:

  BEFORE = Q-routing fix only, tail-UNfixed  (qrouting_fix.mp4)
  AFTER  = Q-routing + partial-tail fix       (tailfix.mp4)

Predicted signature of the fix (from the routing analysis):
  * The partial last K-tile, forced sim=False, is expanded across ALL q tiles
    -> every frame's queries now include the last K-column -> a GLOBAL (all-frames)
       delta, not just the tail.
  * The partial last Q-block (last ~56 query tokens = last frame's corner) forced
    dense -> the LAST frame additionally perturbed -> last frame shows the LARGEST
    delta.

So we expect: per-frame delta > 0 for essentially ALL frames (K-axis global effect),
monotone-ish rise toward the end, last frame the maximum (Q+K compound).

Optionally also compares each sparse output to a dense reference (full attention),
the quantity sparse attention approximates; the tail-fixed output should be no worse
(and at the tail, closer) to dense, because CUDA forces those blocks dense.
"""
import argparse
import sys

import numpy as np


def load_frames(path: str) -> np.ndarray:
    import imageio.v3 as iio

    frames = iio.imread(path, plugin="pyav")  # (T,H,W,C) uint8
    return np.asarray(frames).astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)


def per_frame_stats(a: np.ndarray, b: np.ndarray):
    t = min(a.shape[0], b.shape[0])
    a, b = a[:t], b[:t]
    out = []
    for i in range(t):
        d = a[i] - b[i]
        mse = float(np.mean(d ** 2))
        mae = float(np.mean(np.abs(d)))
        out.append((i, mse, mae, psnr(a[i], b[i])))
    return out


def summarize(label: str, stats):
    mses = np.array([s[1] for s in stats])
    print(f"\n=== {label} ===")
    print(f"  frames: {len(stats)}")
    print(f"  per-frame MSE: mean={mses.mean():.4f} min={mses.min():.4f} max={mses.max():.4f}")
    nonzero = int((mses > 1e-6).sum())
    print(f"  frames with delta>0: {nonzero}/{len(stats)}  "
          f"({'GLOBAL (all-frames)' if nonzero >= len(stats) - 2 else 'localized'})")
    # tail vs interior
    if len(stats) >= 10:
        tail = mses[-5:].mean()
        interior = mses[:-5].mean()
        print(f"  interior(first {len(stats)-5}) mean MSE={interior:.4f} | tail(last 5) mean MSE={tail:.4f} "
              f"| tail/interior ratio={tail/max(interior,1e-9):.2f}x")
    worst = max(stats, key=lambda s: s[1])
    print(f"  worst frame: idx={worst[0]} MSE={worst[1]:.4f} MAE={worst[2]:.4f} PSNR={worst[3]:.2f}dB")
    print(f"  last frame : idx={stats[-1][0]} MSE={stats[-1][1]:.4f} MAE={stats[-1][2]:.4f} PSNR={stats[-1][3]:.2f}dB")
    # compact per-frame MSE sparkline (every frame)
    line = " ".join(f"{s[1]:.2f}" for s in stats)
    print(f"  per-frame MSE: {line}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="tail-UNfixed sparse mp4 (qrouting_fix)")
    ap.add_argument("--after", required=True, help="tail-fixed sparse mp4 (tailfix)")
    ap.add_argument("--dense", default=None, help="optional dense full-attention mp4 (same prompt)")
    args = ap.parse_args()

    before = load_frames(args.before)
    after = load_frames(args.after)
    print(f"before {args.before}: shape={before.shape}")
    print(f"after  {args.after}: shape={after.shape}")
    if before.shape[0] != after.shape[0]:
        print(f"WARNING: frame count differs ({before.shape[0]} vs {after.shape[0]}); comparing overlap.")

    stats = per_frame_stats(before, after)
    summarize("BEFORE vs AFTER (what the tail fix changed)", stats)
    print("\nInterpretation guide:")
    print("  * delta>0 on ~ALL frames  => confirms the K-tile axis is a GLOBAL effect")
    print("    (every frame's queries re-routed over the forced-dense last K-column).")
    print("  * tail/interior ratio > 1 and last-frame = worst => confirms the Q-block")
    print("    axis compounds at the final frame (its own corner forced dense too).")

    if args.dense:
        dense = load_frames(args.dense)
        print(f"\ndense {args.dense}: shape={dense.shape}")
        if dense.shape[0] == before.shape[0]:
            sb = per_frame_stats(dense, before)
            sa = per_frame_stats(dense, after)
            mb = np.array([s[1] for s in sb])
            ma = np.array([s[1] for s in sa])
            print("\n=== distance to DENSE (sparse approximates dense; lower=better) ===")
            print(f"  BEFORE vs dense: mean MSE={mb.mean():.4f}  tail(last5)={mb[-5:].mean():.4f}")
            print(f"  AFTER  vs dense: mean MSE={ma.mean():.4f}  tail(last5)={ma[-5:].mean():.4f}")
            better_overall = ma.mean() < mb.mean()
            better_tail = ma[-5:].mean() < mb[-5:].mean()
            print(f"  AFTER closer to dense overall? {better_overall}  | at tail? {better_tail}")
            if not better_overall:
                print("  NOTE: 'closer to dense' is only a heuristic — CUDA-correct routing need")
                print("        not minimize dense-distance. The authoritative check is the routing")
                print("        match (already proven), not this E2E heuristic.")
        else:
            print(f"  dense frame count {dense.shape[0]} != {before.shape[0]} — likely a different "
                  f"prompt/config; skipping dense comparison.")


if __name__ == "__main__":
    sys.exit(main())

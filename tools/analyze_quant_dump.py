# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize quant_dump npz files into sensitivity rankings (CSV/markdown).

Usage:
  python tools/analyze_quant_dump.py --dump-dir <dir> [--out-dir <dir>]

Reads <dump-dir>/<tag>.npz + <tag>_summary.json for every tag present and
writes, per tag:
  linear_sensitivity.csv       per (expert, block, op): mean/min cos, mean/max rel_l2, mean max_abs
  attention_head_sensitivity.csv  per (expert, block, head): mean/min cos, mean rel_l2, mean q/k rms
  attention_stats.csv          per (expert, block, kind): mean q/k/out rms (stats-only kinds)
  pass_series.csv              per pass: mean linear rel_l2, mean attn rel_l2, mean attn cos
and a combined <tag>_report.md with top-N sensitive layers and heads.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np


def _load_runs(dump_dir: str) -> dict[str, dict]:
    runs = {}
    for name in sorted(os.listdir(dump_dir)):
        if name.endswith(".npz"):
            tag = name[: -len(".npz")]
            runs[tag] = np.load(os.path.join(dump_dir, name), allow_pickle=False)
    return runs


_EXPERT = r"(transformer(?:_2)?|)"


def _parse_lin_key(key: str) -> tuple[str, int, str]:
    # Expert is empty when the module prefix has no transformer root
    # (shared prefixes: "blocks.N.op").
    match = re.match(rf"^lin_{_EXPERT}_(\d+)_(.+)_diff$", key)
    if not match:
        return None
    return match.group(1), int(match.group(2)), match.group(3)


def _parse_attn_key(key: str) -> tuple[str, int, str]:
    match = re.match(rf"^attn_{_EXPERT}_(\d+)_self_diff$", key)
    if not match:
        return None
    return match.group(1), int(match.group(2)), "self"


def analyze(run: np.ndarray | dict, tag: str, out_dir: str) -> str:
    keys = list(run.keys())
    os.makedirs(out_dir, exist_ok=True)
    report_lines = [
        f"# Quant dump report: {tag}",
        "",
        "Pass map: pass 0 = engine warmup (short sequence); each transformer forward",
        "(denoising step x CFG branch: cond, uncond) is one pass. With cache-dit,",
        "cached steps compute only block 0, so non-block-0 rows exist only on",
        "computing steps. Both experts (high-noise / low-noise transformer) share",
        "the same block prefixes, and each denoising step runs exactly one expert,",
        "so rows are per-pass expert-clean; with boundary_ratio=0.875 and 40 steps,",
        "expert 1 covers steps 1..~35 and expert 2 steps ~36..40 (inferred from",
        "the pass index).",
        "",
    ]

    # ---- linear sensitivity -------------------------------------------------
    rows = []
    for key in keys:
        parsed = _parse_lin_key(key)
        if parsed is None:
            continue
        expert, block, op = parsed
        diff = run[f"lin_{expert}_{block}_{op}_diff"]  # [P, 4], NaN on skipped passes
        rows.append(
            {
                "expert": expert,
                "block": block,
                "op": op,
                "n_passes": int(np.count_nonzero(~np.isnan(diff[:, 0]))),
                "mean_cos": float(np.nanmean(diff[:, 0])),
                "min_cos": float(np.nanmin(diff[:, 0])),
                "mean_rel_l2": float(np.nanmean(diff[:, 1])),
                "max_rel_l2": float(np.nanmax(diff[:, 1])),
                "mean_max_abs": float(np.nanmean(diff[:, 2])),
            }
        )
    if rows:
        path = os.path.join(out_dir, f"{tag}_linear_sensitivity.csv")
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in sorted(rows, key=lambda r: (r["expert"], r["block"], r["op"])):
                writer.writerow(row)
        worst = sorted(rows, key=lambda r: r["min_cos"])[:20]
        report_lines += ["## Worst linear layers (by min cos over passes)", "",
                         "| expert | block | op | min_cos | mean_rel_l2 | max_rel_l2 |",
                         "|---|---|---|---|---|---|"]
        for row in worst:
            report_lines.append(
                f"| {row['expert']} | {row['block']} | {row['op']} | "
                f"{row['min_cos']:.6f} | {row['mean_rel_l2']:.6f} | {row['max_rel_l2']:.6f} |"
            )
        report_lines.append("")

    # ---- attention per-head sensitivity --------------------------------------
    rows = []
    for key in keys:
        parsed = _parse_attn_key(key)
        if parsed is None:
            continue
        expert, block, _kind = parsed
        diff = run[f"attn_{expert}_{block}_self_diff"]  # [P, H, 3], NaN on skipped
        q_rms = run[f"attn_{expert}_{block}_self_qk"]  # [P, H]
        k_rms = run[f"attn_{expert}_{block}_self_krms"]  # [P, H]
        for head in range(diff.shape[1]):
            rows.append(
                {
                    "expert": expert,
                    "block": block,
                    "head": head,
                    "n_passes": int(np.count_nonzero(~np.isnan(diff[:, head, 0]))),
                    "mean_cos": float(np.nanmean(diff[:, head, 0])),
                    "min_cos": float(np.nanmin(diff[:, head, 0])),
                    "mean_rel_l2": float(np.nanmean(diff[:, head, 1])),
                    "mean_q_rms": float(np.nanmean(q_rms[:, head])),
                    "mean_k_rms": float(np.nanmean(k_rms[:, head])),
                }
            )
    if rows:
        path = os.path.join(out_dir, f"{tag}_attention_head_sensitivity.csv")
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in sorted(rows, key=lambda r: (r["expert"], r["block"], r["head"])):
                writer.writerow(row)
        worst = sorted(rows, key=lambda r: r["min_cos"])[:20]
        report_lines += ["## Worst attention heads (by min cos over passes)", "",
                         "| expert | block | head | min_cos | mean_rel_l2 | mean_q_rms | mean_k_rms |",
                         "|---|---|---|---|---|---|---|"]
        for row in worst:
            report_lines.append(
                f"| {row['expert']} | {row['block']} | {row['head']} | "
                f"{row['min_cos']:.6f} | {row['mean_rel_l2']:.6f} | "
                f"{row['mean_q_rms']:.4f} | {row['mean_k_rms']:.4f} |"
            )
        report_lines.append("")

    # ---- stats-only attention kinds ------------------------------------------
    stats: dict[tuple, list] = defaultdict(list)
    for key in keys:
        match = re.match(rf"^attn_{_EXPERT}_(\d+)_(\w+)_qk$", key)
        if not match:
            continue
        expert, block, kind = match.groups()
        q_rms = run[key]  # [P, H]
        k_rms_key = f"attn_{expert}_{block}_{kind}_krms"
        k_rms = run[k_rms_key] if k_rms_key in run else None
        outrms_key = f"attn_{expert}_{block}_{kind}_outrms"
        outrms = run[outrms_key] if outrms_key in run else None
        stats[(expert, block, kind)].append((q_rms, k_rms, outrms))
    rows = []
    for (expert, block, kind), entries in sorted(stats.items()):
        q_rms = np.concatenate([e[0] for e in entries], axis=0)  # [n, H]
        k_rms = (
            np.concatenate([e[1] for e in entries], axis=0)
            if entries[0][1] is not None
            else None
        )
        outrms = (
            np.concatenate([e[2] for e in entries], axis=0)
            if entries[0][2] is not None
            else None
        )
        row = {
            "expert": expert,
            "block": block,
            "kind": kind,
            "mean_q_rms": float(np.nanmean(q_rms)),
        }
        if k_rms is not None:
            row["mean_k_rms"] = float(np.nanmean(k_rms))
        if outrms is not None:
            row["mean_out_rms"] = float(np.nanmean(outrms))
        rows.append(row)
    if rows:
        path = os.path.join(out_dir, f"{tag}_attention_stats.csv")
        with open(path, "w", newline="") as fh:
            fieldnames = list(dict.fromkeys(k for r in rows for k in r))
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # ---- per-pass series ------------------------------------------------------
    passes = None
    for key in keys:
        if key.endswith("_diff"):
            passes = run[key].shape[0]
            break
    if passes:
        lin_rell2 = np.array(
            [
                run[f"lin_{p[0]}_{p[1]}_{p[2]}_diff"][:, 1]
                for p in (_parse_lin_key(k) for k in keys)
                if p is not None
            ]
        )
        attn_rell2 = np.array(
            [
                run[f"attn_{p[0]}_{p[1]}_self_diff"][:, :, 1]
                for p in (_parse_attn_key(k) for k in keys)
                if p is not None
            ]
        )
        row = {"pass": np.arange(passes)}
        row["mean_lin_rel_l2"] = np.nanmean(lin_rell2, axis=0)
        row["mean_attn_rel_l2"] = np.nanmean(attn_rell2, axis=0)
        row["min_attn_cos"] = np.nanmin(
            np.array(
                [run[f"attn_{p[0]}_{p[1]}_self_diff"][:, :, 0].min(axis=1) for p in (_parse_attn_key(k) for k in keys) if p is not None]
            ),
            axis=0,
        )
        path = os.path.join(out_dir, f"{tag}_pass_series.csv")
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(list(row.keys()))
            for i in range(passes):
                writer.writerow([row[k][i] for k in row])

    report_path = os.path.join(out_dir, f"{tag}_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(report_lines) + "\n")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(args.dump_dir, "analysis")
    runs = _load_runs(args.dump_dir)
    if not runs:
        raise SystemExit(f"no npz dumps found in {args.dump_dir}")
    for tag, run in runs.items():
        print(analyze(run, tag, out_dir))


if __name__ == "__main__":
    main()

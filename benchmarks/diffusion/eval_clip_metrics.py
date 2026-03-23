# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Standalone CLIP evaluation script using torchmetrics.

Re-evaluates pre-generated text-to-image results using
``torchmetrics.multimodal.CLIPScore`` (standardised, well-tested) and
optionally ``CLIPImageQualityAssessment`` (no-reference perceptual quality).

**Zero dependency on vllm_omni** -- only needs ``torch``, ``torchmetrics``,
``PIL``, and the Python standard library.

Usage:
    # Basic evaluation (re-score with torchmetrics CLIPScore):
    python benchmarks/diffusion/eval_clip_metrics.py \\
        --results-json ./outputs/t2i_baseline/results.json

    # With IQA and custom CLIP model:
    python benchmarks/diffusion/eval_clip_metrics.py \\
        --results-json ./outputs/t2i_baseline/results.json \\
        --clip-model openai/clip-vit-large-patch14 \\
        --enable-iqa --iqa-prompts quality brightness sharpness \\
        --output-json ./outputs/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# A. Imports & dependency check
# ---------------------------------------------------------------------------

# These are TYPE_CHECKING-style placeholders. The actual imports happen in
# ``_check_dependencies()`` (called from ``main()``) so that ``--help``
# works even when the heavy dependencies are missing.

torch: Any = None  # noqa: F811
torchmetrics: Any = None  # noqa: F811
CLIPScore: Any = None  # noqa: F811
CLIPImageQualityAssessment: Any = None  # noqa: F811
Image: Any = None  # noqa: F811
np: Any = None  # noqa: F811

_HAS_IQA = False
_HAS_TQDM = False
tqdm: Any = None  # noqa: F811

# We need ``Any`` for the forward-declared module-level names.
from typing import Any  # noqa: E402


def _check_dependencies() -> None:
    """Import required packages, printing friendly errors on failure.

    Populates the module-level names (``torch``, ``torchmetrics``, etc.)
    so the rest of the module can use them normally.
    """
    global torch, torchmetrics, CLIPScore, Image, np  # noqa: PLW0603
    global CLIPImageQualityAssessment, _HAS_IQA  # noqa: PLW0603
    global tqdm, _HAS_TQDM  # noqa: PLW0603

    try:
        import torch as _torch

        torch = _torch
    except ImportError:
        print(
            "ERROR: PyTorch is required but not installed.\n  pip install torch",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import torchmetrics as _tm
        from torchmetrics.multimodal import CLIPScore as _CLIPScore

        torchmetrics = _tm
        CLIPScore = _CLIPScore
    except ImportError:
        print(
            "ERROR: torchmetrics with multimodal support is required.\n  pip install 'torchmetrics[multimodal]'",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from PIL import Image as _Image

        Image = _Image
    except ImportError:
        print(
            "ERROR: Pillow is required but not installed.\n  pip install Pillow",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import numpy as _np

        np = _np
    except ImportError:
        print(
            "ERROR: NumPy is required but not installed.\n  pip install numpy",
            file=sys.stderr,
        )
        sys.exit(1)

    # Optional: CLIPImageQualityAssessment (only needed with --enable-iqa)
    try:
        from torchmetrics.multimodal import CLIPImageQualityAssessment as _ClipIqa

        CLIPImageQualityAssessment = _ClipIqa
        _HAS_IQA = True
    except ImportError:
        _HAS_IQA = False

    # Optional: tqdm progress bar
    try:
        from tqdm import tqdm as _tqdm

        tqdm = _tqdm
        _HAS_TQDM = True
    except ImportError:
        _HAS_TQDM = False


# ---------------------------------------------------------------------------
# B. Input loading
# ---------------------------------------------------------------------------


def load_results_json(path: str) -> tuple[dict, list[dict]]:
    """Load the results JSON produced by ``benchmark_t2i_accuracy.py``.

    Returns
    -------
    config : dict
        The ``config`` section from the JSON (generation parameters, model
        name, etc.).
    per_prompt : list[dict]
        The ``per_prompt`` list.  Each entry has at least ``index``,
        ``prompt``, ``num_images``, and ``clip_score``.
    """
    json_path = Path(path)
    if not json_path.is_file():
        print(f"ERROR: Results JSON not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"ERROR: Could not parse JSON file {path}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    config = data.get("config", {})
    per_prompt = data.get("per_prompt", [])

    if not per_prompt:
        print(
            f"ERROR: No per_prompt entries found in {path}",
            file=sys.stderr,
        )
        sys.exit(1)

    return config, per_prompt


def discover_images(images_dir: Path, entry: dict) -> list[Path]:
    """Find image files on disk for a given per-prompt entry.

    Image naming convention from the benchmark:
    ``{index:04d}_{sample:02d}.png``
    """
    idx = entry["index"]
    found: list[Path] = []
    # Try sample indices 0..99 (more than enough)
    for s in range(100):
        candidate = images_dir / f"{idx:04d}_{s:02d}.png"
        if candidate.is_file():
            found.append(candidate)
        elif s > 0:
            # No more samples for this prompt
            break
    return found


# ---------------------------------------------------------------------------
# C. Image -> tensor conversion
# ---------------------------------------------------------------------------


def load_image_uint8(path: Path) -> torch.Tensor | None:
    """Load an image as a uint8 tensor ``(C, H, W)`` in ``[0, 255]``.

    This is the format expected by ``torchmetrics.multimodal.CLIPScore``.
    Returns ``None`` if the image cannot be opened.
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        warnings.warn(f"Could not open image {path}: {exc}", stacklevel=2)
        return None
    # (H, W, C) uint8 -> (C, H, W) uint8
    tensor = torch.from_numpy(np.array(img, dtype="uint8")).permute(2, 0, 1)
    return tensor


def load_image_float(path: Path) -> torch.Tensor | None:
    """Load an image as a float32 tensor ``(C, H, W)`` in ``[0, 1]``.

    This is the format expected by ``CLIPImageQualityAssessment``.
    Returns ``None`` if the image cannot be opened.
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        warnings.warn(f"Could not open image {path}: {exc}", stacklevel=2)
        return None
    tensor = torch.from_numpy(np.array(img, dtype="uint8")).permute(2, 0, 1).float() / 255.0
    return tensor


# ---------------------------------------------------------------------------
# D. CLIPScore evaluation
# ---------------------------------------------------------------------------


def evaluate_clip_scores(
    entries: list[dict],
    images_dir: Path,
    clip_model: str,
    device: torch.device,
    disable_tqdm: bool = False,
) -> list[dict]:
    """Compute per-prompt CLIPScore using torchmetrics.

    For each prompt, we update the metric with all images for that prompt,
    compute the score, and reset.  This gives us per-prompt granularity.

    Returns an enriched copy of each entry with ``torchmetrics_clip_score``,
    ``images_found``, and ``images_loaded`` fields added.
    """
    metric = CLIPScore(model_name_or_path=clip_model).to(device)

    results: list[dict] = []
    iterator = entries
    if _HAS_TQDM and not disable_tqdm:
        iterator = tqdm(entries, desc="CLIPScore eval")

    with torch.no_grad():
        for entry in iterator:
            result = dict(entry)  # shallow copy
            num_images_expected = entry.get("num_images", 0)

            # Skip entries with no images generated
            if num_images_expected == 0:
                result["torchmetrics_clip_score"] = None
                result["images_found"] = 0
                result["images_loaded"] = 0
                results.append(result)
                continue

            # Discover images on disk
            image_paths = discover_images(images_dir, entry)
            result["images_found"] = len(image_paths)

            if not image_paths:
                logger.warning(
                    "No images found on disk for prompt index %d (expected pattern: %04d_*.png in %s)",
                    entry["index"],
                    entry["index"],
                    images_dir,
                )
                result["torchmetrics_clip_score"] = None
                result["images_loaded"] = 0
                results.append(result)
                continue

            # Load images
            tensors: list[torch.Tensor] = []
            for p in image_paths:
                t = load_image_uint8(p)
                if t is not None:
                    tensors.append(t)
            result["images_loaded"] = len(tensors)

            if not tensors:
                logger.warning(
                    "All images failed to load for prompt index %d",
                    entry["index"],
                )
                result["torchmetrics_clip_score"] = None
                results.append(result)
                continue

            # Stack images and score
            images_tensor = torch.stack(tensors).to(device)
            prompt = entry["prompt"]
            prompts_list = [prompt] * len(tensors)

            metric.reset()
            metric.update(images_tensor, prompts_list)
            score = metric.compute().item()
            result["torchmetrics_clip_score"] = round(score, 4)

            results.append(result)

    return results


# ---------------------------------------------------------------------------
# E. CLIPImageQualityAssessment evaluation (optional)
# ---------------------------------------------------------------------------


def evaluate_iqa(
    entries: list[dict],
    images_dir: Path,
    iqa_prompts: tuple[str, ...],
    device: torch.device,
    batch_size: int = 32,
    disable_tqdm: bool = False,
) -> list[dict]:
    """Compute per-image IQA scores and aggregate per-prompt.

    Each prompt's images are batched, run through
    ``CLIPImageQualityAssessment``, and the per-prompt score is the mean
    across its images for each IQA aspect.

    Mutates entries in-place (adds ``iqa`` dict to each entry).
    """
    if not _HAS_IQA:
        print(
            "ERROR: CLIPImageQualityAssessment not available. "
            "Upgrade torchmetrics:\n"
            "  pip install 'torchmetrics[multimodal]' --upgrade",
            file=sys.stderr,
        )
        return entries

    iqa_metric = CLIPImageQualityAssessment(prompts=iqa_prompts).to(device)

    iterator = entries
    if _HAS_TQDM and not disable_tqdm:
        iterator = tqdm(entries, desc="IQA eval")

    with torch.no_grad():
        for entry in iterator:
            images_loaded = entry.get("images_loaded", 0)
            if images_loaded == 0:
                entry["iqa"] = {p: None for p in iqa_prompts}
                continue

            image_paths = discover_images(images_dir, entry)
            tensors: list[torch.Tensor] = []
            for p in image_paths:
                t = load_image_float(p)
                if t is not None:
                    tensors.append(t)

            if not tensors:
                entry["iqa"] = {p: None for p in iqa_prompts}
                continue

            # Process in batches
            all_scores: dict[str, list[float]] = {p: [] for p in iqa_prompts}
            for batch_start in range(0, len(tensors), batch_size):
                batch = tensors[batch_start : batch_start + batch_size]
                batch_tensor = torch.stack(batch).to(device)

                iqa_metric.reset()
                iqa_metric.update(batch_tensor)
                scores = iqa_metric.compute()

                # scores is a dict[str, Tensor] when multiple prompts
                if isinstance(scores, dict):
                    for prompt_name, score_tensor in scores.items():
                        vals = score_tensor.tolist()
                        if isinstance(vals, (int, float)):
                            vals = [vals]
                        all_scores[prompt_name].extend(vals)
                else:
                    # Single prompt -- scores is a Tensor
                    vals = scores.tolist()
                    if isinstance(vals, (int, float)):
                        vals = [vals]
                    # Only one prompt name
                    prompt_name = iqa_prompts[0]
                    all_scores[prompt_name].extend(vals)

            entry["iqa"] = {p: round(statistics.mean(v), 4) if v else None for p, v in all_scores.items()}

    return entries


# ---------------------------------------------------------------------------
# F. Results assembly & summary statistics
# ---------------------------------------------------------------------------


def compute_percentile(sorted_values: list[float], p: float) -> float:
    """Simple nearest-rank percentile on a sorted list."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    idx = max(0, int(n * p) - 1)
    return sorted_values[idx]


def compute_summary_stats(values: list[float]) -> dict:
    """Compute mean/median/std/min/max/p10/p90 for a list of floats."""
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "mean": round(statistics.mean(sorted_vals), 4),
        "median": round(statistics.median(sorted_vals), 4),
        "std": round(statistics.stdev(sorted_vals), 4) if n > 1 else 0.0,
        "min": round(sorted_vals[0], 4),
        "max": round(sorted_vals[-1], 4),
        "p10": round(compute_percentile(sorted_vals, 0.10), 4),
        "p90": round(compute_percentile(sorted_vals, 0.90), 4),
    }


def assemble_results(
    entries: list[dict],
    source_config: dict,
    eval_config: dict,
    enable_iqa: bool,
    iqa_prompts: tuple[str, ...],
) -> dict:
    """Build the final results dictionary with summary and per-prompt data."""
    # Collect valid scores
    tm_scores = [e["torchmetrics_clip_score"] for e in entries if e.get("torchmetrics_clip_score") is not None]
    orig_scores = [e["clip_score"] for e in entries if e.get("clip_score") is not None]

    # Count categories
    num_entries = len(entries)
    num_scored = len(tm_scores)
    num_missing = sum(1 for e in entries if e.get("images_found", 0) == 0 and e.get("num_images", 0) > 0)
    num_skipped = sum(1 for e in entries if e.get("num_images", 0) == 0)

    summary: dict = {
        "num_entries": num_entries,
        "num_scored": num_scored,
        "num_missing": num_missing,
        "num_skipped": num_skipped,
        "torchmetrics_clip_score": compute_summary_stats(tm_scores),
        "original_clip_score": compute_summary_stats(orig_scores),
    }

    # IQA summary
    if enable_iqa:
        iqa_summary: dict[str, dict] = {}
        for prompt_name in iqa_prompts:
            vals = [e["iqa"][prompt_name] for e in entries if e.get("iqa") and e["iqa"].get(prompt_name) is not None]
            iqa_summary[prompt_name] = compute_summary_stats(vals)
        summary["iqa"] = iqa_summary

    # Per-prompt output
    per_prompt_out: list[dict] = []
    for entry in entries:
        out: dict = {
            "index": entry["index"],
            "prompt": entry["prompt"],
            "original_clip_score": entry.get("clip_score"),
            "torchmetrics_clip_score": entry.get("torchmetrics_clip_score"),
            "images_found": entry.get("images_found", 0),
            "images_loaded": entry.get("images_loaded", 0),
        }
        if enable_iqa and "iqa" in entry:
            out["iqa"] = entry["iqa"]
        per_prompt_out.append(out)

    return {
        "eval_config": eval_config,
        "source_config": source_config,
        "summary": summary,
        "per_prompt": per_prompt_out,
    }


# ---------------------------------------------------------------------------
# G. Terminal printing
# ---------------------------------------------------------------------------

_PROMPT_DISPLAY_WIDTH = 50


def print_results(results: dict) -> None:
    """Pretty-print per-prompt table and summary statistics."""
    per_prompt = results["per_prompt"]
    summary = results["summary"]
    enable_iqa = "iqa" in summary

    # Determine IQA column names
    iqa_cols: list[str] = []
    if enable_iqa:
        iqa_cols = list(summary["iqa"].keys())

    # --- Per-prompt table ---
    iqa_header = "".join(f"  {col:>10}" for col in iqa_cols)
    header = (
        f"{'Index':>5}  {'Prompt':<{_PROMPT_DISPLAY_WIDTH}}"
        f"  {'Orig CLIP':>10}  {'TM CLIP':>10}  {'Delta':>7}"
        f"{iqa_header}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for entry in per_prompt:
        prompt_display = entry["prompt"]
        if len(prompt_display) > _PROMPT_DISPLAY_WIDTH:
            prompt_display = prompt_display[: _PROMPT_DISPLAY_WIDTH - 3] + "..."

        orig = entry.get("original_clip_score")
        tm = entry.get("torchmetrics_clip_score")

        orig_str = f"{orig:10.2f}" if orig is not None else "       N/A"
        tm_str = f"{tm:10.2f}" if tm is not None else "       N/A"

        if orig is not None and tm is not None:
            delta = tm - orig
            delta_str = f"{delta:+7.2f}"
        else:
            delta_str = "    N/A"

        iqa_str = ""
        if enable_iqa:
            iqa_data = entry.get("iqa", {})
            for col in iqa_cols:
                val = iqa_data.get(col) if iqa_data else None
                if val is not None:
                    iqa_str += f"  {val:10.4f}"
                else:
                    iqa_str += "       N/A  "

        print(
            f"{entry['index']:5d}"
            f"  {prompt_display:<{_PROMPT_DISPLAY_WIDTH}}"
            f"  {orig_str}  {tm_str}  {delta_str}{iqa_str}"
        )

    print(sep)

    # --- Summary block ---
    print("\n=== Evaluation Summary ===")
    print(f"  Entries          : {summary['num_entries']}")
    print(f"  Scored           : {summary['num_scored']}")
    print(f"  Missing images   : {summary['num_missing']}")
    print(f"  Skipped (no gen) : {summary['num_skipped']}")

    # torchmetrics CLIP scores
    tm_stats = summary.get("torchmetrics_clip_score", {})
    if tm_stats:
        print("\n  --- torchmetrics CLIPScore ---")
        print(f"  Mean             : {tm_stats['mean']:.4f}")
        print(f"  Median           : {tm_stats['median']:.4f}")
        print(f"  Std              : {tm_stats['std']:.4f}")
        print(f"  Min / Max        : {tm_stats['min']:.4f} / {tm_stats['max']:.4f}")
        print(f"  P10 / P90        : {tm_stats['p10']:.4f} / {tm_stats['p90']:.4f}")

    # Original CLIP scores
    orig_stats = summary.get("original_clip_score", {})
    if orig_stats:
        print("\n  --- Original CLIPScore (from benchmark) ---")
        print(f"  Mean             : {orig_stats['mean']:.4f}")
        print(f"  Median           : {orig_stats['median']:.4f}")
        print(f"  Std              : {orig_stats['std']:.4f}")
        print(f"  Min / Max        : {orig_stats['min']:.4f} / {orig_stats['max']:.4f}")
        print(f"  P10 / P90        : {orig_stats['p10']:.4f} / {orig_stats['p90']:.4f}")

    # Delta
    if tm_stats and orig_stats:
        delta_mean = tm_stats["mean"] - orig_stats["mean"]
        print(f"\n  Delta (TM - Orig): {delta_mean:+.4f}")

    # IQA summary
    if enable_iqa:
        iqa_summary = summary.get("iqa", {})
        if iqa_summary:
            print("\n  --- IQA (CLIPImageQualityAssessment) ---")
            for prompt_name, stats in iqa_summary.items():
                if stats:
                    print(
                        f"  {prompt_name:<16} : "
                        f"mean={stats['mean']:.4f}  "
                        f"median={stats['median']:.4f}  "
                        f"std={stats['std']:.4f}  "
                        f"[{stats['min']:.4f}, {stats['max']:.4f}]"
                    )
                else:
                    print(f"  {prompt_name:<16} : no data")

    print()


# ---------------------------------------------------------------------------
# H. JSON output
# ---------------------------------------------------------------------------


def save_results(results: dict, path: str) -> None:
    """Write evaluation results to a JSON file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Evaluation results written to %s", path)


# ---------------------------------------------------------------------------
# I. Argument parsing & main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone CLIP evaluation of pre-generated T2I results using torchmetrics. No vllm_omni dependency."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--results-json",
        type=str,
        required=True,
        help=("Path to the JSON output from benchmark_t2i_accuracy.py (must contain 'config' and 'per_prompt' keys)."),
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help=(
            "Directory containing generated images "
            "({idx:04d}_{sample:02d}.png). "
            "Defaults to the directory containing --results-json."
        ),
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="HuggingFace CLIP model for torchmetrics CLIPScore.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for evaluation: auto / cuda / cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for IQA evaluation.",
    )
    parser.add_argument(
        "--enable-iqa",
        action="store_true",
        help="Also run CLIPImageQualityAssessment.",
    )
    parser.add_argument(
        "--iqa-prompts",
        type=str,
        nargs="+",
        default=["quality", "brightness", "sharpness"],
        help="IQA aspects to assess (only used with --enable-iqa).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write evaluation results JSON.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Suppress tqdm progress bars.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    # Import heavy dependencies now (after argparse, so --help works)
    _check_dependencies()

    t_start = time.perf_counter()

    # --- Resolve device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # --- Resolve images directory ---
    if args.images_dir is not None:
        images_dir = Path(args.images_dir)
    else:
        images_dir = Path(args.results_json).parent

    if not images_dir.is_dir():
        print(
            f"ERROR: Images directory does not exist: {images_dir}\n"
            f"  Hint: use --images-dir to specify the correct path.",
            file=sys.stderr,
        )
        sys.exit(1)

    logger.info("Images directory: %s", images_dir)

    # --- Load results JSON ---
    logger.info("Loading results from %s", args.results_json)
    source_config, per_prompt = load_results_json(args.results_json)
    logger.info("Loaded %d per-prompt entries", len(per_prompt))

    # --- IQA check ---
    iqa_prompts = tuple(args.iqa_prompts)
    if args.enable_iqa and not _HAS_IQA:
        print(
            "ERROR: --enable-iqa requested but "
            "CLIPImageQualityAssessment is not available.\n"
            "  pip install 'torchmetrics[multimodal]' --upgrade",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Evaluate CLIPScore ---
    logger.info(
        "Evaluating CLIPScore with model: %s on %s",
        args.clip_model,
        device,
    )
    entries = evaluate_clip_scores(
        entries=per_prompt,
        images_dir=images_dir,
        clip_model=args.clip_model,
        device=device,
        disable_tqdm=args.disable_tqdm,
    )

    # --- Evaluate IQA (optional) ---
    if args.enable_iqa:
        logger.info("Evaluating IQA with prompts: %s", iqa_prompts)
        entries = evaluate_iqa(
            entries=entries,
            images_dir=images_dir,
            iqa_prompts=iqa_prompts,
            device=device,
            batch_size=args.batch_size,
            disable_tqdm=args.disable_tqdm,
        )

    # --- Assemble results ---
    eval_config = {
        "results_json": str(Path(args.results_json).resolve()),
        "images_dir": str(images_dir.resolve()),
        "clip_model": args.clip_model,
        "device": str(device),
        "enable_iqa": args.enable_iqa,
        "iqa_prompts": list(iqa_prompts) if args.enable_iqa else [],
        "torchmetrics_version": torchmetrics.__version__,
    }

    results = assemble_results(
        entries=entries,
        source_config=source_config,
        eval_config=eval_config,
        enable_iqa=args.enable_iqa,
        iqa_prompts=iqa_prompts,
    )

    t_total = time.perf_counter() - t_start
    results["summary"]["eval_time_s"] = round(t_total, 2)

    # --- Print & save ---
    print_results(results)

    if args.output_json:
        save_results(results, args.output_json)

    logger.info("Evaluation completed in %.1fs", t_total)


if __name__ == "__main__":
    main()

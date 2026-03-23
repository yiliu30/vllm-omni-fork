# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark text-to-image accuracy using CLIP score.

Generates images from a prompt file using OmniDiffusion (offline inference)
and measures text-image alignment using CLIP score (Hessel et al. 2021).

By default, prompts are loaded from the MLPerf SD Inference captions_5k.tsv
dataset (COCO-derived, ~5000 captions). Override with ``--prompts-file``.

Usage:
    # Default prompts (auto-downloaded MLPerf captions_5k.tsv):
    python benchmarks/diffusion/benchmark_t2i_accuracy.py \
        --model black-forest-labs/FLUX.1-dev

    # Limit to first 100 prompts:
    python benchmarks/diffusion/benchmark_t2i_accuracy.py \
        --model black-forest-labs/FLUX.1-dev --num-prompts 100

    # Custom prompts file (plain text, one per line):
    python benchmarks/diffusion/benchmark_t2i_accuracy.py \
        --model black-forest-labs/FLUX.1-dev \
        --prompts-file prompts.txt

    # Full options:
    python benchmarks/diffusion/benchmark_t2i_accuracy.py \
        --model black-forest-labs/FLUX.1-dev \
        --prompts-file prompts.txt \
        --height 1024 --width 1024 \
        --num-inference-steps 50 \
        --guidance-scale 3.5 \
        --seed 42 \
        --output-dir ./outputs \
        --output-json results.json
"""

import argparse
import csv
import json
import logging
import os
import statistics
import tempfile
import time
import urllib.request
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# MLPerf SD Inference captions dataset (COCO-derived, ~5000 captions).
# TSV format: image_id \t id \t caption
_DEFAULT_PROMPTS_URL = "https://raw.githubusercontent.com/ahmadki/mlperf_sd_inference/refs/heads/master/captions_5k.tsv"


# ---------------------------------------------------------------------------
# A. Prompt loading
# ---------------------------------------------------------------------------


def load_prompts(path: str | None, num_prompts: int | None = None) -> list[str]:
    """Load prompts from a local file or download the default MLPerf dataset.

    Supports two file formats:
    - **Plain text**: one prompt per line (``#`` comments and blank lines
      are skipped).
    - **TSV** (tab-separated): expects a header row with a ``caption``
      column; the captions are extracted automatically.

    If *path* is ``None``, the MLPerf captions_5k.tsv is downloaded to a
    temporary file and parsed.  *num_prompts* limits how many prompts are
    returned (first *N* after filtering).
    """
    if path is None:
        path = _download_default_prompts()

    prompts = _parse_prompts_file(path)

    if not prompts:
        raise ValueError(f"No prompts found in {path}")

    if num_prompts is not None and num_prompts > 0:
        prompts = prompts[:num_prompts]
    return prompts


def _download_default_prompts() -> str:
    """Download the MLPerf captions_5k.tsv to a cache directory and return
    its local path.  Re-uses a previously downloaded copy if present."""
    cache_dir = os.path.join(tempfile.gettempdir(), "vllm_omni_bench")
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, "captions_5k.tsv")

    if os.path.isfile(local_path):
        logger.info("Using cached default prompts: %s", local_path)
        return local_path

    logger.info("Downloading default prompts from %s …", _DEFAULT_PROMPTS_URL)
    urllib.request.urlretrieve(_DEFAULT_PROMPTS_URL, local_path)
    logger.info("Saved default prompts to %s", local_path)
    return local_path


def _parse_prompts_file(path: str) -> list[str]:
    """Auto-detect plain-text vs. TSV and return a list of prompt strings."""
    with open(path, newline="") as f:
        first_line = f.readline()

    # Heuristic: if the first line looks like a TSV header containing
    # "caption", parse as TSV; otherwise treat as plain text.
    if "\t" in first_line and "caption" in first_line.lower():
        return _parse_tsv_prompts(path)
    return _parse_text_prompts(path)


def _parse_tsv_prompts(path: str) -> list[str]:
    """Extract the ``caption`` column from a TSV file with a header row."""
    prompts: list[str] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "caption" not in reader.fieldnames:
            raise ValueError(f"TSV file {path} does not have a 'caption' column. Found columns: {reader.fieldnames}")
        for row in reader:
            caption = row["caption"].strip()
            if caption:
                prompts.append(caption)
    return prompts


def _parse_text_prompts(path: str) -> list[str]:
    """Read plain-text prompts, one per line, skipping comments/blanks."""
    prompts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


# ---------------------------------------------------------------------------
# B. CLIP scorer
# ---------------------------------------------------------------------------


class CLIPScorer:
    """Compute CLIP score (text-image cosine similarity × 100, clamped ≥ 0).

    Uses ``transformers.CLIPModel`` and ``CLIPProcessor`` — no extra
    dependencies beyond what diffusers already pulls in.
    """

    _IMAGE_MICRO_BATCH = 64  # avoid GPU OOM when scoring many images

    def __init__(self, model_name: str, device: str):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        logger.info("Loading CLIP model %s on %s …", model_name, device)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_batch(
        self,
        prompts: list[str],
        images_per_prompt: list[list[Image.Image]],
    ) -> list[float]:
        """Return one CLIP-S value per prompt (averaged over its images).

        Parameters
        ----------
        prompts:
            ``N`` text prompts.
        images_per_prompt:
            ``N`` lists, each containing one or more PIL images.

        Returns
        -------
        list[float]
            ``N`` CLIP scores.
        """
        scores: list[float] = []
        for prompt, images in zip(prompts, images_per_prompt):
            if not images:
                scores.append(0.0)
                continue
            img_scores = self._score_images(prompt, images)
            scores.append(sum(img_scores) / len(img_scores))
        return scores

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score_images(
        self,
        prompt: str,
        images: list[Image.Image],
    ) -> list[float]:
        """Score each image against *prompt*, micro-batching to limit memory."""
        # Encode text once
        text_inputs = self.processor(
            text=[prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        text_embeds = self.model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        all_scores: list[float] = []
        for start in range(0, len(images), self._IMAGE_MICRO_BATCH):
            batch = images[start : start + self._IMAGE_MICRO_BATCH]
            img_inputs = self.processor(
                images=batch,
                return_tensors="pt",
            ).to(self.device)
            img_embeds = self.model.get_image_features(**img_inputs)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

            # cos_sim: (num_images, 1) → squeeze → per-image float
            cos_sim = (img_embeds @ text_embeds.T).squeeze(-1)
            for val in cos_sim.tolist():
                score = float(val) if isinstance(val, (int, float)) else float(val)
                all_scores.append(max(100.0 * score, 0.0))
        return all_scores


# ---------------------------------------------------------------------------
# C. Image extraction helpers
# ---------------------------------------------------------------------------


def extract_images_from_output(output) -> list[Image.Image]:
    """Extract PIL images from an OmniRequestOutput or similar object.

    Handles both direct diffusion output (``output.images``) and the
    multi-stage pipeline wrapper (``output.request_output[0].images``).
    """
    # Direct diffusion mode — most common
    if hasattr(output, "images") and output.images:
        return list(output.images)

    # Pipeline wrapper mode
    if hasattr(output, "request_output") and output.request_output is not None:
        ro = output.request_output
        if isinstance(ro, (list, tuple)) and len(ro) > 0:
            inner = ro[0]
            if hasattr(inner, "images") and inner.images:
                return list(inner.images)

    return []


# ---------------------------------------------------------------------------
# D. Main benchmark loop
# ---------------------------------------------------------------------------


def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the full benchmark: load → generate → score → summarise."""
    from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    # --- Load prompts -------------------------------------------------------
    prompts = load_prompts(args.prompts_file, num_prompts=args.num_prompts)
    logger.info("Loaded %d prompts", len(prompts))

    # --- Resolve generator device -------------------------------------------
    try:
        from vllm_omni.platforms import current_omni_platform

        gen_device = current_omni_platform.device_type
    except Exception:
        gen_device = "cpu"

    # --- Sampling params ----------------------------------------------------
    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        num_outputs_per_prompt=args.num_outputs_per_prompt,
        generator_device=gen_device,
    )

    # --- Init engine --------------------------------------------------------
    logger.info("Initialising OmniDiffusion with model %s …", args.model)
    engine = OmniDiffusion(model=args.model)

    # --- Init CLIP scorer ---------------------------------------------------
    clip_device = args.clip_device
    if clip_device == "auto":
        clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = CLIPScorer(args.clip_model, device=clip_device)

    # --- Prepare output dir -------------------------------------------------
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Progress bar -------------------------------------------------------
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    # --- Generate & score ---------------------------------------------------
    all_images_per_prompt: list[list[Image.Image]] = []
    prompt_indices: list[int] = []
    gen_times: list[float] = []

    total_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    iterator = range(0, len(prompts), args.batch_size)
    if tqdm is not None and not args.disable_tqdm:
        iterator = tqdm(iterator, total=total_batches, desc="Generating")

    for batch_start in iterator:
        batch_end = min(batch_start + args.batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        t0 = time.perf_counter()
        try:
            outputs = engine.generate(batch_prompts, sampling_params)
        except Exception:
            logger.exception(
                "Generation failed for prompts %d–%d; skipping.",
                batch_start,
                batch_end - 1,
            )
            for idx in range(batch_start, batch_end):
                all_images_per_prompt.append([])
                prompt_indices.append(idx)
                gen_times.append(0.0)
            continue
        elapsed = time.perf_counter() - t0

        for i, output in enumerate(outputs):
            idx = batch_start + i
            images = extract_images_from_output(output)
            all_images_per_prompt.append(images)
            prompt_indices.append(idx)
            gen_times.append(elapsed / len(outputs))

            # Save images
            if args.output_dir and images:
                for s, img in enumerate(images):
                    fname = f"{idx:04d}_{s:02d}.png"
                    img.save(os.path.join(args.output_dir, fname))

    # --- CLIP scoring -------------------------------------------------------
    scored_prompts = [prompts[i] for i in prompt_indices]
    clip_scores = scorer.score_batch(scored_prompts, all_images_per_prompt)

    # --- Assemble per-prompt results ----------------------------------------
    per_prompt: list[dict] = []
    for i, idx in enumerate(prompt_indices):
        entry: dict = {
            "index": idx,
            "prompt": prompts[idx],
            "num_images": len(all_images_per_prompt[i]),
            "generation_time_s": round(gen_times[i], 3),
        }
        if all_images_per_prompt[i]:
            entry["clip_score"] = round(clip_scores[i], 4)
        else:
            entry["clip_score"] = None
        per_prompt.append(entry)

    # --- Summary statistics -------------------------------------------------
    valid_scores = [p["clip_score"] for p in per_prompt if p["clip_score"] is not None]
    summary: dict = {}
    if valid_scores:
        sorted_scores = sorted(valid_scores)
        n = len(sorted_scores)
        summary = {
            "num_prompts": len(prompts),
            "num_scored": n,
            "num_failed": len(prompts) - n,
            "mean": round(statistics.mean(sorted_scores), 4),
            "median": round(statistics.median(sorted_scores), 4),
            "std": round(statistics.stdev(sorted_scores), 4) if n > 1 else 0.0,
            "min": round(sorted_scores[0], 4),
            "max": round(sorted_scores[-1], 4),
            "p10": round(sorted_scores[max(0, int(n * 0.10) - 1)], 4),
            "p25": round(sorted_scores[max(0, int(n * 0.25) - 1)], 4),
            "p75": round(sorted_scores[max(0, int(n * 0.75) - 1)], 4),
            "p90": round(sorted_scores[max(0, int(n * 0.90) - 1)], 4),
        }
    else:
        summary = {
            "num_prompts": len(prompts),
            "num_scored": 0,
            "num_failed": len(prompts),
        }

    # --- Config snapshot for reproducibility --------------------------------
    config = {
        "model": args.model,
        "prompts_file": args.prompts_file or _DEFAULT_PROMPTS_URL,
        "num_prompts": len(prompts),
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "num_outputs_per_prompt": args.num_outputs_per_prompt,
        "batch_size": args.batch_size,
        "clip_model": args.clip_model,
    }

    results = {
        "config": config,
        "summary": summary,
        "per_prompt": per_prompt,
    }

    # --- Cleanup engine -----------------------------------------------------
    engine.close()

    return results


# ---------------------------------------------------------------------------
# E. Printing & saving
# ---------------------------------------------------------------------------

_PROMPT_DISPLAY_WIDTH = 50


def print_results(results: dict) -> None:
    """Pretty-print per-prompt table and summary statistics."""
    per_prompt = results["per_prompt"]
    summary = results["summary"]

    # --- Per-prompt table ---------------------------------------------------
    header = f"{'Index':>5}  {'Prompt':<{_PROMPT_DISPLAY_WIDTH}}  {'CLIP Score':>10}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for entry in per_prompt:
        prompt_display = entry["prompt"]
        if len(prompt_display) > _PROMPT_DISPLAY_WIDTH:
            prompt_display = prompt_display[: _PROMPT_DISPLAY_WIDTH - 3] + "..."
        score_str = f"{entry['clip_score']:10.2f}" if entry["clip_score"] is not None else "    FAILED"
        print(f"{entry['index']:5d}  {prompt_display:<{_PROMPT_DISPLAY_WIDTH}}  {score_str}")
    print(sep)

    # --- Summary block ------------------------------------------------------
    print("\n=== Summary ===")
    print(f"  Prompts scored : {summary.get('num_scored', 0)} / {summary.get('num_prompts', 0)}")
    if summary.get("num_scored", 0) > 0:
        print(f"  Mean           : {summary['mean']:.4f}")
        print(f"  Median         : {summary['median']:.4f}")
        print(f"  Std            : {summary['std']:.4f}")
        print(f"  Min / Max      : {summary['min']:.4f} / {summary['max']:.4f}")
        print(f"  P10 / P25      : {summary['p10']:.4f} / {summary['p25']:.4f}")
        print(f"  P75 / P90      : {summary['p75']:.4f} / {summary['p90']:.4f}")
    if summary.get("num_failed", 0) > 0:
        print(f"  Failed         : {summary['num_failed']}")
    print()


def save_results(results: dict, path: str) -> None:
    """Write results dictionary to a JSON file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results written to %s", path)


# ---------------------------------------------------------------------------
# F. Argument parsing & entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark text-to-image accuracy via CLIP score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model repo ID or local path.",
    )

    # Prompts
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help=(
            "Path to a prompts file (plain text or TSV with a 'caption' "
            "column).  If omitted, the MLPerf captions_5k.tsv dataset is "
            "downloaded automatically."
        ),
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Limit to the first N prompts (default: use all).",
    )

    # Generation
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-outputs-per-prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt (scores averaged).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts per generate() call.",
    )

    # CLIP scoring
    parser.add_argument(
        "--clip-model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="HuggingFace CLIP model for scoring.",
    )
    parser.add_argument(
        "--clip-device",
        type=str,
        default="auto",
        help="Device for CLIP inference (auto / cuda / cpu).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated images as {idx:04d}_{sample:02d}.png.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write full results JSON.",
    )

    # Misc
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Suppress the tqdm progress bar.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    logger.info("Starting T2I accuracy benchmark")
    t_start = time.perf_counter()

    results = run_benchmark(args)

    t_total = time.perf_counter() - t_start
    results["summary"]["total_time_s"] = round(t_total, 2)

    print_results(results)

    if args.output_json:
        save_results(results, args.output_json)

    logger.info("Benchmark completed in %.1fs", t_total)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Standalone evaluation script for offline-quantized diffusion models.

Generates images from both a BF16 baseline and an offline-quantized model,
computes per-prompt LPIPS perceptual distance, and reports time/memory
statistics in a Markdown table.

The two models are loaded **sequentially** (baseline first, then quantized)
to avoid OOM on single-GPU setups.

Requirements:
    pip install lpips Pillow numpy torch torchvision

Usage:

    # Create a prompts file (one prompt per line, # lines are comments)
    cat > prompts.txt <<'EOF'
    a cup of coffee on the table
    a cat sitting on a windowsill
    a sunset over the ocean
    EOF

    # Run evaluation
    python benchmarks/diffusion/benchmark_offline_quant_accuracy.py \\
        --baseline-model black-forest-labs/FLUX.1-dev \\
        --quantized-model Yi30/FLUX.1-dev-AutoRound-w4a16 \\
        --prompts-file prompts.txt \\
        --height 1024 --width 1024 \\
        --num-inference-steps 50 \\
        --seed 42 \\
        --output-dir ./quant_eval_output

    # Re-run quantized only (reuse saved baseline images)
    python benchmarks/diffusion/benchmark_offline_quant_accuracy.py \\
        --baseline-model black-forest-labs/FLUX.1-dev \\
        --quantized-model another-quant-model \\
        --prompts-file prompts.txt \\
        --skip-baseline \\
        --output-dir ./quant_eval_output
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

# ---------------------------------------------------------------------------
# LPIPS metric (inlined from eval_quantization_quality.py to stay standalone)
# ---------------------------------------------------------------------------


def compute_lpips_images(
    baseline_images: list,
    quantized_images: list,
    net: str = "alex",
) -> list[float]:
    """Compute LPIPS between paired lists of PIL images."""
    import lpips
    from torchvision import transforms

    loss_fn = lpips.LPIPS(net=net)
    loss_fn = loss_fn.eval()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    scores: list[float] = []
    for img_bl, img_qt in zip(baseline_images, quantized_images):
        t_bl = transform(img_bl.convert("RGB")).unsqueeze(0)
        t_qt = transform(img_qt.convert("RGB")).unsqueeze(0)
        if torch.cuda.is_available():
            t_bl, t_qt = t_bl.cuda(), t_qt.cuda()
        with torch.no_grad():
            score = loss_fn(t_bl, t_qt).item()
        scores.append(score)
    return scores


# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------


def _load_prompts(path: str) -> list[str]:
    """Read prompts from a text file (one per line, # = comment)."""
    prompts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _get_device_used_memory_gib() -> float:
    """Return device-level GPU memory usage in GiB (visible across processes).

    Uses ``torch.cuda.mem_get_info`` which reports physical device memory,
    unlike ``max_memory_allocated`` which only tracks the calling process.
    The diffusion model lives in a spawned worker process, so per-process
    stats from the main process are meaningless.
    """
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024**3)


def _generate_images(
    model_name: str,
    prompts: list[str],
    *,
    height: int,
    width: int,
    num_inference_steps: int,
    seed: int,
    guidance_scale: float,
    cfg_scale: float,
    enforce_eager: bool,
    parallel_config: DiffusionParallelConfig,
) -> tuple[list, list[float], list[float], float]:
    """Load *model_name*, generate one image per prompt.

    Returns
    -------
    images    : list[PIL.Image]
    times     : list[float]  -- per-prompt wall-clock seconds
    peaks     : list[float]  -- per-prompt peak GPU memory in GiB (device-level)
    model_mem : float        -- model loading memory footprint in GiB
    """
    mem_before_model = _get_device_used_memory_gib()

    omni = Omni(
        model=model_name,
        enforce_eager=enforce_eager,
        parallel_config=parallel_config,
    )

    model_mem = _get_device_used_memory_gib() - mem_before_model

    images: list = []
    times: list[float] = []
    peaks: list[float] = []

    for idx, prompt in enumerate(prompts):
        generator = torch.Generator(
            device=current_omni_platform.device_type,
        ).manual_seed(seed)

        t0 = time.perf_counter()
        outputs = omni.generate(
            {"prompt": prompt},
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                generator=generator,
                guidance_scale=guidance_scale,
                true_cfg_scale=cfg_scale,
                num_inference_steps=num_inference_steps,
            ),
        )
        elapsed = time.perf_counter() - t0

        peak_gib = _get_device_used_memory_gib()

        req_out = outputs[0].request_output
        img = req_out.images[0]
        images.append(img)
        times.append(elapsed)
        peaks.append(peak_gib)

        print(
            f"  [{idx + 1}/{len(prompts)}] {elapsed:.2f}s | "
            f"peak {peak_gib:.2f} GiB (model {model_mem:.2f} GiB) | "
            f"{prompt[:60]}"
        )

    # Unload model
    del omni
    gc.collect()
    current_omni_platform.empty_cache()

    return images, times, peaks, model_mem


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _build_report(
    *,
    baseline_model: str,
    quantized_model: str,
    label: str,
    prompts: list[str],
    height: int,
    width: int,
    num_inference_steps: int,
    seed: int,
    lpips_net: str,
    bl_times: list[float],
    bl_peaks: list[float],
    bl_model_mem: float,
    qt_times: list[float],
    qt_peaks: list[float],
    qt_model_mem: float,
    lpips_scores: list[float],
) -> str:
    """Build a Markdown report string."""
    n = len(prompts)
    avg_bl_time = float(np.mean(bl_times))
    avg_qt_time = float(np.mean(qt_times))
    speedup = avg_bl_time / avg_qt_time if avg_qt_time > 0 else float("inf")
    max_bl_peak = float(np.max(bl_peaks))
    max_qt_peak = float(np.max(qt_peaks))
    mem_reduction = (1.0 - qt_model_mem / bl_model_mem) * 100 if bl_model_mem > 0 else 0.0
    mean_lpips = float(np.mean(lpips_scores))

    lines: list[str] = []
    lines.append("## Offline Quantization Accuracy Evaluation\n")
    lines.append(f"Baseline: `{baseline_model}`  ")
    lines.append(f"Quantized: `{quantized_model}`  ")
    lines.append(
        f"Prompts: {n} | Size: {width}x{height} | "
        f"Steps: {num_inference_steps} | Seed: {seed} | "
        f"LPIPS net: {lpips_net}\n"
    )

    # Summary table
    lines.append("### Summary\n")
    lines.append("| Config | Avg Time | Speedup | Model Mem (GiB) | Peak Mem (GiB) | Mem Reduction | Mean LPIPS |")
    lines.append("|--------|----------|---------|-----------------|----------------|---------------|------------|")
    lines.append(
        f"| BF16 baseline | {avg_bl_time:.2f}s | 1.00x | {bl_model_mem:.2f} | {max_bl_peak:.2f} | -- | (ref) |"
    )
    lines.append(
        f"| {label} | {avg_qt_time:.2f}s | {speedup:.2f}x | "
        f"{qt_model_mem:.2f} | {max_qt_peak:.2f} | {mem_reduction:.0f}% | {mean_lpips:.4f} |"
    )
    lines.append("")
    lines.append(
        "> LPIPS < 0.01 = imperceptible, 0.01-0.05 = minor, 0.05-0.10 = noticeable, > 0.10 = clearly different\n"
    )

    # Per-prompt table
    lines.append("### Per-Prompt Results\n")
    lines.append("| # | Prompt | LPIPS | BL Time | QT Time | BL Peak (GiB) | QT Peak (GiB) |")
    lines.append("|---|--------|-------|---------|---------|----------------|----------------|")
    for i in range(n):
        prompt_short = prompts[i][:50] + ("..." if len(prompts[i]) > 50 else "")
        lines.append(
            f"| {i + 1} | {prompt_short} | {lpips_scores[i]:.4f} | "
            f"{bl_times[i]:.2f}s | {qt_times[i]:.2f}s | "
            f"{bl_peaks[i]:.2f} | {qt_peaks[i]:.2f} |"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate images from a BF16 baseline and an offline-quantized "
            "diffusion model, then compute LPIPS + time + memory metrics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline-model",
        required=True,
        help="BF16 baseline model name or path.",
    )
    parser.add_argument(
        "--quantized-model",
        required=True,
        help="Offline-quantized model name or path.",
    )
    parser.add_argument(
        "--prompts-file",
        required=True,
        help="Text file with one prompt per line (# = comment).",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument(
        "--output-dir",
        default="./quant_eval_output",
        help="Directory for saved images and Markdown report.",
    )
    parser.add_argument(
        "--lpips-net",
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone network.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
    )
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--label",
        default="quantized",
        help="Label for the quantized config in the report.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help=("Skip baseline generation; reuse images already saved in output-dir."),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    prompts = _load_prompts(args.prompts_file)
    output_dir = Path(args.output_dir)
    bl_dir = output_dir / "baseline"
    qt_dir = output_dir / args.label

    bl_dir.mkdir(parents=True, exist_ok=True)
    qt_dir.mkdir(parents=True, exist_ok=True)

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    gen_kwargs = dict(
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        cfg_scale=args.cfg_scale,
        enforce_eager=args.enforce_eager,
        parallel_config=parallel_config,
    )

    # ---- Baseline --------------------------------------------------------
    if args.skip_baseline:
        from PIL import Image

        print(f"\n=== Skipping baseline generation, loading from {bl_dir} ===")
        bl_files = sorted(bl_dir.glob("*.png"))
        if len(bl_files) < len(prompts):
            raise FileNotFoundError(
                f"Expected {len(prompts)} baseline images in {bl_dir}, "
                f"found {len(bl_files)}. "
                f"Run without --skip-baseline first."
            )
        bl_images = [Image.open(p).convert("RGB") for p in bl_files[: len(prompts)]]
        # No timing/memory data when skipping
        bl_times = [0.0] * len(prompts)
        bl_peaks = [0.0] * len(prompts)
        bl_model_mem = 0.0
    else:
        print(f"\n=== Generating baseline images ({args.baseline_model}) ===")
        bl_images, bl_times, bl_peaks, bl_model_mem = _generate_images(
            args.baseline_model,
            prompts,
            **gen_kwargs,
        )
        for idx, img in enumerate(bl_images):
            img.save(bl_dir / f"{idx:04d}.png")
        print(f"Saved {len(bl_images)} baseline images to {bl_dir}")

    # ---- Quantized -------------------------------------------------------
    print(f"\n=== Generating quantized images ({args.quantized_model}) ===")
    qt_images, qt_times, qt_peaks, qt_model_mem = _generate_images(
        args.quantized_model,
        prompts,
        **gen_kwargs,
    )
    for idx, img in enumerate(qt_images):
        img.save(qt_dir / f"{idx:04d}.png")
    print(f"Saved {len(qt_images)} quantized images to {qt_dir}")

    # ---- LPIPS -----------------------------------------------------------
    print(f"\n=== Computing LPIPS ({args.lpips_net}) ===")
    lpips_scores = compute_lpips_images(bl_images, qt_images, net=args.lpips_net)
    mean_lpips = float(np.mean(lpips_scores))
    for idx, (score, prompt) in enumerate(zip(lpips_scores, prompts)):
        print(f"  [{idx + 1}] LPIPS={score:.4f}  {prompt[:60]}")
    print(f"\n  Mean LPIPS: {mean_lpips:.4f}")

    # ---- Report ----------------------------------------------------------
    report = _build_report(
        baseline_model=args.baseline_model,
        quantized_model=args.quantized_model,
        label=args.label,
        prompts=prompts,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        lpips_net=args.lpips_net,
        bl_times=bl_times,
        bl_peaks=bl_peaks,
        bl_model_mem=bl_model_mem,
        qt_times=qt_times,
        qt_peaks=qt_peaks,
        qt_model_mem=qt_model_mem,
        lpips_scores=lpips_scores,
    )

    report_path = output_dir / "results.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(report)
    print(f"{'=' * 60}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def _configure_host_threads() -> None:
    """Clamp host-side thread pools before worker initialization.

    Some Wan sparse debug runs stall in CPU-side tensor conversion paths inside
    worker processes. Keep thread caps explicit here so spawned workers inherit a
    small, stable host-thread configuration.
    """
    intra_threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
    interop_threads = int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "1"))
    torch.set_num_threads(max(1, intra_threads))
    torch.set_num_interop_threads(max(1, interop_threads))
    try:
        import cv2  # type: ignore

        cv2.setNumThreads(0)
    except Exception:
        pass


def _extract_peak_memory_mb(result):
    if isinstance(result, list):
        result = result[0] if result else None
    if result is None:
        return 0.0
    val = getattr(result, "peak_memory_mb", 0.0)
    if not val:
        inner = getattr(result, "request_output", None)
        if isinstance(inner, list):
            inner = inner[0] if inner else None
        val = getattr(inner, "peak_memory_mb", 0.0)
    try:
        return float(val or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _unwrap_frames(result):
    if isinstance(result, list):
        result = result[0] if result else None
    if isinstance(result, OmniRequestOutput):
        if result.images:
            return result.images
        if not result.finished:
            raise RuntimeError("Generation did not finish.")
        final_output = result.outputs
        if isinstance(final_output, list):
            final_output = final_output[0] if final_output else None
        if final_output is None:
            raise RuntimeError("No final output returned.")
        return final_output
    return result


def _normalize_frame(frame):
    if isinstance(frame, torch.Tensor):
        frame_tensor = frame.detach().cpu()
        if frame_tensor.dim() == 4 and frame_tensor.shape[0] == 1:
            frame_tensor = frame_tensor[0]
        if frame_tensor.dim() == 3 and frame_tensor.shape[0] in (3, 4):
            frame_tensor = frame_tensor.permute(1, 2, 0)
        if frame_tensor.is_floating_point():
            frame_tensor = frame_tensor.clamp(-1, 1) * 0.5 + 0.5
        return frame_tensor.float().numpy()
    if isinstance(frame, np.ndarray):
        frame_array = frame
        if frame_array.ndim == 4 and frame_array.shape[0] == 1:
            frame_array = frame_array[0]
        if np.issubdtype(frame_array.dtype, np.integer):
            frame_array = frame_array.astype(np.float32) / 255.0
        return frame_array
    return frame


def _ensure_frame_list(video_array):
    if isinstance(video_array, list):
        if len(video_array) == 0:
            return video_array
        first_item = video_array[0]
        if isinstance(first_item, np.ndarray):
            if first_item.ndim == 5:
                return list(first_item[0])
            if first_item.ndim == 4:
                if len(video_array) == 1:
                    return list(first_item)
                return list(video_array)
            if first_item.ndim == 3:
                return video_array
        return video_array
    if isinstance(video_array, np.ndarray):
        if video_array.ndim == 5:
            return list(video_array[0])
        if video_array.ndim == 4:
            return list(video_array)
        if video_array.ndim == 3:
            return [video_array]
    return video_array


def main():
    _configure_host_threads()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--attention-mode", choices=("dense", "mixed", "sparse_cross"), default="mixed")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--boundary-ratio", type=float, default=0.875)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage-init-timeout", type=int, default=1200)
    args = parser.parse_args()

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    diffusion_attention_config = None
    if args.attention_mode == "dense":
        diffusion_attention_config = {"default": {"backend": "TORCH_SDPA"}}
    elif args.attention_mode == "mixed":
        diffusion_attention_config = {
            "per_role": {
                "self": {"backend": "SAGE_ATTN"},
                "cross": {"backend": "TORCH_SDPA"},
            }
        }
    else:
        diffusion_attention_config = {
            "per_role": {
                "self": {"backend": "SAGE_ATTN"},
                "cross": {"backend": "SAGE_ATTN"},
            }
        }

    omni = Omni(
        model=args.model,
        enable_cpu_offload=True,
        vae_use_slicing=True,
        vae_use_tiling=True,
        enforce_eager=True,
        enable_diffusion_pipeline_profiler=True,
        boundary_ratio=args.boundary_ratio,
        flow_shift=args.flow_shift,
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=4),
        diffusion_attention_config=diffusion_attention_config,
        stage_init_timeout=args.stage_init_timeout,
    )

    sampling = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        generator=generator,
        guidance_scale=4.0,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
    )

    start = time.perf_counter()
    prompt_dict = {"prompt": args.prompt}
    if args.negative_prompt:
        prompt_dict["negative_prompt"] = args.negative_prompt

    result = omni.generate(prompt_dict, sampling)
    end = time.perf_counter()

    total_s = end - start
    peak_mb = _extract_peak_memory_mb(result)
    frames = _unwrap_frames(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frames, torch.Tensor):
        video_tensor = frames.detach().cpu()
        if video_tensor.dim() == 5:
            if video_tensor.shape[1] in (3, 4):
                video_tensor = video_tensor[0].permute(1, 2, 3, 0)
            else:
                video_tensor = video_tensor[0]
        elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
            video_tensor = video_tensor.permute(1, 2, 3, 0)
        if video_tensor.is_floating_point():
            video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
        video_array = video_tensor.float().numpy()
    elif isinstance(frames, np.ndarray):
        video_array = frames
        if video_array.ndim == 5:
            video_array = video_array[0]
        if np.issubdtype(video_array.dtype, np.integer):
            video_array = video_array.astype(np.float32) / 255.0
    elif isinstance(frames, list):
        if len(frames) == 0:
            raise ValueError("No video frames found in output.")
        video_array = [_normalize_frame(frame) for frame in frames]
    else:
        video_array = frames

    video_array = _ensure_frame_list(video_array)
    export_to_video(video_array, str(output_path), fps=args.fps)

    print(f"Total generation time: {total_s:.4f} seconds ({total_s * 1000:.2f} ms)")
    if peak_mb:
        print(f"Worker peak GPU memory (reserved): {peak_mb:.2f} MiB ({peak_mb / 1024:.2f} GiB)")
    print(f"Saved generated video to {output_path}")


if __name__ == "__main__":
    main()

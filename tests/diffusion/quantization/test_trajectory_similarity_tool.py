# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.quantization.tools.compare_diffusion_trajectory_similarity import (
    VariantRun,
    _build_variant_config,
    _expand_frame_container,
    _get_output_frames,
    _request_peak_memory_mb,
    _run_summary,
    compute_tensor_metrics,
    compute_uint8_image_metrics,
    metric_guidance,
    parse_args,
    summarize_output_image_metrics,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_compute_tensor_metrics_identical_tensors():
    metrics = compute_tensor_metrics(torch.ones(2, 3), torch.ones(2, 3))

    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["max_abs"] == 0.0
    assert metrics["l2"] == 0.0
    assert metrics["relative_l2"] == 0.0


def test_compute_uint8_image_metrics_adds_psnr():
    lhs = np.zeros((2, 2, 3), dtype=np.uint8)
    rhs = np.zeros((2, 2, 3), dtype=np.uint8)

    metrics = compute_uint8_image_metrics(lhs, rhs)

    assert math.isinf(metrics["psnr_db"])


def test_summarize_output_image_metrics_stacks_pil_images():
    reference = [Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))]
    candidate = [Image.fromarray(np.ones((2, 2, 3), dtype=np.uint8))]

    summary = summarize_output_image_metrics(reference, candidate)

    assert summary["num_images"] == 1
    assert summary["image0_metrics"]["mae"] == 1.0
    assert summary["all_images_metrics"]["mse"] == 1.0


@pytest.mark.parametrize(
    ("stacked", "expected_frames"),
    [
        ((3, 3, 4, 6), 3),  # video tensor with a frame dimension first
        ((1, 3, 3, 4, 6), 3),  # and with a leading batch dimension
    ],
)
def test_video_tensor_frames_become_pil_frames(stacked, expected_frames):
    """Wan-style video results are one stacked tensor, not a list of PIL images."""
    # Channels-first with a spatial extent above the channel count, as real frames have.
    values = torch.tensor([-1.0, 0.2, 1.0]).view(3, 1, 1, 1).expand(3, 3, 4, 6)
    result = SimpleNamespace(images=[values.reshape(stacked)])

    frames = _get_output_frames(result)

    assert len(frames) == expected_frames
    assert [frame.size for frame in frames] == [(6, 4)] * expected_frames
    # auto range maps [-1, 1] onto uint8; the mid frame is not near either end.
    assert [np.asarray(frame).mean() for frame in frames] == [0.0, 153.0, 255.0]


def test_frame_container_handles_audio_pair_and_dict():
    frames_tensor = torch.zeros(2, 3, 4, 4)
    audio = torch.zeros(8)

    assert len(_expand_frame_container([{"frames": frames_tensor, "audio": audio}])) == 2
    assert len(_expand_frame_container([(frames_tensor, audio)])) == 2
    assert len(_expand_frame_container([[Image.new("RGB", (2, 2))]])) == 1


def test_positive_tensors_are_not_rescaled_and_uint8_passes_through():
    positive = torch.full((1, 3, 2, 2), 0.25)
    assert np.asarray(_get_output_frames(SimpleNamespace(images=[positive]))[0]).mean() == 64.0

    raw = np.full((2, 2, 3), 200, dtype=np.uint8)
    converted = _get_output_frames(SimpleNamespace(images=[raw]))[0]
    assert np.asarray(converted).mean() == 200.0

    # Metrics only accept PIL frames, which is what the conversion guarantees.
    summary = summarize_output_image_metrics(
        _get_output_frames(SimpleNamespace(images=[positive])),
        _get_output_frames(SimpleNamespace(images=[positive])),
    )
    assert summary["all_frames_metrics"]["psnr_db"] == float("inf")


def test_run_summary_reports_worker_peak_memory():
    summary = _run_summary(
        VariantRun(
            label="candidate",
            result=object(),
            generation_times_s=[1.0, 3.0],
            peak_memory_mb=[100.0, 150.0],
        )
    )

    assert summary["peak_memory_mb"] == 150.0
    assert summary["avg_peak_memory_mb"] == 125.0
    assert summary["max_peak_memory_mb"] == 150.0
    assert summary["per_run_peak_memory_mb"] == [100.0, 150.0]


def test_request_peak_memory_reads_peak_memory_mb_directly():
    result = type("Output", (), {"peak_memory_mb": 123.0})()
    assert _request_peak_memory_mb(result) == 123.0


def test_metric_guidance_describes_thresholds():
    guidance = metric_guidance()

    assert "cosine_similarity" in guidance["descriptions"]
    assert guidance["recommended_thresholds"]["output_images_or_frames_uint8"]["psnr_db"]["recommended_min"] == 20.0
    assert (
        guidance["recommended_thresholds"]["performance"]["max_peak_memory_ratio_candidate_over_reference"][
            "recommended_max"
        ]
        == 1.00
    )


def test_candidate_model_can_point_to_offline_checkpoint_without_online_quantization():
    args = SimpleNamespace(
        model="Qwen/Qwen-Image",
        candidate_model="Qwen/Qwen-Image-FP8",
        candidate_quantization=None,
        candidate_quantization_config_json=None,
        candidate_ignored_layers=None,
        ignored_layers=None,
        candidate_load_format="default",
    )

    config = _build_variant_config(args, "candidate")

    assert config.model == "Qwen/Qwen-Image-FP8"
    assert config.quantization is None
    assert config.quantization_config is None


def test_step_execution_defaults_to_false(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diffusion_trajectory_similarity.py",
            "--output-json",
            "result.json",
        ],
    )

    args = parse_args()

    assert args.step_execution is False

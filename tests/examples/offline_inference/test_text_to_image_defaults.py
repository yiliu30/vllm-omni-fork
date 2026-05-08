# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

from examples.offline_inference.text_to_image.text_to_image import resolve_guidance_scale


def test_resolve_guidance_scale_uses_flux_default_when_omitted():
    with patch(
        "examples.offline_inference.text_to_image.text_to_image.detect_model_class_name",
        return_value="FluxPipeline",
    ):
        assert resolve_guidance_scale("unused", None) == 3.5


def test_resolve_guidance_scale_uses_qwen_default_when_omitted():
    with patch(
        "examples.offline_inference.text_to_image.text_to_image.detect_model_class_name",
        return_value="QwenImagePipeline",
    ):
        assert resolve_guidance_scale("unused", None) == 1.0

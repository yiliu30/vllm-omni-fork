from dataclasses import field
from typing import Any

from pydantic import ConfigDict
from vllm.config import ModelConfig
from vllm.config.multimodal import MMCacheType, MMEncoderTPMode
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_text_config
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

import vllm_omni.model_executor.models as me_models

logger = init_logger(__name__)


class OmniModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Config convertor for Omni multi-stage models.

    Pre-quantized checkpoints (e.g. modelopt FP8) store quantization
    config in a stage-specific sub-config (e.g.
    thinker_config.text_config.quantization_config) with correct relative
    prefixes.  The legacy hf_quant_config.json sits at the top level with
    "thinker."-prefixed names that don't match vllm-omni's module names.

    This convertor accepts an optional *stage_config_name* so that only
    the relevant stage's quantization config is surfaced.
    """

    def __init__(
        self,
        hf_config,
        hf_text_config,
        stage_config_name: str | None = None,
    ):
        super().__init__(hf_config, hf_text_config)
        self.stage_config_name = stage_config_name

    def get_quantization_config(self):
        # When a stage_config_name is set, look for quantization config
        # in that stage's text_config first (has correct relative prefixes).
        if self.stage_config_name is not None:
            stage_cfg = getattr(self.hf_config, self.stage_config_name, None)
            if stage_cfg is not None:
                text_cfg = getattr(stage_cfg, "text_config", None)
                if text_cfg is not None:
                    quant_cfg = self._normalize_quantization_config(text_cfg)
                    if quant_cfg is not None:
                        return quant_cfg

            # For non-thinker stages (talker, code2wav) whose text_config
            # has no quantization_config, return None so quantization is
            # not applied to stages that were not quantized.
            return None

        return super().get_quantization_config()


@config(config=ConfigDict(arbitrary_types_allowed=True))
class OmniModelConfig(ModelConfig):
    """Configuration for Omni models, extending the base ModelConfig.

    This configuration class extends the base vLLM ModelConfig with
    omni-specific fields for multi-stage pipeline processing.

    Attributes:
        stage_id: Identifier for the stage in a multi-stage pipeline (default: 0)
        async_chunk: If set to True, perform async chunk
        model_stage: Stage type identifier, e.g., "thinker" or "talker"
            (default: "thinker")
        model_arch: Model architecture name
            (default: "Qwen2_5OmniForConditionalGeneration")
        worker_type: Model Type, e.g., "ar" or "generation"
        engine_output_type: Optional output type specification for the engine.
            Used to route outputs to appropriate processors (e.g., "image",
            "audio", "latents"). If None, output type is inferred.
        stage_connector_config: Stage connector configuration dictionary.
            Contains "name" (connector name), "extra" (extra connector config).
        task_type: Default task type for TTS models (CustomVoice, VoiceDesign, or Base).
            If not specified, will be inferred from model path.

    Example:
        >>> config = OmniModelConfig(
        ...     stage_id=0,
        ...     model_stage="thinker",
        ...     model_arch="Qwen2_5OmniForConditionalGeneration"
        ... )
    """

    stage_id: int = 0
    async_chunk: bool = False
    model_stage: str = "thinker"
    model_arch: str | None = None
    worker_type: str | None = None
    engine_output_type: str | None = None
    hf_config_name: str | None = None
    custom_process_next_stage_input_func: str | None = None
    stage_connector_config: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "SharedMemoryConnector",
            "extra": {},
        }
    )
    omni_kv_config: dict | None = None
    codec_frame_rate_hz: float | None = None
    task_type: str | None = None

    @property
    def registry(self):
        return me_models.OmniModelRegistry

    @property
    def architectures(self) -> list[str]:
        if self.model_arch is not None:
            return [self.model_arch]
        return super().architectures

    @property
    def embedding_size(self):
        if self.hf_config_name is not None:
            stage_config = getattr(self.hf_config, self.hf_config_name, None)
            override = getattr(stage_config, "embedding_size", None)
            if override is not None:
                return override
        return super().embedding_size

    def get_model_arch_config(self):
        # For multi-stage omni models, use a stage-aware convertor so that
        # only the correct stage's quantization config is surfaced.
        # Without this, a pre-quantized thinker checkpoint would also
        # apply quantization to the talker/code2wav stages.
        if self.hf_config_name is not None:
            convertor = OmniModelArchConfigConvertor(
                self.hf_config,
                self.hf_text_config,
                stage_config_name=self.hf_config_name,
            )
            return convertor.convert()
        return super().get_model_arch_config()

    def draw_hf_text_config(self):
        # transformers' get_text_config method is used to get the text config from thinker_config.
        # to handle the case that each model stage has their own text config,
        # we need to draw the text config from the corresponding model stage.
        if self.hf_config_name is None:
            return get_hf_text_config(self.hf_config)
        try:
            # Try to get the stage-specific config (e.g., thinker_config, talker_config)
            stage_config = getattr(self.hf_config, self.hf_config_name)
            return stage_config.get_text_config()
        except AttributeError:
            # Fallback: if the attribute doesn't exist, use the default get_hf_text_config
            logger.warning(
                f"Config attribute '{self.hf_config_name}' not found in hf_config, "
                "falling back to default get_hf_text_config"
            )
            return get_hf_text_config(self.hf_config)

    def __post_init__(
        self,
        # Multimodal config init vars
        language_model_only: bool,
        limit_mm_per_prompt: dict[str, int | dict[str, int]] | None,
        enable_mm_embeds: bool | None,
        media_io_kwargs: dict[str, dict[str, Any]] | None,
        mm_processor_kwargs: dict[str, Any] | None,
        mm_processor_cache_gb: float | None,
        mm_processor_cache_type: MMCacheType | None,
        mm_shm_cache_max_object_size_mb: int | None,
        mm_encoder_only: bool | None,
        mm_encoder_tp_mode: MMEncoderTPMode | None,
        mm_encoder_attn_backend: AttentionBackendEnum | str | None,
        interleave_mm_strings: bool | None,
        skip_mm_profiling: bool | None,
        video_pruning_rate: float | None,
    ) -> None:
        # Call parent's __post_init__ to handle all standard ModelConfig initialization
        super().__post_init__(
            language_model_only=language_model_only,
            limit_mm_per_prompt=limit_mm_per_prompt,
            enable_mm_embeds=enable_mm_embeds,
            media_io_kwargs=media_io_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
            mm_processor_cache_gb=mm_processor_cache_gb,
            mm_processor_cache_type=mm_processor_cache_type,
            mm_shm_cache_max_object_size_mb=mm_shm_cache_max_object_size_mb,
            mm_encoder_only=mm_encoder_only,
            mm_encoder_tp_mode=mm_encoder_tp_mode,
            mm_encoder_attn_backend=mm_encoder_attn_backend,
            interleave_mm_strings=interleave_mm_strings,
            skip_mm_profiling=skip_mm_profiling,
            video_pruning_rate=video_pruning_rate,
        )

        # Qwen3-TTS: infer codec frame rate from the model config for online serving.
        if self.codec_frame_rate_hz is None and self.model_arch == "Qwen3TTSTalkerForConditionalGenerationARVLLM":
            talker_cfg = getattr(self.hf_config, "talker_config", None)
            if isinstance(talker_cfg, dict):
                pos_per_sec = talker_cfg.get("position_id_per_seconds")
            else:
                pos_per_sec = getattr(talker_cfg, "position_id_per_seconds", None)
            if pos_per_sec is not None:
                try:
                    fps = float(pos_per_sec)
                except Exception:
                    fps = None
                if fps is not None and fps > 0:
                    self.codec_frame_rate_hz = fps

        # Override hf_text_config with omni-specific logic for multi-stage models
        # (e.g., thinker_config, talker_config)
        new_hf_text_config = self.draw_hf_text_config()
        if new_hf_text_config is not self.hf_text_config:
            self.hf_text_config = new_hf_text_config
            # Recalculate dependent attributes
            self.attention_chunk_size = getattr(self.hf_text_config, "attention_chunk_size", None)
            # Recalculate max_model_len since it depends on hf_text_config
            self.max_model_len = self.get_and_verify_max_len(self.original_max_model_len)
            # Reset sliding_window if needed
            if self.disable_sliding_window:
                self.hf_text_config.sliding_window = None

from __future__ import annotations


def register() -> None:
    """Register StepAudio2 support into an upstream vLLM installation."""
    from transformers import AutoConfig

    from vllm.model_executor.models import ModelRegistry
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers import TokenizerRegistry
    from vllm.transformers_utils import config as vllm_config

    from .configuration_step_audio import (
        StepAudio2Config,
        StepAudio2EncoderConfig,
        StepAudio2TextConfig,
    )

    AutoConfig.register(StepAudio2Config.model_type, StepAudio2Config, exist_ok=True)
    AutoConfig.register(
        StepAudio2EncoderConfig.model_type,
        StepAudio2EncoderConfig,
        exist_ok=True,
    )
    AutoConfig.register(
        StepAudio2TextConfig.model_type,
        StepAudio2TextConfig,
        exist_ok=True,
    )

    vllm_config._CONFIG_REGISTRY[StepAudio2Config.model_type] = StepAudio2Config
    vllm_config._CONFIG_REGISTRY[StepAudio2EncoderConfig.model_type] = (
        StepAudio2EncoderConfig
    )
    vllm_config._CONFIG_REGISTRY[StepAudio2TextConfig.model_type] = StepAudio2TextConfig

    TokenizerRegistry.register(
        "step_audio_2",
        "vllm.plugins.stepaudio_vllm24.tokenizer",
        "StepAudio2Tokenizer",
    )
    RENDERER_REGISTRY.register("step_audio_2", "vllm.renderers.hf", "HfRenderer")
    ModelRegistry.register_model(
        "StepAudio2ForCausalLM",
        "vllm.plugins.stepaudio_vllm24.mm_step_audio:StepAudio2ForCausalLM",
    )

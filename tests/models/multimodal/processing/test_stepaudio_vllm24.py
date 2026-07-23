# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from transformers import PretrainedConfig

from vllm.multimodal.parse import AudioProcessorItems
from vllm.plugins.stepaudio_vllm24.mm_step_audio import (
    AudioEncoder,
    Step1fAudioProcessingInfo,
    Step1fProcessor,
)


class _Tokenizer:
    _VOCAB = {
        "<audio_start>": 1,
        "<audio_end>": 2,
        "<audio_patch>": 3,
    }

    def get_vocab(self) -> dict[str, int]:
        return self._VOCAB

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._VOCAB[token]

    def __call__(self, texts: list[str]) -> dict[str, list[list[int]]]:
        return {"input_ids": [[0] for _ in texts]}


@pytest.fixture
def processor() -> Step1fProcessor:
    return Step1fProcessor(PretrainedConfig(), _Tokenizer())  # type: ignore[arg-type]


def test_stepaudio_processor_accepts_max_safe_audio(
    processor: Step1fProcessor,
) -> None:
    max_samples = processor.sampling_rate * processor.max_chunk_size

    outputs = processor(
        text="<audio_patch>",
        audios=np.zeros(max_samples, dtype=np.float32),
    )

    assert outputs["audio_waveform_lens"].tolist() == [max_samples]


def test_stepaudio_processor_rejects_oversized_audio(
    processor: Step1fProcessor,
) -> None:
    max_samples = processor.sampling_rate * processor.max_chunk_size

    with pytest.raises(ValueError, match="exceeds the safe processor limit"):
        processor(
            text="<audio_patch>",
            audios=np.zeros(max_samples + 1, dtype=np.float32),
        )


def test_stepaudio_processor_rejects_non_mono_audio(
    processor: Step1fProcessor,
) -> None:
    with pytest.raises(ValueError, match="one-dimensional mono audio waveform"):
        processor(
            text="<audio_patch>",
            audios=np.zeros((2, processor.sampling_rate), dtype=np.float32),
        )


def test_stepaudio_parser_normalizes_stereo_audio_to_mono() -> None:
    info = object.__new__(Step1fAudioProcessingInfo)
    parser = info.get_data_parser()
    stereo = np.stack((np.zeros(16), np.ones(16))).astype(np.float32)

    parsed = parser.parse_mm_data({"audio": (stereo, 16000)})
    audio = parsed.get_items("audio", AudioProcessorItems)[0]

    assert audio.shape == (16,)
    np.testing.assert_allclose(audio, 0.5)


def test_stepaudio_encoder_rejects_oversized_mel_sequence() -> None:
    encoder = AudioEncoder(
        n_mels=2,
        n_ctx=2,
        n_state=4,
        n_head=1,
        n_layer=0,
    )

    with pytest.raises(ValueError, match="3 > 2"):
        encoder(torch.zeros(1, 2, 5), torch.tensor([5]))

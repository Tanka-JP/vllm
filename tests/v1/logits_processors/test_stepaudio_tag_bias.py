# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from vllm.v1.sample.logits_processor import BatchUpdate, MoveDirectionality

from vllm import SamplingParams
from vllm.plugins.stepaudio_vllm24.logits_processor import (
    GATE_AFTER_THINK,
    GATE_ALWAYS,
    TAG_BIAS_ARG,
    TAG_BIAS_GATE_ARG,
    TAG_BIAS_PER_FAMILY_ARG,
    StepAudioTagBiasLogitsProcessor,
    StepAudioTagBiasParams,
    StepAudioTagBiasRequestProcessor,
    build_stepaudio_family_token_bias,
    build_stepaudio_tag_token_spec,
    parse_stepaudio_tag_bias_params,
)


class FakeStepAudioTokenizer:
    _TOKENS = {
        "<": [10],
        "><": [11],
        "<|": [10, 12],
        "</think>": [13, 14, 15],
        "<|emotion=": [10, 12, 20, 30],
        "<|intensity=": [10, 12, 21, 30],
        "<|style=": [10, 12, 22, 30],
        "<|extra_emotion=": [10, 12, 23, 30],
        "<|breath": [10, 12, 24],
        "<|laugh": [10, 12, 25],
        "<|gasp": [10, 12, 24, 26],
    }

    def __call__(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        return {"input_ids": self._TOKENS[text]}


@pytest.fixture
def tokenizer() -> FakeStepAudioTokenizer:
    return FakeStepAudioTokenizer()


def _sampling_params(**extra_args) -> SamplingParams:
    return SamplingParams(extra_args=extra_args or None)


def test_parse_request_params() -> None:
    parsed = parse_stepaudio_tag_bias_params(
        _sampling_params(
            stepaudio_tag_bias=3,
            stepaudio_tag_bias_per_family=["breath=2.5", "style=-1"],
            stepaudio_tag_bias_gate=GATE_ALWAYS,
        )
    )

    assert parsed == StepAudioTagBiasParams(
        global_bias=3.0,
        per_family=(("breath", 2.5), ("style", -1.0)),
        gate=GATE_ALWAYS,
    )


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        ({TAG_BIAS_ARG: "3"}, "must be a number"),
        ({TAG_BIAS_ARG: True}, "must be a number"),
        ({TAG_BIAS_ARG: float("inf")}, "must be finite"),
        ({TAG_BIAS_ARG: 21}, "must be within"),
        ({TAG_BIAS_PER_FAMILY_ARG: "breath=3"}, "must be a list"),
        ({TAG_BIAS_PER_FAMILY_ARG: [3]}, "entries must be strings"),
        ({TAG_BIAS_PER_FAMILY_ARG: ["breath"]}, "expected 'name=bias'"),
        ({TAG_BIAS_PER_FAMILY_ARG: ["breath=x"]}, "invalid bias"),
        ({TAG_BIAS_PER_FAMILY_ARG: ["breath=nan"]}, "must be finite"),
        ({TAG_BIAS_PER_FAMILY_ARG: ["not-valid=1"]}, "invalid StepAudio"),
        (
            {TAG_BIAS_PER_FAMILY_ARG: ["breath=1", "breath=2"]},
            "duplicate",
        ),
        ({TAG_BIAS_GATE_ARG: "sometimes"}, "must be one of"),
    ],
)
def test_invalid_request_params(extra_args: dict, message: str) -> None:
    params = SamplingParams(extra_args=extra_args)
    with pytest.raises(ValueError, match=message):
        StepAudioTagBiasLogitsProcessor.validate_params(params)


def test_tokenizer_probe_and_family_aliases(tokenizer) -> None:
    spec = build_stepaudio_tag_token_spec(tokenizer)
    assert spec.open_token_ids == (10, 11)
    assert spec.pipe_anchor_id == 12
    assert spec.think_close_ids == (13, 14)

    token_bias = build_stepaudio_family_token_bias(
        tokenizer,
        (("style", 2.0), ("breath", 3.0)),
    )
    assert token_bias == {22: 2.0, 23: 2.0, 24: 3.0}


def test_family_token_collision_keeps_larger_absolute_bias(tokenizer) -> None:
    token_bias = build_stepaudio_family_token_bias(
        tokenizer,
        (("breath", 1.0), ("gasp", -3.0)),
    )

    assert token_bias == {24: -3.0}


def test_global_bias_is_gated_after_think(tokenizer) -> None:
    spec = build_stepaudio_tag_token_spec(tokenizer)
    processor = StepAudioTagBiasRequestProcessor(
        StepAudioTagBiasParams(global_bias=3.0, gate=GATE_AFTER_THINK),
        spec,
        {},
    )

    before = torch.zeros(32)
    assert processor([99, 13], before) is before
    assert torch.count_nonzero(before) == 0

    after = torch.zeros(32)
    assert processor([99, 13, 14], after) is after
    assert after[10].item() == 3.0
    assert after[11].item() == 3.0
    assert torch.count_nonzero(after).item() == 2


def test_always_gate_and_per_family_bias(tokenizer) -> None:
    spec = build_stepaudio_tag_token_spec(tokenizer)
    family_bias = build_stepaudio_family_token_bias(
        tokenizer,
        (("breath", 3.0),),
    )
    processor = StepAudioTagBiasRequestProcessor(
        StepAudioTagBiasParams(
            global_bias=-2.0,
            per_family=(("breath", 3.0),),
            gate=GATE_ALWAYS,
        ),
        spec,
        family_bias,
    )

    logits = torch.zeros(32)
    processor([12], logits)
    assert logits[10].item() == -2.0
    assert logits[11].item() == -2.0
    assert logits[24].item() == 3.0

    not_at_pipe = torch.zeros(32)
    processor([99], not_at_pipe)
    assert not_at_pipe[10].item() == -2.0
    assert not_at_pipe[11].item() == -2.0
    assert not_at_pipe[24].item() == 0.0


def test_per_family_bias_is_gated_after_think(tokenizer) -> None:
    spec = build_stepaudio_tag_token_spec(tokenizer)
    family_bias = build_stepaudio_family_token_bias(
        tokenizer,
        (("breath", 3.0),),
    )
    processor = StepAudioTagBiasRequestProcessor(
        StepAudioTagBiasParams(
            per_family=(("breath", 3.0),),
            gate=GATE_AFTER_THINK,
        ),
        spec,
        family_bias,
    )

    before = torch.zeros(32)
    processor([12], before)
    assert before[24].item() == 0.0

    after = torch.zeros(32)
    processor([13, 14, 12], after)
    assert after[24].item() == 3.0


def test_zero_bias_request_is_not_registered(monkeypatch, tokenizer) -> None:
    module = "vllm.plugins.stepaudio_vllm24.logits_processor"
    monkeypatch.setattr(f"{module}.cached_tokenizer_from_config", lambda _: tokenizer)
    wrapper = StepAudioTagBiasLogitsProcessor(Mock(), torch.device("cpu"), False)

    assert wrapper.new_req_logits_processor(_sampling_params()) is None
    assert (
        wrapper.new_req_logits_processor(
            _sampling_params(stepaudio_tag_bias=0, stepaudio_tag_bias_per_family=[])
        )
        is None
    )


def test_batch_requests_are_isolated_and_move_with_slots(
    monkeypatch, tokenizer
) -> None:
    module = "vllm.plugins.stepaudio_vllm24.logits_processor"
    monkeypatch.setattr(f"{module}.cached_tokenizer_from_config", lambda _: tokenizer)
    wrapper = StepAudioTagBiasLogitsProcessor(Mock(), torch.device("cpu"), False)

    output_precision = [13, 14]
    output_recall = [13, 14]
    wrapper.update_state(
        BatchUpdate(
            batch_size=3,
            removed=(),
            added=(
                (0, _sampling_params(stepaudio_tag_bias=-3), None, output_precision),
                (1, _sampling_params(), None, []),
                (2, _sampling_params(stepaudio_tag_bias=3), None, output_recall),
            ),
            moved=(),
        )
    )

    logits = wrapper.apply(torch.zeros((3, 32)))
    assert logits[0, 10].item() == -3.0
    assert logits[1, 10].item() == 0.0
    assert logits[2, 10].item() == 3.0

    wrapper.update_state(
        BatchUpdate(
            batch_size=2,
            removed=(0,),
            added=(),
            moved=((2, 0, MoveDirectionality.UNIDIRECTIONAL),),
        )
    )
    moved_logits = wrapper.apply(torch.zeros((2, 32)))
    assert moved_logits[0, 10].item() == 3.0
    assert moved_logits[1, 10].item() == 0.0

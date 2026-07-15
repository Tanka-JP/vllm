# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.sampling_params import SamplingParams
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


TAG_BIAS_ARG = "stepaudio_tag_bias"
TAG_BIAS_PER_FAMILY_ARG = "stepaudio_tag_bias_per_family"
TAG_BIAS_GATE_ARG = "stepaudio_tag_bias_gate"

GATE_AFTER_THINK = "after_think"
GATE_ALWAYS = "always"
_VALID_GATES = frozenset({GATE_AFTER_THINK, GATE_ALWAYS})
_MAX_ABS_BIAS = 20.0
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# StepAudio checkpoints may emit either spelling for the style family.
_FAMILY_OPEN_PREFIXES: dict[str, tuple[str, ...]] = {
    "emotion": ("<|emotion=",),
    "intensity": ("<|intensity=",),
    "style": ("<|style=", "<|extra_emotion="),
    "extra_emotion": ("<|style=", "<|extra_emotion="),
}


@dataclass(frozen=True)
class StepAudioTagBiasParams:
    global_bias: float = 0.0
    per_family: tuple[tuple[str, float], ...] = ()
    gate: str = GATE_AFTER_THINK

    def is_active(self) -> bool:
        return self.global_bias != 0.0 or bool(self.per_family)


@dataclass(frozen=True)
class StepAudioTagTokenSpec:
    open_token_ids: tuple[int, ...]
    pipe_anchor_id: int
    think_close_ids: tuple[int, ...]


def _parse_bias(value: Any, *, argument: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{argument} must be a number, got {value!r}")
    bias = float(value)
    if not math.isfinite(bias):
        raise ValueError(f"{argument} must be finite, got {value!r}")
    if abs(bias) > _MAX_ABS_BIAS:
        raise ValueError(
            f"{argument} must be within [-{_MAX_ABS_BIAS:g}, "
            f"{_MAX_ABS_BIAS:g}], got {bias:g}"
        )
    return bias


def _parse_per_family(value: Any) -> tuple[tuple[str, float], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(
            f"{TAG_BIAS_PER_FAMILY_ARG} must be a list of 'name=bias' strings"
        )

    parsed: list[tuple[str, float]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                f"{TAG_BIAS_PER_FAMILY_ARG} entries must be strings, got {item!r}"
            )
        name, separator, raw_bias = item.partition("=")
        name = name.strip()
        raw_bias = raw_bias.strip()
        if not separator or not name or not raw_bias:
            raise ValueError(
                f"invalid {TAG_BIAS_PER_FAMILY_ARG} entry {item!r}; "
                "expected 'name=bias'"
            )
        if not _NAME_RE.fullmatch(name):
            raise ValueError(
                f"invalid StepAudio family or marker name {name!r} in "
                f"{TAG_BIAS_PER_FAMILY_ARG}"
            )
        if name in seen:
            raise ValueError(
                f"duplicate StepAudio family or marker {name!r} in "
                f"{TAG_BIAS_PER_FAMILY_ARG}"
            )
        seen.add(name)
        try:
            numeric_bias: int | float = float(raw_bias)
        except ValueError as exc:
            raise ValueError(
                f"invalid bias {raw_bias!r} for StepAudio family or marker {name!r}"
            ) from exc
        bias = _parse_bias(
            numeric_bias,
            argument=f"{TAG_BIAS_PER_FAMILY_ARG}[{name}]",
        )
        if bias != 0.0:
            parsed.append((name, bias))
    return tuple(parsed)


def parse_stepaudio_tag_bias_params(
    sampling_params: SamplingParams,
) -> StepAudioTagBiasParams:
    extra_args = sampling_params.extra_args or {}
    global_bias = _parse_bias(
        extra_args.get(TAG_BIAS_ARG, 0.0),
        argument=TAG_BIAS_ARG,
    )
    per_family = _parse_per_family(extra_args.get(TAG_BIAS_PER_FAMILY_ARG))
    gate = extra_args.get(TAG_BIAS_GATE_ARG, GATE_AFTER_THINK)
    if not isinstance(gate, str) or gate not in _VALID_GATES:
        allowed = ", ".join(sorted(_VALID_GATES))
        raise ValueError(
            f"{TAG_BIAS_GATE_ARG} must be one of [{allowed}], got {gate!r}"
        )
    return StepAudioTagBiasParams(global_bias, per_family, gate)


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def build_stepaudio_tag_token_spec(tokenizer: Any) -> StepAudioTagTokenSpec:
    open_token_ids: list[int] = []
    less_than_ids = _token_ids(tokenizer, "<")
    if not less_than_ids:
        raise ValueError("StepAudio tokenizer produced no token for '<'")
    open_token_ids.append(less_than_ids[-1])

    adjacent_ids = _token_ids(tokenizer, "><")
    if len(adjacent_ids) == 1:
        open_token_ids.append(adjacent_ids[0])

    pipe_ids = _token_ids(tokenizer, "<|")
    if not pipe_ids:
        raise ValueError("StepAudio tokenizer produced no token for '<|'")

    think_close = _token_ids(tokenizer, "</think>")
    if len(think_close) < 2:
        raise ValueError(
            "StepAudio tokenizer must produce at least two tokens for '</think>'"
        )

    return StepAudioTagTokenSpec(
        open_token_ids=tuple(dict.fromkeys(open_token_ids)),
        pipe_anchor_id=pipe_ids[-1],
        think_close_ids=tuple(think_close[:2]),
    )


def _name_first_token_ids(tokenizer: Any, name: str) -> tuple[int, ...]:
    prefixes = _FAMILY_OPEN_PREFIXES.get(name, (f"<|{name}",))
    anchor = _token_ids(tokenizer, "<|")
    token_ids: list[int] = []
    for prefix in prefixes:
        prefix_ids = _token_ids(tokenizer, prefix)
        if prefix_ids[: len(anchor)] != anchor or len(prefix_ids) <= len(anchor):
            raise ValueError(
                f"StepAudio tokenizer cannot isolate the name token in {prefix!r}"
            )
        token_ids.append(prefix_ids[len(anchor)])
    return tuple(dict.fromkeys(token_ids))


def build_stepaudio_family_token_bias(
    tokenizer: Any,
    per_family: tuple[tuple[str, float], ...],
) -> dict[int, float]:
    token_bias: dict[int, float] = {}
    for name, bias in per_family:
        for token_id in _name_first_token_ids(tokenizer, name):
            previous = token_bias.get(token_id)
            if previous is None or abs(bias) > abs(previous):
                token_bias[token_id] = bias
    return token_bias


def _contains_subsequence(sequence: list[int], pattern: tuple[int, ...]) -> bool:
    width = len(pattern)
    if width == 0:
        return True
    if len(sequence) < width:
        return False
    return any(
        tuple(sequence[index : index + width]) == pattern
        for index in range(len(sequence) - width + 1)
    )


class StepAudioTagBiasRequestProcessor:
    def __init__(
        self,
        params: StepAudioTagBiasParams,
        token_spec: StepAudioTagTokenSpec,
        family_token_bias: dict[int, float],
    ) -> None:
        self.global_bias = params.global_bias
        self.open_token_ids = token_spec.open_token_ids
        self.pipe_anchor_id = token_spec.pipe_anchor_id
        self.family_token_bias = family_token_bias
        self.gate_ids = (
            token_spec.think_close_ids if params.gate == GATE_AFTER_THINK else ()
        )

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if not _contains_subsequence(output_ids, self.gate_ids):
            return logits

        if self.global_bias != 0.0:
            for token_id in self.open_token_ids:
                logits[token_id] += self.global_bias

        if (
            self.family_token_bias
            and output_ids
            and output_ids[-1] == self.pipe_anchor_id
        ):
            for token_id, bias in self.family_token_bias.items():
                logits[token_id] += bias
        return logits


class StepAudioTagBiasLogitsProcessor(AdapterLogitsProcessor):
    """Per-request StepAudio precision/recall steering for vLLM V1."""

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        parse_stepaudio_tag_bias_params(params)

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        super().__init__(vllm_config, device, is_pin_memory)
        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        if tokenizer is None:
            raise ValueError(
                "StepAudio tag bias requires tokenizer initialization to be enabled"
            )
        self.tokenizer = tokenizer
        self.token_spec = build_stepaudio_tag_token_spec(tokenizer)

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> RequestLogitsProcessor | None:
        parsed = parse_stepaudio_tag_bias_params(params)
        if not parsed.is_active():
            return None
        family_token_bias = build_stepaudio_family_token_bias(
            self.tokenizer,
            parsed.per_family,
        )
        return StepAudioTagBiasRequestProcessor(
            parsed,
            self.token_spec,
            family_token_bias,
        )


__all__ = [
    "GATE_AFTER_THINK",
    "GATE_ALWAYS",
    "StepAudioTagBiasLogitsProcessor",
    "StepAudioTagBiasParams",
    "StepAudioTagBiasRequestProcessor",
    "StepAudioTagTokenSpec",
    "TAG_BIAS_ARG",
    "TAG_BIAS_GATE_ARG",
    "TAG_BIAS_PER_FAMILY_ARG",
    "build_stepaudio_family_token_bias",
    "build_stepaudio_tag_token_spec",
    "parse_stepaudio_tag_bias_params",
]

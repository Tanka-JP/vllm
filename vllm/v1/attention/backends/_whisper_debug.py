# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic helpers for the Whisper-on-ROCm V2 NaN investigation.

These auto-enable for encoder-decoder models (whisper, etc.) via
``enable_auto()`` (called from ``CrossAttention.__init__``), because the AMD CI
pipeline yaml is generated from the trusted ``main`` ref, so a PR cannot set
env vars on the test step. When auto-enabled, we both dump read-side attention
metadata and force the index tensors contiguous (the contiguity experiment).

Env vars still force-enable independently of model type:
  VLLM_WHISPER_DEBUG=1         metadata dump + per-layer NaN probe
  VLLM_WHISPER_FORCE_CONTIG=1  force query_start_loc/seq_lens/block_table contiguous
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

WHISPER_DEBUG = int(os.environ.get("VLLM_WHISPER_DEBUG", "0"))
FORCE_CONTIG = int(os.environ.get("VLLM_WHISPER_FORCE_CONTIG", "0"))

_auto = False
_dump_counts: dict[str, int] = {}
_MAX_DUMPS_PER_TAG = 4


def enable_auto() -> None:
    """Activate diagnostics for the current process (encoder-decoder model).

    Scoped to ROCm: the bug is ROCm-only and the CUDA path stays the untouched
    passing baseline (and would sync during cudagraph capture otherwise).
    """
    global _auto
    if _auto:
        return
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return
    _auto = True
    logger.warning(
        "[whisper-dbg] AUTO-ENABLED for encoder-decoder model "
        "(dump=on, force_contig=%s)",
        bool(_contig_active()),
    )


def _capturing() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _active() -> bool:
    return bool(WHISPER_DEBUG) or _auto


def _contig_active() -> bool:
    # Auto-enabling also runs the contiguity experiment.
    return bool(FORCE_CONTIG) or _auto


def _fmt(name: str, t: object) -> str:
    if t is None:
        return f"{name}=None"
    if isinstance(t, torch.Tensor):
        try:
            cont = t.is_contiguous()
        except Exception:
            cont = "?"
        s = (
            f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"stride={tuple(t.stride())} contig={cont}"
        )
        if t.numel() and t.is_floating_point():
            s += f" nan={bool(torch.isnan(t).any())} inf={bool(torch.isinf(t).any())}"
        elif t.numel():
            s += f" min={int(t.min())} max={int(t.max())}"
        return s
    return f"{name}={t!r}"


def dump_meta(tag: str, **tensors: object) -> None:
    if not _active() or _capturing():
        return
    n = _dump_counts.get(tag, 0)
    if n >= _MAX_DUMPS_PER_TAG:
        return
    _dump_counts[tag] = n + 1
    logger.info(
        "[whisper-dbg] %s #%d | %s",
        tag,
        n,
        " | ".join(_fmt(k, v) for k, v in tensors.items()),
    )


def maybe_contig(t: object) -> object:
    """Return a contiguous copy when the contiguity experiment is active."""
    if (
        _contig_active()
        and isinstance(t, torch.Tensor)
        and t.numel()
        and not t.is_contiguous()
    ):
        return t.contiguous()
    return t


def check_output_nan(layer_name: str, attn_type: str, output: object) -> None:
    """Per-layer probe: flag NaN/Inf in an attention layer's output."""
    if not _active() or _capturing():
        return
    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        return
    if torch.isnan(output).any() or torch.isinf(output).any():
        logger.warning(
            "[whisper-dbg] NaN/Inf in attn OUTPUT: layer=%s type=%s shape=%s",
            layer_name,
            attn_type,
            tuple(output.shape),
        )

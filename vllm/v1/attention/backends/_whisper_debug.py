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
_MAX_DUMPS_PER_TAG = 40
_nan_warns = 0
_MAX_NAN_WARNS = 40


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
        # Cap reductions: isnan/min/max over a huge tensor (e.g. the multi-GB KV
        # cache) allocates a same-size temporary and OOMs. Only reduce small ones.
        if t.numel() and t.numel() <= 5_000_000:
            if t.is_floating_point():
                nan = bool(torch.isnan(t).any())
                inf = bool(torch.isinf(t).any())
                s += f" nan={nan} inf={inf}"
            else:
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


def check_kv_nan(tag, key_cache, value_cache, block_table, seq_lens) -> None:
    """Check whether the cross-attn KV cache blocks referenced by request 0
    contain NaN/Inf. Bounded to the referenced blocks (capped) to avoid the
    full-cache OOM. Distinguishes a poisoned cache (write-side) from a
    kernel/read-side bug."""
    key = "KVNAN:" + tag
    if not _active() or _capturing():
        return
    n = _dump_counts.get(key, 0)
    if n >= _MAX_DUMPS_PER_TAG:
        return
    _dump_counts[key] = n + 1
    try:
        if seq_lens.numel() == 0 or block_table.numel() == 0:
            return
        length = int(seq_lens[0].item())
        if length <= 0:
            return
        block_size = key_cache.shape[1]
        num_blocks = min((length + block_size - 1) // block_size, 128)
        blk = block_table[0, :num_blocks].to(torch.long)
        k = key_cache[blk]
        v = value_cache[blk]
        kn = bool(torch.isnan(k).any()) or bool(torch.isinf(k).any())
        vn = bool(torch.isnan(v).any()) or bool(torch.isinf(v).any())
        logger.warning(
            "[whisper-dbg] %s KVcache nan/inf: k=%s v=%s (len=%d nblocks=%d)",
            tag,
            kn,
            vn,
            length,
            num_blocks,
        )
    except Exception as e:  # never break the model for a diagnostic
        logger.warning("[whisper-dbg] %s KV check error: %r", tag, e)


def check_output_nan(layer_name: str, attn_type: str, output: object) -> None:
    """Per-layer probe: flag NaN/Inf in an attention layer's output."""
    global _nan_warns
    if not _active() or _capturing() or _nan_warns >= _MAX_NAN_WARNS:
        return
    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        return
    if torch.isnan(output).any() or torch.isinf(output).any():
        _nan_warns += 1
        logger.warning(
            "[whisper-dbg] NaN/Inf in attn OUTPUT: layer=%s type=%s shape=%s",
            layer_name,
            attn_type,
            tuple(output.shape),
        )

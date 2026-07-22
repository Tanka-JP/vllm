#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import safetensors.torch
import soundfile as sf
import torch


def _load_native_modeling(checkpoint_dir: Path):
    package = "native_stepaudio_checkpoint"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(checkpoint_dir)]
        sys.modules[package] = module

    for name in ("configuration_step_audio_2", "modeling_step_audio_2"):
        full_name = f"{package}.{name}"
        if full_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(
            full_name,
            checkpoint_dir / f"{name}.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {full_name}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)

    return sys.modules[f"{package}.modeling_step_audio_2"]


def _decode_audio(raw: bytes, target_sr: int) -> np.ndarray:
    with io.BytesIO(raw) as fp:
        data, sr = sf.read(fp, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(data.astype(np.float32))


def _load_audio_state(
    shard_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    state = safetensors.torch.load_file(str(shard_path), device="cpu")
    encoder = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    adapter = {
        key.removeprefix("adapter."): value
        for key, value in state.items()
        if key.startswith("adapter.")
    }
    return encoder, adapter


def _parse_dtype(value: str) -> torch.dtype:
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    af = a.detach().float().cpu()
    bf = b.detach().float().cpu()
    if af.shape != bf.shape:
        return {
            "shape_a": list(af.shape),
            "shape_b": list(bf.shape),
            "same_shape": False,
        }
    diff = af - bf
    denom = torch.maximum(af.abs(), bf.abs()).clamp_min(1e-12)
    rel = diff.abs() / denom
    return {
        "shape": list(af.shape),
        "exact": bool(torch.equal(a.detach().cpu(), b.detach().cpu())),
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.abs().mean().item()) if diff.numel() else 0.0,
        "rms": float(math.sqrt(torch.mean(diff * diff).item()))
        if diff.numel()
        else 0.0,
        "max_rel": float(rel.max().item()) if rel.numel() else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/data/cbhua/step-audio-finetune/checkpoints"),
    )
    parser.add_argument(
        "--shard",
        type=Path,
        default=Path(
            "/data/liujun/stepaudio-merged/v25_C0_ep3/model-00001-of-00007.safetensors"
        ),
    )
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)

    native = _load_native_modeling(args.checkpoint_dir)
    from step_audio_finetune import _modeling_helpers

    from vllm.plugins.stepaudio_vllm24 import mm_step_audio as vllm_audio

    encoder_sd, adapter_sd = _load_audio_state(args.shard)
    native_encoder = (
        native.AudioEncoder(128, 1500, 1280, 20, 32)
        .to(device=device, dtype=dtype)
        .eval()
    )
    native_adapter = (
        native.Adaptor(1280, 5120, 3, 2).to(device=device, dtype=dtype).eval()
    )
    vllm_encoder = (
        vllm_audio.AudioEncoder(128, 1500, 1280, 20, 32)
        .to(device=device, dtype=dtype)
        .eval()
    )
    vllm_adapter = (
        vllm_audio.Adaptor(1280, 5120, 3, 2).to(device=device, dtype=dtype).eval()
    )
    native_encoder.load_state_dict(encoder_sd, strict=True)
    native_adapter.load_state_dict(adapter_sd, strict=True)
    vllm_encoder.load_state_dict(encoder_sd, strict=True)
    vllm_adapter.load_state_dict(adapter_sd, strict=True)

    for audio_path in args.audio:
        wav = _decode_audio(audio_path.read_bytes(), args.target_sr)
        wav_t = torch.from_numpy(wav).to(device=device)
        native_mel = _modeling_helpers.log_mel_spectrogram(wav_t)
        vllm_mel = vllm_audio._stepaudio_log_mel_spectrogram(wav_t, n_mels=128)
        lens = torch.tensor([native_mel.shape[-1]], device=device, dtype=torch.long)

        with torch.inference_mode():
            native_enc, native_len = native_encoder(
                native_mel.unsqueeze(0).to(dtype), lens
            )
            vllm_enc, vllm_len = vllm_encoder(vllm_mel.unsqueeze(0).to(dtype), lens)
            native_adapt = native_adapter(native_enc)
            vllm_adapt = vllm_adapter(vllm_enc)
            native_feat_len = (native_len - 1) // 2 + 1
            vllm_feat_len = (vllm_len - 1) // 2 + 1

        n_feat = int(min(native_feat_len[0].item(), vllm_feat_len[0].item()))
        print(
            json.dumps(
                {
                    "audio": str(audio_path),
                    "samples": int(len(wav)),
                    "mel_len": int(native_mel.shape[-1]),
                    "encoder_lens": {
                        "native": [int(x) for x in native_len.tolist()],
                        "vllm": [int(x) for x in vllm_len.tolist()],
                    },
                    "feature_lens": {
                        "native": [int(x) for x in native_feat_len.tolist()],
                        "vllm": [int(x) for x in vllm_feat_len.tolist()],
                    },
                    "mel": _stats(native_mel, vllm_mel),
                    "encoder": _stats(native_enc, vllm_enc),
                    "adapter_full": _stats(native_adapt, vllm_adapt),
                    "adapter_trim": _stats(
                        native_adapt[0, :n_feat],
                        vllm_adapt[0, :n_feat],
                    ),
                },
                ensure_ascii=False,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

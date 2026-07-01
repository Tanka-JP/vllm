#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from vllm.multimodal.utils import fetch_audio


def _native_decode(raw: bytes, target_sr: int) -> tuple[np.ndarray, int]:
    with io.BytesIO(raw) as fp:
        data, sr = sf.read(fp, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return np.ascontiguousarray(data.astype(np.float32)), int(sr)


def _vllm_decode(raw: bytes) -> tuple[np.ndarray, int]:
    audio_b64 = base64.b64encode(raw).decode("ascii")
    audio, sr = fetch_audio(f"data:audio/wav;base64,{audio_b64}")
    return np.asarray(audio, dtype=np.float32), int(sr)


def _compare(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    n = min(len(a), len(b))
    diff = a[:n].astype(np.float32) - b[:n].astype(np.float32)
    return {
        "len_equal": len(a) == len(b),
        "len_a": len(a),
        "len_b": len(b),
        "max_abs": float(np.max(np.abs(diff))) if n else None,
        "rms": float(np.sqrt(np.mean(diff * diff))) if n else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--target-sr", type=int, default=16000)
    args = parser.parse_args()

    all_exact = True
    for audio_path in args.audio:
        raw = audio_path.read_bytes()
        native_audio, native_sr = _native_decode(raw, args.target_sr)
        vllm_audio, vllm_sr = _vllm_decode(raw)
        comparison = _compare(native_audio, vllm_audio)
        exact = (
            native_sr == vllm_sr
            and comparison["len_equal"]
            and comparison["max_abs"] == 0.0
        )
        all_exact = all_exact and exact
        print(
            json.dumps(
                {
                    "audio": str(audio_path),
                    "exact": exact,
                    "native": {
                        "sr": native_sr,
                        "shape": list(native_audio.shape),
                        "min": float(native_audio.min()),
                        "max": float(native_audio.max()),
                    },
                    "vllm": {
                        "sr": vllm_sr,
                        "shape": list(vllm_audio.shape),
                        "min": float(vllm_audio.min()),
                        "max": float(vllm_audio.max()),
                    },
                    "native_vs_vllm": comparison,
                },
                ensure_ascii=False,
            )
        )

    print(json.dumps({"all_exact": all_exact}, ensure_ascii=False))
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())

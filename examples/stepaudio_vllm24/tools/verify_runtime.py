#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import torch


def _distribution_report(name: str) -> dict[str, object]:
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False}
    return {
        "installed": True,
        "version": dist.version,
        "location": str(dist.locate_file("")),
    }


def _module_report(name: str) -> dict[str, object]:
    spec = importlib.util.find_spec(name)
    return {
        "available": spec is not None,
        "origin": getattr(spec, "origin", None) if spec is not None else None,
    }


def _source_hit_report(package_root: Path, rel_path: str, needles: list[str]) -> dict[str, object]:
    path = package_root / rel_path
    if not path.exists():
        return {"exists": False, "path": str(path), "hits": {}}
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "exists": True,
        "path": str(path),
        "hits": {needle: needle in text for needle in needles},
    }


def _check_batch_invariant_support() -> dict[str, object]:
    import vllm
    from vllm import envs

    package_root = Path(vllm.__file__).parent
    return {
        "env_enabled": bool(envs.VLLM_BATCH_INVARIANT),
        "source": {
            rel_path: _source_hit_report(package_root, rel_path, needles)
            for rel_path, needles in {
                "model_executor/layers/batch_invariant.py": [
                    "init_batch_invariance",
                    "softmax_batch_invariant",
                    "_batch_invariant_LIB.impl",
                ],
                "model_executor/layers/layernorm.py": [
                    "VLLM_BATCH_INVARIANT",
                    "ops.fused_add_rms_norm",
                ],
                "v1/worker/gpu_worker.py": [
                    "init_batch_invariance",
                ],
                "v1/attention/backends/flash_attn.py": [
                    "VLLM_BATCH_INVARIANT",
                    "num_splits=1 if envs.VLLM_BATCH_INVARIANT",
                ],
                "v1/attention/backends/flashinfer.py": [
                    "VLLM_BATCH_INVARIANT",
                    "fixed_split_size",
                ],
                "v1/attention/backends/flex_attention.py": [
                    "VLLM_BATCH_INVARIANT",
                    "block_m = 16",
                    "block_n = 16",
                ],
            }.items()
        },
    }


def _check_cuda_matmul() -> dict[str, object]:
    x = torch.randn((64, 64), device="cuda", dtype=torch.float16)
    y = torch.randn((64, 64), device="cuda", dtype=torch.float16)
    z = x @ y
    torch.cuda.synchronize()
    return {"shape": list(z.shape), "dtype": str(z.dtype), "finite": bool(torch.isfinite(z).all())}


def _check_rms_norm() -> dict[str, object]:
    from vllm import _custom_ops as ops

    x = torch.randn((8, 128), device="cuda", dtype=torch.float16)
    w = torch.ones((128,), device="cuda", dtype=torch.float16)
    out = torch.empty_like(x)
    ops.rms_norm(out, x, w, 1e-6)
    torch.cuda.synchronize()
    return {"shape": list(out.shape), "dtype": str(out.dtype), "finite": bool(torch.isfinite(out).all())}


def _check_flash_attn(fa_version: int) -> dict[str, object]:
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    q = torch.randn((16, 8, 64), device="cuda", dtype=torch.float16)
    k = torch.randn((16, 8, 64), device="cuda", dtype=torch.float16)
    v = torch.randn((16, 8, 64), device="cuda", dtype=torch.float16)
    cu = torch.tensor([0, 16], device="cuda", dtype=torch.int32)
    out = flash_attn_varlen_func(
        q,
        k,
        v,
        16,
        cu,
        16,
        cu_seqlens_k=cu,
        dropout_p=0.0,
        causal=False,
        fa_version=fa_version,
    )
    torch.cuda.synchronize()
    return {"shape": list(out.shape), "dtype": str(out.dtype), "finite": bool(torch.isfinite(out).all())}


def _check_stepaudio_plugin(model_path: str) -> dict[str, object]:
    from vllm.config import ModelConfig
    from vllm.model_executor.models import ModelRegistry
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.plugins import load_general_plugins
    from vllm.tokenizers import get_tokenizer
    from vllm.transformers_utils.config import get_config

    load_general_plugins()

    cfg = get_config(model_path, trust_remote_code=False)
    tok = get_tokenizer(model_path, tokenizer_mode="step_audio_2", trust_remote_code=False)
    model_config = ModelConfig(
        model=model_path,
        tokenizer_mode="step_audio_2",
        trust_remote_code=False,
        dtype="bfloat16",
        max_model_len=8192,
        limit_mm_per_prompt={"audio": 1},
        interleave_mm_strings=True,
    )
    model_cls, _ = ModelRegistry.resolve_model_cls(
        cfg.architectures,
        model_config,
    )
    dummy = MULTIMODAL_REGISTRY.get_dummy_mm_inputs(model_config, {"audio": 1})
    mm_kwargs = dummy["mm_kwargs"]["audio"][0]

    return {
        "config_type": type(cfg).__name__,
        "model_type": cfg.model_type,
        "architectures": cfg.architectures,
        "tokenizer_type": type(tok).__name__,
        "audio_token_id": tok.convert_tokens_to_ids("<audio_patch>"),
        "eot_token_id": tok.convert_tokens_to_ids("<|EOT|>"),
        "model_cls": model_cls.__name__,
        "dummy_mm_fields": sorted(mm_kwargs.keys()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/liujun/stepaudio-merged/v25_C0_ep3")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    import vllm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    report: dict[str, object] = {
        "wheels": {
            pkg: _distribution_report(pkg)
            for pkg in [
                "vllm",
                "torch",
                "torchaudio",
                "torchvision",
                "flashinfer-python",
                "flashinfer-cubin",
                "triton",
                "nvidia-nccl-cu12",
                "nvidia-cudnn-cu12",
            ]
        },
        "extensions": {
            name: _module_report(name)
            for name in [
                "vllm._C_stable_libtorch",
                "vllm._custom_ops",
                "vllm._flashmla_C",
                "vllm.vllm_flash_attn._vllm_fa2_C",
                "vllm.vllm_flash_attn._vllm_fa3_C",
                "vllm.vllm_flash_attn.flash_attn_interface",
                "flashinfer",
            ]
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
        },
        "vllm": {"version": vllm.__version__},
        "cuda_matmul": _check_cuda_matmul(),
        "custom_ops": {"rms_norm": _check_rms_norm()},
        "flash_attn": {
            "fa2": _check_flash_attn(2),
            "fa3": _check_flash_attn(3),
        },
        "batch_invariant": _check_batch_invariant_support(),
        "stepaudio_plugin": _check_stepaudio_plugin(str(Path(args.model))),
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for section, value in report.items():
            print(f"{section}: {value}")


if __name__ == "__main__":
    main()

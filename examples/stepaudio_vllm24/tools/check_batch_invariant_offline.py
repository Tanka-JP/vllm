#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def _audio_data_url(path: Path) -> str:
    audio_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{audio_b64}"


def _message(audio_path: Path, system_prompt: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": _audio_data_url(audio_path)},
                }
            ],
        },
        {"role": "assistant", "content": "<think>\n"},
    ]


def _completion_row(output: Any) -> dict[str, Any]:
    completion = output.outputs[0]
    return {
        "text": completion.text,
        "token_ids": [int(token_id) for token_id in completion.token_ids],
        "finish_reason": completion.finish_reason,
        "stop_reason": completion.stop_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument(
        "--model-path", default="/data/liujun/stepaudio-merged/v25_C0_ep3"
    )
    parser.add_argument("--served-model-name", default="stepaudio-v25-c0-ep3")
    parser.add_argument("--native-src", default="/data/cbhua/step-audio-finetune/src")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, args.native_src)
    from step_audio_finetune.prompt import v5_v2three_thinking_instruction

    from vllm import LLM, SamplingParams

    system_prompt = v5_v2three_thinking_instruction({})
    messages = [_message(path, system_prompt) for path in args.audio]
    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        stop_token_ids=[151665],
        ignore_eos=True,
        skip_special_tokens=False,
    )
    llm = LLM(
        model=args.model_path,
        served_model_name=args.served_model_name,
        tokenizer_mode="step_audio_2",
        trust_remote_code=False,
        interleave_mm_strings=True,
        limit_mm_per_prompt={"audio": 1},
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=max(2, len(messages)),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
    )

    singles = [
        _completion_row(
            llm.chat(
                [message],
                sampling_params=params,
                add_generation_prompt=False,
                continue_final_message=True,
                use_tqdm=False,
            )[0]
        )
        for message in messages
    ]
    batched = [
        _completion_row(output)
        for output in llm.chat(
            messages,
            sampling_params=params,
            add_generation_prompt=False,
            continue_final_message=True,
            use_tqdm=False,
        )
    ]
    rows = []
    for audio_path, single, batch in zip(args.audio, singles, batched):
        rows.append(
            {
                "audio": str(audio_path),
                "exact_token_match": single["token_ids"] == batch["token_ids"],
                "exact_text_match": single["text"] == batch["text"],
                "single": single,
                "batch": batch,
            }
        )
    report = {
        "batch_invariant_env": os.environ.get("VLLM_BATCH_INVARIANT"),
        "max_tokens": args.max_tokens,
        "total": len(rows),
        "exact_token": sum(int(row["exact_token_match"]) for row in rows),
        "exact_text": sum(int(row["exact_text_match"]) for row in rows),
        "rows": rows,
    }
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["exact_token"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path
from urllib import request


def _post_json(url: str, payload: dict, timeout_s: float) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _vllm_generate(
    *,
    server_url: str,
    model: str,
    audio_path: Path,
    system_prompt: str,
    max_tokens: int,
    timeout_s: float,
) -> dict:
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                    }
                ],
            },
            {"role": "assistant", "content": "<think>\n"},
        ],
        "add_generation_prompt": False,
        "continue_final_message": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "stop_token_ids": [151665],
        "ignore_eos": True,
        "skip_special_tokens": False,
        "return_token_ids": True,
    }
    data = _post_json(
        f"{server_url.rstrip('/')}/chat/completions",
        payload,
        timeout_s,
    )
    choice = data["choices"][0]
    return {
        "text": choice["message"]["content"],
        "token_ids": choice.get("token_ids") or [],
        "finish_reason": choice.get("finish_reason"),
        "stop_reason": choice.get("stop_reason"),
        "usage": data.get("usage") or {},
    }


_TAG_RE = re.compile(r"<\|(?:emotion|intensity|style)=[^>]+?\|>")


def _tag_signature(text: str) -> str:
    return "".join(_TAG_RE.findall(text))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--server-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--model", default="stepaudio-v25-c0-ep3")
    parser.add_argument("--native-src", default="/data/cbhua/step-audio-finetune/src")
    parser.add_argument("--native-checkpoint", default="/data/liujun/stepaudio-merged/v25_C0_ep3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--out-jsonl", type=Path)
    parser.add_argument("--require-token-exact", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, args.native_src)

    from step_audio_finetune.inference import StepAudioR1Runner
    from step_audio_finetune.prompt import v5_v2three_thinking_instruction

    system_prompt = v5_v2three_thinking_instruction({})
    runner = StepAudioR1Runner(args.native_checkpoint, device=args.device)
    tokenizer = runner.tokenizer

    out_fp = None
    if args.out_jsonl is not None:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        out_fp = args.out_jsonl.open("w", encoding="utf-8")

    exact = 0
    tag_exact = 0
    rows = []
    try:
        for audio_path in args.audio:
            t0 = time.time()
            native = runner.generate_for_audio(
                audio_path.read_bytes(),
                {},
                instruction_template=v5_v2three_thinking_instruction,
                max_new_tokens=args.max_tokens,
                use_think_stop=True,
                think_stop_tail_tokens=128,
                add_thinking_prefix=True,
            )
            native_tail = native["pred_raw"]
            if native_tail.startswith("<think>\n"):
                native_tail = native_tail[len("<think>\n") :]
            native_ids = tokenizer(native_tail, add_special_tokens=False)["input_ids"]

            vllm = _vllm_generate(
                server_url=args.server_url,
                model=args.model,
                audio_path=audio_path,
                system_prompt=system_prompt,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout_s,
            )
            is_exact = list(native_ids) == list(vllm["token_ids"])
            is_tag_exact = _tag_signature(native_tail) == _tag_signature(vllm["text"])
            exact += int(is_exact)
            tag_exact += int(is_tag_exact)
            row = {
                "audio": str(audio_path),
                "exact_token_match": is_exact,
                "tag_exact_match": is_tag_exact,
                "native": {
                    "text": native_tail,
                    "token_ids": native_ids,
                    "n_new_tokens": native["n_new_tokens"],
                    "gen_seconds": native["gen_seconds"],
                    "tokens_per_second": native["tokens_per_second"],
                    "prompt_token_count": native["prompt_token_count"],
                },
                "vllm": vllm,
                "elapsed_seconds": time.time() - t0,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "audio": str(audio_path),
                        "exact_token_match": is_exact,
                        "tag_exact_match": is_tag_exact,
                        "native_text": native_tail,
                        "vllm_text": vllm["text"],
                        "native_tokens": len(native_ids),
                        "vllm_tokens": len(vllm["token_ids"]),
                    },
                    ensure_ascii=False,
                )
            )
            if out_fp is not None:
                out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_fp.flush()
    finally:
        if out_fp is not None:
            out_fp.close()

    summary = {
        "total": len(rows),
        "exact": exact,
        "exact_rate": exact / len(rows) if rows else 0.0,
        "tag_exact": tag_exact,
        "tag_exact_rate": tag_exact / len(rows) if rows else 0.0,
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False))
    if args.require_token_exact and exact != len(rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

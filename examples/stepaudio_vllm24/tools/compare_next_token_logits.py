#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from urllib import request

import torch


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
    assistant_prefill: str,
    max_tokens: int,
    timeout_s: float,
    logprobs: bool = False,
    top_logprobs: int = 20,
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
            {"role": "assistant", "content": assistant_prefill},
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
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
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
        "logprobs": choice.get("logprobs"),
    }


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for idx, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return idx
    return n


def _top_logprobs(
    tokenizer,
    logits: torch.Tensor,
    *,
    candidate_ids: list[int],
    top_k: int,
) -> tuple[list[dict], dict[str, float]]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    values, ids = torch.topk(log_probs, k=top_k)
    top = [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
            "logprob": float(value),
        }
        for value, token_id in zip(values.tolist(), ids.tolist())
    ]
    candidates = {
        str(int(token_id)): float(log_probs[int(token_id)].item())
        for token_id in candidate_ids
    }
    return top, candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--server-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--model", default="stepaudio-v25-c0-ep3")
    parser.add_argument("--native-src", default="/data/cbhua/step-audio-finetune/src")
    parser.add_argument("--native-checkpoint", default="/data/liujun/stepaudio-merged/v25_C0_ep3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--out-jsonl", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, args.native_src)

    from step_audio_finetune import _modeling_helpers
    from step_audio_finetune.data import compute_audio_patch_count, decode_audio_bytes
    from step_audio_finetune.inference import StepAudioR1Runner
    from step_audio_finetune.prompt import (
        build_tagged_text_prompt,
        v5_v2three_thinking_instruction,
    )

    system_prompt = v5_v2three_thinking_instruction({})
    runner = StepAudioR1Runner(args.native_checkpoint, device=args.device)
    tokenizer = runner.tokenizer
    device = torch.device(args.device)

    out_fp = None
    if args.out_jsonl is not None:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        out_fp = args.out_jsonl.open("w", encoding="utf-8")

    try:
        for audio_path in args.audio:
            raw = audio_path.read_bytes()
            native = runner.generate_for_audio(
                raw,
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

            vllm_full = _vllm_generate(
                server_url=args.server_url,
                model=args.model,
                audio_path=audio_path,
                system_prompt=system_prompt,
                assistant_prefill="<think>\n",
                max_tokens=args.max_tokens,
                timeout_s=args.timeout_s,
            )
            vllm_ids = [int(x) for x in vllm_full["token_ids"]]
            common = _common_prefix_len(native_ids, vllm_ids)
            exact = native_ids == vllm_ids

            row = {
                "audio": str(audio_path),
                "exact_token_match": exact,
                "common_prefix_len": common,
                "native": {
                    "text": native_tail,
                    "token_ids": native_ids,
                    "prompt_token_count": native["prompt_token_count"],
                },
                "vllm": vllm_full,
            }

            if not exact and common < len(native_ids) and common < len(vllm_ids):
                common_ids = native_ids[:common]
                common_text = tokenizer.decode(common_ids, skip_special_tokens=False)
                retokenized = tokenizer(common_text, add_special_tokens=False)["input_ids"]
                native_next_id = int(native_ids[common])
                vllm_next_id = int(vllm_ids[common])

                wav = decode_audio_bytes(raw)
                n_patches = compute_audio_patch_count(wav)
                prompt_text = build_tagged_text_prompt(
                    {},
                    n_audio_patches=n_patches,
                    instruction_template=v5_v2three_thinking_instruction,
                    transcript=None,
                    add_thinking_prefix=True,
                )
                prompt_ids = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"].to(device)
                ctx_ids = torch.tensor(
                    [prompt_ids[0].tolist() + common_ids],
                    dtype=torch.long,
                    device=device,
                )
                attn = torch.ones_like(ctx_ids)
                wav_t = torch.from_numpy(wav).to(device)
                mel = _modeling_helpers.log_mel_spectrogram(wav_t).to(runner.dtype)
                wavs = mel.unsqueeze(0)
                wav_lens = torch.tensor([mel.shape[-1]], dtype=torch.long, device=device)
                with torch.inference_mode():
                    out = runner.model(
                        input_ids=ctx_ids,
                        attention_mask=attn,
                        wavs=wavs,
                        wav_lens=wav_lens,
                    )
                    logits = out.logits[0, -1]
                native_top, native_candidate_logprobs = _top_logprobs(
                    tokenizer,
                    logits,
                    candidate_ids=[native_next_id, vllm_next_id],
                    top_k=args.top_k,
                )

                vllm_next = _vllm_generate(
                    server_url=args.server_url,
                    model=args.model,
                    audio_path=audio_path,
                    system_prompt=system_prompt,
                    assistant_prefill="<think>\n" + common_text,
                    max_tokens=1,
                    timeout_s=args.timeout_s,
                    logprobs=True,
                    top_logprobs=args.top_k,
                )

                row["next_token"] = {
                    "common_prefix_text": common_text,
                    "common_prefix_retokenizes_exact": retokenized == common_ids,
                    "native_next": {
                        "token_id": native_next_id,
                        "token": tokenizer.decode([native_next_id], skip_special_tokens=False),
                    },
                    "vllm_next": {
                        "token_id": vllm_next_id,
                        "token": tokenizer.decode([vllm_next_id], skip_special_tokens=False),
                    },
                    "native_top_logprobs": native_top,
                    "native_candidate_logprobs": native_candidate_logprobs,
                    "vllm_next_probe": vllm_next,
                }

            print(json.dumps(row, ensure_ascii=False))
            if out_fp is not None:
                out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_fp.flush()
    finally:
        if out_fp is not None:
            out_fp.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

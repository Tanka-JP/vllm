#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib import request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def _native_top_logprobs(
    *,
    model_path: str,
    prompt: str,
    device: str,
    top_k: int,
) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    ).eval()
    input_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0, -1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        values, indices = torch.topk(logprobs, top_k)

    return {
        "prompt_tokens": int(input_ids.shape[-1]),
        "top_logprobs": [
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "logprob": float(logprob),
            }
            for logprob, token_id in zip(values, indices)
        ],
    }


def _vllm_next_token(
    *,
    server_url: str,
    model: str,
    prompt: str,
    top_k: int,
    timeout_s: float,
) -> dict[str, object]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "logprobs": top_k,
        "skip_special_tokens": False,
    }
    data = _post_json(
        f"{server_url.rstrip('/')}/completions",
        payload,
        timeout_s,
    )
    choice = data["choices"][0]
    return {
        "text": choice.get("text"),
        "logprobs": choice.get("logprobs"),
        "usage": data.get("usage"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/data/liujun/stepaudio-merged/v25_C0_ep3")
    parser.add_argument("--server-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--served-model-name", default="stepaudio-v25-c0-ep3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument(
        "--prompt",
        default=(
            "<|BOT|>system\nYou are a helpful assistant.<|EOT|>"
            "<|BOT|>human\nこんにちは<|EOT|>"
            "<|BOT|>assistant\n<think>\n"
        ),
    )
    args = parser.parse_args()

    native = _native_top_logprobs(
        model_path=args.model_path,
        prompt=args.prompt,
        device=args.device,
        top_k=args.top_k,
    )
    vllm = _vllm_next_token(
        server_url=args.server_url,
        model=args.served_model_name,
        prompt=args.prompt,
        top_k=args.top_k,
        timeout_s=args.timeout_s,
    )

    report = {
        "prompt": args.prompt,
        "native": native,
        "vllm": vllm,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

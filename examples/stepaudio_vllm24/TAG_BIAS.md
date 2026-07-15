# StepAudio tag precision/recall steering

This document covers the request-scoped tag-presence bias supported by the
StepAudio vLLM 0.24 plugin. It ports the production HF
`TagPresenceLogitsProcessor` behavior to vLLM's V1 logits processor API.

The control changes the decoding operating point without retraining:

- a positive bias makes tag emission more likely and moves toward recall;
- a negative bias makes tag emission less likely and moves toward precision;
- zero, or omitting the arguments, does not modify logits.

This is the tag-*presence* control. It decides whether to open a tag and can
steer which tag family or point marker follows `<|`. It does not implement the
separate long-tail label-selection adjustment that chooses a value after a
`|family=` anchor.

## Start the server

`serve_stepaudio.sh` registers the processor by default:

```bash
cd /data/liujun/tmp/vllm-stepaudio-tag-bias

MODEL_PATH=/data/liujun/stepaudio-v38-ep4-merged \
SERVED_MODEL_NAME=stepaudio-v38-ep4 \
CUDA_VISIBLE_DEVICES=0 \
examples/stepaudio_vllm24/serve_stepaudio.sh
```

The equivalent vLLM option is:

```text
--logits-processors vllm.plugins.stepaudio_vllm24.logits_processor:StepAudioTagBiasLogitsProcessor
```

Set `ENABLE_TAG_BIAS_PROCESSOR=0` to start a control server without registering
the processor. The StepAudio architecture already uses vLLM's V1 model runner;
custom logits processors are not compatible with speculative decoding in vLLM
0.24.

## Request API

Pass controls in the OpenAI-compatible chat request's `vllm_xargs` object.

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `stepaudio_tag_bias` | number | `0` | Global bias on the tag-open tokens `<` and merged `><` |
| `stepaudio_tag_bias_per_family` | list of strings | `[]` | Entries formatted as `name=bias`, applied after `<|` |
| `stepaudio_tag_bias_gate` | string | `after_think` | `after_think` or `always` |

Bias values must be finite and within `[-20, 20]`. Family or marker names use
letters, digits, and underscores, beginning with a letter. Duplicate names are
rejected.

The default `after_think` gate waits until the generated output contains the
first two tokenizer IDs of `</think>`. This protects the reasoning section and
matches the native HF inference path. Use `always` only with a template that
does not generate `</think>`.

Global example:

```json
{
  "model": "stepaudio-v38-ep4",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}},
        {"type": "text", "text": "请转写音频并标注副语言标签。"}
      ]
    }
  ],
  "temperature": 0,
  "max_tokens": 512,
  "vllm_xargs": {
    "stepaudio_tag_bias": 3
  }
}
```

Useful initial sweep points for v38 ep4 are:

| Operating point | Request value | Expected direction |
| --- | ---: | --- |
| Precision-first | `-3` | fewer emitted tags |
| Baseline | `0` or omitted | unchanged logits |
| Recall-first | `3` | more emitted tags |

These values are evaluation conditions, not universal presets. Freeze a
production value only after measuring the target dataset's precision, recall,
F1, and over-emission rate.

## Per-family and marker steering

Per-family controls fire only immediately after the model has emitted `<|`.
They bias the first tokenizer token of a family or point-marker name:

```json
{
  "vllm_xargs": {
    "stepaudio_tag_bias": 0,
    "stepaudio_tag_bias_per_family": [
      "breath=3",
      "laugh=1.5",
      "emotion=-1"
    ]
  }
}
```

`style` and the legacy `extra_emotion` spelling both steer the first name
tokens for `<|style=` and `<|extra_emotion=`. Other valid names are treated as
point markers, such as `breath`, `laugh`, `sigh`, or `gasp`.

Some names share the same first tokenizer token. When requested entries
collide, the larger absolute bias wins for that token. Use tokenizer-level and
metric-level validation before deploying a new marker list.

## Python client

The OpenAI Python client's `extra_body` field carries `vllm_xargs`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8010/v1", api_key="unused")
response = client.chat.completions.create(
    model="stepaudio-v38-ep4",
    messages=messages,
    temperature=0,
    max_tokens=512,
    extra_body={
        "vllm_xargs": {
            "stepaudio_tag_bias": 3,
            "stepaudio_tag_bias_per_family": ["breath=3"],
        }
    },
)
```

The list-valued per-family option is supported by the chat-completions and
Responses protocols. StepAudio audio inference uses chat completions; the
legacy text-completions protocol accepts only scalar `vllm_xargs` values.

## LoRA interaction

The processor runs after the selected base or LoRA model has produced logits,
so the same controls work with both model names on a LoRA-enabled server:

- `"model": "stepaudio-base"` applies the control to base-model logits;
- `"model": "v38-ep4"` applies it to logits after the v38 ep4 LoRA.

Register the processor together with the normal LoRA flags:

```bash
vllm serve /data/cbhua/step-audio-finetune/checkpoints \
  --served-model-name stepaudio-base \
  --tokenizer-mode step_audio_2 \
  --limit-mm-per-prompt '{"audio":1}' \
  --interleave-mm-strings \
  --enforce-eager \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules v38-ep4=/data/cbhua/step-audio-finetune/outputs/sft_v38_main/checkpoint-18332 \
  --logits-processors vllm.plugins.stepaudio_vllm24.logits_processor:StepAudioTagBiasLogitsProcessor
```

Do not load the v38 ep4 LoRA on top of
`/data/liujun/stepaudio-v38-ep4-merged`; that checkpoint already contains the
adapter weights.

## Validation checklist

Before production use:

1. Send the same deterministic request with the argument omitted and with
   `stepaudio_tag_bias=0`; token IDs and text should match.
2. Send `-3`, `0`, and `3` in one concurrent batch and confirm that each request
   keeps its own operating point.
3. Compare native HF and vLLM on the same tokenizer, checkpoint, prompt,
   sampling parameters, and bias values.
4. Repeat the sweep for both the base model and any served LoRA name.
5. Run the canonical measurement matrix before freezing a production value;
   a smoke test proves wiring, not metric quality.

The processor probes token IDs from the loaded tokenizer at startup rather
than hard-coding checkpoint-specific IDs. For the current v38 ep4 tokenizer,
the expected probes are `<` = `27`, merged `><` = `1784`, and the first two
`</think>` IDs = `[522, 26865]`.

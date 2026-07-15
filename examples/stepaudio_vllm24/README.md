# StepAudio vLLM 0.24 plugin

This branch starts from upstream `vllm==0.24.0` and registers StepAudio2 support
as an in-tree vLLM general plugin. It is meant for the CUDA 12.8 PyTorch wheel
stack while keeping the StepAudio changes scoped to model/plugin code.

Migration scope:

- keep upstream `vllm==0.24.0` as the base release;
- use the CUDA 12.8 PyTorch wheel stack (`torch==2.11.0+cu128`);
- register StepAudio2 as `vllm.plugins.stepaudio_vllm24`;
- port only StepAudio-specific config, tokenizer, chat rendering, audio
  placeholder replacement, audio encoder, and `StepAudio2ForCausalLM`;
- do not port the old fork's broad OpenAI serving/TTS parser changes unless a
  StepAudio labeling use case needs them.

Thinking Machines determinism scope:

- vLLM PR 24583 was a closed, unmerged proof of concept titled
  `[Proof of Concept] Made vllm deterministic (tested for qwen3-8B)`;
- the PoC changed only `layernorm.py`, `flex_attention.py`, and
  `gpu_worker.py`;
- upstream vLLM 0.24 already carries a maintained batch-invariant path behind
  `VLLM_BATCH_INVARIANT=1`, so this branch does not apply the old diff
  directly;
- the 0.24 path maps the PoC requirements to `batch_invariant.py`,
  `layernorm.py`, FlashAttention/FlashInfer fixed split settings, and
  `gpu_worker.py::init_batch_invariance`;
- FlexAttention remains an experiment backend for A/B checks, not the default
  serving backend for StepAudio labels.

Start one server:

```bash
cd /path/to/vllm
CUDA_VISIBLE_DEVICES=0 examples/stepaudio_vllm24/serve_stepaudio.sh
```

Useful overrides:

```bash
PORT=8011 CUDA_VISIBLE_DEVICES=1 examples/stepaudio_vllm24/serve_stepaudio.sh
MAX_NUM_SEQS=16 GPU_MEMORY_UTILIZATION=0.70 examples/stepaudio_vllm24/serve_stepaudio.sh
VLLM_BATCH_INVARIANT=1 examples/stepaudio_vllm24/serve_stepaudio.sh
```

Tag precision/recall steering:

The server helper registers a request-scoped StepAudio tag-presence logits
processor. Requests without steering arguments remain a no-op; pass a bias in
`vllm_xargs` to move the operating point toward precision or recall. See
[TAG_BIAS.md](./TAG_BIAS.md) for the API, per-family controls, LoRA interaction,
and validation commands.

Runtime verification:

```bash
TMPDIR=/data/liujun/tmp/tmp CUDA_VISIBLE_DEVICES=0 \
  uv run --project . python \
  examples/stepaudio_vllm24/tools/verify_runtime.py \
  --json-out /data/liujun/tmp/stepaudio-vllm24_verify_runtime.json \
  --json
```

Batch-invariant single-vs-batch smoke, useful for validating the upstream
replacement for the old Thinking Machines determinism PoC on the StepAudio
path:

```bash
TMPDIR=/data/liujun/tmp/tmp VLLM_RPC_BASE_PATH=/dev/shm \
VLLM_BATCH_INVARIANT=1 CUDA_VISIBLE_DEVICES=1 \
  uv run --project . python \
  examples/stepaudio_vllm24/tools/check_batch_invariant_offline.py \
  --max-tokens 16 \
  --out-json /data/liujun/tmp/stepaudio-vllm24_batch_invariant_smoke.json \
  /path/to/audio_1.wav /path/to/audio_2.wav
```

Native parity verification, with the vLLM server already running on GPU 0 and
the native HF runner loaded on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 STEPAUDIO_DEVICE=cuda:0 \
  /data/cbhua/step-audio-finetune/.venv/bin/python \
  examples/stepaudio_vllm24/tools/compare_native_vllm_api.py \
  --device cuda:0 \
  --out-jsonl /data/liujun/tmp/stepaudio-vllm24_parity.jsonl \
  /path/to/audio_1.wav /path/to/audio_2.wav
```

Audio decode parity, useful when API data URLs are suspected:

```bash
TMPDIR=/data/liujun/tmp/tmp \
  uv run --project . python \
  examples/stepaudio_vllm24/tools/compare_audio_decode.py \
  /path/to/audio_1.wav /path/to/audio_2.wav
```

Audio tower parity, useful before blaming the text decoder:

```bash
PYTHONPATH=/data/cbhua/step-audio-finetune/src \
TMPDIR=/data/liujun/tmp/tmp CUDA_VISIBLE_DEVICES=1 \
  uv run --project . python \
  examples/stepaudio_vllm24/tools/compare_audio_tower.py \
  /path/to/audio_1.wav /path/to/audio_2.wav
```

Next-token native/vLLM logprob comparison at the first drift point:

```bash
CUDA_VISIBLE_DEVICES=1 STEPAUDIO_DEVICE=cuda:0 \
  /data/cbhua/step-audio-finetune/.venv/bin/python \
  examples/stepaudio_vllm24/tools/compare_next_token_logits.py \
  --device cuda:0 \
  --out-jsonl /data/liujun/tmp/stepaudio-vllm24_next_token.jsonl \
  /path/to/audio_1.wav /path/to/audio_2.wav
```

Text-only native/vLLM next-token comparison, useful for separating decoder
math drift from audio conditioning drift:

```bash
CUDA_VISIBLE_DEVICES=1 STEPAUDIO_DEVICE=cuda:0 \
  /data/cbhua/step-audio-finetune/.venv/bin/python \
  examples/stepaudio_vllm24/tools/compare_text_only_logits.py \
  --device cuda:0 \
  --out-json /data/liujun/tmp/stepaudio-vllm24_text_only_logits.json
```

Audio-conditioned layer probe, using chuanbo's native HF environment for the
baseline dump and the vLLM 0.24 environment for the vLLM side:

```bash
CUDA_VISIBLE_DEVICES=1 STEPAUDIO_DEVICE=cuda:0 \
  /data/cbhua/step-audio-finetune/.venv/bin/python \
  examples/stepaudio_vllm24/tools/compare_audio_conditioned_layers.py \
  --device cuda:0 \
  --native-only \
  --trace-layer0-sublayers \
  --trace-layer0-weights \
  --audio /path/to/audio.wav \
  --native-pt-out /data/liujun/tmp/stepaudio-native_layer.pt

TMPDIR=/data/liujun/tmp/tmp VLLM_RPC_BASE_PATH=/dev/shm \
CUDA_VISIBLE_DEVICES=1 \
  uv run --project . python \
  examples/stepaudio_vllm24/tools/compare_audio_conditioned_layers.py \
  --device cuda:0 \
  --trace-layer0-sublayers \
  --native-pt-in /data/liujun/tmp/stepaudio-native_layer.pt \
  --out-json /data/liujun/tmp/stepaudio-vllm24_layer_probe.json
```

Optional diagnostic switches for the vLLM-side pass:

```bash
--hf-style-rmsnorm
--hf-style-layer0-rope
--hf-style-layer0-sdpa
--hf-style-layer0-sdpa-backend auto|math|flash|mem_efficient|cudnn
```

Latest local run, 2026-07-01 UTC:

- PyPI and GitHub latest `vllm` release checked: `0.24.0`.
- vLLM PR 24583 checked through the GitHub API: it is closed, unmerged, and its
  file list is limited to `layernorm.py`, `flex_attention.py`, and
  `gpu_worker.py`.
- Runtime verification passed for `torch==2.11.0+cu128`, vLLM custom
  `rms_norm`, FlashAttention 2, FlashAttention 3, StepAudio config/tokenizer
  registration, and dummy audio multimodal inputs.
- Runtime verification now also records the wheel and extension provenance. In
  the current environment, FA2 and FA3 come from the upstream vLLM wheel at
  `vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so` and
  `vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so`; no separate source rebuild of
  `vllm-flash-attn` is required for this CUDA 12.8 wheel stack.
- Runtime verification now supports `--json-out` for a clean machine-readable
  report even when vLLM/Transformers log to stdout. The latest clean artifact is
  `/data/liujun/tmp/stepaudio-vllm24_verify_runtime_latest_clean.json`.
- With `VLLM_BATCH_INVARIANT=1`, the same runtime verification reported
  `env_enabled=true` and found the upstream 0.24 batch-invariant hooks in
  `batch_invariant.py`, `layernorm.py`, `gpu_worker.py`, FlashAttention,
  FlashInfer, and FlexAttention. This is the maintained upstream replacement
  for directly applying the closed/unmerged PR 24583 diff.
- `VLLM_BATCH_INVARIANT=1` offline StepAudio smoke passed on GPU 1 with
  `max_num_batched_tokens=8192`, `max_num_seqs=1`, `enforce_eager=True`, and
  `enable_prefix_caching=False`; logs showed the CUDA softmax override from
  `batch_invariant.py`, FlashAttention 3 selection, successful 7-shard weight
  load, and an 8-token decode.
- `check_batch_invariant_offline.py` was run with `VLLM_BATCH_INVARIANT=1`,
  `max_num_seqs=2`, two drift samples, and `max_tokens=16`
  (`/data/liujun/tmp/stepaudio-vllm24_batch_invariant_smoke.json`). Both
  samples were exact for single-vs-batch token ids and text (`2/2`), and the
  log showed the batch-invariant CUDA softmax override registration.
- Native HF vs current vLLM API parity on 20 short anime wav samples
  (`/data/liujun/tmp/stepaudio-vllm24_parity_20_rerun_current.jsonl`):
  token-exact `18/20`; emotion/intensity/style tag-exact `20/20`.
- The two token drift samples were rerun after restarting the current sidecar:
  prompt token counts still matched native (`798` and `802`), tag exact stayed
  `2/2`, and token exact stayed `0/2`.
- For those two drift samples, native/soundfile decode and vLLM `fetch_audio`
  decode were byte-for-float identical (`max_abs=0.0`, equal lengths), so the
  remaining drift is in the vLLM model execution path rather than API audio
  decoding or prompt length.
- For the same two drift samples, native and vLLM24 audio towers were exact at
  every checked boundary: log-mel, `AudioEncoder`, full `Adaptor`, and trimmed
  audio embeddings all had `max_abs=0.0`.
- `VLLM_BATCH_INVARIANT=1` and forced `TRITON_ATTN` both failed to recover
  token-exact native output on the two drift samples; labels stayed aligned.
- Forced `FLEX_ATTENTION` also failed to recover token-exact native output on
  the same two drift samples; both completions matched the current FA3 path.
- A text-only next-token probe, using no audio input, matched the native/vLLM
  prompt length (`21`) and top next token (`Okay`); top-20 ranking was broadly
  aligned with only small logprob drift, so no gross text decoder config or
  weight mapping error was found.
- Repeating the current API request for `anime_v2_0172__old_model_SE.wav`
  8 times produced one unique token sequence after warmup.
- Chuanbo-native-vs-vLLM audio-conditioned layer probes were run with native
  hidden states dumped from `/data/cbhua/step-audio-finetune/.venv` and vLLM
  hidden states dumped from the CUDA 12.8 vLLM 0.24 environment:
  - `/data/liujun/tmp/stepaudio-vllm24_layer_probe_0006_chuanbo_native.json`:
    prompt tokens matched (`822` vs `822`), native candidate logits ranked `き`
    id `49734` over `あ` id `29491` (`16.375` vs `16.125`), and vLLM generated
    `29491`. The first layer over `1e-2` max error was decoder layer `0`
    (`max_abs=0.03125`, `cosine=0.9999882`); final hidden max error was
    `2.8125`.
  - `/data/liujun/tmp/stepaudio-vllm24_layer_probe_0172_chuanbo_native.json`:
    prompt tokens matched (`823` vs `823`), native candidate logits ranked
    `来了` id `101161` over vLLM token id `60596` (`18.0` vs `17.875`), and
    vLLM generated `60596`. The first layer over `1e-2` max error was decoder
    layer `0` (`max_abs=0.015625`, `cosine=0.9999945`); final hidden max error
    was `1.0`.
- A follow-up layer 0 sublayer probe for `anime_v2_0006__old_model_SE.wav`
  (`/data/liujun/tmp/stepaudio-vllm24_layer0_sublayers_0006.json`) found:
  `layer_input` exact (`max_abs=0.0`), `input_layernorm` below the `1e-2`
  threshold (`max_abs=0.0078125`), then the first execution-order sublayer over
  threshold at `q_proj` (`max_abs=0.03125`, `mean_abs=0.0004473`,
  `cosine=1.0000001`). `k_proj` also showed max error `0.0625`. This points at
  layer 0 normalization/position/attention math rather than input assembly.
- A QKV attribution rerun with native layer 0 weights
  (`/data/liujun/tmp/stepaudio-vllm24_layer0_qkv_attr_0006.json`) ruled out the
  packed QKV layout as the direct cause: vLLM packed QKV matched manual
  `F.linear`, native and vLLM Q/K/V weights and biases matched, and the same
  input through native-vs-vLLM Q/K/V weights was exact.
- Replacing vLLM RMSNorm with an HF-style fp32-variance RMSNorm for the probe
  made `input_layernorm`, `q_proj`, `k_proj`, and `v_proj` exact on layer 0.
  The first remaining sublayer over `1e-2` moved to RoPE
  (`q_rope max_abs=0.03125`, `k_rope max_abs=0.03125`) when compared against
  chuanbo-native
  (`/data/liujun/tmp/stepaudio-native_layer0_deep_0006_chuanboenv.pt`).
- Adding an HF-style layer 0 RoPE implementation to the probe made `q_rope` and
  `k_rope` exact against chuanbo-native. The first remaining layer 0 drift then
  moved to `attn_output` (`max_abs=0.015625`), while the generated first token
  still stayed on the vLLM side (`29491`, `あ`).
- Chuanbo native is runtime-sensitive. The same native probe in the sidecar
  CUDA 12.8 environment (`torch==2.11.0+cu128`, `transformers==5.12.1`) flips
  the first-token candidate logits on `anime_v2_0006`: `あ` becomes higher than
  `き` (`16.25` vs `15.875`). The chuanbo environment
  (`torch==2.6.0+cu124`, `transformers==5.6.2`) keeps the original ranking
  (`き` over `あ`, `16.375` vs `16.125`), so precision validation must use the
  chuanbo `.venv` dump as the baseline.
- With `--hf-style-rmsnorm --hf-style-layer0-rope --hf-style-layer0-sdpa`,
  vLLM layer 0 is below the `1e-2` sublayer threshold against a sidecar-native
  dump (`attn_output max_abs=0.001953125`,
  `layer_output max_abs=0.00390625`); the first over-threshold layer moves to
  decoder layer 2. Against the chuanbo `.venv` native dump, the same probe still
  shows `attn_output max_abs=0.015625`. That isolates the remaining chuanbo
  parity gap to the old torch2.6/cu124 SDPA runtime versus the new
  torch2.11/cu128 runtime, not to StepAudio audio features, prompt assembly,
  QKV weight mapping, or RoPE formula.
- SDPA backend A/B on the same chuanbo-native dump did not find a torch2.11
  backend that recovers the old torch2.6 output. `auto`, `flash`, `math`, and
  `cudnn` all kept the first layer 0 drift at `attn_output max_abs=0.015625`
  and generated the vLLM-side first token `29491`; `mem_efficient` is not
  available for this dense GQA shape because Q heads (`40`) and KV heads (`8`)
  differ, ending with `RuntimeError: No available kernel`.
- Next-token logprob comparison at the first drift point shows a small-margin
  candidate swap, not a prompt/audio mismatch:
  - `anime_v2_0006`: native ranks `き` over `あ` by about `0.25` logprob;
    vLLM ranks `あ` over `き` by about `0.44`.
  - `anime_v2_0172`: native ranks `来了` over the vLLM continuation token by
    about `0.13`; vLLM ranks its continuation token over `来了` by about `0.25`.
- `VLLM_BATCH_INVARIANT=1` smoke passed with upstream vLLM 0.24's maintained
  batch-invariant implementation; it initialized, profiled, and decoded through
  the StepAudio plugin.

Defaults:

- model path: `/data/liujun/stepaudio-merged/v25_C0_ep3`
- served model name: `stepaudio-v25-c0-ep3`
- tokenizer mode: `step_audio_2`
- max model length: `8192`
- max batched tokens: `8192`
- max sequences: `8`
- GPU memory utilization: `0.55`
- multimodal limit: `{"audio":1}`

Operational notes:

- `TMPDIR` defaults to `/data/liujun/tmp/tmp` because `/tmp` has been full in
  this environment.
- `VLLM_RPC_BASE_PATH` defaults to `/dev/shm` because the CPFS-backed `/data`
  path cannot bind vLLM's `ipc://` sockets.
- `VLLM_USE_FLASHINFER_SAMPLER=0` is kept by default for easier parity checks.
- `VLLM_BATCH_INVARIANT=1` enables upstream vLLM's batch-invariant path, the
  maintained replacement for the old Thinking Machines PoC patch.
- The remaining chuanbo-token drift is a small-margin runtime parity issue. The
  next production decision is whether to accept tag-level parity on the CUDA
  12.8 stack, or add slower HF-style RMSNorm/RoPE diagnostic gates for closer
  native reproduction. Exact token parity with chuanbo's historical
  torch2.6/cu124 SDPA path is not guaranteed by upstream vLLM 0.24's CUDA 12.8
  wheel stack.

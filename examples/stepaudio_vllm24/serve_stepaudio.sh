#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

UV_BIN="${UV_BIN:-}"
if [[ -z "$UV_BIN" ]]; then
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  elif [[ -x /root/.local/bin/uv ]]; then
    UV_BIN="/root/.local/bin/uv"
  else
    echo "uv not found; set UV_BIN=/path/to/uv" >&2
    exit 127
  fi
fi

MODEL_PATH="${MODEL_PATH:-/data/liujun/stepaudio-merged/v25_C0_ep3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-stepaudio-v25-c0-ep3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8010}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"audio\":1}}"
TMPDIR="${TMPDIR:-/data/liujun/tmp/tmp}"
VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH:-/dev/shm}"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

export TMPDIR
export VLLM_RPC_BASE_PATH
export VLLM_USE_FLASHINFER_SAMPLER

mkdir -p "$TMPDIR"

args=(
  vllm serve "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --tokenizer-mode step_audio_2
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT"
  --interleave-mm-strings
)

if [[ "${ENFORCE_EAGER:-1}" == "1" ]]; then
  args+=(--enforce-eager)
fi

exec "$UV_BIN" run --project "$REPO_ROOT" "${args[@]}"

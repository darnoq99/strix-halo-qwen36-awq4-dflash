#!/usr/bin/env bash
set -euo pipefail

# Best current Strix Halo vLLM variant found locally:
# Qwen3.6-27B AWQ4 target + z-lab DFlash speculative drafter.
# Full context is preserved: 262144 tokens.

NAME="${NAME:-vllm-awq4-qwen}"
IMAGE="${IMAGE:-localhost/vllm-awq4-qwen:builder}"
PORT="${PORT:-18141}"
HOST="${HOST:-0.0.0.0}"
API_KEY="${API_KEY:-change-me}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-3}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
DFLASH_TOKENS="${DFLASH_TOKENS:-8}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"

ROOT="${VLLM_AWQ4_ROOT:-/mnt/bigdrive/strix-qwen36-vllm-awq4-qwen}"
MODELS_DIR="$ROOT/models"
DFLASH_DIR="${DFLASH_DIR:-/mnt/bigdrive/strix-qwen36-vllm-dflash/models/Qwen3.6-27B-DFlash}"
TRITON_CACHE="$ROOT/.triton-cache"
VLLM_CACHE="$ROOT/.vllm-cache"
LOG_DIR="$ROOT/logs"

mkdir -p "$TRITON_CACHE" "$VLLM_CACHE" "$LOG_DIR"

if [[ ! -d "$MODELS_DIR/hub/models--cyankiwi--Qwen3.6-27B-AWQ-INT4" ]]; then
  echo "Missing AWQ4 model under $MODELS_DIR" >&2
  exit 1
fi

if [[ ! -d "$DFLASH_DIR" ]]; then
  echo "Missing DFlash drafter under $DFLASH_DIR" >&2
  exit 1
fi

if ! podman image exists "$IMAGE"; then
  echo "Missing container image: $IMAGE" >&2
  exit 1
fi

podman rm -f "$NAME" >/dev/null 2>&1 || true

podman run -d \
  --name "$NAME" \
  --restart unless-stopped \
  --privileged \
  --network host \
  --device /dev/kfd \
  --device /dev/dri \
  --ipc host \
  -v "$MODELS_DIR:/models:ro" \
  -v "$DFLASH_DIR:/dflash/Qwen3.6-27B-DFlash:ro" \
  -v "$TRITON_CACHE:/root/.triton/cache" \
  -v "$VLLM_CACHE:/root/.cache/vllm" \
  -e HF_HOME=/models \
  -e HF_HUB_OFFLINE=1 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_USE_TRITON_AWQ=1 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e MIOPEN_FIND_MODE=FAST \
  "$IMAGE" \
  vllm serve cyankiwi/Qwen3.6-27B-AWQ-INT4 \
    --host "$HOST" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --served-model-name Qwen3.6-27B-AWQ4 \
    --attention-backend ROCM_ATTN \
    --mm-encoder-attn-backend TRITON_ATTN \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --enforce-eager \
    --safetensors-load-strategy eager \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"/dflash/Qwen3.6-27B-DFlash\",\"num_speculative_tokens\":$DFLASH_TOKENS}"

echo "Started $NAME on $HOST:$PORT"
echo "Model: Qwen3.6-27B-AWQ4"
echo "Base URL: http://127.0.0.1:$PORT/v1"
echo "API key: $API_KEY"
echo "Logs: podman logs -f $NAME"

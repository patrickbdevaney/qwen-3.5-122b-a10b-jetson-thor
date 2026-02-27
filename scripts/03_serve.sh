#!/bin/bash
# reproduce/03_serve.sh
#
# Launches the vLLM inference server for Qwen3.5-122B-A10B-NVFP4
# on Jetson AGX Thor.
#
# Prerequisites:
#   - Model weights at ~/Qwen3.5-122B-A10B-NVFP4/resharded/
#   - Docker image: vllm-thor:qwen35-latest (built via 02_build_docker.sh)
#   - ~97GB VRAM available (97GB peak at init)
#
# First run:
#   CUDA graphs are captured and cached to ~/thor-vllm-cache on first startup.
#   This adds 10-20 minutes to the initial boot. Subsequent starts are fast.
#   Server is ready when you see:
#     INFO:     Uvicorn running on http://0.0.0.0:8000
#
# Observed performance:
#   - Decode: 18.9 t/s
#   - TTFT: ~10-20s (without --enforce-eager, after CUDA graph warmup)
#   - Context: up to 16,384 tokens

set -euo pipefail

MODEL_PATH="$HOME/Qwen3.5-122B-A10B-NVFP4/resharded"
CACHE_DIR="$HOME/thor-vllm-cache"
IMAGE="vllm-thor:qwen35-latest"
PORT=8000

# ── Preflight checks ──────────────────────────────────────────────────────────

echo "=== Preflight checks ==="

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model weights not found at $MODEL_PATH"
    echo "Run: bash reproduce/01_reshard_nvfp4.sh"
    exit 1
fi

if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "ERROR: Docker image $IMAGE not found."
    echo "Run: bash reproduce/02_build_docker.sh"
    exit 1
fi

if ss -tlnp | grep -q ":$PORT "; then
    echo "WARNING: Port $PORT is already in use."
    echo "Stop any existing server before starting a new one:"
    echo "  docker stop vllm-qwen35"
    exit 1
fi

echo "Model path: $MODEL_PATH"
echo "Cache dir:  $CACHE_DIR"
echo "Image:      $IMAGE"
echo "Port:       $PORT"
echo ""

# ── Free page cache to maximize available memory ─────────────────────────────
echo "=== Freeing page cache ==="
sudo sync
sudo sysctl -w vm.drop_caches=3
echo "Done."
echo ""

# ── Create cache directory ────────────────────────────────────────────────────
mkdir -p "$CACHE_DIR"

# ── Launch ────────────────────────────────────────────────────────────────────
echo "=== Launching vLLM server ==="
echo "Server will be available at: http://0.0.0.0:$PORT"
echo "API endpoint: http://0.0.0.0:$PORT/v1/chat/completions"
echo ""
echo "IMPORTANT: First run captures CUDA graphs (~10-20 min)."
echo "Watch for: 'INFO: Uvicorn running on http://0.0.0.0:$PORT'"
echo ""

docker run --rm -it \
    --name vllm-qwen35 \
    --runtime nvidia \
    --gpus all \
    --ipc=host \
    --network host \
    -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
    -e HF_HUB_DISABLE_XET=1 \
    -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
    -v "$MODEL_PATH:/model:ro" \
    -v "$CACHE_DIR:/root/.cache/vllm" \
    "$IMAGE" \
    vllm serve /model \
        --host 0.0.0.0 \
        --port "$PORT" \
        --max-model-len 16384 \
        --max-num-seqs 2 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.72 \
        --trust-remote-code \
        --quantization compressed-tensors \
        --attention-backend FLASHINFER \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder

# ── Flag notes ────────────────────────────────────────────────────────────────
# --max-model-len 16384         : Maximum context window. Higher values OOM.
# --max-num-seqs 2              : Max concurrent sequences. Limits CUDA graph
#                                 capture range, reducing warmup time.
# --max-num-batched-tokens 8192 : Chunked prefill size. Fine for 16K context.
# --gpu-memory-utilization 0.72 : Do NOT increase — causes OOM at load.
# --trust-remote-code           : Required for Qwen3.5 custom model code.
# --quantization compressed-tensors : Loads NVFP4 weights correctly.
# --attention-backend FLASHINFER: FlashInfer attention (NOT MoE FP4 kernel).
# --reasoning-parser qwen3      : Parses <think> blocks from output.
# --enable-auto-tool-choice     : Enables tool/function calling.
# --tool-call-parser qwen3_coder: Parses tool calls in Qwen3.5 format.
#
# DELIBERATELY OMITTED:
# --enforce-eager               : Causes ~120s TTFT. Never use.
#!/bin/bash
# reproduce/01_reshard_nvfp4.sh
#
# Downloads Qwen3.5-122B-A10B base weights from HuggingFace and converts
# them to NVFP4 compressed-tensors format, resharded for single-GPU inference
# on Jetson AGX Thor.
#
# Prerequisites:
#   - huggingface-cli installed and authenticated (huggingface-cli login)
#   - ~300GB free disk space during conversion (source + output + temp)
#   - vllm-thor:qwen35-latest Docker image built (for the conversion tools)
#
# Output:
#   ~/Qwen3.5-122B-A10B-NVFP4/resharded/
#
# This script is the most compute-intensive step. Allow several hours.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$HOME/Qwen3.5-122B-A10B-NVFP4"
SOURCE_DIR="$BASE_DIR/source"
OUTPUT_DIR="$BASE_DIR/resharded"
HF_MODEL="Qwen/Qwen3.5-122B-A10B"

echo "=== Step 1: Download base weights from HuggingFace ==="
echo "Source model: $HF_MODEL"
echo "Destination: $SOURCE_DIR"
echo ""

mkdir -p "$SOURCE_DIR"


# find the Hugging Face CLI cmd
if command -v hf &> /dev/null; then
    HF_AUTH_CMD="hf auth"
elif command -v huggingface-cli &> /dev/null; then
    HF_AUTH_CMD="huggingface-cli"
else
    echo "ERROR: HuggingFace CLI is not installed."
    echo "Please install it first."
    exit 1
fi
# Check HuggingFace auth
if ! $HF_AUTH_CMD whoami &>/dev/null; then
    echo "ERROR: Not authenticated with HuggingFace."
    echo "Run: $HF_AUTH_CMD login"
    exit 1
fi

# Download base BF16 weights
# --include filters to only model weights (skip LFS pointers for non-weight files)
huggingface-cli download "$HF_MODEL" \
    --local-dir "$SOURCE_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "=== Step 2: Convert to NVFP4 compressed-tensors ==="
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# The conversion uses the TensorRT Model Optimizer (torch-trt / modelopt) tooling
# baked into the vLLM Jetson container.
#
# The convert_to_nvfp4_moe_kernel_format script handles:
#   - Weight quantization to FP4 (W4A4 format)
#   - Expert weight resharding for single-GPU MoE dispatch
#   - compressed-tensors metadata generation for vLLM
#
# NOTE: If this script path differs in your container version, inspect:
#   docker run --rm vllm-thor:qwen35-latest find / -name "*nvfp4*" 2>/dev/null
#   docker run --rm vllm-thor:qwen35-latest find / -name "*reshard*" 2>/dev/null

# Locate the optional NVFP4 MoE kernel-format repack tool *inside the container*
# (the tool ships within the vLLM source tree under /tmp/vllm, and it has moved
# across vLLM versions). This step is an offline optimization only — vLLM repacks
# NVFP4 MoE weights into kernel format automatically at load time if it is skipped.
# Detection runs inside the container because /tmp/vllm does not exist on the host.
docker run --rm \
    --runtime nvidia --gpus all \
    --ipc=host \
    -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
    -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
    -v "$SOURCE_DIR:/source:ro" \
    -v "$OUTPUT_DIR:/output" \
    vllm-thor:qwen35-latest \
    bash -c '
        set -e
        REPACK_TOOL=""
        for p in \
            /tmp/vllm/tools/convert_to_nvfp4_moe_kernel_format.py \
            /tmp/vllm/vllm/model_executor/layers/fused_moe/tools/convert_to_nvfp4_moe_kernel_format.py; do
            if [ -f "$p" ]; then REPACK_TOOL="$p"; break; fi
        done
        if [ -z "$REPACK_TOOL" ]; then
            REPACK_TOOL="$(find /tmp/vllm -name convert_to_nvfp4_moe_kernel_format.py 2>/dev/null | head -1)"
        fi

        if [ -n "$REPACK_TOOL" ] && [ -f "$REPACK_TOOL" ]; then
            echo "Found NVFP4 MoE repack tool at $REPACK_TOOL — running offline repack."
            python3 "$REPACK_TOOL" \
                --model-path /source \
                --output-path /output \
                --dtype fp4
        else
            echo "NOTE: convert_to_nvfp4_moe_kernel_format.py not present in this vLLM build."
            echo "This is expected on newer vLLM commits. The offline MoE repack is being skipped."
            echo "vLLM will repack NVFP4 MoE weights into kernel format automatically at model load"
            echo "(you will see '\''Using MoEPrepareAndFinalizeNoDPEPModular'\'' in the serve logs)."
        fi
    '

echo ""
echo "=== Step 3: Verify output ==="
echo ""

# Check that key files were created
if [ ! -f "$OUTPUT_DIR/config.json" ]; then
    echo "ERROR: config.json not found in output. Conversion may have failed."
    exit 1
fi

if ! ls "$OUTPUT_DIR"/*.safetensors &>/dev/null; then
    echo "ERROR: No .safetensors files found in output."
    exit 1
fi

echo "Resharded weights written to: $OUTPUT_DIR"
echo "Files:"
ls -lh "$OUTPUT_DIR"/*.safetensors | head -20
echo ""
echo "Total size:"
du -sh "$OUTPUT_DIR"
echo ""
echo "=== Resharding complete ==="
echo "Next step: bash reproduce/02_build_docker.sh"

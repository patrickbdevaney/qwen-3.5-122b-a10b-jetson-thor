#!/bin/bash
# reproduce/02_build_docker.sh
#
# Builds the patched vLLM Docker image for Qwen3.5-122B-A10B-NVFP4 on Jetson Thor.
#
# What this does:
#   1. Identifies the correct base image (NVIDIA Jetson vLLM container)
#   2. Applies Patch 1: RMSNormGated activation parameter to layernorm.py
#   3. Sets VLLM_USE_FLASHINFER_MOE_FP4=0 as a baked-in environment variable
#   4. Verifies patches are present before tagging
#
# Output image: vllm-thor:qwen35-latest
#
# IMPORTANT: Must be run on the Jetson Thor itself (aarch64).
# Cross-building for aarch64 from x86 is not supported for this image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_TAG="vllm-thor:qwen35-latest"
DOCKERHUB_TAG="patrickbdevaney/vllm-thor:qwen35-122b-a10b-nvfp4"

echo "=== Building patched vLLM image for Qwen3.5-122B-A10B-NVFP4 ==="
echo "Repo root: $REPO_ROOT"
echo "Target tag: $IMAGE_TAG"
echo ""

# ── Option A: Build from Dockerfile (clean rebuild) ──────────────────────────
# This requires the correct base image tag in the Dockerfile.
# If you know the exact base image your working container was derived from,
# update the FROM line in Dockerfile first.
#
# To find the base image of your current working container:
#   docker inspect vllm-thor:qwen35-latest \
#     --format '{{.Config.Image}} {{index .Config.Labels "base_image"}}'
#
# If the base image is available locally or on NGC, use Option A:

if docker image inspect vllm-thor:qwen35-latest &>/dev/null; then
    echo "Existing image found. Applying patches via container commit method."
    echo "(This preserves the exact base layers from your working image.)"
    echo ""
    APPLY_VIA_COMMIT=true
else
    echo "No existing image found. Building from Dockerfile."
    APPLY_VIA_COMMIT=false
fi

# ── Option B: Patch-and-commit (preserves exact base layers) ─────────────────
# This is the safer option when you already have a working base image.
# It applies patches to a running container then commits the result.

if [ "$APPLY_VIA_COMMIT" = true ]; then
    echo "=== Applying patches via container commit ==="

    # Clean up any leftover patch container
    docker rm -f vllm-build-patch 2>/dev/null || true

    # Start a temporary container
    docker run -d --name vllm-build-patch vllm-thor:qwen35-latest sleep 600
    sleep 3

    echo "Applying Patch 1: RMSNormGated activation parameter..."
    docker exec vllm-build-patch python3 << 'PATCHEOF'
path = '/tmp/vllm/vllm/model_executor/layers/layernorm.py'
content = open(path).read()

# Check if patch already applied
if 'activation: str = "silu"' in content and 'self.activation = activation' in content:
    print('Patch 1 already applied — skipping')
    exit(0)

# Add activation param to __init__ signature
old = '        norm_before_gate: bool = False,\n        device: torch.device | None = None,'
new = '        norm_before_gate: bool = False,\n        activation: str = "silu",\n        device: torch.device | None = None,'
if old not in content:
    print('ERROR: Signature pattern not found. Check layernorm.py version.')
    exit(1)
content = content.replace(old, new, 1)

# Add self.activation = activation to __init__ body
old2 = '        self.norm_before_gate = norm_before_gate\n        self.reset_parameters()'
new2 = '        self.norm_before_gate = norm_before_gate\n        self.activation = activation\n        self.reset_parameters()'
if old2 not in content:
    old2 = '        self.norm_before_gate = norm_before_gate'
    new2 = '        self.norm_before_gate = norm_before_gate\n        self.activation = activation'
    if old2 not in content:
        print('ERROR: Body pattern not found.')
        exit(1)
content = content.replace(old2, new2, 1)

open(path, 'w').write(content)
c = open(path).read()
assert 'activation: str = "silu"' in c, 'Signature verification failed'
assert 'self.activation = activation' in c, 'Body verification failed'
print('Patch 1 applied successfully')
PATCHEOF

    echo "Verifying patches in container..."
    docker exec vllm-build-patch grep -n \
        "self.activation\|activation: str" \
        /tmp/vllm/vllm/model_executor/layers/layernorm.py

    echo "Committing patched container as $IMAGE_TAG..."
    docker commit \
        --change "ENV VLLM_USE_FLASHINFER_MOE_FP4=0" \
        --change "ENV HF_HUB_DISABLE_XET=1" \
        --change "LABEL patch.layernorm=RMSNormGated_activation_added" \
        --change "LABEL model=Qwen3.5-122B-A10B-NVFP4" \
        --change "LABEL platform=Jetson-AGX-Thor-aarch64" \
        vllm-build-patch \
        "$IMAGE_TAG"

    docker rm -f vllm-build-patch

else
    echo "=== Building from Dockerfile ==="
    echo "NOTE: Update the FROM line in Dockerfile to match your base image."
    echo ""
    docker build \
        --platform linux/arm64 \
        -t "$IMAGE_TAG" \
        -f "$REPO_ROOT/Dockerfile" \
        "$REPO_ROOT"
fi

echo ""
echo "=== Build complete ==="
echo ""
echo "=== Final patch verification ==="
docker run --rm "$IMAGE_TAG" \
    grep -n "self.activation\|activation: str" \
    /tmp/vllm/vllm/model_executor/layers/layernorm.py

echo ""
echo "=== Environment variable verification ==="
docker run --rm "$IMAGE_TAG" \
    env | grep -E "VLLM_USE_FLASHINFER|HF_HUB"

echo ""
echo "Image built successfully: $IMAGE_TAG"
echo ""
echo "To also tag for Docker Hub:"
echo "  docker tag $IMAGE_TAG $DOCKERHUB_TAG"
echo "  docker push $DOCKERHUB_TAG"
echo ""
echo "Next step: bash reproduce/03_serve.sh"
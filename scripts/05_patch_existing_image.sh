#!/bin/bash
# reproduce/05_patch_existing_image.sh
#
# Applies all required patches to an existing NVIDIA Jetson vLLM base image
# and commits the result as vllm-thor:qwen35-latest.
#
# Use this if:
#   - You have pulled a fresh Jetson vLLM base image and want to patch it
#   - You need to reapply patches after a base image update
#   - You want to patch a different base image tag
#
# Usage:
#   bash reproduce/05_patch_existing_image.sh [BASE_IMAGE_TAG]
#
# Example:
#   bash reproduce/05_patch_existing_image.sh nvcr.io/nvidia/jetson/vllm:latest-jetson
#
# If BASE_IMAGE_TAG is not provided, patches are applied to the current
# vllm-thor:qwen35-latest (idempotent — safe to run on already-patched image).

set -euo pipefail

BASE_IMAGE="${1:-vllm-thor:qwen35-latest}"
OUTPUT_TAG="vllm-thor:qwen35-latest"
PATCH_CONTAINER="vllm-patch-work"

echo "=== Patching image: $BASE_IMAGE ==="
echo "Output tag: $OUTPUT_TAG"
echo ""

# ── Clean up any leftover container ──────────────────────────────────────────
docker rm -f "$PATCH_CONTAINER" 2>/dev/null || true

# ── Start temporary container ─────────────────────────────────────────────────
echo "Starting temporary container..."
docker run -d --name "$PATCH_CONTAINER" "$BASE_IMAGE" sleep 600
sleep 3

# ── Inspect current state of layernorm.py ────────────────────────────────────
echo ""
echo "=== Current state of layernorm.py (lines 505-545) ==="
docker exec "$PATCH_CONTAINER" \
    sed -n '505,545p' /tmp/vllm/vllm/model_executor/layers/layernorm.py
echo ""

# ── Apply Patch 1: RMSNormGated activation parameter ─────────────────────────
echo "=== Applying Patch 1: RMSNormGated activation parameter ==="

docker exec "$PATCH_CONTAINER" python3 << 'PATCHEOF'
import sys

path = '/tmp/vllm/vllm/model_executor/layers/layernorm.py'
content = open(path).read()

# ── Check if already patched ──────────────────────────────────────────────────
sig_patched = 'activation: str = "silu"' in content
body_patched = 'self.activation = activation' in content

if sig_patched and body_patched:
    print('Patch 1 already applied — no changes needed.')
    sys.exit(0)

print(f'Current state: sig_patched={sig_patched}, body_patched={body_patched}')

# ── Patch signature ───────────────────────────────────────────────────────────
if not sig_patched:
    old = '        norm_before_gate: bool = False,\n        device: torch.device | None = None,'
    new = '        norm_before_gate: bool = False,\n        activation: str = "silu",\n        device: torch.device | None = None,'
    if old not in content:
        print('ERROR: Signature pattern not found.')
        print('Expected to find:')
        print(repr(old))
        print()
        print('Searching for similar patterns...')
        for i, line in enumerate(content.splitlines()):
            if 'norm_before_gate' in line or 'device' in line:
                print(f'  Line {i+1}: {repr(line)}')
        sys.exit(1)
    content = content.replace(old, new, 1)
    print('  ✓ Signature patched: activation: str = "silu" added')

# ── Patch body ────────────────────────────────────────────────────────────────
if not body_patched:
    # Try with reset_parameters on next line first
    old2 = '        self.norm_before_gate = norm_before_gate\n        self.reset_parameters()'
    new2 = '        self.norm_before_gate = norm_before_gate\n        self.activation = activation\n        self.reset_parameters()'

    if old2 in content:
        content = content.replace(old2, new2, 1)
        print('  ✓ Body patched (with reset_parameters): self.activation = activation added')
    else:
        # Fallback: just after norm_before_gate assignment
        old3 = '        self.norm_before_gate = norm_before_gate'
        new3 = '        self.norm_before_gate = norm_before_gate\n        self.activation = activation'
        if old3 in content:
            content = content.replace(old3, new3, 1)
            print('  ✓ Body patched (fallback): self.activation = activation added')
        else:
            print('ERROR: Body pattern not found. Manual inspection required.')
            print('Searching for norm_before_gate assignments...')
            for i, line in enumerate(content.splitlines()):
                if 'norm_before_gate' in line:
                    print(f'  Line {i+1}: {repr(line)}')
            sys.exit(1)

# ── Write and verify ──────────────────────────────────────────────────────────
open(path, 'w').write(content)
c = open(path).read()

ok_sig = 'activation: str = "silu"' in c
ok_body = 'self.activation = activation' in c

print(f'  Verification: sig={ok_sig}, body={ok_body}')

if not (ok_sig and ok_body):
    print('ERROR: Patch verification failed after write.')
    sys.exit(1)

print('Patch 1 complete.')
PATCHEOF

# ── Verify patch in container ─────────────────────────────────────────────────
echo ""
echo "=== Verifying patches ==="
docker exec "$PATCH_CONTAINER" \
    grep -n "self.activation\|activation: str" \
    /tmp/vllm/vllm/model_executor/layers/layernorm.py

EXPECTED_LINES=3  # line 511, 536, 597
ACTUAL_LINES=$(docker exec "$PATCH_CONTAINER" \
    grep -c "self.activation\|activation: str" \
    /tmp/vllm/vllm/model_executor/layers/layernorm.py 2>/dev/null || echo 0)

if [ "$ACTUAL_LINES" -ge 2 ]; then
    echo ""
    echo "  ✓ Patch verified ($ACTUAL_LINES matching lines found)"
else
    echo ""
    echo "  ✗ Patch verification failed (expected ≥2 lines, found $ACTUAL_LINES)"
    docker rm -f "$PATCH_CONTAINER"
    exit 1
fi

# ── Commit patched container ──────────────────────────────────────────────────
echo ""
echo "=== Committing patched image as $OUTPUT_TAG ==="

docker commit \
    --change 'ENV VLLM_USE_FLASHINFER_MOE_FP4=0' \
    --change 'ENV HF_HUB_DISABLE_XET=1' \
    --change 'LABEL patch.layernorm=RMSNormGated_activation_added' \
    --change 'LABEL patch.flashinfer_moe=VLLM_USE_FLASHINFER_MOE_FP4_disabled' \
    --change 'LABEL model=Qwen3.5-122B-A10B-NVFP4' \
    --change 'LABEL platform=Jetson-AGX-Thor-aarch64' \
    --change "LABEL patched_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$PATCH_CONTAINER" \
    "$OUTPUT_TAG"

docker rm -f "$PATCH_CONTAINER"

echo ""
echo "=== Final verification on committed image ==="
docker run --rm "$OUTPUT_TAG" \
    grep -n "self.activation\|activation: str" \
    /tmp/vllm/vllm/model_executor/layers/layernorm.py

docker run --rm "$OUTPUT_TAG" \
    env | grep -E "VLLM_USE_FLASHINFER|HF_HUB"

echo ""
echo "=== Done ==="
echo "Image ready: $OUTPUT_TAG"
echo "Next: bash reproduce/03_serve.sh"
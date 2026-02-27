# vllm-thor-qwen35 Dockerfile
# Builds a patched vLLM runtime for Qwen3.5-122B-A10B-NVFP4 on Jetson AGX Thor
#
# Base: NVIDIA Jetson vLLM container (aarch64, JetPack CUDA stack)
# Patches applied:
#   1. RMSNormGated activation parameter (layernorm.py)
#   2. VLLM_USE_FLASHINFER_MOE_FP4=0 set as default env var
#
# Build: docker build -t vllm-thor:qwen35-latest .
# Note: Must be built ON the Jetson Thor or on an aarch64 host with matching
#       CUDA/JetPack environment. Cross-compilation not supported.

# ── Base image ────────────────────────────────────────────────────────────────
# Replace this tag with the exact base image tag from your working build.
# To find it: docker inspect vllm-thor:qwen35-latest | grep -i "parent\|base"
# Typical Jetson vLLM base images follow the pattern:
#   nvcr.io/nvidia/jetson/vllm:<version>-jetson-<jetpack>
#
# If rebuilding from scratch, start from the official Jetson vLLM image:
#   https://catalog.ngc.nvidia.com/orgs/nvidia/teams/jetson/containers/vllm
FROM nvcr.io/nvidia/jetson/vllm:latest-jetson

# ── Environment ───────────────────────────────────────────────────────────────
# VLLM_USE_FLASHINFER_MOE_FP4=0: MoE FP4 FlashInfer kernel broken on Thor.
# Must be set — without this the model fails to load with kernel errors.
ENV VLLM_USE_FLASHINFER_MOE_FP4=0
ENV HF_HUB_DISABLE_XET=1

# ── Patch 1: RMSNormGated activation parameter ────────────────────────────────
# Problem:  RMSNormGated.forward() references self.activation at line ~595
#           but __init__ never accepted or stored this parameter.
# Error:    AttributeError: 'RMSNormGated' object has no attribute 'activation'
# Fix:      Add activation: str = "silu" to __init__ signature and body.
#
# This patch is applied via inline Python to avoid fragile line-number offsets.
RUN python3 -c " \
path = '/tmp/vllm/vllm/model_executor/layers/layernorm.py'; \
content = open(path).read(); \
\
# Add activation param to __init__ signature \
old = '        norm_before_gate: bool = False,\n        device: torch.device | None = None,'; \
new = '        norm_before_gate: bool = False,\n        activation: str = \"silu\",\n        device: torch.device | None = None,'; \
if old not in content: \
    raise RuntimeError('SIGNATURE PATTERN NOT FOUND - vLLM version mismatch. Check layernorm.py manually.'); \
content = content.replace(old, new, 1); \
\
# Add self.activation = activation to __init__ body \
old2 = '        self.norm_before_gate = norm_before_gate\n        self.reset_parameters()'; \
new2 = '        self.norm_before_gate = norm_before_gate\n        self.activation = activation\n        self.reset_parameters()'; \
if old2 not in content: \
    # Fallback: try without reset_parameters on same line \
    old2 = '        self.norm_before_gate = norm_before_gate'; \
    new2 = '        self.norm_before_gate = norm_before_gate\n        self.activation = activation'; \
    if old2 not in content: \
        raise RuntimeError('ASSIGNMENT PATTERN NOT FOUND - patch failed. Inspect layernorm.py manually.'); \
content = content.replace(old2, new2, 1); \
\
open(path, 'w').write(content); \
c = open(path).read(); \
assert 'activation: str = \"silu\"' in c, 'Signature patch verification failed'; \
assert 'self.activation = activation' in c, 'Body patch verification failed'; \
print('Patch 1 applied and verified: RMSNormGated activation parameter'); \
"

# ── Verify patches ────────────────────────────────────────────────────────────
RUN echo "=== Patch verification ===" && \
    grep -n "self.activation\|activation: str" \
      /tmp/vllm/vllm/model_executor/layers/layernorm.py && \
    echo "=== Patch verification passed ==="

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL maintainer="patrickbdevaney"
LABEL description="vLLM patched for Qwen3.5-122B-A10B-NVFP4 on Jetson AGX Thor"
LABEL model="Qwen3.5-122B-A10B-NVFP4"
LABEL platform="Jetson AGX Thor aarch64"
LABEL patch.layernorm="RMSNormGated activation parameter added"
LABEL patch.flashinfer_moe="VLLM_USE_FLASHINFER_MOE_FP4=0"
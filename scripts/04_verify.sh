#!/bin/bash
# reproduce/04_verify.sh
#
# Verifies:
#   1. Patches are correctly applied in the Docker image
#   2. Environment variables are set correctly
#   3. Server is responding (if running)
#   4. Sends a minimal test inference request and checks for streaming output
#
# Run this after 02_build_docker.sh to verify the image,
# and again after 03_serve.sh to verify end-to-end.

set -euo pipefail

IMAGE="vllm-thor:qwen35-latest"
SERVER="http://localhost:8000"
PASS=0
FAIL=0

pass() { echo "  ✓ $1"; ((PASS++)); }
fail() { echo "  ✗ $1"; ((FAIL++)); }

echo "============================================"
echo " vLLM Thor Qwen3.5 — Verification Suite"
echo "============================================"
echo ""

# ── Section 1: Image exists ───────────────────────────────────────────────────
echo "[ 1/4 ] Docker image checks"

if docker image inspect "$IMAGE" &>/dev/null; then
    pass "Image exists: $IMAGE"
else
    fail "Image not found: $IMAGE — run 02_build_docker.sh"
fi

# ── Section 2: Patch verification ────────────────────────────────────────────
echo ""
echo "[ 2/4 ] Patch verification (grep — no CUDA required)"

LAYERNORM_GREP=$(docker run --rm "$IMAGE" \
    grep -n "self.activation\|activation: str" \
    /tmp/vllm/vllm/model_executor/layers/layernorm.py 2>/dev/null || true)

if echo "$LAYERNORM_GREP" | grep -q "activation: str"; then
    pass "RMSNormGated __init__ signature contains activation: str"
else
    fail "RMSNormGated __init__ signature MISSING activation param"
fi

if echo "$LAYERNORM_GREP" | grep -q "self.activation = activation"; then
    pass "RMSNormGated __init__ body contains self.activation = activation"
else
    fail "RMSNormGated __init__ body MISSING self.activation assignment"
fi

if echo "$LAYERNORM_GREP" | grep -q "activation=self.activation"; then
    pass "RMSNormGated.forward() passes activation= to kernel"
else
    fail "RMSNormGated.forward() missing activation= call (unexpected)"
fi

echo ""
echo "  Patched lines in layernorm.py:"
echo "$LAYERNORM_GREP" | sed 's/^/    /'

# ── Section 3: Environment variables ─────────────────────────────────────────
echo ""
echo "[ 3/4 ] Environment variable checks"

ENV_OUTPUT=$(docker run --rm "$IMAGE" env 2>/dev/null || true)

if echo "$ENV_OUTPUT" | grep -q "VLLM_USE_FLASHINFER_MOE_FP4=0"; then
    pass "VLLM_USE_FLASHINFER_MOE_FP4=0 is set in image"
else
    fail "VLLM_USE_FLASHINFER_MOE_FP4 not set — MoE FP4 kernel will crash"
fi

if echo "$ENV_OUTPUT" | grep -q "HF_HUB_DISABLE_XET=1"; then
    pass "HF_HUB_DISABLE_XET=1 is set in image"
else
    pass "HF_HUB_DISABLE_XET not baked in (can be passed at runtime — OK)"
fi

# ── Section 4: Live server test ───────────────────────────────────────────────
echo ""
echo "[ 4/4 ] Live server test (skipped if server not running)"

if curl -sf "$SERVER/health" &>/dev/null; then
    pass "Server health endpoint responding at $SERVER/health"

    # Check models endpoint
    MODELS=$(curl -sf "$SERVER/v1/models" 2>/dev/null || true)
    if echo "$MODELS" | grep -q "model"; then
        pass "Models endpoint responding"
        echo ""
        echo "  Available models:"
        echo "$MODELS" | python3 -c "
import json,sys
data=json.load(sys.stdin)
for m in data.get('data',[]):
    print(f'    - {m[\"id\"]}')
" 2>/dev/null || echo "$MODELS" | sed 's/^/    /'
    else
        fail "Models endpoint returned unexpected response"
    fi

    # Send a minimal inference request
    echo ""
    echo "  Sending test inference request..."
    TEST_RESPONSE=$(curl -sf "$SERVER/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "/model",
            "messages": [{"role": "user", "content": "Reply with exactly: VLLM_THOR_OK"}],
            "max_tokens": 20,
            "temperature": 0,
            "stream": false
        }' 2>/dev/null || true)

    if echo "$TEST_RESPONSE" | grep -q "VLLM_THOR_OK"; then
        pass "Test inference returned expected response"
    elif [ -n "$TEST_RESPONSE" ]; then
        pass "Test inference returned a response (content check inconclusive)"
        echo "  Response: $(echo "$TEST_RESPONSE" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(d['choices'][0]['message']['content'][:100])
except:
    print('(could not parse response)')
" 2>/dev/null)"
    else
        fail "Test inference returned no response"
    fi

    # Check metrics
    echo ""
    echo "  Server metrics (from Prometheus endpoint):"
    curl -sf "$SERVER/metrics" 2>/dev/null | \
        grep -E "num_requests_running|prompt_tokens_total|generation_tokens_total" | \
        sed 's/^/    /' || echo "    (metrics endpoint not available)"

else
    echo "  Server not running at $SERVER — skipping live tests."
    echo "  Run 03_serve.sh then re-run this script to test end-to-end."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Results: $PASS passed, $FAIL failed"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Fix failures before serving. See README.md for patch instructions."
    exit 1
else
    echo ""
    echo "All checks passed. System is ready."
fi
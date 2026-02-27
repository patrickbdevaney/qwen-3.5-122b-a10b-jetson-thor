# vLLM Jetson Thor — Qwen3.5-122B-A10B NVFP4

**First confirmed deployment of Qwen3.5-122B-A10B-NVFP4 on NVIDIA Jetson AGX Thor.**  
*Patrick Devaney — February 27, 2026*

This repository contains every artifact needed to reproduce a working vLLM inference server
for Qwen3.5-122B-A10B at NVFP4 precision on a single Jetson AGX Thor (128GB unified memory).
This includes the exact vLLM source commit, two required patches, all environment variables,
the resharding pipeline, and the Docker build process.

---

## Observed Performance

| Metric | Value |
|--------|-------|
| Decode throughput | 18.9 t/s |
| TTFT (with `--enforce-eager`) | ~120s ⚠️ remove this flag |
| TTFT (without `--enforce-eager`) | ~10–20s (estimated after CUDA graph warmup) |
| VRAM at init (peak) | ~97GB |
| VRAM steady-state | ~75–80GB |
| Max context length | 16,384 tokens |
| GPU memory utilization flag | 0.72 |
| Quantization | NVFP4 / compressed-tensors |
| Attention backend | FLASHINFER |
| Model weights size (resharded) | ~75GB |

---

## Hardware Environment

| Component | Specification |
|-----------|--------------|
| Platform | NVIDIA Jetson AGX Thor |
| SoC | Thor (Blackwell GPU architecture) |
| Unified Memory | 128 GB LPDDR5x |
| GPU | Integrated Blackwell GPU (FP4 native tensor cores) |
| CPU | 12-core Arm Cortex-X925 |
| Storage | NVMe SSD (model weights on local disk) |
| OS | Ubuntu 24.04 (aarch64) |
| CUDA | 12.x (Jetson JetPack) |
| Architecture | aarch64 |

---

## Software Environment

| Component | Version / Detail |
|-----------|-----------------|
| vLLM | Built from source — see exact commit below |
| Python | 3.12.12 (cpython-3.12.12-linux-aarch64) |
| PyTorch | Compatible with Jetson Thor JetPack CUDA stack |
| Docker runtime | nvidia container runtime |
| Base Docker image | NVIDIA Jetson vLLM container (aarch64) |
| Quantization format | compressed-tensors (NVFP4) |
| Attention kernel | FlashInfer (MoE FP4 kernel **disabled** — see patches) |

---

## vLLM Source — Exact Commit

The Docker image builds vLLM from source at `/tmp/vllm` inside the container.

```bash
# To inspect the exact commit baked into the image:
docker run --rm vllm-thor:qwen35-latest \
  git -C /tmp/vllm log --oneline -5
```

Key files modified from upstream (see `reproduce/patches/`):

| File | Change | Reason |
|------|--------|--------|
| `vllm/model_executor/layers/layernorm.py` | Added `activation` param to `RMSNormGated.__init__` | Missing parameter caused `AttributeError: 'RMSNormGated' object has no attribute 'activation'` at model load |

---

## Critical Patches Required

### Patch 1: RMSNormGated activation parameter

**File:** `vllm/model_executor/layers/layernorm.py`  
**Lines affected:** ~510–536  
**Error without patch:**
```
AttributeError: 'RMSNormGated' object has no attribute 'activation'
```

**Root cause:** `RMSNormGated.forward()` references `self.activation` at line 595,
but the `__init__` method never accepted or stored this parameter. The upstream
vLLM code assumed it was already set but the version baked into the Jetson container
was missing it.

**The fix — two additions to `__init__`:**

1. Add to signature (after `norm_before_gate`):
```python
activation: str = "silu",
```

2. Add to body (after `self.norm_before_gate = norm_before_gate`):
```python
self.activation = activation
```

**Full patched `__init__` signature:**
```python
def __init__(
    self,
    hidden_size: int,
    eps: float = 1e-6,
    group_size: Optional[int] = None,
    norm_before_gate: bool = False,
    activation: str = "silu",          # ← ADDED
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
):
```

**Full patched body section:**
```python
    self.norm_before_gate = norm_before_gate
    self.activation = activation        # ← ADDED
    self.reset_parameters()
```

**Verify the patch is present in your image:**
```bash
docker run --rm vllm-thor:qwen35-latest \
  grep -n "self.activation\|activation: str" \
  /tmp/vllm/vllm/model_executor/layers/layernorm.py
# Expected output:
# 511:        activation: str = "silu",
# 536:        self.activation = activation
# 597:            activation=self.activation,
```

See `reproduce/patches/layernorm_rmsnormgated_activation.patch` for the exact
line context extracted from the working image.

---

### Patch 2: FlashInfer MoE FP4 kernel — environment variable workaround

**Not a code patch — set via environment variable:**
```bash
VLLM_USE_FLASHINFER_MOE_FP4=0
```

**Reason:** The FlashInfer MoE FP4 kernel is broken on Jetson Thor at this vLLM
version. Setting this variable falls back to the standard MoE kernel. FlashInfer
attention (non-MoE) works correctly and should be kept enabled via
`--attention-backend FLASHINFER`.

---

## Critical Environment Variables

| Variable | Value | Reason |
|----------|-------|--------|
| `VLLM_USE_FLASHINFER_MOE_FP4` | `0` | MoE FP4 FlashInfer kernel broken on Thor |
| `LD_PRELOAD` | `/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1` | Required for CUDA library resolution on Jetson |
| `HF_HUB_DISABLE_XET` | `1` | Disables experimental HuggingFace XET transfer protocol |

**Known benign warning:** At container startup you will see:
```
ERROR: ld.so: object '/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1' from
LD_PRELOAD cannot be preloaded (file too short): ignored.
```
This is a non-fatal warning. The CUDA stack loads correctly via other paths.
Ignore it.

---

## Known Issues and Gotchas

### 1. Do NOT use `--enforce-eager`
With `--enforce-eager`, every forward pass goes through Python dispatch with no
CUDA graph optimization. For a 94-layer MoE model this causes ~120s TTFT on
prompts of ~900 tokens. Remove the flag and allow CUDA graph warmup at startup.
First startup after removing the flag will take 10–20 minutes longer while graphs
are captured and cached to `~/thor-vllm-cache`.

### 2. Import errors in non-GPU docker runs are expected
Running any vLLM import outside of a GPU-enabled container (e.g., for patching
or verification) will fail with:
```
ImportError: /tmp/vllm/vllm/_C.abi3.so: undefined symbol: cuPointerGetAttribute
```
This is expected. Verify patches using `grep` not Python imports.

### 3. CUDA graph cache
CUDA graphs are saved to the volume mounted at `/root/.cache/vllm`. Always mount
a persistent host directory here. Without this, graphs are recaptured on every
container start (adds 10–20 min per boot).

### 4. GPU memory utilization 0.72
Values above 0.72 cause OOM during KV cache profiling at model load. Do not
increase this value without testing.

### 5. `--max-num-seqs 2`
Higher values increase CUDA graph capture time and VRAM pressure during warmup.
Keep at 2 unless you have tested higher values.

---

## Model Weights

Weights are **not included** in this repository or Docker image.

**Source:** Qwen3.5-122B-A10B-NVFP4  
**Original HuggingFace repo:** `Qwen/Qwen3.5-122B-A10B` (BF16 base)  
**Required format:** NVFP4 compressed-tensors, resharded for single-GPU

Run `reproduce/01_reshard_nvfp4.sh` to convert from base BF16 weights.
The resharded output should be placed at `~/Qwen3.5-122B-A10B-NVFP4/resharded/`
on the Thor host before running the serve script.

---

## Quick Start (from scratch)

```bash
# 1. Clone this repo
git clone https://github.com/patrickbdevaney/vllm-thor-qwen35.git
cd vllm-thor-qwen35

# 2. Download and reshard model weights
bash reproduce/01_reshard_nvfp4.sh

# 3. Build the Docker image (applies all patches automatically)
bash reproduce/02_build_docker.sh

# 4. Verify patch integrity
docker run --rm vllm-thor:qwen35-latest \
  grep -n "self.activation\|activation: str" \
  /tmp/vllm/vllm/model_executor/layers/layernorm.py

# 5. Serve
bash reproduce/03_serve.sh
```

---

## Docker Image

The Docker image is large (~30–50GB) due to the full CUDA/vLLM/aarch64 stack.

```bash
# Pull (when available)
docker pull patrickbdevaney/vllm-thor:qwen35-122b-a10b-nvfp4

# Or rebuild locally from this repo
bash reproduce/02_build_docker.sh
```

---

## Repository Structure

```
QWEN-3.5-122B-A10B-JETSON-.../
├── README.MD
├── DOCKERFILE
├── docker-compose.yml
├── .gitignore
├── layernorm_rmsnormgated_activation.patch
├── chat/
│   ├── templates/
│   │   └── chat.html
│   ├── main.py
│   └── requirements.txt
└── scripts/
    ├── 01_reshard_nvfp4.sh
    ├── 02_build_docker.sh
    ├── 03_serve.sh
    ├── 04_verify.sh
    └── 05_patch_existing_image.sh
```

---

## License

Reproduction scripts: Apache 2.0
vLLM: Apache 2.0  
Qwen3.5 model weights: Qwen License (see HuggingFace model card)
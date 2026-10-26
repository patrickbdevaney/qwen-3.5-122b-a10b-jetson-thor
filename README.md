# Qwen3.5-122B-A10B NVFP4 on Jetson AGX Thor: vLLM Container

**A patched vLLM container image, an NVFP4 checkpoint and a serving recipe for running Qwen3.5-122B-A10B on a single
NVIDIA Jetson AGX Thor**

| | |
|---|---|
| **Model** | [`Qwen/Qwen3.5-122B-A10B`](https://huggingface.co/Qwen/Qwen3.5-122B-A10B): 122B total and 10B active parameters, multimodal MoE (`qwen3_5_moe`) |
| **Precision** | NVFP4 (W4A4, group size 16, FP8 E4M3 scales), compressed-tensors format; vision tower unquantized |
| **Serving stack** | vLLM, OpenAI-compatible API, run as a Docker container on the NVIDIA container runtime |
| **Target** | NVIDIA Jetson AGX Thor, 128 GB unified memory, aarch64 |
| **Image** | `vllm-thor:qwen35-latest`, distributed as a tarball on [Hugging Face](https://huggingface.co/patrickbdevaney/Qwen3.5-122B-A10B-NVFP4-vllm-docker-jetson-agx-thor) |
| **Weights** | [`patrickbdevaney/qwen-3.5-122b-a10b-nvfp4-resharded`](https://huggingface.co/patrickbdevaney/qwen-3.5-122b-a10b-nvfp4-resharded) (71.2 GiB, 8 shards) |
| **Validated** | 2026-02-27, three days after the model's release |

---

## Overview

Qwen3.5-122B-A10B was released on 2026-02-24 with a new model type, `qwen3_5_moe`. It is a 48-layer hybrid that
alternates three linear-attention layers with one full-attention layer, routes over 256 experts with 8 active, and
includes a vision tower. At that point no packaged vLLM build for Jetson could load it.

This repository records how it was brought up on Jetson AGX Thor in the days after release:

1. **A vLLM source build with `qwen3_5_moe` support**, compiled for Thor inside an NVIDIA Jetson vLLM container.
   The source tree is at `/tmp/vllm` in the image.
2. **A runtime patch to `RMSNormGated`.** The gated-norm layer used by the linear-attention blocks read
   `self.activation` but never set it. Model load failed until the attribute was backfilled.
3. **A kernel-selection override for Thor.** `VLLM_USE_FLASHINFER_MOE_FP4=0` routes the NVFP4 MoE away from
   FlashInfer's FP4 MoE kernels, which ship no `sm_110` build.
4. **A tuned serving configuration** that fits the 122B model, its KV cache and CUDA graphs inside Thor's unified
   memory.

The result is packaged as a ready-to-load container image and a matching weights repository. The scripts here
rebuild or re-patch the image and launch the server.

---

## Performance

Single Jetson AGX Thor, NVFP4 weights, configuration as in [Serving configuration](#serving-configuration).

| Metric | Value | Notes |
|---|---:|---|
| Decode throughput | **18.9 tok/s** | Single stream, measured |
| Time to first token, `--enforce-eager` | ~120 s | Measured on a ~900-token prompt; eager mode is not recommended |
| Time to first token, CUDA graphs | ~10–20 s | **Estimated, not measured** |
| Unified memory, peak at initialization | ~97 GB | Measured |
| Unified memory, steady state | ~75–80 GB | Measured |
| Maximum context length | 16,384 tokens | Configured limit |
| Weights on disk | 71.2 GiB | 8 safetensors shards |

These numbers come from the 2026-02 deployment and a small number of requests. They are not a benchmark sweep.
Accuracy against the BF16 model was not evaluated here.

---

## Support matrix

| Component | Validated configuration |
|---|---|
| Hardware | NVIDIA Jetson AGX Thor Developer Kit, 128 GB unified memory, Blackwell GPU (`sm_110`) |
| Host OS | Jetson Linux (JetPack 7 series), Ubuntu 24.04 aarch64 |
| Container runtime | Docker with the NVIDIA container runtime (`--runtime nvidia`) |
| Base image | NVIDIA Jetson vLLM container (aarch64) |
| vLLM | Pre-release source build with `qwen3_5_moe` support, at `/tmp/vllm` in the image |
| Quantization format | compressed-tensors `nvfp4-pack-quantized` |
| Python (in image) | 3.12 |

> **Note:** the image is aarch64-only and depends on Jetson's integrated-GPU driver stack. It does not run on
> x86-64 hosts or on discrete-GPU systems.

To read the exact vLLM commit baked into the image:

```bash
docker run --rm vllm-thor:qwen35-latest git -C /tmp/vllm log --oneline -5
```

---

## Quick start: prebuilt image

### 1. Load the container image

```bash
hf download patrickbdevaney/Qwen3.5-122B-A10B-NVFP4-vllm-docker-jetson-agx-thor \
    vllm-thor-qwen35-latest.tar.gz --local-dir ~/vllm-thor-image
docker load -i ~/vllm-thor-image/vllm-thor-qwen35-latest.tar.gz
docker image inspect vllm-thor:qwen35-latest --format '{{.Id}}'
```

The repository also carries a zstd-compressed tarball of the same size class, `vllm-thor-qwen35-nvfp4.tar.zst`.
Load it with `zstd -dc <file> | docker load`.

### 2. Download the weights

```bash
hf download patrickbdevaney/qwen-3.5-122b-a10b-nvfp4-resharded --local-dir ~/Qwen3.5-122B-A10B-NVFP4
ls ~/Qwen3.5-122B-A10B-NVFP4/resharded/      # config.json, 8 x *.safetensors, tokenizer files
```

### 3. Verify the image

```bash
bash scripts/04_verify.sh
```

This checks that the image exists, that the `RMSNormGated` patch is present (by `grep`; no GPU needed) and that the
required environment variables are set. If a server is running, it also checks `/health` and `/v1/models` and sends
a test completion.

### 4. Serve

```bash
bash scripts/03_serve.sh          # or: docker compose up
```

The server is ready when the log shows `Uvicorn running on http://0.0.0.0:8000`. The first start captures CUDA
graphs into `~/thor-vllm-cache` and takes an extra 10–20 minutes; later starts reuse the cache.

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "/model",
  "messages": [{"role": "user", "content": "Summarize the Jetson AGX Thor platform in two sentences."}],
  "max_tokens": 256
}'
```

---

## Building the image

Use this path to rebuild the image, or to apply the patch to a newer Jetson vLLM base.

| Script | Purpose |
|---|---|
| [`scripts/02_build_docker.sh`](scripts/02_build_docker.sh) | Re-patches an existing `vllm-thor:qwen35-latest` and commits it in place, preserving its base layers. With no local image, it builds from the [`Dockerfile`](Dockerfile). |
| [`scripts/05_patch_existing_image.sh`](scripts/05_patch_existing_image.sh) | Applies the patch to any base image: `bash scripts/05_patch_existing_image.sh <base-image>`. Idempotent. |
| [`Dockerfile`](Dockerfile) | Declarative form of the same patch and environment. |

> **Important:** the `Dockerfile`'s `FROM` line is a generic placeholder (`nvcr.io/nvidia/jetson/vllm:latest-jetson`).
> The published image was produced by patching and committing a container (`02_build_docker.sh`), not by a clean
> `docker build`. For an exact reproduction, start from the published image. To rebuild from scratch, set `FROM` to
> a Jetson vLLM base whose `/tmp/vllm` tree includes `qwen3_5_moe`.

The patch is applied by exact string match against `layernorm.py`. It fails loudly instead of editing the wrong
place when the upstream source differs, and every build path re-verifies the result with `grep`.

---

## Patches and runtime overrides

### `RMSNormGated` activation attribute

| | |
|---|---|
| File | `vllm/model_executor/layers/layernorm.py` |
| Symptom | `AttributeError: 'RMSNormGated' object has no attribute 'activation'` at model load |
| Cause | `forward()` passes `activation=self.activation` to the fused gated-norm kernel, but `__init__` neither accepted nor stored it |
| Fix | Add `activation: str = "silu"` to the `__init__` signature and assign `self.activation = activation` |
| Reference | [`patches/layernorm_rmsnormgated_activation.patch`](patches/layernorm_rmsnormgated_activation.patch), annotated lines from the working image |

Expected verification output in a patched image:

```text
$ docker run --rm vllm-thor:qwen35-latest grep -n "self.activation\|activation: str" \
    /tmp/vllm/vllm/model_executor/layers/layernorm.py
511:        activation: str = "silu",
536:        self.activation = activation
597:            activation=self.activation,
```

### Environment variables

| Variable | Value | Purpose |
|---|---|---|
| `VLLM_USE_FLASHINFER_MOE_FP4` | `0` | Disables FlashInfer's FP4 MoE kernels, which have no Thor (`sm_110`) build. Baked into the image. |
| `LD_PRELOAD` | `/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1` | CUDA driver resolution on Jetson. |
| `HF_HUB_DISABLE_XET` | `1` | Uses the standard Hugging Face transfer path. Baked into the image. |

> **Note:** at container start, `ld.so` may report that the `LD_PRELOAD` object `cannot be preloaded (file too short)`.
> This is benign; the CUDA driver loads through the runtime's own path.

---

## Serving configuration

The same configuration is used by [`scripts/03_serve.sh`](scripts/03_serve.sh) and
[`docker-compose.yml`](docker-compose.yml).

| Flag | Value | Rationale |
|---|---|---|
| `--quantization` | `compressed-tensors` | Loads the NVFP4 checkpoint |
| `--attention-backend` | `FLASHINFER` | FlashInfer attention (not its MoE kernels) |
| `--max-model-len` | `16384` | Larger values did not fit alongside the weights |
| `--max-num-seqs` | `2` | Bounds the CUDA-graph capture range and warm-up memory |
| `--max-num-batched-tokens` | `8192` | Chunked-prefill size |
| `--gpu-memory-utilization` | `0.72` | Higher values ran out of memory during KV-cache profiling |
| `--reasoning-parser` | `qwen3` | Separates `<think>` content |
| `--enable-auto-tool-choice`, `--tool-call-parser` | `qwen3_coder` | Tool calling in the Qwen3.5 format |
| `--trust-remote-code` | set | Part of the validated configuration |
| Volume `/root/.cache/vllm` | `~/thor-vllm-cache` | Persists compiled CUDA graphs across restarts |

> **Important:** do not add `--enforce-eager`. Without CUDA graphs, every forward pass of the 48-layer MoE goes
> through Python dispatch, and time to first token rose to about 120 s.

`03_serve.sh` drops the page cache (`vm.drop_caches=3`, needs `sudo`) before launch to give vLLM the most free
unified memory. Run no other GPU workload on the Thor while the server is up.

---

## Compatibility with newer vLLM builds

The recipe above is pinned to the February 2026 image. In June 2026, the same checkpoint was served from a newer
vLLM build (a DFlash speculative-decoding fork) and needed different kernel selection:

- **Marlin NVFP4 MoE faults at load** for this 256-expert model on Thor, during `process_weights_after_loading`, with
  no Python traceback. Models with fewer experts were not affected. This matches open upstream reports of FP4 MoE
  illegal memory access on large Blackwell MoE models (vLLM #35519, #35566, #35922).
- **Working configuration on that build:** `--moe-backend cutlass` with `--attention-backend TRITON_ATTN`. On that
  build FlashInfer paged-decode attention failed with an API mismatch (`kv_cache_sf`).
- **Measured on that build:** 10.9 tok/s for the base model; 18.6 tok/s with a DFlash draft (12 speculative tokens,
  acceptance τ 4.20, `--gpu-memory-utilization 0.78`).

If you move this checkpoint to a newer vLLM, start from those flags rather than the ones above.

---

## Troubleshooting

| Symptom | Cause | Resolution |
|---|---|---|
| `AttributeError: ... no attribute 'activation'` | `RMSNormGated` patch absent | Rebuild with `02_build_docker.sh`, or patch with `05_patch_existing_image.sh` |
| Out of memory during KV-cache profiling | `--gpu-memory-utilization` too high, or another process is holding memory | Keep 0.72; stop other GPU work; drop caches |
| Every start spends 10–20 minutes capturing graphs | Graph cache not persisted | Mount a host directory at `/root/.cache/vllm` |
| ~120 s to first token | `--enforce-eager` set | Remove it |
| `undefined symbol: cuPointerGetAttribute` when importing vLLM | Container started without the NVIDIA runtime | Expected outside a GPU container; verify patches with `grep`, not imports |
| Process exits during weight processing (exit 255) on a newer vLLM | Marlin NVFP4 MoE fault at 256 experts | See [Compatibility with newer vLLM builds](#compatibility-with-newer-vllm-builds) |

---

## Weights

The served checkpoint is
[`patrickbdevaney/qwen-3.5-122b-a10b-nvfp4-resharded`](https://huggingface.co/patrickbdevaney/qwen-3.5-122b-a10b-nvfp4-resharded):

- format: compressed-tensors `nvfp4-pack-quantized`, W4A4;
- size: 71.2 GiB in 8 shards;
- the vision tower is excluded from quantization.

> **Note:** [`scripts/01_reshard_nvfp4.sh`](scripts/01_reshard_nvfp4.sh) does **not** quantize BF16 weights. It
> downloads the BF16 base model and runs vLLM's offline NVFP4 MoE kernel-format repack *if that tool exists in the
> image*. Recent vLLM builds no longer ship the tool, and vLLM repacks the experts at load time anyway. To obtain
> serveable weights, download the published NVFP4 checkpoint above.

---

## Reference chat client

[`chat/`](chat/) holds a small FastAPI proxy with a browser UI. It streams chat completions and shows the server's
health and metrics.

```bash
cd chat && pip install -r requirements.txt
VLLM_BASE_URL=http://<thor-host>:8000 uvicorn main:app --host 127.0.0.1 --port 7860
```

---

## Security considerations

- `vllm serve` here runs with host networking, binds `0.0.0.0:8000`, and has **no API key**. Expose it only on a
  trusted network, or add `--api-key` and a TLS reverse proxy.
- `--trust-remote-code` executes Python shipped with the model repository. Use only checkpoints you trust.
- The image runs as root inside the container.

---

## Repository layout

| Path | Contents |
|---|---|
| [`Dockerfile`](Dockerfile) | Declarative image definition: patch, environment and labels |
| [`docker-compose.yml`](docker-compose.yml) | Compose service with the validated serving flags, graph-cache volume and health check |
| [`scripts/`](scripts/) | `01` weights repack (optional), `02` image build, `03` serve, `04` verification suite, `05` patch any base image |
| [`patches/`](patches/) | Annotated `RMSNormGated` patch |
| [`chat/`](chat/) | Reference FastAPI chat client |

---

## License

- **This repository** (scripts, Dockerfile, compose file, chat client, documentation): Apache-2.0.
- **vLLM:** Apache-2.0.
- **Qwen3.5-122B-A10B:** Apache-2.0 ([model card](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)).
- **NVIDIA Jetson container images, CUDA and TensorRT components** are governed by their NVIDIA license agreements.

This is an independent community project. It is not affiliated with, endorsed by, or supported by NVIDIA, the vLLM
project or the Qwen team.

## Citation

```bibtex
@misc{devaney2026qwen35thor,
  title  = {Qwen3.5-122B-A10B NVFP4 on Jetson AGX Thor: a patched vLLM container},
  author = {Devaney, Patrick},
  year   = {2026},
  url    = {https://github.com/patrickbdevaney/qwen-3.5-122b-a10b-jetson-thor}
}
```

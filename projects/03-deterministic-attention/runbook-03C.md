Runbook — 03C KV‑Cache Invariance
=================================

Prereqs
-------
- Docker with NVIDIA Container Toolkit for CUDA runs (optional).
- This repo root mounted into the container.

Build
-----

# CPU-only (quick):
scripts/build-in-container.sh

# CUDA (Ampere example):
ENGINE=docker IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
CMAKE_ARGS='-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86' \
scripts/build-in-container.sh

Run tests
---------

# CPU backend
GGML_DETERMINISTIC=1 build-container/bin/test-kvcache-invariance

# CUDA backend (GPU index 0)
ENGINE=docker IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
$ENGINE run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 \
  -v "$(pwd):/src" -w /src/build-container/bin "$IMAGE" \
  bash -lc 'GGML_DETERMINISTIC=1 ./test-kvcache-invariance'

Two flows (what we compare)
---------------------------
- Single-shot prefill to position P:
  - Inputs: Q has N=P, K/V have KV=P (padded to 256), mask is [KV, PAD(P,64), 1, 1].
  - Output slice compared: logits for the last token (column P-1).
- Incremental decode up to P:
  - Steps s=1..P with N=1; at each step, KV is padded up to the next multiple of 256.
  - Mask is [KVp, PAD(1,64), 1, 1] with 0 for [0..s-1] and -inf for padded [s..KVp-1].
- Deterministic policy: in det mode we require KV multiple-of-256 and mask N padded to 64. If shapes do not satisfy this, the graph aborts with guidance.

Notes
-----
- Deterministic mode forces KV padding to 256 across flows to keep reduction order fixed.
- Masks are padded to GGML_KQ_MASK_PAD (64) and at least N.
- For CUDA FlashAttention determinism, ensure KV length is a multiple of 256; otherwise the test may abort with guidance.

Debug toggles
-------------
- `GGML_DETERMINISTIC=1` — enable deterministic mode (required).
- `LLAMA_GRAPH_INPUT_DEBUG=1` — optional verbose graph input info.

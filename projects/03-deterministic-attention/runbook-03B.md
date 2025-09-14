Runbook — 03B Deterministic Attention (CUDA Ada/Ampere)
======================================================

Prereqs
-------
- Docker with NVIDIA Container Toolkit.
- This repo root mounted into the container.

Build (Ada)
-----------

ENGINE=docker IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
CMAKE_ARGS='-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89' \
scripts/build-in-container.sh

Build (Ampere)
--------------

ENGINE=docker IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
CMAKE_ARGS='-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86' \
scripts/build-in-container.sh

Run tests on a specific GPU
---------------------------

# Example: restrict to GPU index 2
ENGINE=docker IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
$ENGINE run --rm --gpus all -e CUDA_VISIBLE_DEVICES=2 \
  -v "$(pwd):/src" -w /src/build-container/bin "$IMAGE" \
  bash -lc 'GGML_DETERMINISTIC=1 ./test-attention-determinism && GGML_DETERMINISTIC=1 ./test-matmul-determinism'

Notes
-----
- Deterministic attention relies on a single-block accumulation (no stream-k) for fixed reduction order.
- Quantized K/V coverage is limited to supported vec kernels (e.g., D=128 with q4_0/q8_0 pairs). Unsupported pairs will be skipped by the tests.
- For DeepSeek (D=576/DV=512), deterministic mode calls MMA and remains deterministic via single-block accumulation.
Build (mixed Ada + Ampere)
--------------------------

ENGINE=docker IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
CMAKE_ARGS='-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86;89' \
scripts/build-in-container.sh

- Mixed-arch hosts: either build for both archs (`-DCMAKE_CUDA_ARCHITECTURES=86;89`) or set `CUDA_VISIBLE_DEVICES` to a single architecture during test runs.

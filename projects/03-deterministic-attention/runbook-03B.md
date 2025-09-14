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
- Quantized K/V coverage is limited to supported vec kernels (e.g., D=128 with q4_0/q4_0 and q8_0/q8_0). If `GGML_CUDA_FA_ALL_QUANTS=ON`, a few more pairs are exercised. Unsupported pairs error with guidance.
- F16 K/V may automatically fall back to the deterministic tile path; quantized K/V does not have a tile fallback.
- Special head sizes 80/96/112 are supported in deterministic mode via a single‑column tile path (F16 K/V only). Throughput is lower than vec at 64/128/256. D=576 remains experimental and requires `GGML_DETERMINISTIC_ATTENTION_ALLOW_MMA=1`.

Optional builds
---------------
- Full quant vec instances:

ENGINE=docker IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
CMAKE_ARGS='-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_CUDA_FA_ALL_QUANTS=ON' \
scripts/build-in-container.sh

Debug toggles
-------------
- `GGML_DETERMINISTIC_ATTENTION_FORCE_VEC=1` or `GGML_DETERMINISTIC_ATTENTION_FORCE_TILE=1` (F16‑only)
- `GGML_DETERMINISTIC_ATTENTION_ALLOW_MMA=1` (experimental)
- `RUN_FORCE_TOGGLE_TESTS=1` enables FORCE_* determinism smokes in the tests
- `RUN_MMA_HEADSIZE_TESTS=1` probes D=576 behavior (no assertions by default)
Build (mixed Ada + Ampere)
--------------------------

ENGINE=docker IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
CMAKE_ARGS='-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86;89' \
scripts/build-in-container.sh

- Mixed-arch hosts: either build for both archs (`-DCMAKE_CUDA_ARCHITECTURES=86;89`) or set `CUDA_VISIBLE_DEVICES` to a single architecture during test runs.

Deterministic Numerics (RMSNorm, MatMul, Attention)
========================================

This document describes the deterministic mode added for ggml/llama.cpp and the guarantees we currently make for RMSNorm.

Overview
--------

- Run‑to‑run determinism means: same inputs, same software stack → bitwise‑identical outputs.
- Batch invariance means: the result for a given row does not change when other rows are present in the batch (i.e., reduction order per row is fixed and independent of batch size).
- Current scope: RMSNorm (all backends), MatMul (CUDA), and Attention forward (CUDA) under `GGML_DETERMINISTIC`.

What We Guarantee (Current Scope)
---------------------------------

- RMSNorm forward (and its common fused variants RMSNorm+MUL[+ADD]) are batch‑invariant and bitwise deterministic on supported backends (CPU, CUDA, Vulkan, Metal, SYCL/OpenCL) for a fixed model shape.
- Within a given backend on a given machine and build, re‑running the same RMSNorm invocation yields identical bits.

What We Do Not Guarantee (Yet)
------------------------------

- Cross‑device or cross‑driver bitwise parity. Different GPU models/driver versions or CPU instruction sets may produce different bit patterns. For parity across hosts, pin container image, drivers, compiler versions, and disable/align fast‑math or codegen heuristics as needed.
- Determinism for attention on non‑CUDA backends (Metal/Vulkan/OpenCL/HIP) and for quantized K/V in all cases (planned in 03B/03C).

How To Enable Deterministic Mode
--------------------------------

You can enable determinism at runtime or build time.

- Runtime (recommended):
  - CLI: add `--deterministic` to `llama-cli` or `llama-server`. This sets `GGML_DETERMINISTIC=1` in the process.
  - Environment variable: `GGML_DETERMINISTIC=1` before running any tool using ggml.

- Build time (forces it across the library):
  - `-DGGML_DETERMINISTIC=ON` to CMake.

Examples
--------

- Default CPU build with runtime determinism:

```
scripts/build-in-container.sh
build-container/bin/llama-cli --deterministic -m <model.gguf> -p "Hello" -n 32
```

- Enable at build time:

```
CMAKE_ARGS='-DGGML_DETERMINISTIC=ON' scripts/build-in-container.sh
```

- With CUDA (example arch=86):

```
CMAKE_ARGS='-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86' scripts/build-in-container.sh
GGML_DETERMINISTIC=1 build-container/bin/test-rmsnorm-determinism
```

What Changes Under The Hood
---------------------------

- A new helper `ggml_is_deterministic()` returns true if either the library was built with `GGML_DETERMINISTIC` or the `GGML_DETERMINISTIC` environment variable is set to a truthy value.
- RMSNorm: implementations are already batch‑invariant: per‑row reductions are kept within a single block/workgroup or a serial loop, avoiding atomics or split‑reductions that would change reduction order with batch size.
- The CLI adds `--deterministic` which sets the environment flag.

MatMul (CUDA)
--------------

- Policy: when `ggml_is_deterministic()` is true, CUDA matmul never uses cuBLAS GEMM (including strided/batched). This avoids split‑K and algorithmic variance in accumulation order.
- Dispatcher changes:
  - Prefer `mmf` when eligible (N ≤ 16, alignment holds). This path is already batch‑invariant.
  - Otherwise, use a deterministic `mmvf` fallback that tiles output columns in fixed 8‑wide groups left→right, calling a stable reduction kernel per tile.
  - Quantized matmul is unchanged for now (stretch goal).
- Supported dtypes: F32, F16, BF16 for `mul_mat`; `src1` is promoted/handled as F32.

Testing
-------

- Unit tests:
  - `tests/test-rmsnorm-determinism.cpp` (RMSNorm invariance).
  - `tests/test-matmul-determinism.cpp` (CUDA only; program skips if CUDA not present):
    - Batch‑size invariance: compare first output column for B=1 vs B∈{4,16,33}.
    - Cross‑run determinism: same inputs twice → identical bits.
    - Dtypes: F32, F16, BF16; shapes chosen to exercise both `mmf` and wide `mmvf` tiling.

Testing
-------

- Unit test: `tests/test-rmsnorm-determinism.cpp`.
  - Batch‑size invariance: compares the first row of outputs for `B=1` and `B∈{3,8,32}` bitwise.
  - Cross‑run determinism: repeats the same call and compares outputs bitwise.
  - Enumerates all available backends; prints `[OK] BACKEND_NAME` on success.

Run the test in the container after building:

```
scripts/build-in-container.sh
ENGINE=${ENGINE:-podman} IMAGE=${IMAGE:-docker.io/library/fedora:41} \
  $ENGINE run --rm -v "$(pwd):/src:Z" -w /src/build-container/bin "$IMAGE" \
  bash -lc "./test-rmsnorm-determinism"
```

Notes & Caveats
---------------

- Determinism currently covers RMSNorm, MatMul (CUDA), and Attention forward (CUDA) when enabled. End‑to‑end inference also depends on scheduler choices and fused kernels.
- Performance: deterministic RMSNorm uses the existing per‑row reduction tree, which is already efficient. We do not change performance characteristics in this scope.
- Performance (MatMul/CUDA): avoiding cuBLAS may reduce throughput for some shapes; disable determinism to restore peak speed.
- If you add new RMSNorm variants, keep reductions per row within a single block/workgroup and avoid batch‑size‑dependent split strategies. In deterministic mode, prefer a single reduction policy per row.

Attention (CUDA)
----------------

- Policy in deterministic mode:
  - Dispatch avoids algorithm switching and uses kernels with one query column per block (vector paths) when available; otherwise a tile variant.
  - `launch_fattn` enforces `parallel_blocks = 1` and disables `stream_k`, so no cross‑block combination occurs. This fixes the reduction order and batch invariance.
  - Masks, ALiBi, sinks, and GQA are supported.
  - K/V dtypes:
    - F16 K/V: preferred path is vec‑f16 (or vec‑f32 if precision is forced to F32); tile fallback remains deterministic but slower.
    - Quantized K/V: supported via vec kernels for selected shapes. Minimal guaranteed coverage: D=128 with pairs q4_0/q4_0 and q8_0/q8_0. Unsupported quantized shapes will error in det mode (no tile fallback for quantized K/V).
    - Note: F16 K/V may automatically fall back to the deterministic tile path; quantized K/V does not have a tile fallback.
  - Special head sizes: D ∈ {80, 96, 112, 576} are not yet supported in deterministic mode because current MMA kernels process multiple columns per block (not batch‑invariant). Use D∈{64,128,256} or disable determinism. This is planned follow‑up work.
- Supported shapes (03A):
  - Head sizes D ∈ {64, 128, 256}; KV length must be a multiple of 256.
  - Typical LLaMA head counts and GQA ratios (e.g., 8 heads; GQA {1,2,4}).
  - Mask must be padded to `GGML_KQ_MASK_PAD` (64) and be at least `N` (queries) in length.
  - 03B additions:
    - Quantized K/V: D=128 with q4_0/q4_0 and q8_0/q8_0, KV ∈ {256, 1024}, B ∈ {1,2,8,33}. Additional pairs may be available when built with `GGML_CUDA_FA_ALL_QUANTS`.
    - Special head sizes: not supported in deterministic mode; experimental via `GGML_DETERMINISTIC_ATTENTION_ALLOW_MMA=1` only.
- Caveats:
  - Throughput is lower than default (no multi‑block combine and no stream‑k).
  - Some shapes may fall back to deterministic tile with additional slowdown.

Quick test run (CUDA)
---------------------

Build with CUDA (choose correct arch id, e.g., 86=Ampere, 89=Ada):

```
ENGINE=docker IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
CMAKE_ARGS='-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86' \
scripts/build-in-container.sh
```

Run the attention determinism test on a specific GPU (index 2 in this example):

```
ENGINE=docker IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
$ENGINE run --rm --gpus all -e CUDA_VISIBLE_DEVICES=2 \
  -v "$(pwd):/src" -w /src/build-container/bin "$IMAGE" \
  bash -lc './test-attention-determinism'
```

Debug controls (optional)
-------------------------

- `GGML_DETERMINISTIC_ATTENTION_FORCE_VEC=1` forces the deterministic dispatcher to take a vec path when possible.
- `GGML_DETERMINISTIC_ATTENTION_FORCE_TILE=1` forces the deterministic dispatcher to take the tile path (F16 K/V only) and logs an info message once.
- `GGML_DETERMINISTIC_ATTENTION_ALLOW_MMA=1` experimental: allows MMA path for special head sizes when available. Not guaranteed batch‑invariant yet; prefer OFF for strict determinism.


Roadmap
-------

- Broaden deterministic attention coverage (quantized K/V; additional head sizes) and extend to other backends (HIP/Metal/Vulkan/OpenCL).

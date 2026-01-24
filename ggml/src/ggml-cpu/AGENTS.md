# ggml/src/ggml-cpu/AGENTS.md (iFairy ARM 3W LUT rules)

This directory is hot-path CPU code. Prefer minimal diffs, keep routing conditions explicit, and always preserve correctness.

## Non-negotiable semantic invariant

Must match the ggml baseline exactly: compute `w * conj(x)` (NOT `w * x`). See `IFAIRY_ARM_3W_LUT_DESIGN.md`.

## Current routing constraints (as implemented)

- Compile-time: `GGML_IFAIRY_ARM_LUT` (CPU-only; CMake disables other backends when enabled)
- Platform: LUT route requires `__aarch64__` + `__ARM_NEON__` (otherwise fall back)
- Shape gate: `K % QK_K == 0` with `QK_K=256`
- Supported activations: `GGML_TYPE_F32` (bf16-pair complex container) or `GGML_TYPE_IFAIRY_Q16`
- Output type: `GGML_TYPE_F32` (written as bf16-pair when `pack_bf16=true`)

Primary integration points:
- `ggml/src/ggml-cpu/ggml-cpu.c::ggml_compute_forward_mul_mat()`
- LUT implementation: `ggml/src/ggml-ifairy-lut.h`, `ggml/src/ggml-ifairy-lut.cpp`, `ggml/src/ggml-ifairy-lut-{transform,preprocess,qgemm}.cpp`
- Index encoding: `ggml/src/ggml-quants.c` (3W 6-bit pattern)

## Runtime toggles (current implementation)

- `GGML_IFAIRY_LUT=0/1` (enable unless explicitly set to `0`)
- `GGML_IFAIRY_LUT_DEBUG=0/1`

V2 keeps a single production LUT path and removes layout/kernel/tiling knobs to reduce surface area. Do not add new knobs unless strictly necessary.

If adding a new knob, document it in `IFAIRY_ARM_3W_LUT_V2_STATUS.md` and keep a safe default.

## Formatting & Static Analysis (required)

Follow repo-root `AGENTS.md` for `git clang-format` / `clang-tidy` (diff-only, and only on the `.c/.cpp` files you touched).

## Validation gates (required for any LUT change)

1) Release build:
- `cmake --build build-rel -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)` (multi-config: add `--config Release`)
- `cmake --build build-rel-lut -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)` (multi-config: add `--config Release`)

2) Unit/functional test:
- `./build-rel/bin/test-ifairy`
- `./build-rel-lut/bin/test-ifairy`

3) CLI sanity (quick smoke) + bench tok/s baseline:
- `GGML_IFAIRY_LUT=1 ./build-rel-lut/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv`
- `GGML_IFAIRY_LUT=1 ./build-rel-lut/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 1 --no-warmup`

Edge-case regression coverage is in `tests/test-ifairy.cpp` (alignment, small/large dims, env semantics, transform concurrency).

## Performance claims

- Use `eval tok/s` only, and always include the full command + env.
- Record results in `IFAIRY_ARM_3W_LUT_V2_STATUS.md` (or link to raw logs/TSV paths).

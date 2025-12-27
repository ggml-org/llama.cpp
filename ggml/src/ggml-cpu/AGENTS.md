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

## Runtime toggles (must preserve)

- `GGML_IFAIRY_LUT=0/1`
- `GGML_IFAIRY_LUT_LAYOUT=legacy|compact|tbl64|merged64|auto`
- `GGML_IFAIRY_LUT_KERNEL=auto|sdot|tbl|merged64`
- `GGML_IFAIRY_LUT_BK_BLOCKS=<int>`
- `GGML_IFAIRY_LUT_BM=<int>`
- `GGML_IFAIRY_LUT_FULLACC=0/1`
- `GGML_IFAIRY_LUT_VALIDATE_STRICT=0/1`
- `GGML_IFAIRY_LUT_DEBUG=0/1`
- `GGML_IFAIRY_LUT_PREFETCH=0/1`
- `GGML_IFAIRY_LUT_PREFETCH_DIST=<int>`
- `GGML_IFAIRY_LUT_PREFETCH_INDEX=0/1`
- `GGML_IFAIRY_LUT_N1_FASTPATH=0/1`
- `GGML_IFAIRY_LUT_COMPACT_N1_UNROLL=2|4`
- `GGML_IFAIRY_LUT_MERGED64_ACC16=0/1`
- `GGML_IFAIRY_LUT_MERGED64_ACC_F32X2=0/1`
- `GGML_IFAIRY_LUT_MERGED64_N1_STREAM_ADD=0/1`
- `GGML_IFAIRY_LUT_MERGED64_N1_FASTPATH=0/1`
- `GGML_IFAIRY_LUT_MERGED64_UNROLL=4|8`
- `GGML_IFAIRY_LUT_MERGED64_UNROLL8_2X4=0/1`
- `GGML_IFAIRY_LUT_DECODE_NTH=<int>`
- `GGML_IFAIRY_LUT_DECODE_THRESHOLD=<int>`

If adding a new knob, document it in `IFAIRY_ARM_3W_LUT_STATUS.md` and keep a safe default.

## Formatting & Static Analysis (required)

Follow repo-root `AGENTS.md` for `git clang-format` / `clang-tidy` (diff-only, and only on the `.c/.cpp` files you touched).

## Validation gates (required for any LUT change)

1) Release build:
- `cmake --build build-rel -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)` (multi-config: add `--config Release`)

2) Unit/functional test:
- `./build-rel/bin/test-ifairy`

3) Strict validation (slow but required):
- `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_VALIDATE_STRICT=1 ./build-rel/bin/test-ifairy`

4) CLI sanity (quick smoke) + bench tok/s baseline:
- `GGML_IFAIRY_LUT=1 ./build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv`
- `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 ./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 1 --no-warmup`

Edge-case regression coverage is in `tests/test-ifairy.cpp` (alignment, small/large dims, env semantics, transform concurrency).

## Performance claims

- Use `eval tok/s` only, and always include the full command + env.
- Record results in `IFAIRY_ARM_3W_LUT_STATUS.md` (or link to raw logs/TSV paths).

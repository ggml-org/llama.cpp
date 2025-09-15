**Deterministic RMSNorm Plan**

**Objectives**
- Make RMSNorm execution batch-invariant and bitwise deterministic on supported backends.
- Provide an opt-in deterministic mode (build- and run-time) without regressing default performance.
- Add tests that guard batch invariance across batch sizes and fused/unfused paths.

**Non‑Goals**
- Attention and matmul determinism (handled in follow-up projects).
- Cross-device bitwise parity (different GPUs/CPUs may still differ unless the full stack is pinned).

**Scope**
- Backends: CUDA, CPU, Vulkan, Metal, SYCL/OpenCL.
- Ops: `RMS_NORM` forward (and fused RMS_NORM+MUL[+ADD]); keep existing backward intact.

**Deterministic Mode Design**
- Build flag: add CMake option `GGML_DETERMINISTIC` (OFF by default) and compile definition `GGML_DETERMINISTIC` for ggml and backends.
- Runtime flag: environment variable `GGML_DETERMINISTIC=1` and an API getter `ggml_is_deterministic()`; CLI alias `--deterministic` in `tools/main` and `tools/server` that sets the env var.
- Behavior in deterministic mode:
  - Never change a row’s reduction strategy based on batch size or transient fusion decisions.
  - Prefer a single stable kernel configuration per shape; existing RMSNorm already uses a per-row, single-block reduction with a fixed intra-block tree. Keep that but assert invariance where appropriate.

**Backends Audit & Decisions**
- CUDA (`ggml/src/ggml-cuda/norm.cu`)
  - Kernels `rms_norm_f32<block_size,...>` reduce per-row inside one block; block_size chosen by `ncols` (hidden size). No atomics or split reductions. Batch-invariant as-is.
  - Deterministic mode: leave algorithm; add comments/asserts to prevent future split reductions or batch-size–dependent changes. Optionally pin block size for each `ncols` branch.
- CPU (`ggml/src/ggml-cpu/ops.cpp`)
  - `ggml_compute_forward_rms_norm_f32` loops per-row and sums serially per row. Batch-invariant.
  - Deterministic mode: no change; add unit tests and comments.
- Vulkan (`ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp` and dispatch in `ggml-vulkan.cpp`)
  - Workgroup does per-row reduction with fixed shared-memory halving; loop count depends on `ncols`. Batch-invariant.
  - Deterministic mode: keep; document the invariant.
- Metal (`ggml/src/ggml-metal/ggml-metal.m`)
  - RMSNorm kernels use SIMD-group reductions per row. Batch-invariant; verify fused paths.
  - Deterministic mode: keep; document and test.
- SYCL/OpenCL
  - Similar per-row patterns; ensure no atomics are used in forward RMSNorm; test.

**Implementation Steps**
1) Add deterministic switches
   - `CMakeLists.txt`: `option(GGML_DETERMINISTIC "Enable deterministic numerics" OFF)`; `target_compile_definitions(ggml PRIVATE GGML_DETERMINISTIC=$<BOOL:${GGML_DETERMINISTIC}>)` and propagate to backends.
   - New helper API: `ggml/include/ggml.h` + `ggml/src/ggml.c`: `bool ggml_is_deterministic();` reads compile define and env var.
   - CLI flags: `tools/main` and `tools/server` add `--deterministic` to set `GGML_DETERMINISTIC=1` in process env.

2) Guard invariance in backends
   - CUDA: add comments and `GGML_ASSERT` that RMSNorm forward does not use atomics or cross-block split reductions; ensure launch parameters depend only on `ncols` and per-row indexing.
   - Vulkan/Metal/SYCL/OpenCL: annotate kernels and dispatch code with one-liner invariance notes; avoid adding batch-size–conditioned variants in deterministic mode.

3) Tests (unit + integration)
   - Location: `tests/test_rmsnorm_determinism.cpp`.
   - Cases (run for each available backend):
     - Batch-size invariance: construct tensor X with shapes `(B,H)` for `B∈{1,3,8,32}`, constant `H` (model dim). Compare `rms_norm(X)[:1]` vs `rms_norm(X[:1])` bitwise.
     - Fused vs unfused: compare `rms_norm(x)*w` vs fused kernel output bitwise for same inputs across `B∈{1,8,32}`.
     - Cross-run determinism: run the same call twice and compare bitwise.
   - Deterministic mode enforcement: tests run with and without `GGML_DETERMINISTIC=1`; bitwise equality required in both for RMSNorm.

4) CI wiring
   - Add a CTest target `rmsnorm_determinism` compiled in standard and CUDA-enabled builds when possible.
   - Update `ci/run.sh` to run the test with CPU-only and with CUDA if `GG_BUILD_CUDA=1`.

5) Documentation
   - Add `docs/DETERMINISM.md` section for RMSNorm guarantees and how to enable deterministic mode.
   - Note caveats: cross-driver/architecture parity not guaranteed unless the full software/hardware stack is pinned.

**Milestones**
- M1: Deterministic flag plumbed; CUDA+CPU tests passing.
- M2: Vulkan+Metal tests passing; CI job green on at least one GPU runner.
- M3: Docs merged; pinning guidance included.

**Risks & Mitigations**
- Future performance optimizations might introduce batch-size–dependent strategies. Mitigate with tests that block non-invariant changes.
- Backend or driver updates could change numerics. Mitigate with CI and clear pinning guidance.

**Acceptance Criteria**
- For any `B1,B2` and fixed `H`, `rms_norm(X[B1])[:1]` equals `rms_norm(X[B2])[:1]` bitwise on all enabled backends.
- Fused and unfused paths produce bitwise identical outputs for RMSNorm+MUL[+ADD].
- Re-running the same RMSNorm invocation yields identical bits.

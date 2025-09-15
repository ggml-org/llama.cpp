**Deterministic MatMul (CUDA) — Plan & TODOs (updated)**

**Scope**
- Deterministic, batch-invariant matmul for `GGML_OP_MUL_MAT` and `GGML_OP_MUL_MAT_ID` on CUDA.
- Dtypes: `F32`, `F16`, and `BF16`; quantized (`mmq`) remains a stretch goal.
- Determinism is opt-in via `ggml_is_deterministic()`.

**Definition Of Deterministic**
- Cross-run determinism: identical bitwise output for the same inputs on the same device/driver.
- Batch-size invariance: the first token’s output for B=1 equals the first row when B∈{4,16,32} bitwise.

**Policy (Deterministic Mode)**
- Forbid cuBLAS-based GEMM (including strided/batched) for matmul when `ggml_is_deterministic()` is true.
- Forbid split-K or any algorithm relying on non-deterministic reductions.
- Use custom kernels (`mmf`, `mmvf`) with fixed reduction and iteration order.
- If an eligible shape would normally hit cuBLAS, force a deterministic fallback that tiles columns and calls custom kernels in a fixed order.

**Code Inventory (Where We Touch)**
- Dispatcher: `ggml/src/ggml-cuda/ggml-cuda.cu`
  - `ggml_cuda_mul_mat(...)` (main selector)
  - `ggml_cuda_mul_mat_id(...)` (Mixture-of-Experts path)
  - cuBLAS helpers: `ggml_cuda_op_mul_mat_cublas(...)`, `ggml_cuda_mul_mat_batched_cublas(...)`
- Custom kernels used deterministically:
  - `mmf.*` (tensor-core tile matmul for up to 16 columns): `ggml/src/ggml-cuda/mmf.cu`, `mmf.cuh`
  - `mmvf.*` (vector/column kernels, groups up to 8 cols): `ggml/src/ggml-cuda/mmvf.cu`, `mmvf.cuh`
- Determinism toggle: `ggml_is_deterministic()` in `ggml/src/ggml.c` (already present).

**Design Notes**
- Normal path may choose cuBLAS (including batched). In deterministic mode we will:
  1) Hard-disable cuBLAS selection in the dispatcher (set the `use_batched_cublas_*` flags to false and skip the cuBLAS fallback branch).
  2) Prefer `mmf` when `ggml_cuda_should_use_mmf(...)` passes (N ≤ 16, dims aligned).
  3) Otherwise, use `mmvf` to process N in fixed tiles (≤ 8 columns per launch) in a deterministic left→right order.
  4) For `MUL_MAT_ID`, route to the same deterministic kernels after the expert/tokens reordering phase (no split-K).
- Both `mmf` and `mmvf` choose block sizes based on K and warp size only; this does not depend on batch size, so batch invariance holds.

**Implementation Steps (status)**
1) Dispatcher gating (deterministic):
   - In `ggml_cuda_mul_mat(...)` and `ggml_cuda_mul_mat_id(...)`, when `ggml_is_deterministic()` is true:
     - Force `use_batched_cublas_* = false`.
     - Never call `ggml_cuda_op_mul_mat_cublas` (route to custom kernels instead).
2) Deterministic fallback (wide-N):
   - Add a helper that tiles `src1`/`dst` along columns and invokes `mmf` (preferred) or `mmvf` in a fixed, serial order.
   - Ensure the helper handles broadcasting/layout the same as the cuBLAS path.
3) Guardrails & visibility:
   - If an internal branch would reach cuBLAS in det mode, log once (debug) and assert in Debug builds.
4) Tests:
   - New `tests/test-matmul-determinism.cpp` (skips if CUDA unavailable):
     - Types: `F32`, `F16`, `BF16`.
     - Shapes: K∈{4096, 8192}, M∈{4096}, N∈{8, 32}. Include a case that normally triggers batched cuBLAS (e.g., multiple samples) to prove we forced custom kernels.
     - Batch invariance: compare row 0 results for B=1 vs B∈{4,16,32}.
     - Cross-run determinism: run twice, compare bitwise.
     - `MUL_MAT_ID` coverage: small MoE-shaped test to exercise the id path.
5) Docs:
   - Extend `docs/DETERMINISM.md` with a MatMul section: policy, supported types/shapes, perf caveats, and how to enable.
   - Mention CLI `--deterministic` effect now applies to matmul.
6) CI:
   - Hook the new test into CTest. Skip gracefully when CUDA isn’t present.

**Acceptance Criteria**
- In deterministic mode, no cuBLAS calls for matmul paths; outputs are bitwise equal across runs and batch sizes for covered shapes.
- Tests pass locally (CPU-only builds skip CUDA tests) and in CI where CUDA is available.
- Documentation clearly states scope and caveats.

**Risks & Mitigations**
- Performance regressions when avoiding cuBLAS: document trade-offs; allow users to disable determinism to regain speed.
- Shape misalignment for `mmf`: fallback to `mmvf` tiling path to retain determinism.

**TODO Checklist**
- [x] Gate cuBLAS in `ggml_cuda_mul_mat(...)` when deterministic.
- [x] Gate cuBLAS in `ggml_cuda_mul_mat_id(...)` when deterministic.
- [x] Implement deterministic column-tiling fallback helper.
- [x] Route dispatcher to fallback when `mmf` not eligible and det mode on.
- [x] Add `tests/test-matmul-determinism.cpp` (F32/F16/BF16; multiple shapes and batch sizes).
- [~] Add `MUL_MAT_ID` deterministic test (optional via `TEST_MATMUL_ID=1`; flip on after broader CI soak).
- [x] Update `docs/DETERMINISM.md` (MatMul).
- [x] Wire CTest target and conditional CUDA skip.

**Follow-ups & Interlocks with Project 03/03C**
- Ensure end-to-end determinism in attention blocks that embed small matmul variants; reuse mmvf tiling where applicable.
- When porting determinism to other backends (03C), mirror matmul policy: forbid backend BLAS planners and split‑K, prefer fixed-order kernels.

**Deterministic RMSNorm — Design and Code Pointers**

**Summary**
- RMSNorm in ggml/llama.cpp already uses per-row reductions with a fixed intra-block tree (or serial loop on CPU). That means it is batch-invariant as implemented today across CUDA, CPU, Vulkan, Metal, and SYCL/OpenCL.
- Work to ship: add an explicit “deterministic mode” switch, write invariance tests, and document guarantees so future optimizations don’t reintroduce batch-size–dependent strategies.

**Why RMSNorm Is Batch‑Invariant Today**
- CUDA (`ggml/src/ggml-cuda/norm.cu`)
  - Kernels `rms_norm_f32<256, ...>` and `rms_norm_f32<1024, ...>` reduce per row within a single block using warp + shared-memory tree reduction. No atomics or cross-block split reductions.
  - Launch config toggles only on `ncols` (hidden size). For a given model, `ncols` is fixed; batch size only affects grid size (number of rows), not the per-row reduction order.
- CPU (`ggml/src/ggml-cpu/ops.cpp`)
  - `ggml_compute_forward_rms_norm_f32` iterates a row, accumulates in high-precision scalar, then scales. Deterministic and batch-invariant.
- Vulkan (`ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp`)
  - One workgroup per row; shared-memory halving reduction; loop trip count depends on `ncols` only.
- Metal (`ggml/src/ggml-metal/ggml-metal.m`)
  - SIMDGROUP reduction per row; no atomics; per-row order fixed.
- SYCL/OpenCL
  - Similar per-row design; no atomics in forward RMSNorm.

**Code Locations**
- API: `ggml/include/ggml.h` (`ggml_rms_norm`, `ggml_rms_norm_inplace`).
- CPU: `ggml/src/ggml-cpu/ops.cpp` — `ggml_compute_forward_rms_norm_f32` and dispatcher `ggml_compute_forward_rms_norm`.
- CUDA: `ggml/src/ggml-cuda/norm.cu` — kernels and entry points `ggml_cuda_op_rms_norm`, `ggml_cuda_op_rms_norm_fused`, `ggml_cuda_op_rms_norm_fused_add`.
- Vulkan: `ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp` and dispatch in `ggml-vulkan.cpp`.
- Metal: `ggml/src/ggml-metal/ggml-metal.m` RMS norm kernels and dispatch.
- SYCL/OpenCL: `ggml/src/ggml-sycl/norm.cpp` and `ggml/src/ggml-opencl/ggml-opencl.cpp`.

**Proposed Controls**
- Build flag: `GGML_DETERMINISTIC` (CMake option + compile define) to declare determinism intent across ggml and backends.
- Runtime: `GGML_DETERMINISTIC=1` env var and `ggml_is_deterministic()` helper to let backends pin or reject non-invariant variants if added later.
- CLI: `--deterministic` in `tools/main` and `tools/server` as a convenience alias to set the env var.

**Tests To Add**
- Batch invariance: for `B∈{1,3,8,32}`, fixed `H`, check `rms_norm(X)[:1]` equals `rms_norm(X[:1])` bitwise.
- Fused path equivalence: `rms_norm(x)*w` equals fused RMSNorm+MUL (and +ADD) bitwise.
- Cross-run stability: same inputs → identical bits across repeated runs.

**Risks**
- Future perf work could introduce batch-size–conditioned strategies (e.g., split reductions). Tests and assertions in deterministic mode will block this.
- Cross-driver/arch variance isn’t solved by this change; must pin stack for cross-machine parity.

**Decision Record (ADR)**
- We will not change RMSNorm algorithms; we will formalize deterministic mode and add tests. This keeps performance for default builds and adds guarantees for evaluation and RL workflows when enabled.

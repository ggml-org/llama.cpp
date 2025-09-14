Deterministic MatMul (CUDA) — executive summary and next steps

Summary
- We will make CUDA matmul paths deterministic and batch-invariant by disabling cuBLAS usage in deterministic mode and routing to custom kernels (`mmf`, `mmvf`) with fixed reduction order and fixed column tiling. This avoids split-K and batched GEMM reductions that can introduce non-deterministic accumulation order.

Key files and selectors
- Dispatcher: `ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_mul_mat`, `ggml_cuda_mul_mat_id`).
- cuBLAS helpers: `ggml_cuda_op_mul_mat_cublas`, `ggml_cuda_mul_mat_batched_cublas_*` (will be bypassed in deterministic mode).
- Custom deterministic kernels: `mmf.*` (tile matmul up to 16 cols), `mmvf.*` (vector/column groups up to 8 cols).
- Determinism toggle exists: `ggml_is_deterministic()` (Project 01).

Plan & TODOs
- A full plan with acceptance criteria and a checkbox TODO list is in `projects/02-deterministic-matmul/plan.md`.
- Implemented: Dispatcher gating, deterministic MMVF tiling fallback, tests, and docs. Optional MoE (`mul_mat_id`) test is included but off by default pending further evaluation; enable with `TEST_MATMUL_ID=1`.

Tests & Docs
- New CUDA tests validate batch-size invariance and cross-run determinism for F32/F16/BF16 across multiple shapes and batch sizes. Docs extended in `docs/DETERMINISM.md`.

Status
- Built and ran in container with GPU passthrough on mixed Ampere (A4000) and Ada (RTX 2000E Ada) GPUs. All CUDA matmul determinism tests passed; RMSNorm determinism tests pass on CPU and CUDA.

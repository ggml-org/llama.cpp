# Vulkan QLoRA Training Implementation Plan

## Goal

Make QLoRA training stay on the Vulkan backend for the backward graph, especially
through quantized frozen weights and LoRA gradient computation. The main target is
to avoid CPU fallback and GPU/CPU synchronization during each optimizer step.

This is a backend parity project with CUDA's training-critical paths, not a broad
generic Vulkan tuning pass.

## Current Assessment

CUDA is fast because it has dedicated training paths for:

- Quantized `OUT_PROD`, including GPU dequantization and cuBLAS GEMM.
- `OUT_PROD_ID` for scattered MoE expert gradients.
- `MUL_MAT_ID` support for MoE paths.

Vulkan already has substantial `MUL_MAT` and `MUL_MAT_ID` coverage, including
quantized variants. The major missing training op is `GGML_OP_OUT_PROD`. Dense
QLoRA backward uses this heavily for:

- LoRA weight gradients.
- Activation gradients through frozen quantized base weights.

Because Vulkan does not currently support `OUT_PROD`, the scheduler must place
those nodes on CPU or another backend. That creates synchronization and transfer
costs that dominate training time.

For dense Gemma/Llama-style QLoRA, plain `OUT_PROD` is the first priority.
`OUT_PROD_ID` matters for MoE models and can be implemented after dense models
are working well.

## Phase 0: Baseline And Instrumentation

Before changing kernels, add enough logging to prove where time is going.

Tasks:

- Add optional QLoRA graph placement logging:
  - op name
  - tensor shape
  - tensor type
  - selected backend
  - transfer/copy boundaries
- Add optional per-op timing where backend support exists.
- Record step time broken down by forward, backward, optimizer, and save/checkpoint.

Baseline cases:

- Dense Gemma or Llama, Q4_K.
- LoRA rank 8, 16, and 32.
- `n_ubatch` 128, 256, and 512.
- Vulkan, CUDA, and CPU comparison where available.

Expected fallback ops:

- `GGML_OP_OUT_PROD`
- Quantized `OUT_PROD`
- Possibly small backward elementwise chains
- `OUT_PROD_ID` only for MoE models

Acceptance:

- We can list every CPU fallback op in a QLoRA step.
- We can show how much time is spent around `OUT_PROD` and backend transfers.

## Phase 1: Vulkan OUT_PROD For F32

Status: done for `F32 x F32 -> F32`, including batched/broadcasted shapes and
transposed `src1` coverage in `test-backend-ops`.

Implement basic `GGML_OP_OUT_PROD` on Vulkan for:

- `src0 = F32`
- `src1 = F32`
- `dst = F32`

Files likely touched:

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- `ggml/src/ggml-vulkan/vulkan-shaders/out_prod.comp`
- `tests/test-backend-ops.cpp`

Implementation outline:

- Add a Vulkan compute shader for tiled outer-product/GEMM behavior.
- Add pipeline creation for `out_prod`.
- Add dispatch code in `ggml_vk_graph_compute`.
- Add `GGML_OP_OUT_PROD` to `ggml_backend_vk_device_supports_op`.
- Handle basic 2D and batched 3D/4D shapes used by existing tests.

Important shape behavior:

- `src0`: `[n, k, q1, r1]`
- `src1`: `[m, k, qq, rr]`
- `dst`: `[n, m, qq, rr]`
- `src0` can be broadcast across `qq` and `rr`.
- `src1` may be transposed.

Acceptance:

- Existing `test_out_prod` passes on Vulkan for F32.
- Dense QLoRA can keep F32 LoRA weight-gradient `OUT_PROD` nodes on Vulkan.

## Phase 2: Batched And Broadcast OUT_PROD Coverage

Status: done for F32 coverage exercised by backend tests.

Expand the F32 implementation to match CPU/CUDA shape coverage.

Tasks:

- Support `ne2` and `ne3` batching.
- Support `src0` broadcast factors in dimensions 2 and 3.
- Support transposed `src1`.
- Support nontrivial strides only where Vulkan can do so safely.
- Add tests for broadcast and transposed cases.

Acceptance:

- Existing backend `test_out_prod` variants pass on Vulkan:
  - multiple `bs` values
  - multiple `nr` broadcast values
  - `trans_b = true`
- QLoRA dense backward does not fall back for F32 `OUT_PROD`.

## Phase 3: Quantized src0 OUT_PROD

Status: done for all Vulkan quantized `src0 x F32 -> F32` formats that already
have dequant helpers: `Q1_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`,
`Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`,
`IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`, `MXFP4`, and `NVFP4`.
Backend tests cover quantized shapes accepted by the CPU reference path,
including transposed `src1`; CPU currently rejects quantized broadcast cases
where `src0` has smaller `ne2` or `ne3` than `src1`.

Implement quantized `src0 x F32 -> F32` for `GGML_OP_OUT_PROD`.

Priority order:

1. `Q4_K`
2. `Q5_K`
3. `Q6_K`
4. `Q4_0`
5. `Q4_1`
6. `Q5_0`
7. `Q5_1`
8. `Q8_0`
9. IQ formats later

Implementation options:

- Preferred first pass: dequantize `src0` inside the `OUT_PROD` shader tile and
  accumulate in F32.
- Alternative: dequantize `src0` to a temporary Vulkan buffer, then reuse the F32
  `OUT_PROD` path. This is easier to reason about but may cost extra memory
  bandwidth and allocation overhead.

For QLoRA, start with the quantization type actually used in training. If the
model is Q4_K, implement Q4_K first and validate the training step before adding
the rest.

Acceptance:

- New backend tests pass for quantized `src0` with `src1 = F32` and `dst = F32`.
- Dense QLoRA no longer falls back for activation-gradient `OUT_PROD` through
  frozen quantized base weights.
- Vulkan step time improves materially versus the pre-implementation baseline.

## Phase 4: OUT_PROD_ID For MoE QLoRA

Implement `GGML_OP_OUT_PROD_ID` for MoE expert LoRA gradients.

Status: initial Vulkan correctness path implemented for `F32 x F32` inputs with
`I32` expert ids and `F32` output. Backend tests cover scattered expert-gradient
shapes. This keeps covered MoE expert LoRA gradients on Vulkan, but it is not
CUDA-parity performance; CUDA-style gather-plus-GEMM remains future work.

Inputs:

- `src0 = F32`: token activations
- `src1 = F32`: upstream gradients
- `src2 = I32`: expert ids
- `dst = F32`: gradient tensor per expert

Possible algorithms:

- Simple atomic accumulation shader:
  - easiest correctness path
  - one workgroup per expert/tile
  - atomically accumulate into `dst`
  - likely acceptable as an initial version
- Gather-plus-GEMM:
  - build compact per-expert activation/gradient buffers
  - run tiled GEMM per expert
  - faster, but more complex

Recommendation:

- Start with atomics for correctness.
- Benchmark on MoE QLoRA.
- Replace or augment with gather-plus-GEMM only if atomics are too slow.

Acceptance:

- Add backend tests for `OUT_PROD_ID`.
- MoE QLoRA backward keeps expert LoRA gradients on Vulkan.
- No per-step CPU fallback for `OUT_PROD_ID`.

## Phase 5: Optimizer State Residency

Verify that optimizer state remains on Vulkan.

Status: Vulkan already had an `OPT_STEP_ADAMW` shader. The backend support
predicate now verifies all AdamW state tensors and the 8-element parameter tensor
before advertising support, and the backend AdamW test now matches the core
8-parameter op contract. Focused Vulkan backend testing passes for
`OPT_STEP_ADAMW`.

Audit:

- LoRA tensor buffer placement.
- Gradient tensor placement.
- AdamW moment tensor placement.
- `GGML_OP_OPT_STEP_ADAMW` scheduling.
- Save/checkpoint readback behavior.

Vulkan already has `OPT_STEP_ADAMW`; the important part is verifying that QLoRA
actually schedules it there and does not copy optimizer state to host each step.

Acceptance:

- No per-step CPU copies for optimizer updates.
- Host readback occurs only for save/checkpoint/export.

## Phase 6: Backward Graph Fusion

After `OUT_PROD` is on Vulkan, many small backward ops may become visible in the
profile. Fuse only after the large fallback is gone.

Candidate fusions:

- GELU/GEGLU backward expression chains.
- `mul + mul + add` activation-gradient chains.
- LoRA gradient accumulation patterns.
- RMS norm backward chains.
- Small scale/bias/mul chains around FFN activations.

Acceptance:

- Fusions preserve backend test correctness.
- Fusions improve measured QLoRA step time, not just microbenchmarks.

## Phase 7: Vendor And Shape Tuning

Tune after correctness.

Targets:

- AMD discrete GPUs.
- NVIDIA through Vulkan.
- Intel Arc where available.

Shape priorities:

- hidden sizes 2048, 3072, 4096, 5120
- LoRA rank 8, 16, 32, 64
- `n_ubatch` 128, 256, 512
- Q4_K dense models

Tuning work:

- Tile size selection.
- Subgroup reductions.
- Shared memory usage.
- Cooperative matrix paths where appropriate.
- Separate paths for small rank LoRA gradients versus large base-weight
  activation gradients.

Acceptance:

- Dense Vulkan QLoRA is within a reasonable factor of CUDA on NVIDIA.
- Vulkan QLoRA is competitive and usable on AMD.
- No correctness regressions in backend tests.

## Branch Strategy

Use separate branches so each milestone can be reviewed and benchmarked.

Suggested branches:

- `vulkan-qlora-profile`
  - graph placement and timing logs only
- `vulkan-out-prod-f32`
  - F32 `OUT_PROD`
- `vulkan-out-prod-quant`
  - quantized `src0` support, starting with Q4_K
- `vulkan-out-prod-id`
  - MoE `OUT_PROD_ID`
- `vulkan-qlora-fusion`
  - optional fusion and tuning work

## Dense QLoRA Definition Of Done

- `GGML_OP_OUT_PROD` stays on Vulkan.
- Quantized `src0 x F32 -> F32` `OUT_PROD` stays on Vulkan.
- No repeated GPU/CPU sync inside backward.
- QLoRA loss matches CUDA/CPU within expected tolerance.
- Save/checkpoint remains functional.
- Vulkan step time improves substantially over the current fallback behavior.

## MoE QLoRA Definition Of Done

- `MUL_MAT_ID` remains on Vulkan.
- `OUT_PROD_ID` stays on Vulkan.
- Scattered expert gradients match CPU/CUDA reference.
- No per-step CPU fallback for expert LoRA gradients.

## First Engineering Milestone

Implement and test Vulkan `GGML_OP_OUT_PROD` for:

- `F32 x F32 -> F32`
- batched 3D/4D shapes
- broadcasted `src0`
- transposed `src1`

Then extend that same path to:

- `Q4_K x F32 -> F32`

That is the first branch that should materially improve dense QLoRA training on
Vulkan.

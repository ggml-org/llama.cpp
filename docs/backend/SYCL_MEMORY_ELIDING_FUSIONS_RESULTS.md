# SYCL fusion results

## Scope

This branch is based on master `3d82ef62d`. It contains only SYCL backend compute changes:

- F16 cast plus F32 add fusion
- QSA score generation, ReLU, and head reduction fusion
- QSA gather fusion
- QSA gather, mask add, and radix top-k fusion

There are no changes to `ggml-alloc`, the backend scheduler, or other backends. Intermediate graph tensors are still allocated normally. This preserves the compute-buffer size of master and limits the branch to SYCL execution optimizations.

The QSA mask construction, sparse flash-attention, and generic in-place-view experiments are not included.

## Implementation

The existing SYCL graph execution loop calls `ggml_sycl_fuse()` before dispatching a node. A matched fusion computes the final tensor directly and returns the number of following graph nodes it consumed. The loop skips those nodes. No generic backend callback is needed because all graph tensors retain normal storage.

- Cast plus add reads the F16 input directly in the F32 add kernel instead of executing the cast first.
- QSA gather reads selected score values into the final contiguous layout in one kernel.
- QSA top-k extends the radix selector with row accessors. The selector reconstructs each gathered score-plus-mask value when a radix pass reads it, avoiding the gather and add kernel launches.
- QSA score executes the F32 GEMM in column tiles, applies ReLU, reduces the heads, and writes the final score tensor. This avoids writing and reading the full unreduced product during execution.

The matchers check graph order, tensor types, shapes, strides, view relationships, use counts, and input/output flags. If a graph does not match, normal SYCL dispatch remains unchanged.

All paths use the existing `GGML_SYCL_ENABLE_FUSION` switch.

## Test system

- 3 x Intel Arc Pro B60, driver `1.17.39395+13`
- Intel oneAPI compiler `2026.1`
- Release build with `GGML_SYCL=ON`, `GGML_SYCL_F16=ON`, and `GGML_VULKAN=ON`
- Model: `unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS`
- Layer split: `31,33,36`
- Context: 131072
- Batch/ubatch: 4096/1024
- Flash attention enabled, Q8_0 K/V cache

GPU commands ran through the serialized `gpu_run.sh` harness. No compiler process overlapped a measured arm.

## Correctness

Focused graph tests used `ggml_backend_sched` and compared the fused paths with their unfused equivalents.

| Test | Result |
| --- | --- |
| QSA gather, 1024 x 257 score and 4099 indices | Exact, 0 mismatches |
| QSA top-k, F16 mask, k=131 | Exact indices and values |
| QSA top-k, F32 mask, k=131 | Exact indices and values |
| QSA score, one stream | PASS, max absolute error 1.097e-05 |
| QSA score, two streams | PASS, max absolute error 1.097e-05 |
| QSA score, fusion disabled | PASS, max absolute error 1.097e-05 |

The official SYCL backend operation tests passed for `ADD`, `GET_ROWS`, and `TOP_K`. `CPY` reported four randomized tolerance failures on this branch and three on a clean master build in back-to-back runs, so it is not classified as a branch regression.

The full `test-backend-ops test -b SYCL0` run on this revision is blocked by the same upstream batched-F16 `MUL_MAT_HADAMARD` assertion on master and this branch:

```text
GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1))
```

## Compute buffers

The corrected SYCL-only scope intentionally keeps normal graph allocation.

| Device | Master | SYCL fusion branch | Change |
| --- | ---: | ---: | ---: |
| SYCL0 | 1893.85 MiB | 1893.85 MiB | 0 MiB |
| SYCL1 | 1901.10 MiB | 1901.10 MiB | 0 MiB |
| SYCL2 | 1901.10 MiB | 1901.10 MiB | 0 MiB |
| SYCL host | 415.87 MiB | 415.87 MiB | 0 MiB |

The fusion kernels reduce executed intermediate work and memory traffic, but they do not reduce reserved graph memory in this branch.

## Performance

The master arm ran first. The final SYCL-only arm ran after the allocator and backend-interface changes were removed. Each arm discarded one 16384-token warmup, retained three 16384-token samples, and retained one 65535-token sample. Every request generated 64 tokens and passed the text sanity check.

### 16384-token retained samples

| Build | Prompt processing, tok/s | Decode, tok/s |
| --- | --- | --- |
| Master | 298.327, 295.980, 299.100 | 15.203, 15.391, 15.429 |
| SYCL fusion | 304.005, 300.292, 301.403 | 15.331, 15.560, 15.439 |

| Metric | Master median | SYCL fusion median | Change |
| --- | ---: | ---: | ---: |
| Prompt processing | 298.327 tok/s | 301.403 tok/s | +1.03% |
| Decode | 15.391 tok/s | 15.439 tok/s | +0.31% |

### 65535-token retained sample

| Metric | Master | SYCL fusion | Change |
| --- | ---: | ---: | ---: |
| Prompt processing | 246.356 tok/s | 254.295 tok/s | +3.22% |
| Decode | 8.830 tok/s | 8.799 tok/s | -0.35% |

The 65535-token values are single retained samples, so they do not establish a decode regression or improvement. Startup wall time was 156.4 seconds for master and 141.4 seconds for the SYCL fusion branch, also with one sample per build.

## Result

The final branch is confined to the SYCL backend and preserves master compute-buffer reservations. Focused correctness checks pass. The retained measurements show a small prompt-processing improvement, approximately neutral decode at 16384 tokens, and a single 65535-token decode result within 0.35% of master.

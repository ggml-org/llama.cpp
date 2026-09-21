# SYCL fusion and allocator-elision results

## Scope

This branch extends the SYCL fusion branch with allocator support for eliding fused intermediate tensors:

- F16 cast plus F32 add fusion
- QSA score generation, ReLU, and head reduction fusion
- QSA gather fusion
- QSA gather, mask add, and radix top-k fusion

The generic backend interface lets a backend report which consecutive node outputs its fusion replaces. The scheduler maps those outputs into the allocation graph, and `ggml-alloc` omits their storage while retaining the original sources until the fused consumer runs. Backends without a fusion callback keep their existing behavior. The allocator also lets an in-place operation reuse the storage owned by a full-size, zero-offset view when that view is the only remaining consumer.

The QSA mask construction, sparse flash-attention, and flash-attention staging cap are not included.

## Implementation

The existing SYCL graph execution loop calls `ggml_sycl_fuse()` before dispatching a node. A matched fusion computes the final tensor directly and returns the number of following graph nodes it consumed. The loop skips those nodes. The SYCL callback uses the same structural matchers to report which intermediate outputs will not be written.

- Cast plus add reads the F16 input directly in the F32 add kernel instead of executing the cast first.
- QSA gather reads selected score values into the final contiguous layout in one kernel.
- QSA top-k extends the radix selector with row accessors. The selector reconstructs each gathered score-plus-mask value when a radix pass reads it, avoiding the gather and add kernel launches.
- QSA score executes the F32 GEMM in column tiles, applies ReLU, reduces the heads, and writes the final score tensor. This avoids writing and reading the full unreduced product during execution.

The matchers check graph order, tensor types, shapes, strides, view relationships, use counts, and input/output flags. If a graph does not match, normal SYCL dispatch remains unchanged.

In-place reuse resolves a view to its storage owner for the ownership test. Reuse is allowed only when the parent covers the complete owner allocation at offset zero, and neither the parent nor its owner has another live consumer or view. This behavior is unconditional and does not add an environment flag.

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

| Test | Result | Fusion on | Fusion off |
| --- | --- | ---: | ---: |
| QSA gather, 1024 x 257 score and 4099 indices | Exact, 0 mismatches | 4.02 MiB | 9.04 MiB |
| QSA top-k, F16 mask, k=131 | Exact indices and values | 0.13 MiB | 9.04 MiB |
| QSA top-k, F32 mask, k=131 | Exact indices and values | 0.13 MiB | - |
| QSA score, one stream | PASS, max absolute error 1.097e-05 | 1.00 MiB | 8.03 MiB |
| QSA score, two streams | PASS, max absolute error 1.097e-05 | 1.01 MiB | - |

The official SYCL backend operation tests passed for `ADD`, `GET_ROWS`, and `TOP_K`. `CPY` reported four randomized tolerance failures on this branch and three on a clean master build in back-to-back runs, so it is not classified as a branch regression.

After adding in-place view reuse, the allocator test suite passed all 14 cases. The official SYCL operation tests were repeated: `ADD` passed 101/101 cases, `GET_ROWS` passed all supported cases, and `TOP_K` passed 525/525 cases.

The full `test-backend-ops test -b SYCL0` run on this revision is blocked by the same upstream batched-F16 `MUL_MAT_HADAMARD` assertion on master and this branch:

```text
GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1))
```

## Compute buffers

The server startup reservation used the same model and runtime arguments for master and the allocator-enabled branch.

| Device | Master | Allocator branch | Reduction |
| --- | ---: | ---: | ---: |
| SYCL0 | 1893.85 MiB | 1173.85 MiB | 720.00 MiB (38.02%) |
| SYCL1 | 1901.10 MiB | 1181.10 MiB | 720.00 MiB (37.87%) |
| SYCL2 | 1901.10 MiB | 1181.10 MiB | 720.00 MiB (37.87%) |
| Total | 5696.05 MiB | 3536.05 MiB | 2160.00 MiB (37.92%) |
| SYCL host | 415.87 MiB | 415.87 MiB | 0 MiB |

Fusion-aware allocation removes 464 MiB per GPU. In-place view reuse removes another 256 MiB per GPU in the combined graph, for a total reduction of 720 MiB per GPU.

The earlier combined development branch reached 869.85/877.10/877.10 MiB. The remaining difference is exactly 304 MiB per GPU and came from its separate SYCL flash-attention staging cap. That cap is outside this allocator branch.

## Performance

Each arm discarded one 16384-token warmup, retained three 16384-token samples, and retained one 65535-token sample. The final allocator arm was repeated after enabling in-place view reuse. Every request generated 64 tokens and passed the text sanity check.

### 16384-token retained samples

| Build | Prompt processing, tok/s | Decode, tok/s |
| --- | --- | --- |
| Master | 298.327, 295.980, 299.100 | 15.203, 15.391, 15.429 |
| Allocator fusion with in-place views | 308.682, 303.652, 305.876 | 15.638, 15.555, 15.527 |

| Metric | Master median | Allocator fusion median | Change |
| --- | ---: | ---: | ---: |
| Prompt processing | 298.327 tok/s | 305.876 tok/s | +2.53% |
| Decode | 15.391 tok/s | 15.555 tok/s | +1.07% |

### 65535-token retained sample

| Metric | Master | Allocator fusion | Change |
| --- | ---: | ---: | ---: |
| Prompt processing | 246.356 tok/s | 255.423 tok/s | +3.68% |
| Decode | 8.830 tok/s | 8.885 tok/s | +0.62% |

The 65535-token values are single retained samples, so they do not establish a decode regression or improvement. Startup wall time was 156.4 seconds for master and 147.2 seconds for the allocator-enabled branch, also with one sample per build.

## Result

The allocator-enabled branch removes 720 MiB of compute storage per GPU, or 2.109 GiB across three GPUs, while preserving allocator and focused SYCL correctness. The retained measurements show no throughput regression for this workload. Reaching the earlier 869.85/877.10/877.10 MiB reservation also requires the separate 304 MiB flash-attention staging reduction.

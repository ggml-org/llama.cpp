# SYCL Q8_0 wide-load and ESIMD results

## Scope

The branch is based on commit `3d82ef62d`. It adds two Q8_0 matrix-vector paths:

- A wide-load MMVQ dot product selected by `GGML_SYCL_MMVQ_WIDE=1`.
- An ESIMD DMMV kernel selected by `GGML_SYCL_ESIMD_Q8_0=1` when reorder support is active.

Both flags default to enabled. The ESIMD kernel takes priority for single-vector DMMV. Its work-group size depends on the number of Q8_0 blocks per row. The wide-load MMVQ implementation remains the fallback when the Q8_0 ESIMD path is disabled or not selected.

## Test system

- Three Intel Arc Pro B60 GPUs, driver `1.17.39395+13`.
- Intel oneAPI DPC++/C++ Compiler 2026.1.1.
- Release build with `GGML_SYCL=ON`, `GGML_SYCL_TARGET=INTEL`, and `GGML_SYCL_F16=ON`.
- GPU tests ran with an exclusive GPU lock and no concurrent compiler workload.

## Correctness

The custom Q8_0 matrix-vector harness covered 28 shapes at batch sizes 1, 4, and 64, for 84 cases per mode. It included odd block counts, odd row counts, short rows, and model-sized shapes. All cases passed in each configuration:

| Configuration | `GGML_SYCL_ESIMD_Q8_0` | `GGML_SYCL_MMVQ_WIDE` | Result |
| --- | ---: | ---: | --- |
| Existing paths | 0 | 0 | 84/84 passed |
| Wide-load MMVQ | 0 | 1 | 84/84 passed |
| ESIMD DMMV and wide-load MMVQ | 1 | 1 | 84/84 passed |

The filtered backend test command was `test-backend-ops test -b SYCL0 -o MUL_MAT -p q8_0`. It passed 43/43 cases with the wide-load configuration and 43/43 cases with both features enabled.

The unfiltered SYCL0 backend suite was also run with both feature flags disabled and with both enabled. Both runs completed 9,460 operations and then reached the same assertion in the existing `MUL_MAT_HADAMARD` coverage:

```text
ggml-sycl.cpp:3326: GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1)) failed
```

The identical feature-off and feature-on stopping point indicates that this full-suite limitation is not caused by these Q8_0 paths.

## End-to-end performance

The test used `unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS` across all three GPUs with tensor split `31,33,36`, 131072 context, 4096 batch, 1024 microbatch, flash attention, and Q8_0 K/V cache. Each server start ran three requests. The first request was discarded as a cold prefill, leaving four retained measurements per configuration. Test order was baseline, wide, ESIMD, ESIMD, wide, baseline. All 18 generated responses passed the output sanity check.

| Configuration | Prompt tokens/s, median | Change | Generation tokens/s, median | Change |
| --- | ---: | ---: | ---: | ---: |
| Existing paths | 308.910 | - | 18.407 | - |
| Wide-load MMVQ | 313.133 | +1.37% | 19.437 | +5.60% |
| ESIMD DMMV and wide-load MMVQ | 307.033 | -0.61% | 19.749 | +7.29% |

The prompt result for the ESIMD configuration is within run-to-run variation. The generation result is the relevant improvement for these matrix-vector kernels.

Retained per-run measurements:

| Configuration | Prompt tokens/s | Generation tokens/s |
| --- | ---: | ---: |
| Existing paths | 305.309 | 18.453 |
| Existing paths | 307.332 | 18.361 |
| Existing paths | 311.612 | 18.208 |
| Existing paths | 310.488 | 18.854 |
| Wide-load MMVQ | 311.477 | 19.416 |
| Wide-load MMVQ | 313.156 | 19.316 |
| Wide-load MMVQ | 313.110 | 19.559 |
| Wide-load MMVQ | 315.324 | 19.458 |
| ESIMD DMMV and wide-load MMVQ | 308.576 | 19.858 |
| ESIMD DMMV and wide-load MMVQ | 310.659 | 19.698 |
| ESIMD DMMV and wide-load MMVQ | 305.215 | 19.578 |
| ESIMD DMMV and wide-load MMVQ | 305.490 | 19.799 |

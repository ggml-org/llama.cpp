# P11 - Per-tile fp32 fold for the F16 FA output

**Kind:** port
**Depends on:** P03, which decides whether the 2.4% is worth paying

## Purpose

Kmic-68 found that the attention output accumulated over the whole KV cache in a
`half2` register, and that the error grew as the square root of the context:

| KV length | `half2` accumulator | per-tile fp32 fold |
| --- | --- | --- |
| 4096 | 3.3e-06 | 2.9e-06 |
| 65536 | 2.8e-05 | 3.1e-06 |

Folding into fp32 once per tile kept their fast inner loop and cost 2.4%.

**This is the one Kmic-68 finding that applies to us unchanged**, and it applies
to more of our build than the F16 flag suggests. Their warning that
`fattn-vec.cuh` looks like the same bug but is HIP-only does not apply: we are
the `half2` build. `fattn-vec.hpp:170` declares `sycl::half2 VKQ[...]` on the F16
branch, selected by `-DGGML_SYCL_F16=ON`, and the accumulation is at `:480` and
`:499-500`, inside the KV loop. The fp32 branch at `:184` uses `float2`.

**TILE is `half2` unconditionally.** `fattn-tile.hpp` declares both a
`sycl::half2 VKQ` at `:781` and a `sycl::float2 VKQ` at `:799`, selected by
`SYCL_FAST_FP16`, which `ggml-sycl/common.hpp:52` defines unconditionally (the
comment there says removing it breaks the file). So the default FP32 build's
TILE path accumulates in fp16 across the whole KV cache exactly as the F16 VEC
path does, and scoping this plan to the F16 build would leave that untouched.

## Source

- Kmic-68 `p100-docs/FINDINGS.md`, *Bound fp16 accumulation error*, plus the
  `fattn-tile.cuh` +390/-38 change.

## In this fork

- `ggml/src/ggml-sycl/fattn-vec.hpp:170` (half2), `:184` (float2), `:480` and
  `:499-500` (accumulation), `:445-451` and `:527-534` (the online rescale that
  must survive the fold).
- `ggml/src/ggml-sycl/fattn-tile.hpp`, the TILE family, same question.

## Requirements

- **R11.1** (event-driven) WHEN the FA F16 build accumulates `VKQ` across the KV loop, the <kernel> shall fold the running accumulator into fp32 once per KV tile rather than rounding the running sum to fp16 at every cell.
- **R11.2** (ubiquitous) The <fold> shall preserve the online rescaling the kernel applies when `KQ_max` changes mid-loop.
- **R11.3** (ubiquitous) The VEC kernel shall remain unchanged on its fp32 path, which already accumulates in `float2` at `fattn-vec.hpp:184`. This exemption does not extend to TILE.
- **R11.3a** (ubiquitous) The TILE kernel shall fold its `sycl::half2 VKQ` accumulator declared at `fattn-tile.hpp:781` into fp32 per KV tile, because `SYCL_FAST_FP16` is unconditionally defined and that branch is selected regardless of `GGML_SYCL_F16`.
- **R11.4** (unwanted) IF the per-tile fold costs more than 5% of FA throughput, THEN the <change> shall be reverted and the measurement recorded in the research note.
- **R11.5** (ubiquitous) The <fold> shall not change the shared-memory footprint of the kernel.
- **R11.6** (event-driven) WHEN the fold is enabled, the VEC and TILE kernels shall be measured separately, and TILE shall be tested with `GGML_SYCL_F16` both on and off.

## Acceptance

Turbo oracle nmse and cosine at depths 4096 and 16384 with `GGML_SYCL_F16=ON`
before and after, plus FA throughput over three reps, A770, named driver. TILE
additionally with the flag off, since that is the build where its `half2`
accumulator would otherwise go unaddressed. P03's floor tells us whether the
memory work or the arithmetic is the binding constraint here.

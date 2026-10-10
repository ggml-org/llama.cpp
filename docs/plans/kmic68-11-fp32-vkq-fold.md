# P11 - Per-tile fp32 fold for F16 FA output

**Kind:** SYCL backend feature
**Depends on:** P03

## Context

The `GGML_SYCL_F16` VEC flash-attention branch keeps the running attention
output in `sycl::half2 VKQ` and currently rounds every per-cell contribution
to fp16. Kmic-68 bounded a comparable long-context error by accumulating one KV
tile in fp32 and folding that tile into the running fp16 output once. Its 2.4%
cost was measured on a P100 and is not an A770 expectation.

P11 applies that structure only to `flash_attn_ext_vec` under
`GGML_SYCL_F16`. The persistent running output remains `sycl::half2`; each
outer `k_VKQ_0` iteration gets an fp32 tile accumulator. The non-F16
`sycl::float2` branch and `fattn-tile.hpp` are explicitly unchanged. The
accepted implementation is unconditional in the F16 VEC branch; there is no
runtime knob to maintain after the A770 gates pass.

P03 must first publish
`docs/research/kmic68-a770-fa-memory-floor.md` with valid paired proxy
measurements. Missing or invalid measurement evidence stops P11 before kernel
editing; the proxy timing gap cannot establish that P11 is unreachable.

Source provenance: `Kmic-68/llama.cpp` branch `p100-optimizations`,
`p100-docs/FINDINGS.md`, "Bound fp16 accumulation error".

## EARS requirements

- **R11.1 (Event-driven):** WHEN the F16 VEC FA kernel processes one KV tile, `flash_attn_ext_vec` shall accumulate that tile's V contributions in fp32 registers.
- **R11.2 (Event-driven):** WHEN one KV tile is complete, `flash_attn_ext_vec` shall fold its fp32 tile accumulator into the existing fp16 running output exactly once.
- **R11.3 (Event-driven):** WHEN a tile changes `KQ_max`, the F16 VEC FA kernel shall apply the existing online scale to the prior running output before folding the new tile contribution.
- **R11.4 (Ubiquitous):** The non-F16 VEC FA branch shall retain its existing `sycl::float2` accumulation path.
- **R11.5 (Ubiquitous):** The F16 VEC tile fold shall leave `local_share_mem_size` unchanged.
- **R11.6 (Event-driven):** WHEN the candidate build enumerates affected F16 VEC instantiations, the P11 harness shall require correctness and throughput evidence for every enumerated specialization.
- **R11.7 (Unwanted behaviour):** IF a measured row does not report `GGML_SYCL_F16=1` and `route=VEC` for both builds, THEN the P11 harness shall reject that row as scope evidence.

## Approach

1. Read the completed P03 report before editing. Continue only when schema version,
   build/driver keys, paired VEC manifests, and `p11_eligible=true` all match
   the baseline used here. This is P03's measurement-readiness predicate,
   without a proxy-gap threshold.
2. Keep `sycl::half2 VKQ[ncols][...]` as persistent cross-tile output. Create
   zeroed private `sycl::float2 VKQ_tile[ncols][...]` inside each
   `k_VKQ_0` iteration; add no local/shared-memory array.
3. Preserve KQ softmax, `KQ_sum`, `KQ_max`, and online scale ordering.
   Redirect only F16-branch per-cell V multiply-adds to `VKQ_tile`. At tile end,
   add tile lanes to the already-rescaled persistent value in fp32 and cast to
   `half2` once. Preserve sink rescale, combine, normalization, and output.
4. Leave the non-F16 branch, TILE family, dispatch, work-group dimensions, and
   `local_share_mem_size` expression unchanged. The accepted change is
   unconditional only after the complete affected F16 VEC matrix passes; no
   runtime/build opt-out remains.
5. Discover the affected matrix from the baseline/candidate
   `compile_commands.json` and VEC template-instance/dispatch tables rather
   than a hand-maintained Turbo list. Include every F16 VEC specialization that
   can instantiate the changed branch: standard F16, q4/q5/q8 families, and all
   supported Turbo K/V combinations, head dimensions, query widths, and GQA
   shapes selected by production dispatch.
6. Add a manifest row keyed by build, driver, `type_K`, `type_V`, head
   dimension, query width, GQA ratio, depth, and route. Both binaries must report
   `GGML_SYCL_F16=1` and `route=VEC`; a row routed to TILE, XMX, oneMKL, or
   another implementation is excluded and causes missing-matrix failure, not a
   passing measurement.
7. Extend backend-op/oracle coverage to the full discovered matrix at KV lengths
   4096 and 16384. Use the correct existing tolerance class per type and record
   NMSE, cosine, norm ratio, route, register count, spills, and shared-memory size.
8. Run three synchronized throughput repetitions for every discovered row.
   Register/spill growth is recorded and every row is subject to the same 5%
   median regression rejection gate.

## Critical files & anchors

- `ggml/src/ggml-sycl/fattn-vec.hpp:40-46` - `flash_attn_ext_vec` entry.
- `ggml/src/ggml-sycl/fattn-vec.hpp:167-196` - F16 `half2`, fp32 `float2`, and shared-memory layout.
- `ggml/src/ggml-sycl/fattn-vec.hpp:338-453` - outer KV tile, KQ maximum, and online rescale.
- `ggml/src/ggml-sycl/fattn-vec.hpp:458-503` - V contribution loops and current per-cell running accumulation.
- `ggml/src/ggml-sycl/fattn-vec.hpp:517-536` - sink-time online rescale to preserve.
- `ggml/src/ggml-sycl/fattn-vec.hpp:563-646` - combine, normalization, and output.
- `ggml/src/ggml-sycl/fattn-tile.hpp` - explicit unchanged scope boundary.
- `ggml/CMakeLists.txt:204` and `ggml/src/ggml-sycl/CMakeLists.txt:346-348` - `GGML_SYCL_F16` build selection.
- `ggml/src/ggml-sycl/template-instances/` and `ggml/src/ggml-sycl/fattn.cpp` - discoverable VEC specialization and route matrix.
- `tests/test-backend-ops.cpp` and `tests/test-sycl-turbo-correctness.cpp` - full operator/oracle matrix.
- `docs/research/kmic68-a770-fa-memory-floor.md` - planned P03 gate report consumed by P11.
- `scripts/perf/bench-sycl-fa-fold.py` - planned matrix discovery, deep oracle, and throughput comparison.

## Verification

Build baseline/candidate with identical compiler/runtime flags and
`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, differing only by P11. Run existing
operator gates first:

```bash
timeout 180 ./build-sycl/bin/test-backend-ops -b CPU -o FLASH_ATTN_EXT
timeout 180 ./build-sycl/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT
LLAMA_TEST_TURBO_FA=1 timeout 180 ./build-sycl/bin/test-sycl-turbo-correctness
```

After A770 sole-tenancy setup, run the discovered matrix:

```bash
timeout 3600 python3 scripts/perf/bench-sycl-fa-fold.py --baseline-build ./build-sycl-fa-baseline --candidate-build ./build-sycl --discover-from compile_commands.json --all-affected-f16-vec --depths 4096,16384 --repetitions 3 --require-f16 1 --require-route VEC --p03-report docs/research/kmic68-a770-fa-memory-floor.md --report /tmp/p11-fa-fold.json
```

Expected evidence:

- the script refuses missing/stale/false P03 eligibility;
- baseline and candidate manifests enumerate the same complete affected F16 VEC
  specialization keys with no unmeasured row;
- every row proves `GGML_SYCL_F16=1` and `route=VEC` in both builds;
- every standard, quantized, and Turbo row passes its existing correctness class
  at both depths, with baseline/candidate NMSE, cosine, and norm ratio recorded;
- candidate NMSE is no worse for every row and improves at least one 16384-depth
  Turbo case without reducing cosine;
- every row has three throughput samples and candidate median is no more than 5%
  below its matching baseline;
- compiler data shows unchanged `local_share_mem_size`; registers/spills are
  recorded;
- non-F16 VEC and `fattn-tile.hpp` remain unchanged;
- the post-run two-driver fault gate passes and the service is restarted.

Reject the change on any missing specialization, route/build mismatch, >5%
regression, correctness regression, or false P03 gate. Record rejection rather
than leave a dormant knob or TILE follow-up.

## Assumptions & contingencies

- The fp32 tile accumulator consumes private registers, not shared memory;
  occupancy is measured for every affected specialization.
- P100 figures establish mechanism only; A770 acceptance uses the discovered
  matrix.
- No runtime/build opt-out remains after acceptance; rollback reverts the
  isolated VEC change.
- A TILE-family experiment is a separate plan.

# Fork vs upstream overview on Arc A770

Date: 2026-07-09
Branch: `benchmark/fork-vs-upstream-a770`
Fork tree: `/mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork` at `24ce99ec1`
Upstream comparison tree: `/mnt/mrgr/llama-cpp-sycl-turbo/compare/llama.cpp` at `259f2e2`
Target hardware: Intel Arc A770 via SYCL / oneAPI Level Zero

## Scope

This document compares fork-specific code against upstream master checkout, then maps which fork-only paths are reachable and benchmarkable on Arc A770.

Comparison basis:
- tracked Git diff only, not no-index directory diff
- fork and upstream both built with oneAPI `icx`/`icpx`, `Release`, `GGML_SYCL=ON`, `GGML_SYCL_F16=ON`, `GGML_NATIVE=ON`
- fork build dir: `build-port/`
- upstream build dir: `compare/llama.cpp/build-sycl-a770/`

## High-level diff shape

Tracked diff from upstream commit to fork HEAD large, but not random.
Major churn buckets from scoped diff:
- `tools/`: 291 files, 6520 insertions, 15301 deletions
- `docs/`: 47 files, 4501 insertions, 7089 deletions
- `src/`: 51 files, 5829 insertions, 5138 deletions
- `ggml/`: 76 files, 4331 insertions, 5208 deletions
- `tests/`: 23 files, 3055 insertions, 1221 deletions

Interpretation: fork not tiny patchset. Fork re-shapes runtime, SYCL backend, tests, scripts, docs, plus repo surface pruning.

## Biggest fork-only technical surfaces

### 1. New KV and weight types

Fork adds TurboQuant types in `ggml/include/ggml.h:432-436`:
- `GGML_TYPE_TURBO2_0`
- `GGML_TYPE_TURBO3_0`
- `GGML_TYPE_TURBO4_0`
- `GGML_TYPE_TQ3_1S`
- `GGML_TYPE_TQ4_1S`

This alone creates benchmark surface upstream does not have. Upstream cannot run turbo KV control cases because types do not exist there.

### 2. New graph/operator surface

Fork adds turbo WHT graph path and support checks:
- `GGML_OP_TURBO_WHT` support gate in `ggml/src/ggml-sycl/ggml-sycl.cpp:5698-5706`
- graph-side inverse WHT after attention output in `src/llama-graph.cpp:2104-2115` and `src/llama-graph.cpp:2202-2214`

Meaning: turbo path not only quant type add. Fork changes graph semantics around attention output for turbo V.

### 3. SYCL TurboQuant kernels

Fork adds or materially changes SYCL files for turbo path:
- `ggml/src/ggml-sycl/turbo-quants.hpp`
- `ggml/src/ggml-sycl/turbo-wht.cpp`
- `ggml/src/ggml-sycl/innerq.cpp`
- `ggml/src/ggml-turbo-quant.c`
- changed dispatch in `convert.cpp`, `cpy.cpp`, `set_rows.cpp`, `mmvq.cpp`, `fattn.cpp`, `fattn-vec.hpp`, `ggml-sycl.cpp`

Reachable stages proven by harness:
- WHT
- turbo decode / copy
- turbo quantize-store (`SET_ROWS`)
- turbo `mul_mat`
- non-FA turbo attention path
- turbo FA path
- XMX FA path

See benchmark/results doc for actual runs.

### 4. Flash-attention router changed

Fork router in `ggml/src/ggml-sycl/fattn.cpp:288-318` adds three fork-only facts:
- turbo KV routed through turbo-aware path
- turbo XMX path opt-in with `GGML_SYCL_FA_XMX`
- XMX limited to same-type K/V and `D in {128,256}`

Exact router facts from source:
- turbo KV requires `K->ne[0] % 128 == 0`
- turbo + `GGML_SYCL_FA_XMX` + same K/V type + `D == 128 || D == 256` selects `BEST_FATTN_KERNEL_XMX`
- otherwise turbo falls back to `BEST_FATTN_KERNEL_VEC`

This made combined harness run mandatory: `LLAMA_TEST_TURBO_FA=1 GGML_SYCL_FA_XMX=1`.

### 5. Core KV-cache policy changed

Fork `src/llama-kv-cache.cpp` adds runtime policy, not only kernels:
- auto-asymmetric K downgrade for high GQA in `:125-166`
- extra turbo rotation tensor overhead in `:185-193`
- layer-adaptive turbo/q8 boundary modes in `:311-372`
- turbo head-dim zero-padding in `:374-411`

Important benchmark consequence:
labels like `turbo2/turbo2` can lie unless env fixed.
Two env knobs must be controlled per run:
- `TURBO_AUTO_ASYMMETRIC=0`
- `TURBO_LAYER_ADAPTIVE=0`

Without them, fork may mutate requested KV layout at runtime.

### 6. Non-FA path changed for quantized V correctness

Fork `src/llama-graph.cpp:2175-2196` fixes non-FA quantized V handling by dequantizing V to F32 before transpose, then applying inverse WHT after contraction in `:2202-2214`.

This path benchmarkable, distinct from FA. Harness probes it explicitly. Good. Needed, because `llama-bench` alone would miss graph-correctness regressions.

### 7. InnerQ groundwork present

Fork adds SYCL-side InnerQ wrapper in `ggml/src/ggml-sycl/innerq.cpp:3-71`.
Current state:
- real SYCL kernel wrapper exists
- fallback to C reference exists when SYCL device unavailable
- harness section still opt-in, not default benchmark path yet

So InnerQ work reachable in code, but still partially gated in validation flow.

### 8. Reachability/benchmark harness surface much larger than upstream

Fork test surface in `tests/CMakeLists.txt:207-233` adds:
- `test-sycl-turbo`
- `test-stress-context`
- `test-sycl-fuzz`
- `test-sycl-stress-deep`
- `test-sycl-turbo-correctness`

Fork also adds `scripts/turbo-quality-gate.sh`, but script defaults wrong for this tree: `scripts/turbo-quality-gate.sh:21-44` points `CORRECTNESS_BIN` at `../build-sycl-fp32/bin/test-sycl-turbo-correctness`, while actual harness here lives under `build-port/bin/`.

## Reachability map for fork-only work

### Reachable now, proven

1. Turbo WHT kernel
   - support gate: `ggml_backend_sycl_device_supports_op`
   - exercised by `test-sycl-turbo-correctness`
   - passed on A770

2. Turbo decode / copy kernels
   - routes in `ggml/src/ggml-sycl/cpy.cpp`
   - exercised by harness `[2]`
   - passed on A770

3. Turbo quantize-store kernels
   - routes in `ggml/src/ggml-sycl/set_rows.cpp`
   - exercised by harness `[2b]`
   - passed on A770

4. Turbo `mul_mat`
   - routes in `ggml/src/ggml-sycl/mmvq.cpp`
   - exercised by harness `[3]`
   - passed on A770

5. Turbo non-FA attention
   - graph path in `src/llama-graph.cpp`
   - exercised by harness `[3b]` and `[3c]`
   - turbo3/turbo4 pass, turbo2 warns

6. Turbo FA VEC path, plus opt-in turbo XMX
   - router in `ggml/src/ggml-sycl/fattn.cpp:288-305`
   - default turbo FA selects VEC; XMX only when `GGML_SYCL_FA_XMX=1`, same-type K/V, `D in {128,256}`
   - exercised by `LLAMA_TEST_TURBO_FA=1 ./build-port/bin/test-sycl-turbo-correctness` and combined XMX run
   - turbo3/turbo4 pass; turbo2 remains expected lossy xfail

7. XMX path
   - router in `ggml/src/ggml-sycl/fattn.cpp:298-317`
   - implementation in `ggml/src/ggml-sycl/fattn-xmx.cpp`
   - exercised by `GGML_SYCL_FA_XMX=1 ./build-port/bin/test-sycl-turbo-correctness`
   - f16 XMX passes oracle

8. Turbo XMX path
   - same router, but needs both env gates
   - exercised by `LLAMA_TEST_TURBO_FA=1 GGML_SYCL_FA_XMX=1 ./build-port/bin/test-sycl-turbo-correctness`
   - reachable and passing for turbo3/turbo4 covered probes

### Reachable in code, not fully benchmarked yet

1. Full multi-model fork-vs-upstream matrix
   - runner: `scripts/bench-a770-fork-unique.py`
   - output dir: `docs/research/a770-fork-unique-2026-07-09/`
   - running during this write

2. InnerQ FA path
   - harness gate exists, but default run still skips
   - needs `LLAMA_TEST_INNERQ=1`

### Intentionally not run yet

1. d=256 generic FA opt-in section
   - harness gate: `LLAMA_TEST_FA256=1`
   - reason: documented A770 hang risk

2. Upstream turbo controls
   - impossible; upstream lacks turbo types

## Benchmarking rule set for this fork

Fork needs per-process benchmarking. Reason:
`src/llama-kv-cache.cpp` uses `static const int adaptive_mode`, initialized once per process. Multi-case one-process runs can become order-dependent for turbo modes.

Safe rule:
- one process per KV/env case
- always log env
- distinguish `default` from `pure`
- distinguish `XMX default` from `XMX pure`

Runner now does this.

## Prior validation results (from RALPH/ASSUMPTIONS docs)

The fork's own docs already carry quality/capacity results this session's throughput bench does not. Attributed, not re-run:

- PPL (CPU-FA, 564-chunk): turbo4 within +0.27% (mistral) / +1.58% (llama31) of f16, beats q4_0 on both. turbo2/turbo3 KILLED on qwen3 MoE (exp divergence / NaN) - turbo4 only on MoE.
- Capacity: turbo4/f16 = 3.79x on both dense 7-8B models, model-invariant. turbo2 = 6.38x max.
- Reframe (binding): turbo KV is a CAPACITY feature, not a speed feature. Prior perf-findings measured turbo losing to f16 at depth.
- InnerQ (P3.2.2): producer path fires live, consumer path never proven (`scale_inv tensor updated finalized=1` = 0). Inapplicable to qwen3 by design (auto-asymmetric K downgrade). Prior runtime proof blocked by an AOT/IGC offload-link failure on the separate `build-turbo-aot` tree; the JIT `build-port` used here builds and runs clean.
- Speed work closed: Tier-1 SLM LUT (-8% at depth, reverted); Tier-3 XMX (SG=16 IGC ICE, fixed at SG=8, still bring-up-slow).

See `2026-07-09-a770-benchmark-results-incremental.md` for the folded numbers and this session's throughput/coherence runs.

## Current verdict

Fork not small delta. Fork adds real runtime policy, real graph transformations, real SYCL kernels, real dispatch changes, real harnesses.

Fork-only work is source-reachable and A770-benchmarkable through:
- `test-sycl-turbo-correctness` (kernel-stage + FA + XMX correctness)
- `llama-bench` (throughput, 60-case matrix complete)
- `llama-completion` deterministic coherence smoke (15/15 coherent)
- fork-only env switches controlling router and KV policy

Reachability proven. Throughput matrix complete. Quality/capacity carried by prior RALPH results. Open: long-context crossover, fresh full-corpus PPL, live InnerQ consumer proof.

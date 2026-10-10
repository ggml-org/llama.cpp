# P03 - Memory-only floor ceiling analysis

**Kind:** experimental methodology
**Depends on:** none
**Gates:** P11

## Context

Kmic-68 measured a 114.5 us loads-only floor for a 164.5 us CUDA q6_K kernel
and used it to prove that a requested 84 us target was unreachable. Those
numbers and that kernel do not transfer to the A770 FA path, but the method does:
retain memory traffic and synchronization, remove the arithmetic under study,
and measure the lower bound instead of arguing from mechanism.

P03 applies that method only to the TurboQuant SYCL VEC flash-attention path.
It introduces an experimental build option,
`GGML_SYCL_FA_MEMORY_FLOOR`, default `OFF`. An enabled build is not
correctness-capable and must never be a production default. The concrete output
is `docs/research/kmic68-a770-fa-memory-floor.md`, the turbo FA ceiling report.

Source provenance: `Kmic-68/llama.cpp` branch `p100-optimizations`,
`OPTLOG.md`, "Ceiling analysis: why 40 t/s is not reachable".

## EARS requirements

- **R03.1 (Ubiquitous):** The `GGML_SYCL_FA_MEMORY_FLOOR` build option shall default to `OFF`.
- **R03.2 (Optional feature):** WHERE `GGML_SYCL_FA_MEMORY_FLOOR` is included in a build, the TurboQuant SYCL VEC FA kernel shall stub all FA arithmetic except a minimal observable anti-elision dependency.
- **R03.3 (Event-driven):** WHEN the memory-floor benchmark completes, the `bench-sycl-fa-floor.py` harness shall report turbo3 and turbo4 floor time in microseconds per iteration for every requested depth.
- **R03.4 (Event-driven):** WHEN a valid floor time is recorded, the turbo FA ceiling report shall compute the corresponding maximum throughput from the matching full-kernel control.
- **R03.5 (Unwanted behaviour):** IF an optimization target requires an iteration time below the measured floor, THEN the turbo FA ceiling report shall classify that target as unavailable for the measured kernel design.
- **R03.6 (State-driven):** WHILE a memory-floor build is running, the `test-sycl-turbo-correctness` harness shall label every emitted result as non-correctness data.
- **R03.7 (State-driven):** WHILE `GGML_SYCL_FA_MEMORY_FLOOR` is `OFF`, the SYCL VEC FA translation unit shall compile the existing kernel body without a floor-mode runtime branch.
- **R03.8 (Unwanted behaviour):** IF the paired device-code manifest differs in a retained memory-path category, THEN the `bench-sycl-fa-floor.py` harness shall reject the floor measurement.
- **R03.9 (Ubiquitous):** The floor kernel shall retain the matching full kernel's global K/V load path.
- **R03.10 (Ubiquitous):** The floor kernel shall retain the matching full kernel's local-memory staging and barrier path.
- **R03.11 (Ubiquitous):** The floor kernel shall retain the matching full kernel's indexing and output-store path.
- **R03.12 (Event-driven):** WHEN the ceiling report is finalized, the P03 report writer shall compute `p11_eligible` from the versioned floor-validity and arithmetic-headroom rule.

## Approach

1. Add `GGML_SYCL_FA_MEMORY_FLOOR` as an `OFF` CMake option beside the
   existing SYCL options in `ggml/CMakeLists.txt`, and propagate a preprocessor
   compile definition through `ggml/src/ggml-sycl/CMakeLists.txt`; do not add a
   runtime or environment-controlled kernel branch.
2. In `flash_attn_ext_vec`, guard only the measured TurboQuant compile-time
   instantiations. Stub KQ/V dequantization and dot products, softmax/exp,
   online rescaling, normalization, and running output accumulation. Preserve
   the original load, staging, barrier, traversal, index, and store statements.
3. Prevent dead-code elimination with one private lane-local integer hash that
   consumes every retained loaded/staged value by bit pattern and is folded into
   the existing non-correctness output store. The hash is the only arithmetic
   allowed beyond address/index work; its instruction cost makes the floor
   conservative and is recorded, never subtracted.
4. Add `scripts/perf/probe-fa-memory-floor.sh` by reusing the real compile-
   command, `IGC_ShaderDumpEnable=1`, LLVM-SPIR-V, `ocloc`, and summary
   pattern from `scripts/perf/probe-q8-load-width.sh`. Probe the actual
   `fattn-vec-instance-turbo3_0-turbo3_0.cpp` and
   `fattn-vec-instance-turbo4_0-turbo4_0.cpp` translation units from both
   builds; do not assume a LUT or staging path that the selected specialization
   does not instantiate.
5. For each full/floor kernel pair, emit a versioned JSON manifest with kernel
   symbol, build ID, compile command hash, route, type, and counts for global
   load messages, SLM load/store messages, gateway/barrier instructions, index/
   address instructions, and global output stores. Require nonzero applicable
   categories and exact full/floor equality for all retained categories; require
   the floor's floating arithmetic count to be lower as a negative control.
6. Add a manifest-validator fixture-pair test, not another kernel mode. First
   require an intact parsed manifest fixture to pass. Copy that fixture, remove
   one required load/barrier/store-category record, and require the validator to
   reject it with exit code 42 and the exact diagnostic
   `missing-required-category`. This distinguishes the expected fail-closed
   predicate from a missing fixture, unknown option, timeout, or parser crash,
   without compiling or shipping a hashless kernel variant.
7. Keep non-turbo VEC, normal builds, and TILE unchanged. Add the early
   `LLAMA_TEST_TURBO_FA_BENCH=1` mode to
   `tests/test-sycl-turbo-correctness.cpp`: fixed VEC shapes, warmup,
   synchronized timing, JSONL, and `correctness_valid=false`, bypassing rather
   than suppressing the ordinary correctness suite.
8. Add `scripts/perf/bench-sycl-fa-floor.py` using the Python standard
   library. It requires a passing paired manifest, runs matching full/floor
   binaries three times for turbo3 and turbo4 at depths 4096 and 16384, and
   writes report schema version 1.
9. Record, per build/driver/route/type/depth,
   `arithmetic_headroom_pct=100*(full_time_us-floor_time_us)/full_time_us` and
   `ceiling_tps=full_tps*full_time_us/floor_time_us`. Set
   `p11_eligible=true` only when every required row is `floor_valid=true`,
   reports `route=VEC`, matches the report build/driver, and has arithmetic
   headroom of at least 5%; otherwise set it false with the failed predicate.
   This boolean permits P11 to run; it does not predict an accuracy improvement
   or replace P11's oracle and 5% performance gates.

## Critical files & anchors

- `ggml/CMakeLists.txt:204-210` - existing SYCL option pattern.
- `ggml/src/ggml-sycl/CMakeLists.txt:332-348` - SYCL compile-definition propagation.
- `ggml/src/ggml-sycl/fattn-vec.hpp:43` - `flash_attn_ext_vec` entry.
- `ggml/src/ggml-sycl/fattn-vec.hpp:94-100` - TurboQuant K/V selection.
- `ggml/src/ggml-sycl/fattn-vec.hpp:167-196` - local-memory layout that the floor build must retain.
- `ggml/src/ggml-sycl/fattn-vec.hpp:331-536` - retained traversal, barriers, and output stores plus every arithmetic site the floor build must stub.
- `tests/test-sycl-turbo-correctness.cpp:188-195` - existing exact opt-in pattern for hang-prone Turbo FA probes.
- `scripts/perf/probe-q8-load-width.sh` - existing compile-command, IGC dump, LLVM-SPIR-V, and `ocloc` pattern.
- `scripts/perf/probe-fa-memory-floor.sh` - planned retained-path manifest parser/validator.
- `scripts/perf/fixtures/fa-floor-valid.json` - planned intact validator fixture accepted before the negative control.
- `scripts/perf/fixtures/fa-floor-missing-category.json` - planned validator negative-control fixture; no extra kernel mode.
- `scripts/perf/bench-sycl-fa-floor.py` - planned full-versus-floor orchestrator.
- `docs/research/kmic68-a770-fa-memory-floor.md` - planned versioned P03 report consumed by P11.

## Verification

First prove the normal AOT build still passes the existing Turbo FA gate:

```bash
LLAMA_TEST_TURBO_FA=1 timeout 180 ./build-sycl/bin/test-sycl-turbo-correctness
```

Configure two otherwise identical AOT builds with the current `AGENTS.md`
compiler/device flags and `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`:

```bash
cmake -B build-sycl-fa-full -GNinja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=acm-g10 -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_FA_MEMORY_FLOOR=OFF
cmake -B build-sycl-fa-floor -GNinja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=acm-g10 -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_FA_MEMORY_FLOOR=ON
ninja -C build-sycl-fa-full test-sycl-turbo-correctness
ninja -C build-sycl-fa-floor test-sycl-turbo-correctness
```

Generate and validate the paired device-code manifest before timing:

```bash
IGC_ShaderDumpEnable=1 timeout 900 scripts/perf/probe-fa-memory-floor.sh build-sycl-fa-full build-sycl-fa-floor /tmp/p03-fa-floor-probe
timeout 60 scripts/perf/probe-fa-memory-floor.sh --check-manifest scripts/perf/fixtures/fa-floor-valid.json || exit 1
set +e
out=$(timeout 60 scripts/perf/probe-fa-memory-floor.sh --check-manifest scripts/perf/fixtures/fa-floor-missing-category.json 2>&1)
rc=$?
set -e
test "$rc" -eq 42
test "$out" = 'missing-required-category'
```

After stopping the service, verifying sole tenancy, naming the driver, and
setting the same AOT/cache environment for both builds, run:

```bash
timeout 900 python3 scripts/perf/bench-sycl-fa-floor.py --full build-sycl-fa-full/bin/test-sycl-turbo-correctness --floor build-sycl-fa-floor/bin/test-sycl-turbo-correctness --manifest /tmp/p03-fa-floor-probe/manifest.json --types turbo3,turbo4 --depths 4096,16384 --repetitions 3 --output docs/research/kmic68-a770-fa-memory-floor.md
```

Expected evidence:

- four keyed shape rows each contain three full times, three floor times,
  medians, derived ceiling/headroom, build IDs, route, and named kernel driver;
- every floor JSON row contains `correctness_valid=false` and no PASS/FAIL
  correctness verdict;
- the manifest reports equal nonzero retained global-load, SLM, barrier,
  indexing, and output-store categories for each actual full/floor VEC pair;
- the manifest validator accepts the intact fixture, then rejects the fixture
  with one required category removed using exit code 42 and the exact
  `missing-required-category` diagnostic, without compiling or executing
  another kernel variant;
- the report labels any target faster than the measured floor as unavailable
  and emits the versioned `p11_eligible` predicate result;
- the post-run two-driver fault gate is clean and the service is restarted.

## Assumptions & contingencies

- The floor is a measured lower bound for the exact VEC shapes and software
  stack in the report, not a universal A770 bandwidth limit.
- The lane-local anti-elision dependency adds some work, so the measured floor
  is conservative. Record its generated instructions rather than subtracting an
  unmeasured correction.
- A floor build that reaches an ordinary correctness test is a harness error;
  fail the run instead of weakening the correctness oracle.
- P12 is not gated by this experiment. P03 gates only P11, which changes the
  same VEC accumulation path.

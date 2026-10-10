# P03 - Memory-path timing proxy

**Kind:** experimental methodology
**Depends on:** none
**Gates:** P11

## Context

Kmic-68 measured a 114.5 us loads-only floor for a 164.5 us CUDA q6_K kernel
and used it to prove that a requested 84 us target was unreachable. Those
numbers and that kernel do not transfer to the A770 FA path, but the method does:
retain memory traffic and synchronization and remove the arithmetic under study.
The anti-elision hash below adds work absent from a correct optimized kernel,
so this plan measures a timing proxy, not a proven lower bound.

P03 applies that method only to the TurboQuant SYCL VEC flash-attention path.
It introduces an experimental build option,
`GGML_SYCL_FA_MEMORY_FLOOR`, default `OFF`. An enabled build is not
correctness-capable and must never be a production default. The concrete output
is `docs/research/kmic68-a770-fa-memory-floor.md`, the turbo FA proxy report.
The option and artifact names retain `floor` for continuity; they do not confer
lower-bound validity.

Source provenance: `Kmic-68/llama.cpp` branch `p100-optimizations`,
`OPTLOG.md`, "Ceiling analysis: why 40 t/s is not reachable".

## EARS requirements

- **R03.1 (Ubiquitous):** The `GGML_SYCL_FA_MEMORY_FLOOR` build option shall default to `OFF`.
- **R03.2 (Optional feature):** WHERE `GGML_SYCL_FA_MEMORY_FLOOR` is included in a build, the TurboQuant SYCL VEC FA kernel shall stub all FA arithmetic except a minimal observable anti-elision dependency.
- **R03.3 (Event-driven):** WHEN the memory-floor benchmark completes, the `bench-sycl-fa-floor.py` harness shall report turbo3 and turbo4 proxy time in microseconds per iteration for every requested depth.
- **R03.4 (Event-driven):** WHEN a valid proxy time is recorded, the turbo FA proxy report shall compute its timing gap from the matching full-kernel control.
- **R03.5 (Ubiquitous):** The turbo FA proxy report shall label target reachability as undetermined by proxy timing.
- **R03.6 (State-driven):** WHILE a memory-floor build is running, the `test-sycl-turbo-correctness` harness shall label every emitted result as non-correctness data.
- **R03.7 (State-driven):** WHILE `GGML_SYCL_FA_MEMORY_FLOOR` is `OFF`, the SYCL VEC FA translation unit shall compile the existing kernel body without a floor-mode runtime branch.
- **R03.8 (Unwanted behaviour):** IF the paired device-code manifest differs in a retained category or normalized retained-path fingerprint, or cannot establish that fingerprint, THEN the `bench-sycl-fa-floor.py` harness shall reject the floor measurement.
- **R03.9 (Ubiquitous):** The floor kernel shall retain the matching full kernel's global K/V load path.
- **R03.10 (Ubiquitous):** The floor kernel shall retain the matching full kernel's local-memory staging and barrier path.
- **R03.11 (Ubiquitous):** The floor kernel shall retain the matching full kernel's indexing and output-store path.
- **R03.12 (Event-driven):** WHEN the proxy report is finalized, the P03 report writer shall compute `p11_eligible` from the versioned measurement-validity rule without a proxy timing threshold.

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
   allowed beyond address/index work. Record its generated instructions, but
   do not infer its latency or subtract an unmeasured correction. This extra
   work prevents treating the measured proxy as an unreachable timing floor.
4. Add `scripts/perf/probe-fa-memory-floor.sh` by reusing the real compile-
   command, `IGC_ShaderDumpEnable=1`, LLVM-SPIR-V, `ocloc`, and summary
   pattern from `scripts/perf/probe-q8-load-width.sh`. Probe the actual
   `template-instances/fattn-vec-instance-tq3-tq3.cpp` and
   `template-instances/fattn-vec-instance-tq4-tq4.cpp` translation units from both
   builds; do not assume a LUT or staging path that the selected specialization
   does not instantiate.
5. For each full/floor kernel pair, emit a versioned JSON manifest with kernel
   symbol, build ID, compile command hash, route, type, and counts for global
   load messages, SLM load/store messages, gateway/barrier instructions, index/
   address instructions, and global output stores. Require nonzero applicable
   categories and exact full/floor equality for all retained categories. Also
   retain a normalized instruction/dataflow record and fingerprint covering each
   memory message's operation, address space/surface, cache policy, access width,
   vector length, lane mask/predicate, and addressed bytes as a function of
   lane/query/KV-loop indices. Include address/index dependencies, loop bounds,
   control-flow edges, barriers, and output-store destinations. Normalize only
   register/label names and incidental code addresses; do not erase stride,
   offset, predication, or ordering differences. Compare the records as well as
   their fingerprints. Unknown descriptors or unprovable address/control
   equivalence fail closed; equal category counts alone never pass. Require the
   floor's floating arithmetic count to be lower as a negative control.
6. Add a manifest-validator fixture-pair test, not another kernel mode. First
   require an intact parsed manifest fixture to pass. Copy that fixture, remove
   one required load/barrier/store-category record, and require the validator to
   reject it with exit code 42 and the exact diagnostic
   `missing-required-category`. This distinguishes the expected fail-closed
   predicate from a missing fixture, unknown option, timeout, or parser crash,
   without compiling or shipping a hashless kernel variant. Additional mutated
   copies preserve all category counts but change one message width/surface,
   address stride, lane predicate, or loop bound. Each must fail with exit 42 and
   `retained-path-mismatch`; an unparseable descriptor must fail with exit 42 and
   `unverifiable-retained-path`. Register/label-only renaming must still pass.
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
   `proxy_gap_pct=100*(full_time_us-proxy_time_us)/full_time_us`. Label this
   as a descriptive gap, not arithmetic headroom or a throughput ceiling. Set
   `p11_eligible=true` only when every required row is `proxy_valid=true`,
   has complete finite positive paired timings and a passing manifest, reports
   `route=VEC`, and matches the report build/driver; otherwise set it false
   with the failed predicate. A small or negative proxy gap cannot reject P11.
   This boolean establishes measurement readiness only; P11's own oracle and
   5% performance gates decide whether to retain that change.

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
cmake -B build-sycl-fa-full -GNinja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=acm-g10 -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER= -DGGML_SYCL_FA_MEMORY_FLOOR=OFF
cmake -B build-sycl-fa-floor -GNinja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=acm-g10 -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER= -DGGML_SYCL_FA_MEMORY_FLOOR=ON
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

- four keyed shape rows each contain three full times, three proxy times,
  medians, descriptive proxy gap, build IDs, route, and named kernel driver;
- every floor JSON row contains `correctness_valid=false` and no PASS/FAIL
  correctness verdict;
- the manifest reports equal nonzero applicable retained categories and matching
  normalized memory-descriptor, byte/address, and control-path records and
  fingerprints for each actual full/floor VEC pair;
- the manifest validator accepts the intact fixture, then rejects the fixture
  with one required category removed using exit code 42 and the exact
  `missing-required-category` diagnostic, without compiling or executing
  another kernel variant;
- same-count descriptor/address/predicate/loop mutations fail with
  `retained-path-mismatch`, unknown descriptors fail with
  `unverifiable-retained-path`, and register/label-only renaming passes;
- the report labels target reachability as undetermined by the proxy and emits
  the versioned `p11_eligible` measurement-readiness result;
- valid fixtures with small, zero, and negative proxy gaps remain eligible for
  P11; invalid manifests or missing timings fail regardless of their gap;
- the post-run two-driver fault gate is clean and the service is restarted.

## Assumptions & contingencies

- The proxy describes only the exact VEC shapes and software stack measured.
  It proves neither a lower time bound nor a maximum achievable throughput.
- A future lower-bound claim needs an independently justified bound on the
  anti-elision overhead and its scheduling effects; instruction counts alone
  do not supply one. Such a proof is outside P03.
- A floor build that reaches an ordinary correctness test is a harness error;
  fail the run instead of weakening the correctness oracle.
- P12 is not gated by this experiment. P03 gates only P11, which changes the
  same VEC accumulation path.

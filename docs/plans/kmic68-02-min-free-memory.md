# P02 - Minimum free memory over a run

**Kind:** measurement methodology
**Depends on:** none

## Context

Kmic-68 found that roughly 250 MiB of apparent headroom at 8% context fill
became negative at full fill. A startup or final sample can therefore miss the
allocation peak; the evidence-bearing value is the minimum free device memory
observed over the complete launch.

This plan adds opt-in benchmark instrumentation, not a production allocation or
throughput policy. `LLAMA_BENCH_MEM_LOG=<path>` enables one JSONL record per
`llama-server` process. `LLAMA_BENCH_MEM_INTERVAL_MS` defaults to 50 and
accepts only whole decimal milliseconds in `[10,1000]`. The sampler uses a new
status-bearing wrapper around `ggml_backend_dev_memory`; the existing void API
cannot distinguish an exact query from SYCL's total-as-free fallback. With the
log variable absent, current server behavior is unchanged.

Source provenance: `Kmic-68/llama.cpp` branch `p100-optimizations`,
`p100-docs/FINDINGS.md`, "How the measurements lied", item 5.

## EARS requirements

- **R02.1 (Ubiquitous):** The `server_memory_sampler` shall retain the minimum free-device-memory value observed during each server process.
- **R02.2 (Event-driven):** WHEN an instrumented server process ends its request loop, the `server_memory_sampler` shall publish that process's minimum and final free-device-memory values.
- **R02.3 (Unwanted behaviour):** IF an A770 headroom claim uses only a sample taken before the first long-context decode, THEN the P02 result validator shall reject the claim as insufficient evidence.
- **R02.4 (State-driven):** WHILE an instrumented server process is active, the `server_memory_sampler` shall sample from after backend initialization and before model allocation through a final sample after the request loop and before cleanup.
- **R02.5 (State-driven):** WHILE `LLAMA_BENCH_MEM_LOG` is configured, the `bench_spec.py` result writer shall persist one record per server process with launch identity, device, interval, sample count, minimum bytes, final bytes, provider provenance, and query-failure count.
- **R02.6 (Unwanted behaviour):** IF a device-memory query, log open, append, or flush fails, THEN the `bench_spec.py` launch result shall mark memory telemetry invalid instead of reporting headroom.
- **R02.7 (State-driven):** WHILE `LLAMA_BENCH_MEM_LOG` is absent, the `server_memory_sampler` shall remain inactive.
- **R02.8 (Ubiquitous):** The status-bearing device-memory query shall distinguish exact Level Zero, exact SYCL, total-as-free fallback, and query failure outcomes.
- **R02.9 (Unwanted behaviour):** IF a sample has fallback or failure provenance, zero total bytes, `free_bytes > total_bytes`, or `0/0` values, THEN the P02 result validator shall invalidate the entire launch record.
- **R02.10 (Unwanted behaviour):** IF `LLAMA_BENCH_MEM_INTERVAL_MS` is empty, malformed, outside `[10,1000]`, or overflows, THEN the server initializer shall fail with the rejected value.

## Approach

1. Add a status-bearing backend query next to `ggml_backend_dev_memory` in
   `ggml-backend.h`. Preserve the existing void wrapper for current callers,
   but return provider provenance and exact/fallback/failure status to P02. In the
   SYCL provider, propagate success from Level Zero or
   `ext_intel_free_memory`; identify the line 145-150 total-as-free path as
   fallback, never exact telemetry.
2. Add a small `server_memory_sampler` owned by `llama_server` in
   `tools/server/server.cpp`, not by `server_context_impl`. For non-router
   servers, resolve the configured SYCL device and start after
   `llama_backend_init()` at line 143 but before
   `ctx_server.load_model(params)` at line 498.
3. Keep the sampler alive across model sleep, destroy, reload, and wake events.
   After `ctx_server.start_loop()` returns at line 565, take the final sample,
   join the polling thread, and append exactly one process record before
   `clean_up()` at line 567. RAII cleanup must also emit an explicitly invalid
   record on model-load or HTTP-start failure without creating a second record.
4. Require `ZES_ENABLE_SYSMAN=1` for P02 launches. Parse the interval with
   complete string consumption into a bounded unsigned integer; do not accept
   signs, whitespace, suffixes, zero, or overflow.
5. On every poll, update minimum/final values only from exact samples. Any
   fallback, failure, impossible size relation, or log failure permanently marks
   the process record invalid; later valid samples cannot clear it.
6. Extend `scripts/perf/bench_spec.py:start_server` to assign one unique log
   path and launch ID per server process, and extend `run_arm` to require
   exactly one matching terminal record. Preserve every launch record instead of
   collapsing to an arm-only aggregate.
7. Add parser and lifecycle tests to `scripts/test_bench_spec.py`: exact Level
   Zero/SYCL records, total-as-free fallback, `0/0`, malformed intervals,
   missing/duplicate records, write failure, and sleep-wake-shutdown with one
   process record.
8. Treat instrumented launches as memory evidence only. Discard their throughput
   fields from performance comparisons; any performance claim must come from a
   separate launch with `LLAMA_BENCH_MEM_LOG` absent.

## Critical files & anchors

- `tools/server/server.cpp:140-145,484-503,564-567` - backend init, pre-load start, request-loop completion, and pre-cleanup finalization.
- `ggml/include/ggml-backend.h` - existing void query and planned status-bearing compatibility API.
- `ggml/src/ggml-sycl/mem.cpp:122-150` - exact providers and total-as-free fallback provenance.
- `common/speculative.cpp:3199-3244` - existing instantaneous checkpoint query; not whole-process telemetry.
- `scripts/perf/bench_spec.py:442-520` - `start_server` and `run_arm` process ownership.
- `scripts/test_bench_spec.py` - parser, failure-policy, interval, and lifecycle tests.

## Verification

Create two one-prompt JSONL fixtures for `CTX=16384`: one whose server-reported
prompt-token count is 8% of the usable context and one that fills the usable
context while leaving room for the requested decode. Record those token counts
with the fixtures. Use the same model, binary, build, polling interval, kernel
driver, and boot for both launches.

Prerequisites: stop `llama-sycl.cpp.service`, verify sole tenancy on
`/dev/dri/renderD128`, name the xe or i915 driver, set
`ZES_ENABLE_SYSMAN=1`, and apply the post-run fault gate from `AGENTS.md`.
Run one instrumented process per fixture:

```bash
ZES_ENABLE_SYSMAN=1 LLAMA_BENCH_MEM_LOG=/tmp/p02-8pct.jsonl LLAMA_BENCH_MEM_INTERVAL_MS=50 MODE=baseline CTX=16384 REPEATS=1 LAUNCHES=1 PROMPTS=/tmp/p02-8pct-prompts.jsonl OUT_TAG=p02-8pct timeout 1800 python3 scripts/perf/bench_spec.py
ZES_ENABLE_SYSMAN=1 LLAMA_BENCH_MEM_LOG=/tmp/p02-full.jsonl LLAMA_BENCH_MEM_INTERVAL_MS=50 MODE=baseline CTX=16384 REPEATS=1 LAUNCHES=1 PROMPTS=/tmp/p02-full-prompts.jsonl OUT_TAG=p02-full timeout 1800 python3 scripts/perf/bench_spec.py
```

Expected evidence:

- each summary contains exactly one process-lifetime memory record with positive
  exact-sample count, provider provenance, `min_free_bytes`, and
  `final_free_bytes`;
- the full-context process reports no more minimum free memory than the 8%-fill
  process under the same setup;
- at least the full-context record demonstrates the intended distinction by
  reporting `min_free_bytes < final_free_bytes`;
- injected total-as-free, `0/0`, query, and write failures invalidate the
  process and produce no numeric headroom claim;
- a sleep-wake-shutdown scenario emits one record and preserves the minimum
  across reloads;
- invalid interval strings fail startup, while an unsampled control creates no
  sampler record; sampled throughput is excluded from performance evidence.

## Assumptions & contingencies

- The 50 ms default is a measurement knob, not a claim that shorter peaks cannot
  occur. If repeated memory-only runs show interval sensitivity, record it and
  lower the interval within the declared range before using the result as a
  floor.
- Per launch means one `llama-server` process lifetime. Multiple model-load
  epochs and prompts share one record.
- Exact Level Zero or SYCL provider provenance is mandatory. Never infer free
  VRAM from process RSS, unrelated sysfs files, or the total-as-free fallback.
- Periodic polling can perturb timing; P02 records headroom only, and performance
  comparisons use a separate unsampled process.

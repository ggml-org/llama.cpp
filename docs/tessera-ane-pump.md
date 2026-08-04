# Tessera ANE pump

The Tessera ANE pump is the E-core (low-power cluster) lock-free
state machine that drives the multifunction .mlmodelc dispatch
path. It is the W6/W6.5/W7/W8 progression that turns the IOSurface-
backed stateful bundle into a first-class peer of Metal on Apple
Silicon: the host signals input readiness, the pump drives the
Core ML prediction on the E-core, and downstream consumers
(Metal, the host's read path) observe the pump's monotonic
completion counter via the per-slot MTLSharedEvent handles.

This document covers the follow-up work that landed on top of the
initial pump architecture (W6.5 part 2, integrate/ane-iosurface-state).
The five follow-ups are:

| ID | What | Landed in |
|----|------|-----------|
| F4.1 | Route `dispatch_pinned_function` through the pump's atomic CAS | W6.5 part 2 |
| F4.2 | Monotonic completion counter as the per-slot signal value | W7 |
| F4.3 | E-core thread affinity (QOS_CLASS_BACKGROUND) | W6.5 |
| F4.4 | MTP/DFlash bundle manifest sidecar export | this doc |
| F4.5 | Phase 0 profile NDJSON emit | this doc |

The first three landed on the `integrate/ane-iosurface-state` branch
in commits d2d6eebad (F4.1), ca6972630 (F4.3), and f163d4f38 (W7
MTLSharedEvent handoff with the monotonic counter as the signal
value, F4.2). The last two are documented below; the code lives in
`tools/ane-mtp/export_manifest.py` (F4.4) and
`common/ane-mtp.mm` / `common/ane-mtp.h` (F4.5).

## State machine recap

The pump's state machine has four states:

```
   IDLE                 host can write inputs to the pinned slots
      |                 (caller calls ane_pump_signal_input_ready)
      v
   INPUT_READY          pump thread is awakened; it submits the
      |                 Core ML prediction with outputBackings =
      v                 pinned output slots
   ANE_BUSY             Core ML is processing; pump waits via the
      |                 MLFeatureProvider completion path
      v
   OUTPUT_READY         Core ML wrote outputs into the pinned
      |                 slots; pump signals the per-slot
      v                 MTLSharedEvent for downstream (Metal)
   IDLE                 pump returns to idle; the next host
                        transition restarts the cycle.
```

All four transitions are atomic CAS (the weak variant on Apple
Silicon, where the LL/SC primitive makes the spurious-failure
retry loop faster than the strong variant). The pump is
single-producer / single-consumer (SPSC): the host (producer)
signals input readiness; the pump (consumer) drives the cycle.

## F4.1: in-band caller routed through the pump

`dispatch_pinned_function` in `common/ane-mtp.mm` is the canonical
multifunction dispatch path. Before W6.5 part 2 it used
`dispatch_sync` on the program's serial queue directly:

```cpp
// pre-W6.5: direct dispatch on the program's queue
dispatch_sync(program->queue, ^{
    dispatch_pinned_function_locked(program, function_name, ...);
});
```

This worked but the pump's state machine (IDLE / INPUT_READY / ANE_BUSY
/ OUTPUT_READY) was unused on the in-band path: the host's call
synthetically drove the entire cycle without ever transitioning the
state, and the pump's monotonic counter was not incremented. The
asynchronous path went through the pump; the synchronous path did
not, so a host that mixed async and sync would see inconsistent
counters and inconsistent signal values.

W6.5 part 2 routes the synchronous path through the pump:

```cpp
// W6.5 part 2: route through the pump's CAS state machine
if (!ane_pump::signal_input_ready(instance.pump)) {
    return false;  // pump is busy with another submission
}
ane_pump_dispatch_context ctx = { &function_name, ... };
return ane_pump::run(instance.pump, program, instance,
                     dispatch_pinned_function_submit,
                     ane_signal_slot_events,
                     &ctx);
```

The external API is unchanged (still synchronous from the caller's
view). The submit_fn `dispatch_pinned_function_submit` is the
existing `dispatch_pinned_function_locked`; the signal_fn
`ane_signal_slot_events` is the per-slot MTLSharedEvent signaller.
The state machine is the single source of truth for both the
async and sync paths: the host's `signal_input_ready` CASes
IDLE -> INPUT_READY, the pump's `run` drives INPUT_READY ->
ANE_BUSY -> OUTPUT_READY -> IDLE, and the counter increments
on every successful run.

The lock-free contract: the host may safely mix async and sync
calls (they share the same `common_ane_pump` instance) as long as
the host waits for the pump to return to IDLE before signaling
again. The CAS guarantees only one transition lands per state
change; the host's `wait_idle` helper spins until the state
machine reports IDLE.

## F4.2: monotonic completion counter

Before W7, the `MTLSharedEvent` signaled on OUTPUT_READY used
`steady_clock` nanoseconds as the signal value:

```cpp
// pre-W7: signal value is the steady_clock nanosecond timestamp
const auto now = std::chrono::steady_clock::now();
const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(
    now.time_since_epoch()).count();
[event setSignaledValue:value];
```

This worked but the signal value was opaque: a Metal consumer
that observed the event had no way to reason about ordering
except by comparing timestamps, and two signals on the same
event would always have distinct values (the timestamp is
monotonic but two events at the same nanosecond are vanishingly
rare).

W7 replaces the timestamp with the pump's monotonic completion
counter:

```cpp
// W7: signal value is the pump's monotonic completion counter
const uint64_t value = pump.completions.load(
    std::memory_order_acquire) + 1;
dispatch_sync(q, ^{
    signal(program, pump.function_id, value, context);
});
pump.completions.fetch_add(1, std::memory_order_acq_rel);
```

The counter is a `std::atomic<uint64_t>` on `common_ane_pump`,
incremented once per successful `run`. Two signals on the same
pump have strictly distinct values; the counter is dense
(1, 2, 3, ...) so consumers can detect "the Nth completion" by
comparing against the expected value. The counter is the
canonical "completion N" identifier; the timestamp is gone.

Metal consumers can use the counter directly:

```cpp
const uint64_t expected = pump.completions.load() + 1;
[cmd_buf encodeWaitForEvent:event value:expected];
// the IOSurface bytes the ANE wrote are now visible
```

The wait is strict-order: encoding with `value:expected` waits
until the event's signal value is >= `expected`. The pump's
counter is monotonic, so the wait returns only after the
specific Nth completion lands.

## F4.3: E-core thread affinity

The pump's per-function E-core dispatch queue is created with
`QOS_CLASS_BACKGROUND` affinity. The thread that services the
queue inherits the QoS; the OS scheduler places it on the
low-power cluster (the E-cores) on Apple Silicon, off the
critical dispatch path on the main thread.

```cpp
// common/ane-pump.mm:init
dispatch_queue_t q = dispatch_queue_create(label,
    dispatch_queue_attr_make_with_qos_class(
        DISPATCH_QUEUE_SERIAL, QOS_CLASS_BACKGROUND, 0));
dispatch_sync(q, ^{
    pin_current_thread_to_ecore();
});
```

The `pin_current_thread_to_ecore` helper is a
`pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0)` call.
The dispatch_sync on the queue ensures the QoS is set on the
worker thread (not the caller). Once set, the worker thread
stays on the E-core for the lifetime of the pump; the OS
scheduler does not preempt the pump for foreground work
because the QoS is the lowest tier.

The runtime payoff: the host's main thread (which is usually
`QOS_CLASS_USER_INITIATED` or `QOS_CLASS_USER_INTERACTIVE`)
issues `signal_input_ready` and immediately returns; the pump
thread takes over, runs the Core ML prediction on the E-core,
signals the per-slot MTLSharedEvent, and returns to IDLE. The
host's main thread is never blocked on Core ML work; the E-core
absorbs the dispatch cost.

`ane_pump::ecore_qos_class(pump)` reads back the current
thread's QoS class via `pthread_get_qos_class_np`. The
test-ane-pump.cpp test 9 calls the helper from a submit
callback running on the E-core thread and asserts the result
is `QOS_CLASS_BACKGROUND`.

## F4.4: MTP/DFlash bundle manifest sidecar

The multifunction ANE architecture pivot requires every
.mlmodelc to ship with an `ane_state_layout.v1.json` sidecar.
The gemma4 prefill bundle does this; the MTP, DFlash, and
hybrid bundles (built by `tools/ane-mtp/export-gemma4-mtp.py`)
did not.

`tools/ane-mtp/export_manifest.py` is the post-export adapter
that fills the gap:

```sh
python3 tools/ane-mtp/export_manifest.py \
    --gguf MTP/mtp-gemma-4-12b-it-BF16.gguf \
    --mlpackage MTP/batch-1.mlpackage
```

The script:
1. Reads the source GGUF and locates the multifunction bundle
   prefix (`mtp.ane.bucket.N.*` for the per-bucket layout, or
   `mtp.ane.bundle.*` for the multifunction layout).
2. Materializes the embedded .mlmodelc to a temp directory by
   walking the `prefix.file.NNNN` tensors and the
   `prefix.file.NNNN.path` string kvs.
3. Reads the materialized .mlmodelc's `metadata.json` and
   builds the `ane_state_layout.v1` manifest via the same
   helper the prefill bundle uses
   (`emit_manifest_from_mlmodelc.build_manifest`).
4. Validates the manifest via `state_layout.StateLayout.from_dict`
   to catch schema violations before writing.
5. Writes the manifest next to the user-supplied .mlpackage
   (default: `<mlpackage_stem>.ane_state.v1.json` in the
   .mlpackage's parent directory).

The emitted manifest carries an `_experimental: true` flag so
downstream consumers can detect the unstable schema. The flag
is additive: the runtime reader ignores unknown fields. The
flag will be removed once the Studio UI consumer (the planned
F5 deliverable) lands and the schema is frozen.

The script is ungated: it works on any of the existing bundles
(MTP, DFlash, hybrid, prefill). The validation step catches
malformed manifests before the sidecar is written, so a bad
export doesn't ship a broken sidecar.

## F4.5: Phase 0 profile NDJSON emit

The dispatch path's `dispatch_pinned_function_locked` measures
three phases via `ggml_time_us`:

| Phase | Window |
|-------|--------|
| `input_prep` | building the input MLFeatureProvider (slot loop + extra_inputs merge + MLDictionaryFeatureProvider construction) |
| `ane_dispatch` | the Core ML `predictionFromFeatures` call (the actual ANE work) |
| `output_read` | reading outputs from the feature provider + zero-copy verification (`dataPointer == pinned.dataPointer` for every output slot) |

The per-phase totals and maxes are accumulated atomically on
the function's `common_ane_compute_instance::phase_stats`. The
host reads them via `common_ane_mtp_program_phase_stats` (a
snapshot of the atomics into a plain `common_ane_phase_stats`
struct).

`--tessera-ane-profile-out PATH` adds streaming NDJSON emit on
top of the atomic accumulation. When the path is set, each
successful dispatch appends three lines (one per phase) to the
file. The line shape:

```json
{"phase":"input_prep","us":1234,"n_tokens":128,"ts":"2026-07-31T12:34:56.123456Z"}
{"phase":"ane_dispatch","us":5678,"n_tokens":128,"ts":"2026-07-31T12:34:56.123456Z"}
{"phase":"output_read","us":90,"n_tokens":128,"ts":"2026-07-31T12:34:56.123456Z"}
```

`n_tokens` is the function's first non-batch input dim
(inferred from the manifest's slot shape). `ts` is an ISO 8601
string with microsecond precision. The file is opened lazily
on the first emit; the dispatch path makes a single atomic
branch on every call.

The CLI flag is added as a global tessera option (not a
subcommand) so any subcommand that issues ANE dispatches (the
prefill path, MTP dispatch, DFlash) can opt in. The C++ side
also reads `TESSERA_ANE_PROFILE_OUT` as a fallback env var so
ad-hoc benchmark scripts can set it without touching the
parser.

The emit is marked experimental. The schema may evolve as
F5 (the Studio UI consumer) and downstream tooling land.
The `_experimental` flag on the sidecar manifest (F4.4) and
the field-stable shape on the profile lines are the two
stabilization points; both will be removed when the consumer
lands.

## Tests

The five ANE tests required by the follow-up:

| Test | Covers |
|------|--------|
| `test-ane-pinned-slot-dispatch` | End-to-end prefill dispatch through the pinned-slot path |
| `test-ane-pump` | Pump state machine, monotonic counter, QOS background |
| `test-ane-slot-event` | Per-slot MTLSharedEvent handoff on OUTPUT_READY |
| `test-ane-state-layout` | ane_state_layout.v1 manifest reader (C++) |
| `test-ane-matmul-w0-spike` | W0 standalone matmul spike (smoke test) |

New tests added by the follow-ups:

| Test | Covers |
|------|--------|
| `test-ane-phase-profile-emit` | Phase 0 profile NDJSON emit (set_output, lazy open, line shape) |
| `test_export_manifest.py` | MTP/DFlash manifest sidecar export (find_bundle_prefix, materialize, validate) |

The `test-ane-pump` tests 1-7 cover the state machine; test 8
covers the monotonic counter (F4.2); test 9 covers the QOS
background affinity (F4.3). The pump's routing of
`dispatch_pinned_function` (F4.1) is exercised by the
end-to-end `test-ane-pinned-slot-dispatch` and
`test-ane-phase-profile` tests, which load a real gemma4
prefill bundle and observe the dispatch go through the pump.

## References

- `docs/tessera-ane-matmul-research.md` - the W0/W1/W2 spike
  that established the multifunction ANE path.
- `docs/tessera-coreml-conversion-design.md` - the
  `ane_state_layout.v1` manifest contract.
- `docs/ane-backend-deep-study.md` - the deep study that
  motivated the E-core pump architecture.
- `common/ane-pump.h` / `common/ane-pump.mm` - the pump
  implementation (lock-free SPSC state machine).
- `common/ane-mtp.h` / `common/ane-mtp.mm` - the dispatch
  path that uses the pump (multifunction `.mlmodelc`).
- `tools/ane-mtp/export_manifest.py` - the F4.4 adapter.
- `tests/test-ane-pump.cpp` - the pump state machine tests.
- `tests/test-ane-phase-profile-emit.cpp` - the F4.5 emit
  test.
- `tools/ane-mtp/test_export_manifest.py` - the F4.4
  adapter tests.

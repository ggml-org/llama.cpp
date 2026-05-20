---
name: llama-threading-optimization-2026-05-20
description: Multi-threading optimization for core llama library: adaptive barrier to eliminate CPU busy-wait, per-thread timing accumulators for observability, and 8 bug fixes (threading + correctness).
metadata:
  type: project-spec
---

# Core Llama Library — Multi-Threading Optimization Design

**Date:** 2026-05-20
**Scope:** `ggml/src/ggml-cpu/ggml-cpu.c`, `src/llama-context.cpp`, `src/llama-graph.cpp`, `src/llama-kv-cache.cpp`, `src/llama-kv-cells.h`
**Approach:** A — Adaptive barrier + per-thread timing + all identified bug fixes

## Problem Statement

The core llama library's CPU threadpool workers in `ggml-cpu.c` use a pure-spin barrier (`ggml_barrier()`) that holds CPU cores at 100% while waiting for slower threads to arrive. Additionally, there is no per-thread timing visibility — it is impossible to observe how long each worker thread spends computing vs. waiting. Under multi-threaded multi-sequence workloads, this wastes CPU cycles and makes performance debugging blind.

## Threading Analysis Findings (Summary)

- **Server/CLI layer**: Clean — uses `std::condition_variable`, `std::mutex`, proper yielding. No busy-wait.
- **ggml CPU backend**: Two busy-wait patterns:
  1. `ggml_barrier()` — indefinite spin loop, 50+ call sites in hot computation paths
  2. `ggml_graph_compute_poll_for_work()` — bounded spin + cond_wait hybrid (already configurable)
- **Timing**: Server has per-slot timing. CPU backend has zero per-thread metrics. llama core tracks only main-thread timing.

## Performance Optimizations (2 Items)

### P1 — Adaptive Barrier in `ggml_barrier()`

**File:** `ggml/src/ggml-cpu/ggml-cpu.c:566-601`

**Current behavior:**

```c
// wait for other threads
while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
    ggml_thread_cpu_relax();
}
```

Threads spin indefinitely with `pause`/`yield` instruction. On imbalanced workloads, faster threads burn 100% CPU waiting.

**New algorithm:**

1. Thread atomically increments `n_barrier`
2. If last thread (`n_barrier == n_threads - 1`): reset counters, increment `n_barrier_passed`, broadcast `cond`, return
3. Otherwise: **spin phase** — loop `tp->poll * 128 * 1024` iterations checking `n_barrier_passed`, with `ggml_thread_cpu_relax()` each iteration
4. If spin expires: **sleep phase** — lock `tp->mutex`, `cond_wait(tp->cond, tp->mutex, 1ms timeout)`, re-check, repeat
5. Last thread's `cond_broadcast` wakes all sleepers

**New threadpool member:**

```c
// Added to struct ggml_threadpool:
atomic_int barrier_passed_round;  // tracks which barrier round is active (avoids ABA)
```

The existing `mutex` and `cond` members are reused for barrier signaling — no new synchronization primitive.

**Trade-off:** The fast path (all threads arrive within ~128K relax iterations ≈ sub-microsecond) is ~5% slower due to the extra spin-counter comparison. The slow path (threads arrive at different speeds, common with NUMA or CFS scheduling) saves ~100% CPU per waiting thread.

### P2 — Per-Thread Timing Accumulators

**Files:** `ggml/src/ggml-cpu/ggml-cpu.c`, `src/llama-context.cpp`, `src/llama-impl.h`

**Metrics per worker thread:**

| Metric | Accumulated At | Meaning |
|--------|---------------|---------|
| `t_compute_us` | `ggml_graph_compute_thread()` entry/exit | Time spent executing kernel operations |
| `t_barrier_us` | `ggml_barrier()` entry/exit | Time spent waiting at thread barriers |
| `t_poll_us` | `ggml_graph_compute_check_for_work()` entry/exit | Time spent polling + waiting for work dispatch |
| `t_idle_us` | Worker loop gap between graph completion and next work | Time idle (sleeping on cond_wait) |
| `n_compute` | Incremented per graph computed | Number of graph evaluations |
| `n_barrier` | Incremented per barrier hit | Number of barrier synchronizations |

**Data structure** — per-thread arrays in `struct ggml_threadpool`:

```c
int64_t * t_compute_us;    // [n_threads + 1] (index 0 = main thread)
int64_t * t_barrier_us;
int64_t * t_poll_us;
int64_t * t_idle_us;
uint64_t * n_compute;
uint64_t * n_barrier;
```

Each worker thread writes to its own index — no contention, no mutex needed. Values are accumulated via `atomic_fetch_add`.

**Instrumentation points** in `ggml_graph_compute_secondary_thread()`:

```
while (true) {
    // ... pause handling ...

    t0 = ggml_time_us();
    ggml_graph_compute_check_for_work(state);  // wait for work
    t_poll += ggml_time_us() - t0;

    t0 = ggml_time_us();
    ggml_graph_compute_thread(threadpool);     // execute kernels
    t_compute += ggml_time_us() - t0;

    t_idle += ggml_time_us() - t_graph_end;    // gap to next iteration
}
```

Inside `ggml_barrier()`:

```c
t0 = ggml_time_us();
// ... barrier wait logic ...
atomic_fetch_add(&tp->t_barrier_us[my_thread_id], ggml_time_us() - t0);
```

**Reporting:**

1. **ggml API**: `ggml_threadpool_print_timings(struct ggml_threadpool)` — prints per-thread table to log
2. **Auto-report**: Call `ggml_threadpool_print_timings()` in `ggml_threadpool_free()` on every shutdown
3. **llama context**: Extend `llama_print_timings()` to print CPU threadpool summary (min/max/avg per metric)
4. **Server**: Extend `/metrics` endpoint with per-thread labels: `llama_cpu_thread_{metric}_{thread_id}_us`

**Reset**: Timings reset on `llama_reset()` and at threadpool creation. New function `ggml_threadpool_reset_timings()` allows explicit reset from any backend. Timings are NOT reset per-graph-eval — they accumulate across the session for observability.

## Bug Fixes (8 Items)

### T1 — Barrier Memory Ordering (Thread Safety)

- **File:** `ggml/src/ggml-cpu/ggml-cpu.c:590-591`
- **Issue:** `n_barrier_passed` loaded with `memory_order_relaxed` in the barrier spin loop. On ARM/POWER (weakly-ordered architectures), the thread can cache a stale value and miss the signal from the last thread, causing indefinite hang.
- **Fix:** Change the load in the spin loop to `memory_order_acquire`. The store by the last thread already uses `memory_order_seq_cst`, which implies release — the acquire load forms the necessary synchronizes-with pair.

### T2 — `~llama_io_read_device()` Not `noexcept`

- **File:** `src/llama-context.cpp:2589-2636`
- **Issue:** Destructor can throw during stack unwinding via `ggml_backend_tensor_copy` or `std::map::at`, calling `std::terminate`. Mirror of the `llama_io_write_host` fix from commit `eedad9cab` that was missed.
- **Fix:** Mark destructor `noexcept`. Wrap buffer copy loop in try-catch, `LOG_ERROR` on failure.

### T3 — `type_v()` Null Dereference on MLA Models

- **File:** `src/llama-kv-cache.cpp:1117`
- **Issue:** `type_v()` unconditionally accesses `layers[0].v->type`, but MLA models set `v = nullptr`. Crashes when any thread inspects cache type.
- **Fix:** `return layers[0].v ? layers[0].v->type : GGML_TYPE_COUNT;`

### T4 — `llama_kv_cells::cp()` Leaves `used` Set Empty

- **File:** `src/llama-kv-cells.h:141-157`
- **Issue:** `cp()` copies `pos`, `ext`, `seq` arrays but not the `used` set. Any future call to `get_used()`, `used_min()`, or `used_max_p1()` on a copied object returns incorrect results.
- **Fix:** After copying arrays, rebuild `used`: `if (pos[j] != -1) used.insert(j);`. Apply to both `cp` overloads.

### T5 — `const_cast` Undefined Behavior in `copy_tensor_async_rows`

- **File:** `src/llama-context.cpp:1548`
- **Issue:** Parameter `const void * dst_data` is mutated via `const_cast<uint8_t *>`. Undefined behavior if caller passes truly const memory.
- **Fix:** Change parameter to `void * dst_data`. Update all callers. Remove `const_cast`.

### T6 — Unsigned Underflow in `can_reuse` Checks

- **File:** `src/llama-graph.cpp:620, 663, 747`
- **Issue:** `mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs` performs unsigned subtraction. When `n_seqs > n_rs`, result wraps to a huge number, silently passing reuse check.
- **Fix:** Guard: `mctx->get_recr()->get_n_rs() >= params.ubatch.n_seqs && (mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs) == s_copy_extra->ne[0]`

### T7 — Null Pointer in `llm_graph_input_pos::can_reuse`

- **File:** `src/llama-graph.cpp:130`
- **Issue:** `pos->ne[0]` accessed without null check. `llm_graph_input_embd::can_reuse` guards with `(!params.ubatch.token) || (tokens && ...)`, but `llm_graph_input_pos` does not.
- **Fix:** `(!pos) || (pos->ne[0] == params.ubatch.n_tokens * n_pos_per_embd())`

### T8 — Missing `v_idxs` Reuse Checks

- **File:** `src/llama-graph.cpp:475, 538, 541, 613, 731, 739`
- **Issue:** Six commented-out `self_v_idxs` reuse checks allow graph reuse with wrong V cache indices — silent data corruption under varying batch sizes.
- **Fix:** Re-enable: `res &= (!self_v_idxs) || (self_v_idxs->ne[0] == params.ubatch.n_tokens);` (null-guarded for models without v_idxs).

## Error Handling

- Bug fixes T1-T8 add defensive guards that prevent crashes, UB, and silent corruption under concurrent or varying workloads.
- Per-thread timing uses `atomic_fetch_add` — if `ggml_time_us()` fails (should not on Linux), values remain 0. No crash path.
- Adaptive barrier: if `cond_broadcast` fails (extremely rare), threads will eventually wake from the 1ms timeout. No deadlock possible.

## Testing

1. **Build:** `cmake -B build -DLLAMA_BUILD_TESTS=ON && cmake --build build -j`
2. **Regression:** `cd build && ctest -C Release --output-on-failure`
3. **Targeted tests:**
   - T1 (barrier memory order): Run with `GGML_CPU_THREADS=8` on any multi-core system, verify no hang under load
   - T2 (destructor noexcept): Force GPU error during IO read, verify no `std::terminate`
   - T3 (MLA type_v): Load DeepSeek model, inspect cache type, verify no crash
   - T6-T8 (graph reuse): Run multi-sequence batch with varying token counts, verify correct output
   - P1 (adaptive barrier): Enable `GGML_CPU_FRONT=1` + multi-sequence load, check CPU utilization per core with `htop` — idle cores should drop to ~0%
   - P2 (timing): Run server with `curl /metrics`, verify per-thread timing metrics appear
4. **Performance validation:**
   - Run `llama-cli --threads 8 -m <model> -p "test"` and compare prompt/generation timing
   - Run `llama-server` with concurrent multi-stream requests (8-16 parallel), verify no regression
   - Check that barrier wait time (`t_barrier_us`) appears in `/metrics` and log output

## Out of Scope

- Per-layer per-thread timing (Approach B) — would require API changes
- RISC-V `spine_barrier_wait()` — platform-specific, low impact
- Medium/low priority items from previous analysis (duplication, code quality)
- `llama-model.cpp`, `llama-sampler.cpp`, `llama-vocab.cpp` (not on threading hot path)

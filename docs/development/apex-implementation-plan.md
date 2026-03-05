# APEX Runtime Scheduling — Implementation Plan

Reference: APEX (arXiv:2506.03296), User Stories in `apex-runtime-scheduling-stories.md`

## Architecture Constraints (from codebase analysis)

Before diving into the plan, key constraints discovered during research:

1. **No runtime tensor reassignment** — Model weight tensors are allocated to backends at
   load time and cannot be moved. This is a hard constraint.
2. **Graph-level backend control** — `graph_get_cb()` in `llama_context` can call
   `ggml_backend_sched_set_tensor_backend()` to force specific ops to specific backends
   during graph build. This is our primary runtime control lever.
3. **Splits execute sequentially** — `ggml_backend_sched_compute_splits()` runs splits
   one after another. True async overlap requires scheduler modifications.
4. **Eval callback exists** — `ggml_backend_sched_set_eval_callback()` fires per-node
   with ask/done phases. Good for profiling but adds sync overhead.
5. **Pipeline parallelism** — `n_copies` enables overlapping input prep with compute
   across iterations, but not within a single graph's splits.

## Implementation Phases

---

### Phase 1: Profiler Hardening (US-5)

**Goal:** Make the existing UMA profiler accurate enough to drive scheduling decisions.

**Files:** `common/uma-profiler.{h,cpp}`

**Changes:**

1. **Add GPU synchronization before timing**
   - The eval callback fires after `ggml_backend_graph_compute_async()` + `ggml_backend_synchronize()`
   - Verify that the scheduler's existing sync (line 1618 in ggml-backend.cpp) is sufficient
   - If not, add explicit sync in the callback's "done" phase

2. **Track batch size in profiler data**
   ```cpp
   struct uma_profiler_data {
       // existing fields...
       int32_t profiled_batch_size = 0;    // batch size during profiling
       bool    needs_reprofile = false;     // set if batch size changed >2x
   };
   ```

3. **Separate decode vs prefill timing**
   - Add a `phase` field to `uma_op_stats` (decode=1 token, prefill=N tokens)
   - APEX's inequality depends on knowing `T_glinear` and `T_gatt` for decode specifically

4. **Validate arithmetic intensity calculations**
   - For MUL_MAT with batch=1: AI ≈ 1 (bandwidth-bound) — verify this
   - For MUL_MAT with batch=512: AI ≈ 512 (compute-bound) — verify this
   - Add a self-test that logs warnings if classification seems wrong

**Test:** Run profiler on a known model, verify that decode MUL_MAT ops classify as
bandwidth-bound and prefill MUL_MAT ops classify as compute-bound.

---

### Phase 2: Critical Inequality Gate (US-2)

**Goal:** Implement APEX's throughput inequality to decide if CPU offload helps.

**Files:** `common/uma-profiler.{h,cpp}`, new `common/apex-scheduler.{h,cpp}`

**Changes:**

1. **Define the APEX decision structure**
   ```cpp
   struct apex_decision {
       bool   cpu_offload_profitable = false;
       double gpu_throughput;     // tokens/sec GPU-only
       double hybrid_throughput;  // tokens/sec with CPU offload
       double ratio;              // N_G / N_C
       double threshold;          // 2*(T_glinear/T_gatt) + 3 + (T_gatt/T_glinear)

       enum strategy_t {
           GPU_ONLY,
           ASYNC_OVERLAP,
           ASYMMETRIC_PIPELINE,
       } strategy = GPU_ONLY;
   };
   ```

2. **Implement the inequality evaluation**
   ```cpp
   apex_decision apex_evaluate_offload(
       double T_glinear_us,  // measured GPU linear layer time
       double T_gatt_us,     // measured GPU attention time
       double T_clinear_us,  // measured CPU linear layer time (if available)
       double T_catt_us,     // measured CPU attention time
       bool   has_prefill    // whether mixed workload
   );
   ```

   For decode-only: `N_G/N_C < 2*(T_glinear/T_gatt) + 3 + (T_gatt/T_glinear)`
   For mixed: use extended inequality with prefill times

3. **Integrate with profiler output**
   - After profiling iterations complete, automatically evaluate the inequality
   - Log the decision: `"APEX gate: cpu_offload=%s ratio=%.2f threshold=%.2f strategy=%s"`

**Test:** Unit test with known timing values, verify inequality produces correct decisions.

---

### Phase 3: Runtime Layer Reassignment via Graph Callback (US-1)

**Goal:** Use profiler + inequality gate to route attention ops to CPU at graph build time.

**Files:** `src/llama-context.cpp`, `common/apex-scheduler.{h,cpp}`

**Changes:**

1. **Extend `graph_get_cb()` for APEX-driven backend selection**

   Currently `graph_get_cb()` only handles norm layers. Extend it:
   ```cpp
   llm_graph_cb llama_context::graph_get_cb() const {
       return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
           ggml_format_name(cur, ...);

           // Existing norm layer logic...

           // NEW: APEX-driven attention offload to CPU
           if (apex_state.active && il >= 0 && apex_state.should_offload_to_cpu(name, il)) {
               ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend_cpu);
           }
       };
   }
   ```

2. **Define offload policy in apex_scheduler**
   ```cpp
   bool apex_state::should_offload_to_cpu(const char * name, int layer) const {
       if (!decision.cpu_offload_profitable) return false;
       // Only offload attention ops for layers in the offload set
       if (!is_attention_op(name)) return false;
       if (layer < offload_start_layer || layer > offload_end_layer) return false;
       return true;
   }
   ```

3. **Add `apex_state` to `llama_context`**
   ```cpp
   struct llama_context::impl {
       // existing fields...
       apex_scheduler_state apex_state;
   };
   ```

4. **Trigger reassignment after profiling completes**
   - In `process_ubatch()`, after profiling iterations are done:
     - Call `apex_evaluate_offload()` with profiled timings
     - If decision changed, set `sched_need_reserve = true` to force graph rebuild
     - The next graph build will use the updated `graph_get_cb()` routing

**Constraint:** This only affects compute graph ops, NOT weight storage. Weights stay
on their original backend. The scheduler handles data transfer (copies) automatically
when an op's backend differs from its input tensor's backend.

**Test:** Verify that after profiling, attention ops for specified layers execute on CPU
backend (check via scheduler split logs).

---

### Phase 4: Asynchronous CPU-GPU Overlap (US-3)

**Goal:** Overlap CPU attention with GPU FFN across layers.

This is the most complex phase and requires scheduler-level changes.

**Files:** `ggml/src/ggml-backend.cpp`, `ggml/include/ggml-backend.h`

**Approach A: Split-level overlap (preferred)**

Currently `ggml_backend_sched_compute_splits()` executes splits sequentially:
```
Split 0 (GPU: FFN layer i)  →  Split 1 (CPU: attn layer i)  →  Split 2 (GPU: FFN layer i+1)  → ...
```

Modify to overlap adjacent splits on different backends:
```
Split 0 (GPU: FFN layer i)  ──────────────────→
                   Split 1 (CPU: attn layer i)  ──────────→
                                          Split 2 (GPU: FFN layer i+1)  ──→
```

**Changes:**

1. **Add `op_overlap` flag to scheduler** (alongside existing `op_offload`)
   ```cpp
   struct ggml_backend_sched {
       // existing...
       bool op_overlap;  // allow overlapping splits on different backends
   };
   ```

2. **Modify `ggml_backend_sched_compute_splits()`**
   - Before launching split N, check if split N-1 is on a different backend
   - If so, launch split N without waiting for N-1 to complete
   - Use events to track completion: record event after each split
   - Before a split reads inputs from a previous split's backend, wait on that event
   - This is exactly how the existing event infrastructure works — we just need to
     not block between splits on different backends

3. **Add public API**
   ```cpp
   GGML_API void ggml_backend_sched_set_overlap(ggml_backend_sched_t sched, bool overlap);
   ```

4. **Enable in llama_context when APEX is active**
   - After APEX decision says `ASYNC_OVERLAP`, enable split overlap on the scheduler

**Approach B: Dual-thread (fallback)**

If split-level overlap is too invasive:
- Use a separate thread for CPU splits (similar to APEX's Python approach)
- Main thread drives GPU splits, worker thread drives CPU splits
- Synchronize via events or condition variables

**Test:**
- Compare output of overlapped vs sequential execution (must be bit-identical)
- Measure decode throughput with overlap enabled vs disabled
- Verify no data races with thread sanitizer

---

### Phase 5: Dynamic Mixed-Workload Scheduling (US-4)

**Goal:** Server mode dynamically selects strategy per batch.

**Files:** `common/apex-scheduler.{h,cpp}`, `tools/server/server.cpp`

**Changes:**

1. **Per-batch strategy evaluation**
   ```cpp
   apex_decision apex_evaluate_batch(
       const apex_scheduler_state & state,
       int n_prefill_tokens,    // tokens in prefill requests this batch
       int n_decode_tokens,     // tokens in decode requests this batch
       double T_glinear_us,     // from profiler
       double T_gatt_us
   );
   ```

2. **Strategy selection logic** (from APEX Algorithm 2):
   - If no CPU decode queue → GPU_ONLY
   - If decode-only → evaluate decode inequality
   - If mixed → evaluate mixed inequality (longer `T_overlap` gives CPU more time)
   - Log strategy transitions

3. **Server integration**
   - Add `--apex-scheduling` flag (default: auto-detect on UMA)
   - Server's batch formation already separates prefill/decode
   - Hook into batch dispatch to apply strategy selection

**Test:** Server benchmark with mixed prefill/decode load, verify strategy switching
and throughput improvement.

---

## File Summary

| File | Phase | Description |
|------|-------|-------------|
| `common/uma-profiler.h` | 1 | Add batch size tracking, decode/prefill separation |
| `common/uma-profiler.cpp` | 1 | GPU sync verification, AI validation |
| `common/apex-scheduler.h` | 2,3,5 | New: APEX decision structures and API |
| `common/apex-scheduler.cpp` | 2,3,5 | New: inequality evaluation, strategy selection |
| `common/CMakeLists.txt` | 2 | Add apex-scheduler to build |
| `src/llama-context.h` | 3 | Add apex_state to context impl |
| `src/llama-context.cpp` | 3 | Extend graph_get_cb, trigger after profiling |
| `ggml/src/ggml-backend.cpp` | 4 | Split overlap in compute_splits |
| `ggml/include/ggml-backend.h` | 4 | Public API for overlap mode |
| `tools/server/server.cpp` | 5 | Server integration, --apex-scheduling flag |

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Profiler overhead during warmup | Low | Self-disabling after N iterations |
| Incorrect inequality evaluation | Medium | Conservative default (GPU-only), logging |
| Data race in split overlap | High | Event-based sync, thread sanitizer testing |
| Weight transfer overhead | Medium | Scheduler handles copies automatically; on UMA this is zero-copy |
| Graph rebuild after reassignment | Low | Only triggers once after profiling, not per-batch |

## Implementation Order

```
Phase 1 (Profiler) → Phase 2 (Inequality) → Phase 3 (Reassignment) → Phase 4 (Overlap) → Phase 5 (Server)
         ↑                    ↑                      ↑                       ↑
     Foundation          Decision logic        Uses decisions         Core APEX win
```

Each phase is independently valuable and testable. Phase 3 alone (without Phase 4)
provides benefit by routing bandwidth-heavy ops to GPU and compute-tolerant ops to CPU.
Phase 4 provides the async overlap which is APEX's primary throughput contribution.

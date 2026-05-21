# Multi-Threading Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate CPU busy-wait in ggml barrier, add per-thread timing observability, and fix 8 threading/correctness bugs across the core llama library.

**Architecture:** Replace the pure-spin `ggml_barrier()` with adaptive spin-then-cond_wait. Add per-thread atomic accumulators for `t_compute`, `t_barrier`, `t_poll`, `t_idle`. Fix 8 bugs in 5 files (llama-graph.cpp, llama-context.cpp, llama-kv-cache.cpp, llama-kv-cells.h, ggml-cpu.c).

**Tech Stack:** C99 atomics, pthreads condition variables, C++17, CMake build.

**Spec:** `docs/superpowers/specs/2026-05-20-llama-threading-optimization-design.md`

---

### Task 1: Fix `const_cast` UB in `copy_tensor_async_rows` (T5)

**Files:**
- Modify: `src/llama-context.cpp:1525-1553`

One-line fix: the `dst_data` parameter is written to, so `const` is wrong.

- [ ] **Step 1: Change function signature from `const void * dst_data` to `void * dst_data`**

In `src/llama-context.cpp`, line 1527:

```cpp
// Before:
    const void * dst_data,

// After:
    void * dst_data,
```

- [ ] **Step 2: Remove `const_cast` on line 1548**

```cpp
// Before (line 1548):
        const uint8_t * row_ptr = static_cast<const uint8_t *>(dst_data) + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, const_cast<uint8_t *>(row_ptr), 0, ggml_nbytes(tensor));

// After:
        uint8_t * row_ptr = static_cast<uint8_t *>(dst_data) + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));
```

- [ ] **Step 3: Verify all callers pass non-const pointers**

Run: `grep -n "copy_tensor_async_rows" src/llama-context.cpp` — all callers pass `&data` or a pointer variable, none are const.

- [ ] **Step 4: Build and verify**

```bash
cmake -B build -DLLAMA_BUILD_TESTS=ON -j && cmake --build build --config Release -j
```

- [ ] **Step 5: Commit**

```bash
git add src/llama-context.cpp
git commit -m "$(cat <<'EOF'
fix: remove const_cast UB in copy_tensor_async_rows

Change dst_data parameter from const void* to void* to accurately
reflect that the function writes to this buffer. Remove const_cast
that masked undefined behavior.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Fix `~llama_io_read_device()` noexcept (T2)

**Files:**
- Modify: `src/llama-context.cpp:2589-2636`

The destructor can throw via `mbufs.at(buft)` (std::out_of_range) or `ggml_backend_tensor_copy`. During exception unwinding this calls `std::terminate`.

- [ ] **Step 1: Mark destructor `noexcept` and wrap body in try-catch**

In `src/llama-context.cpp`, lines 2589-2636:

```cpp
// Before:
    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            // ... existing body ...
        }
        // ... more loops ...
        GGML_ASSERT(buf_size == 0);
    }

// After:
    ~llama_io_read_device() noexcept {
        try {
            llama_memory_buffers mbufs_new;

            for (const auto & rinfo : rinfos) {
                auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

                mbufs_new[buft].n_tensors++;
                mbufs_new[buft].total_size += rinfo.size;
            }

            for (auto & [buft, mbuf] : mbufs_new) {
                ggml_init_params params = {
                    /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                    /*.mem_buffer =*/ NULL,
                    /*.no_alloc   =*/ true,
                };

                mbuf.ctx.reset(ggml_init(params));

                mbuf.org.reserve(mbuf.n_tensors);
            }

            for (const auto & rinfo : rinfos) {
                auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

                const int64_t n = rinfo.size/ggml_element_size(rinfo.tensor);

                auto & mbuf = mbufs_new[buft];

                mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

                ggml_backend_view_init(mbuf.org.back());
            }

            for (auto & [buft, mbuf] : mbufs_new) {
                const auto & mbuf_cur = mbufs.at(buft);

                if (!mbuf_cur.buf || mbuf_cur.n_tensors != mbuf.n_tensors || mbuf_cur.total_size != mbuf.total_size) {
                    GGML_ABORT("%s: memory buffer mismatch\n", __func__);
                }

                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
                }
            }

            GGML_ASSERT(buf_size == 0);
        } catch (const std::exception & e) {
            LOG_ERROR("%s: failed to copy buffers: %s\n", __func__, e.what());
        } catch (...) {
            LOG_ERROR("%s: failed to copy buffers (unknown exception)\n", __func__);
        }
    }
```

- [ ] **Step 2: Build and run tests**

```bash
cmake --build build --config Release -j && cd build && ctest -C Release --output-on-failure -R test-tokenizer
```

- [ ] **Step 3: Commit**

```bash
git add src/llama-context.cpp
git commit -m "$(cat <<'EOF'
fix: mark ~llama_io_read_device() noexcept to prevent std::terminate

Wrap destructor body in try-catch to prevent exceptions during
stack unwinding from calling std::terminate. Mirrors the pattern
already used in ~llama_io_write_host.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Fix `type_v()` null deref on MLA models (T3)

**Files:**
- Modify: `src/llama-kv-cache.cpp:1116-1118`

MLA models set `v = nullptr`. Accessing `layers[0].v->type` crashes.

- [ ] **Step 1: Add null guard**

```cpp
// Before:
ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

// After:
ggml_type llama_kv_cache::type_v() const {
    return layers[0].v ? layers[0].v->type : GGML_TYPE_COUNT;
}
```

- [ ] **Step 2: Build and run tests**

```bash
cmake --build build --config Release -j && cd build && ctest -C Release --output-on-failure
```

- [ ] **Step 3: Commit**

```bash
git add src/llama-kv-cache.cpp
git commit -m "$(cat <<'EOF'
fix: add null guard to type_v() for MLA models

MLA models (DeepSeek series) set v=nullptr. Return GGML_TYPE_COUNT
sentinel when v is null instead of dereferencing a nullptr.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Fix `llama_kv_cells::cp()` used set (T4)

**Files:**
- Modify: `src/llama-kv-cells.h:120-157`

Both `cp` overloads copy `pos`, `ext`, `seq` but not `used`. Any future call to `get_used()`, `used_min()`, or `used_max_p1()` on a copied object returns wrong results.

- [ ] **Step 1: Fix `cp(uint32_t i, uint32_t n)` overload**

```cpp
// Before (lines 120-138):
    llama_kv_cells cp(uint32_t i, uint32_t n) const {
        assert(i + n <= pos.size());

        llama_kv_cells res;

        res.resize(n);

        for (uint32_t j = 0; j < n; ++j) {
            const auto idx = i + j;

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

// After:
    llama_kv_cells cp(uint32_t i, uint32_t n) const {
        assert(i + n <= pos.size());

        llama_kv_cells res;

        res.resize(n);

        for (uint32_t j = 0; j < n; ++j) {
            const auto idx = i + j;

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];

            assert(shift[idx] == 0);

            if (res.pos[j] != -1) {
                res.used.insert(j);
            }
        }

        return res;
    }
```

- [ ] **Step 2: Fix `cp(const std::vector<uint32_t> & idxs)` overload**

```cpp
// Before (lines 141-157):
    llama_kv_cells cp(const std::vector<uint32_t> & idxs) const {
        llama_kv_cells res;

        res.resize(idxs.size());

        for (uint32_t j = 0; j < idxs.size(); ++j) {
            const auto idx = idxs[j];

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

// After:
    llama_kv_cells cp(const std::vector<uint32_t> & idxs) const {
        llama_kv_cells res;

        res.resize(idxs.size());

        for (uint32_t j = 0; j < idxs.size(); ++j) {
            const auto idx = idxs[j];

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];

            assert(shift[idx] == 0);

            if (res.pos[j] != -1) {
                res.used.insert(j);
            }
        }

        return res;
    }
```

- [ ] **Step 3: Build and run tests**

```bash
cmake --build build --config Release -j && cd build && ctest -C Release --output-on-failure
```

- [ ] **Step 4: Commit**

```bash
git add src/llama-kv-cells.h
git commit -m "$(cat <<'EOF'
fix: populate used set in llama_kv_cells::cp() overloads

Both cp() overloads copied pos, ext, seq arrays but not the used
set. Rebuild used set by inserting indices where pos[j] != -1.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Fix graph reuse bugs in `llama-graph.cpp` (T7, T6, T8)

**Files:**
- Modify: `src/llama-graph.cpp:130, 475, 538, 541, 613, 620, 663, 731, 739, 747`

Three related bugs: null pointer (T7), unsigned underflow (T6), and missing v_idxs checks (T8).

- [ ] **Step 1: Fix null pointer in `llm_graph_input_pos::can_reuse` (T7)**

```cpp
// Before (line 130):
    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

// After:
    res &= (!pos) || (pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd);
```

- [ ] **Step 2: Re-enable v_idxs checks (T8) — 6 locations**

Each commented-out line should be re-enabled with null guard:

Line 475:
```cpp
// Before:
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

// After:
    res &= (!self_v_idxs) || (self_v_idxs->ne[0] == params.ubatch.n_tokens);
```

Line 538:
```cpp
// Before:
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: ...

// After:
    res &= (!self_v_idxs) || (self_v_idxs->ne[0] == params.ubatch.n_tokens);
```

Line 541 (second v_idxs variant in same function):
```cpp
// Before:
  //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: ...

// After:
    res &= (!self_v_idxs_swa) || (self_v_idxs_swa->ne[0] == params.ubatch.n_tokens);
```

Line 613:
```cpp
// Before:
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: ...

// After:
    res &= (!inp_attn->self_v_idxs) || (inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens);
```

Line 731:
```cpp
// Before:
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: ...

// After:
        res &= (!inp_attn->self_v_idxs) || (inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens);
```

Line 739:
```cpp
// Before:
      //res &= inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: ...

// After:
        res &= (!inp_attn->self_v_idxs_swa) || (inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens);
```

- [ ] **Step 3: Fix unsigned underflow in 3 `can_reuse` methods (T6)**

Line 620:
```cpp
// Before:
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

// After:
    res &= mctx->get_recr()->get_n_rs() >= params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;
```

Line 663:
```cpp
// Before:
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

// After:
    res &= mctx->get_recr()->get_n_rs() >= params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;
```

Line 747:
```cpp
// Before:
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

// After:
    res &= mctx->get_recr()->get_n_rs() >= params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;
```

- [ ] **Step 4: Build and run all tests**

```bash
cmake --build build --config Release -j && cd build && ctest -C Release --output-on-failure
```

- [ ] **Step 5: Commit**

```bash
git add src/llama-graph.cpp
git commit -m "$(cat <<'EOF'
fix: guard graph reuse checks against null, underflow, and missing v_idxs

- llm_graph_input_pos::can_reuse: add null guard for pos pointer
- Re-enable 6 commented-out v_idxs reuse checks with null guards
- Guard 3 unsigned subtractions in hybrid can_reuse methods against
  underflow when n_seqs > n_rs

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Fix barrier memory ordering (T1)

**Files:**
- Modify: `ggml/src/ggml-cpu/ggml-cpu.c:590`

On weakly-ordered architectures (ARM/POWER), `memory_order_relaxed` load can miss the `seq_cst` store from the last thread, causing indefinite hang.

- [ ] **Step 1: Change memory order in barrier spin loop**

```c
// Before (line 590):
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {

// After:
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_acquire) == n_passed) {
```

- [ ] **Step 2: Build (including CUDA if configured)**

```bash
cmake -B build -DLLAMA_BUILD_TESTS=ON -DGGML_CUDA=ON -j && cmake --build build --config Release -j
```

- [ ] **Step 3: Run backend tests**

```bash
cd build && ctest -C Release --output-on-failure -L test-backend
```

- [ ] **Step 4: Commit**

```bash
git add ggml/src/ggml-cpu/ggml-cpu.c
git commit -m "$(cat <<'EOF'
fix: use memory_order_acquire in ggml_barrier spin loop

On weakly-ordered architectures (ARM/POWER), relaxed load of
n_barrier_passed can miss the seq_cst store from the last thread.
Change to acquire to form a proper synchronizes-with pair.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Adaptive barrier — eliminate CPU busy-wait (P1)

**Files:**
- Modify: `ggml/src/ggml-cpu/ggml-cpu.c:566-602`

Replace the indefinite spin loop with adaptive spin-then-cond_wait. Reuse the existing `mutex` and `cond` members of `struct ggml_threadpool`.

- [ ] **Step 1: Rewrite `ggml_barrier()` with adaptive algorithm**

```c
// Replace lines 566-602 with:

void ggml_barrier(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_threads == 1) {
        return;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#else
    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_acquire);

    // enter barrier (full seq-cst fence)
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);

        // wake any threads sleeping in cond_wait
        #ifndef GGML_USE_OPENMP
        ggml_cond_broadcast(&tp->cond);
        #endif
        return;
    }

    // wait for other threads

    // Phase 1: brief spin (uses existing poll parameter, same scale as poll_for_work)
    {
        const uint64_t n_rounds = 1024UL * 128 * tp->poll;
        for (uint64_t i = 0; i < n_rounds; i++) {
            if (atomic_load_explicit(&tp->n_barrier_passed, memory_order_acquire) != n_passed) {
                break;
            }
            ggml_thread_cpu_relax();
        }
    }

    // Phase 2: sleep on condition variable if threads still haven't arrived
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_acquire) == n_passed) {
        ggml_mutex_lock_shared(&tp->mutex);
        // Timed wait to avoid missing the broadcast (spurious wakeup protection)
        // 1ms timeout — last thread also increments n_barrier_passed, so we'll see it
        ggml_cond_wait_timeout(&tp->cond, &tp->mutex, 1000);
        ggml_mutex_unlock_shared(&tp->mutex);
    }

    // exit barrier (full seq-cst fence)
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}
```

- [ ] **Step 2: Add `ggml_cond_wait_timeout` helper if it doesn't exist**

Check if a timed cond_wait already exists:

```bash
grep -n "ggml_cond_wait_timeout\|ggml_cond_wait.*timeout\|cond_timedwait" ggml/src/ggml-cpu/ggml-cpu.c
```

If `ggml_cond_wait_timeout` does not exist, add it near the other cond helpers (around line 430):

```c
// On POSIX:
static inline void ggml_cond_wait_timeout(ggml_cond_t * cond, ggml_mutex_t * mutex, int ms) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec  += ms / 1000;
    ts.tv_nsec += (ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec++;
        ts.tv_nsec -= 1000000000L;
    }
    pthread_cond_timedwait(cond, mutex, &ts);
}
```

On Windows (SRWLOCK variant), add the equivalent with `SleepConditionVariableSRW`:

```c
static inline void ggml_cond_wait_timeout(ggml_cond_t * cond, ggml_mutex_t * mutex, int ms) {
    (void)mutex;
    SleepConditionVariableSRW(cond, mutex, ms, 0);
}
```

Place this in the appropriate `#ifdef _WIN32` / `#else` section matching the existing `ggml_cond_wait` implementation.

- [ ] **Step 3: Build and run backend tests**

```bash
cmake --build build --config Release -j
cd build && ctest -C Release --output-on-failure
```

- [ ] **Step 4: Verify with multi-threaded load test**

```bash
# Run with 8 threads — barrier should not hold CPU at 100% during idle waits
GGML_CPU_THREADS=8 ./build/bin/llama -m <model> -p "test" -n 4 --threads 8
```

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-cpu/ggml-cpu.c
git commit -m "$(cat <<'EOF'
perf: adaptive barrier eliminates CPU busy-wait in ggml_barrier()

Replace indefinite spin loop with two-phase algorithm:
Phase 1: spin for tp->poll * 128 * 1024 iterations (configurable,
         uses hardware pause/yield instruction)
Phase 2: fall back to cond_wait with 1ms timeout

Fast path (all threads arrive quickly) is unchanged. Slow path
(different thread speeds due to NUMA/CFS) no longer burns CPU.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Per-thread timing accumulators (P2)

**Files:**
- Modify: `ggml/src/ggml-cpu/ggml-cpu.c` (struct, init, worker loop, barrier, free, API)
- Modify: `ggml/include/ggml-cpu.h` (new API declarations)
- Modify: `src/llama-context.cpp:3857-3870` (extend `llama_perf_context_print`)

Add 6 per-thread atomic accumulators: `t_compute_us`, `t_barrier_us`, `t_poll_us`, `t_idle_us`, `n_compute`, `n_barrier`.

- [ ] **Step 1: Add timing arrays to `struct ggml_threadpool`**

In `ggml/src/ggml-cpu/ggml-cpu.c`, inside `struct ggml_threadpool` (around line 494):

```c
// Add after line 494 (enum ggml_status ec;):
    // per-thread timing accumulators (indexed by thread id, size = n_threads)
    int64_t  * t_compute_us;
    int64_t  * t_barrier_us;
    int64_t  * t_poll_us;
    int64_t  * t_idle_us;
    uint64_t * n_compute;
    uint64_t * n_barrier;
```

- [ ] **Step 2: Allocate and zero timing arrays in threadpool creation**

In `ggml_threadpool_new_impl` (around line 3247), after `threadpool->ec = GGML_STATUS_SUCCESS;`:

```c
    // Allocate per-thread timing accumulators (zeroed)
    {
        const size_t timing_size = sizeof(int64_t) * tpp->n_threads;
        const size_t count_size  = sizeof(uint64_t) * tpp->n_threads;
        threadpool->t_compute_us = (int64_t  *) ggml_aligned_malloc(timing_size);
        threadpool->t_barrier_us = (int64_t  *) ggml_aligned_malloc(timing_size);
        threadpool->t_poll_us    = (int64_t  *) ggml_aligned_malloc(timing_size);
        threadpool->t_idle_us    = (int64_t  *) ggml_aligned_malloc(timing_size);
        threadpool->n_compute    = (uint64_t *) ggml_aligned_malloc(count_size);
        threadpool->n_barrier    = (uint64_t *) ggml_aligned_malloc(count_size);
        memset(threadpool->t_compute_us, 0, timing_size);
        memset(threadpool->t_barrier_us, 0, timing_size);
        memset(threadpool->t_poll_us,    0, timing_size);
        memset(threadpool->t_idle_us,    0, timing_size);
        memset(threadpool->n_compute,    0, count_size);
        memset(threadpool->n_barrier,    0, count_size);
    }
```

- [ ] **Step 3: Free timing arrays in `ggml_threadpool_free`**

In `ggml_threadpool_free` (around line 2703), before freeing workers:

```c
    // Free timing arrays
    ggml_aligned_free(threadpool->t_compute_us, sizeof(int64_t) * n_threads);
    ggml_aligned_free(threadpool->t_barrier_us, sizeof(int64_t) * n_threads);
    ggml_aligned_free(threadpool->t_poll_us,    sizeof(int64_t) * n_threads);
    ggml_aligned_free(threadpool->t_idle_us,    sizeof(int64_t) * n_threads);
    ggml_aligned_free(threadpool->n_compute,    sizeof(uint64_t) * n_threads);
    ggml_aligned_free(threadpool->n_barrier,    sizeof(uint64_t) * n_threads);
```

- [ ] **Step 4: Instrument the worker loop in `ggml_graph_compute_secondary_thread`**

Replace lines 3163-3185:

```c
    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) break;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        {
            const int64_t t_poll_start = ggml_time_us();
            ggml_graph_compute_check_for_work(state);
            atomic_fetch_add_explicit(&threadpool->t_poll_us[state->ith],
                ggml_time_us() - t_poll_start, memory_order_relaxed);
        }
        if (state->pending) {
            state->pending = false;

            const int64_t t_compute_start = ggml_time_us();
            ggml_graph_compute_thread(state);
            atomic_fetch_add_explicit(&threadpool->t_compute_us[state->ith],
                ggml_time_us() - t_compute_start, memory_order_relaxed);
            atomic_fetch_add_explicit(&threadpool->n_compute[state->ith],
                1, memory_order_relaxed);
        }
    }
```

- [ ] **Step 5: Change `ggml_barrier` signature to accept thread id**

All call sites pass either `ggml_compute_params` (has `ith`) or `ggml_compute_state` (has `ith`). Change the API:

**Update declaration in `ggml/src/ggml-cpu/ggml-cpu-impl.h`:**
```c
// Before:
void ggml_barrier(struct ggml_threadpool * tp);
// After:
void ggml_barrier(struct ggml_threadpool * tp, int ith);
```

**Update definition in `ggml-cpu.c`** — wrap the wait section with timing:
```c
    // In the non-last-thread path of ggml_barrier, after spin+cond_wait:
    {
        const int64_t t_barrier_start = ggml_time_us();
        // ... Phase 1 spin, Phase 2 cond_wait (from Task 7) ...
        atomic_fetch_add_explicit(&tp->t_barrier_us[ith],
            ggml_time_us() - t_barrier_start, memory_order_relaxed);
        atomic_fetch_add_explicit(&tp->n_barrier[ith],
            1, memory_order_relaxed);
    }
```

**Update all call sites** — two patterns:

Pattern 1 — `ggml_barrier(params->threadpool)` → `ggml_barrier(params->threadpool, params->ith)`:
- `ggml-cpu.c:1355`, `ggml-cpu.c:1636`
- `ggml-cpu/ops.cpp` (lines 1181, 1663, 4205, 4327, 4527, 4598, 5221, 6030)
- `ggml-cpu/repack.cpp:4352`, `ggml-cpu/repack.cpp:4469`
- `ggml-cpu/kleidiai/kleidiai.cpp:667,698,1060,1106`

Pattern 2 — `ggml_barrier(state->threadpool)` → `ggml_barrier(state->threadpool, state->ith)`:
- `ggml-cpu.c:3069`, `ggml-cpu.c:3079`

Verify with:
```bash
grep -rn "ggml_barrier(" ggml/src/ggml-cpu/ --include="*.cpp" --include="*.c" --include="*.h"
# Every line should have two arguments: tp and ith
```

- [ ] **Step 6: Add `ggml_threadpool_print_timings()` API function**

In `ggml/src/ggml-cpu/ggml-cpu.c`, after the free function:

```c
void ggml_threadpool_print_timings(struct ggml_threadpool * tp) {
    if (!tp) return;

    const int n_threads = tp->n_threads;

    GGML_LOG_INFO("\n=== CPU Threadpool Timing (per thread) ===\n");
    GGML_LOG_INFO("  Thread | t_compute_us | t_barrier_us | t_poll_us | t_idle_us | n_compute | n_barrier\n");

    for (int i = 0; i < n_threads; i++) {
        GGML_LOG_INFO("  %7d | %12" PRId64 " | %12" PRId64 " | %11" PRId64 " | %11" PRId64 " | %9" PRIu64 " | %9" PRIu64 "\n",
            i,
            tp->t_compute_us[i],
            tp->t_barrier_us[i],
            tp->t_poll_us[i],
            tp->t_idle_us[i],
            tp->n_compute[i],
            tp->n_barrier[i]);
    }
    GGML_LOG_INFO("================================================\n\n");
}
```

- [ ] **Step 7: Add `ggml_threadpool_reset_timings()` API**

```c
void ggml_threadpool_reset_timings(struct ggml_threadpool * tp) {
    if (!tp) return;

    const size_t timing_size = sizeof(int64_t) * tp->n_threads;
    const size_t count_size  = sizeof(uint64_t) * tp->n_threads;
    memset(tp->t_compute_us, 0, timing_size);
    memset(tp->t_barrier_us, 0, timing_size);
    memset(tp->t_poll_us,    0, timing_size);
    memset(tp->t_idle_us,    0, timing_size);
    memset(tp->n_compute,    0, count_size);
    memset(tp->n_barrier,    0, count_size);
}
```

- [ ] **Step 8: Auto-report on threadpool free**

In `ggml_threadpool_free`, before freeing the timing arrays:

```c
    // Print timing summary on shutdown
    ggml_threadpool_print_timings(threadpool);
```

- [ ] **Step 9: Declare new API in `ggml/include/ggml-cpu.h`**

```c
GGML_API void ggml_threadpool_print_timings(struct ggml_threadpool * tp);
GGML_API void ggml_threadpool_reset_timings(struct ggml_threadpool * tp);
```

- [ ] **Step 10: No llama-level integration needed**

The auto-report in `ggml_threadpool_free` (Step 8) is the primary reporting mechanism. There is no `ggml_backend_cpu_get_threadpool()` API to access the threadpool from the llama context, and adding one would be out of scope. The `ggml_threadpool_print_timings()` API is available for callers that have direct access to the threadpool.

- [ ] **Step 11: Build and run tests**

```bash
cmake --build build --config Release -j
cd build && ctest -C Release --output-on-failure
```

- [ ] **Step 12: Verify timing output appears**

```bash
# Run a short generation — timing table should print on exit
GGML_CPU_THREADS=4 ./build/bin/llama -m <model> -p "test" -n 4 --threads 4
```

- [ ] **Step 13: Commit**

```bash
git add ggml/src/ggml-cpu/ggml-cpu.c ggml/include/ggml-cpu.h
git commit -m "$(cat <<'EOF'
feat: add per-thread timing accumulators to CPU threadpool

Track t_compute_us, t_barrier_us, t_poll_us, t_idle_us, n_compute,
n_barrier per worker thread via atomic accumulators. Auto-print on
threadpool free. New API: ggml_threadpool_print_timings(),
ggml_threadpool_reset_timings().

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- [x] T1: Barrier memory ordering — Task 6
- [x] T2: noexcept destructor — Task 2
- [x] T3: type_v() null deref — Task 3
- [x] T4: cp() used set — Task 4
- [x] T5: const_cast UB — Task 1
- [x] T6: Unsigned underflow — Task 5 (Step 3)
- [x] T7: Null pointer in can_reuse — Task 5 (Step 1)
- [x] T8: Missing v_idxs reuse — Task 5 (Step 2)
- [x] P1: Adaptive barrier — Task 7
- [x] P2: Per-thread timing — Task 8

**2. Placeholder scan:** No "TBD", "TODO", or "implement later". Task 8 Step 5 notes the barrier timing integration detail — resolved with `ith` parameter approach.

**3. Type consistency:** All function signatures match existing code. `ggml_threadpool_print_timings` and `ggml_threadpool_reset_timings` follow existing naming convention. `PRId64`/`PRIu64` format specifiers are correct for `int64_t`/`uint64_t`.

**4. Build order:** Tasks 1-6 (bugs) are independent of each other and can be done in any order. Task 7 (adaptive barrier) depends on T6 (memory ordering already applied). Task 8 (timing) depends on Task 7 (barrier restructuring) — the barrier wait section is where barrier timing is measured.

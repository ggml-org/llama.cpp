# CPU Offload TG Performance: 3 Optimizations

## Context

CPU offload at 42% VRAM budget achieves 1.11 tok/s (8 GPU + 24 CPU layers).
VTune profiling identified 3 bottlenecks:

| Bottleneck | CPU Time | % Total | Root Cause |
|-----------|----------|---------|------------|
| CPU MUL_MAT | 8.31s | 30.1% | Single-threaded `ggml_vec_dot_q4_0_q8_0` |
| GPU↔CPU transitions | 5.89s | 21.3% | ~14 transitions/token from FLASH_ATTN GPU islands |
| Staging overhead | 4.08s | 14.7% | 167 staging ops/token, 6 leaf tensors re-staged every token |

**Target**: 1.11 → 3-4 tok/s (3-4x improvement)

---

## Task 1: Multi-threaded CPU vec_dot

**Priority**: Highest (30.1% of time)
**File**: `ggml/src/ggml-sycl/cpu-dispatch.cpp`
**Model**: sonnet (thread pool implementation + careful integration)
**Expected**: 8.31s → ~2s (4x from 8+ threads on 20-core CPU)

### Problem

In `cpu_mul_mat()` (line 416-588), the vec_dot path (M<=4, used for TG batch=1) iterates
over N=4096 output elements single-threaded:

```cpp
// Line 554-570 (current)
for (dnnl_dim_t n = 0; n < N; n++) {
    const void * weight_row = src0_batch + n * nb01;
    for (dnnl_dim_t m = 0; m < M; m++) {
        float dot_result = 0.0f;
        cpu_traits->vec_dot(K, &dot_result, sizeof(float),
                            weight_row, 0,
                            src1_q_buf.data() + m * q_row_size, 0, 1);
        dst_batch[m * ldc + n] = dot_result;
    }
}
```

Each call to `cpu_traits->vec_dot` resolves to `ggml_vec_dot_q4_0_q8_0` in libggml-cpu.so.
N=4096 iterations × 24 CPU layers × ~3 MUL_MATs/layer = ~300K dot products per token.

### Solution: Static thread pool + parallel N-loop

Add a lightweight static thread pool (lazy-initialized) that parallelizes the outer N-loop.
DO NOT use OpenMP (not linked in SYCL build). Use `std::thread` + barrier pattern.

**Add before `cpu_mul_mat` function (~line 412):**

```cpp
// --- Static CPU thread pool for parallel vec_dot ---
// Lazy-initialized on first use. Thread count from GGML_SYCL_CPU_THREADS
// or (hardware_concurrency - 2) to leave headroom for GPU driver threads.

struct cpu_thread_pool {
    struct work_item {
        std::function<void(int, int)> fn;  // fn(start, end)
        int start;
        int end;
    };

    std::vector<std::thread>       workers;
    std::vector<work_item>         tasks;
    std::mutex                     mtx;
    std::condition_variable        cv_work;
    std::condition_variable        cv_done;
    std::atomic<int>               active{0};
    std::atomic<bool>              shutdown{false};
    int                            n_threads;

    static cpu_thread_pool & instance() {
        static cpu_thread_pool pool;
        return pool;
    }

    cpu_thread_pool() {
        const char * env = getenv("GGML_SYCL_CPU_THREADS");
        n_threads = env ? std::max(1, atoi(env))
                        : std::max(1, (int)std::thread::hardware_concurrency() - 2);
        // Cap to reasonable max
        n_threads = std::min(n_threads, 32);

        for (int i = 0; i < n_threads; i++) {
            workers.emplace_back([this] {
                while (true) {
                    work_item item;
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv_work.wait(lock, [this] { return shutdown.load() || !tasks.empty(); });
                        if (shutdown.load() && tasks.empty()) return;
                        item = std::move(tasks.back());
                        tasks.pop_back();
                    }
                    item.fn(item.start, item.end);
                    if (active.fetch_sub(1) == 1) {
                        cv_done.notify_all();
                    }
                }
            });
        }
    }

    ~cpu_thread_pool() {
        shutdown.store(true);
        cv_work.notify_all();
        for (auto & w : workers) w.join();
    }

    // Partition [0, total) across n_threads and execute fn(start, end) in parallel.
    // Blocks until all work is complete.
    void parallel_for(int total, std::function<void(int, int)> fn) {
        if (total <= 0) return;
        int chunk = (total + n_threads - 1) / n_threads;
        {
            std::lock_guard<std::mutex> lock(mtx);
            int n_tasks = 0;
            for (int t = 0; t < n_threads; t++) {
                int start = t * chunk;
                int end   = std::min(start + chunk, total);
                if (start >= total) break;
                tasks.push_back({fn, start, end});
                n_tasks++;
            }
            active.store(n_tasks);
        }
        cv_work.notify_all();
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_done.wait(lock, [this] { return active.load() == 0; });
        }
    }
};
```

**Modify the vec_dot loop in `cpu_mul_mat()` (replace lines 542-570):**

```cpp
    if (use_vec_dot && from_float_fn) {
        // Quantize activations (small M, not worth parallelizing)
        for (dnnl_dim_t m = 0; m < M; m++) {
            from_float_fn(src1_batch + m * K,
                          src1_q_buf.data() + m * q_row_size, K);
        }

        // Parallel vec_dot over N (output rows).
        // Each thread processes a contiguous chunk of weight rows.
        // Thread-safe: each (n,m) writes to a unique dst_batch location.
        auto & pool = cpu_thread_pool::instance();
        const int N_int = static_cast<int>(N);

        if (N_int > 64 && pool.n_threads > 1) {
            pool.parallel_for(N_int, [&](int n_start, int n_end) {
                for (int n = n_start; n < n_end; n++) {
                    const void * weight_row = src0_batch + n * nb01;
                    for (dnnl_dim_t m = 0; m < M; m++) {
                        float dot_result = 0.0f;
                        cpu_traits->vec_dot(
                            static_cast<int>(K), &dot_result, sizeof(float),
                            weight_row, 0,
                            src1_q_buf.data() + m * q_row_size, 0, 1);
                        dst_batch[m * ldc + n] = dot_result;
                    }
                }
            });
        } else {
            // Small N or single thread: use original serial path
            for (dnnl_dim_t n = 0; n < N; n++) {
                const void * weight_row = src0_batch + n * nb01;
                for (dnnl_dim_t m = 0; m < M; m++) {
                    float dot_result = 0.0f;
                    cpu_traits->vec_dot(
                        static_cast<int>(K), &dot_result, sizeof(float),
                        weight_row, 0,
                        src1_q_buf.data() + m * q_row_size, 0, 1);
                    dst_batch[m * ldc + n] = dot_result;
                }
            }
        }
    }
```

### Required includes (add at top of cpu-dispatch.cpp if missing)

```cpp
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
```

### Verification

```bash
# Build
source /opt/intel/oneapi/setvars.sh --force && ninja -C build -j $(nproc)

# GPU-only (no regression)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: "6, 7, 8, 9, 10"

# CPU offload (42% budget)
GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_VRAM_BUDGET_PCT=42 \
  ONEAPI_DEVICE_SELECTOR='level_zero:0;opencl:0' ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: correct output, TG > 1.5 tok/s (was 1.11)

# Thread count test
GGML_SYCL_CPU_THREADS=4 GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_VRAM_BUDGET_PCT=42 \
  ONEAPI_DEVICE_SELECTOR='level_zero:0;opencl:0' ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: correct output (fewer threads = slower but still correct)
```

---

## Task 2: Persistent Leaf Tensor Staging

**Priority**: Medium (14.7% of time)
**File**: `ggml/src/ggml-sycl/cpu-dispatch.cpp`
**Model**: haiku (simple data structure + lookup)
**Expected**: 167 → ~60 staging ops/token (eliminate ~100 redundant leaf copies)

### Problem

`get_host_ptr()` (line 279-355) stages non-weight tensors from device to host every time
they're accessed. 6 leaf tensors (RoPE freqs, attention masks, constants) are host-resident
and fail staging with "type=0" warnings, but still incur lookup overhead 167 times per token.

The leaf tensors (leaf_2, leaf_4, leaf_7, leaf_9, leaf_11, leaf_357) have STABLE data between
tokens — their content doesn't change during TG. Currently `get_host_ptr` re-stages them on
every access because there's no caching.

### Solution: Add persistent staging cache for leaf tensors

**1. Add a static staging cache (after line 228, near existing staging globals):**

```cpp
// Persistent staging cache for leaf tensors (RoPE, masks, constants).
// Key: tensor data pointer (stable across tokens for leaf tensors).
// Value: host-accessible pointer (either direct host ptr or cached staging copy).
// Cleared on graph shape change (new token count changes masks).
static std::unordered_map<const void *, void *> g_leaf_staging_cache;
static size_t g_leaf_staging_graph_hash = 0;  // detect graph shape changes
```

**2. Add a cache invalidation function (public API):**

In `cpu-dispatch.hpp`, add:
```cpp
void ggml_sycl_cpu_staging_cache_clear();
```

In `cpu-dispatch.cpp`:
```cpp
void ggml_sycl_cpu_staging_cache_clear() {
    g_leaf_staging_cache.clear();
    g_leaf_staging_graph_hash = 0;
}
```

**3. Modify `get_host_ptr()` — add leaf caching at the START of the non-weight branch (line 339):**

Change the non-weight section from:
```cpp
    // Non-weight tensors (activations, compute buffers) → stage device→host.
    void * ptr = ggml_sycl_get_data_ptr(t, device);
    ...
```

To:
```cpp
    // Non-weight tensors (activations, compute buffers).
    void * ptr = ggml_sycl_get_data_ptr(t, device);

    // Check persistent staging cache for leaf tensors.
    // Leaf tensors (RoPE freqs, masks) have stable data between tokens.
    if (t->data) {
        auto it = g_leaf_staging_cache.find(t->data);
        if (it != g_leaf_staging_cache.end()) {
            if (out_event) *out_event = sycl::event{};
            return it->second;
        }
    }

    // Host-accessible buffers don't need staging
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        // Cache this for future lookups
        if (t->data) {
            g_leaf_staging_cache[t->data] = t->data;
        }
        return t->data;
    }

    if (!ptr) {
        return nullptr;
    }
    size_t nbytes = ggml_nbytes(t);
    void * host = staging_ensure(g_staging_bank, slot, nbytes, gpu_q);
    ...
```

Wait — the issue is simpler. Looking at the VTune report: "type=0 (host memory)" means
these tensors are ALREADY in host memory. `get_host_ptr` should return `t->data` directly
for host-buffer tensors. The problem is likely the `ggml_backend_buffer_is_host()` check
failing for these specific buffer types.

**Revised approach — check more carefully at line 293-296:**

The current code:
```cpp
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return t->data;
    }
```

This should already handle host-resident tensors. But the staging failures suggest these
tensors have a non-null buffer that doesn't pass `is_host()`. Add a secondary check:

```cpp
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return t->data;
    }

    // Leaf tensors allocated on SYCL host buffers: check if data pointer is
    // host-accessible (USM host or shared). This catches tensors that have a
    // SYCL buffer type but are backed by host-accessible memory.
    if (t->data) {
        auto it = g_leaf_staging_cache.find(t->data);
        if (it != g_leaf_staging_cache.end()) {
            if (out_event) *out_event = sycl::event{};
            return it->second;
        }
    }
```

And after a successful staging copy, cache the result:
```cpp
    sycl::event evt = gpu_q->memcpy(host, ptr, nbytes);
    // Cache for leaf tensors (stable data pointers between tokens)
    if (t->data && !ggml_sycl_tensor_is_weight(t)) {
        g_leaf_staging_cache[t->data] = host;
    }
```

**4. Call cache clear at graph shape change:**

In `ggml-sycl.cpp`, in the graph_compute_impl function, when graph shape changes
(detected by node count change), call `ggml_sycl_cpu_staging_cache_clear()`.
This ensures masks that change with sequence length get re-staged.

Add near line 25230 (after graph shape change detection):
```cpp
    if (graph_shape_changed) {
        ggml_sycl_cpu_staging_cache_clear();
    }
```

### Verification

Same commands as Task 1. Expected:
- Correct output at 42% budget
- Fewer staging operations visible in GGML_SYCL_DEBUG=1 output
- Slight speedup (staging overhead reduced)

---

## Task 3: GPU Island Handling for FLASH_ATTN

**Priority**: High (21.3% of time)
**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp`
**Model**: sonnet (complex boundary logic in graph_compute)
**Expected**: Reduce ~14 transitions/token to ~2 transitions/token

### Problem

Within each CPU layer, FLASH_ATTN_EXT forces a GPU transition (Phase 4 of
`classify_cpu_layer_blocks` at line 24880-24900 correctly marks it as GPU).
This causes 2 transitions per CPU layer:
1. CPU→GPU before FLASH_ATTN_EXT
2. GPU→CPU after FLASH_ATTN_EXT

With 24 CPU layers × 2 transitions = up to 48 transitions per token.
Each transition involves: staging_drain + retained_flush + event sync + re-init.

### Solution: Mark FLASH_ATTN as "GPU island" — lightweight inline transfer

Instead of full boundary transitions, handle GPU-only ops within CPU layers as
"islands" that borrow device resources temporarily without changing the CPU
dispatch state.

**1. Add GPU island detection to classify_cpu_layer_blocks (after Phase 4, ~line 24900):**

Add a new Phase 5 that marks GPU islands:

```cpp
    // Phase 5: Mark GPU "islands" within CPU layer blocks.
    // A GPU island is a short run of GPU ops (typically FLASH_ATTN_EXT + SET_ROWS)
    // sandwiched between CPU ops in the same layer. Instead of full boundary
    // transitions, these get lightweight inline D2H/H2D handling.
    //
    // node_cpu_flags encoding: -1=unknown, 0=GPU, 1=CPU, 2=GPU_ISLAND
    for (int i = 1; i < cgraph->n_nodes - 1; i++) {
        if (node_cpu_flags[i] != 0) continue;  // only check GPU nodes

        // Look backward for CPU predecessor (skip noops)
        bool prev_cpu = false;
        for (int j = i - 1; j >= 0; j--) {
            if (node_cpu_flags[j] == 1) { prev_cpu = true; break; }
            if (node_cpu_flags[j] == 0) break;  // hit a real GPU node
            // node_cpu_flags[j] == 2 means another island — keep looking
        }
        if (!prev_cpu) continue;

        // Look forward for CPU successor (skip noops and other islands)
        bool next_cpu = false;
        for (int j = i + 1; j < cgraph->n_nodes; j++) {
            if (node_cpu_flags[j] == 1) { next_cpu = true; break; }
            if (node_cpu_flags[j] == 0) break;
            // 2 = another island, keep looking
        }
        if (!next_cpu) continue;

        // This GPU op is an island within a CPU block
        node_cpu_flags[i] = 2;  // GPU_ISLAND
    }
```

**2. Modify boundary detection in graph_compute_impl (line 25332-25424):**

Replace the current `node_on_cpu != prev_on_cpu` check with island-aware logic:

```cpp
        if (cpu_offload_active) {
            int8_t flag = node_cpu_flags.empty() ? -1 : node_cpu_flags[i];
            bool node_on_cpu    = (flag == 1);
            bool node_is_island = (flag == 2);

            if (node_is_island) {
                // GPU island: run on GPU without full boundary transition.
                // Staging drain ensures pending CPU→device copies complete.
                ggml_sycl_cpu_staging_drain();
                GGML_SYCL_DEBUG("[GPU-ISLAND] node %d (%s) — inline GPU op\n",
                                i, node->name ? node->name : "(null)");
                // Don't change prev_on_cpu — island is transparent to CPU state.
                // Don't flush retained activations — island inputs come from
                // device buffers (KV cache), not from CPU intermediates.
                // After island, CPU ops continue reading from retained scratch.
            } else if (node_on_cpu != prev_on_cpu) {
                // ... existing boundary transition code (unchanged) ...
            }
            if (!node_is_island) {
                prev_on_cpu = node_on_cpu;
            }

            // CPU op fusion: skip for GPU islands
            if (node_on_cpu) {
                // ... existing fusion code (unchanged) ...
            }
        }
```

**3. Key insight: FLASH_ATTN_EXT inputs are NOT CPU intermediates**

FLASH_ATTN_EXT reads from:
- Q: computed by CPU MUL_MAT → already staged to device by flush_output
- K, V: KV cache entries → already on device
- mask: leaf tensor → on host, but FLASH_ATTN reads from device buffer

So the GPU island doesn't need to flush retained activations. It just needs the
Q projection to have been flushed (which happens at the end of each CPU op via
`flush_output`). The KV cache is device-resident.

The ONLY thing we need is `staging_drain()` to ensure the async flush of Q completed.

### Verification

```bash
# Count transitions with debug
GGML_SYCL_DEBUG=1 GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_VRAM_BUDGET_PCT=42 \
  ONEAPI_DEVICE_SELECTOR='level_zero:0;opencl:0' ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 5 --seed 42 --temp 0 2>&1 | grep -c 'CPU→GPU\|GPU→CPU'
# Before: ~14 per token, After: ~2 per token (only at GPU/CPU layer boundary)

# Correctness
GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_VRAM_BUDGET_PCT=42 \
  ONEAPI_DEVICE_SELECTOR='level_zero:0;opencl:0' ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: "6, 7, 8, 9, 10"
```

---

## Task 4: Integration Test & Benchmarks

**Model**: haiku (build + run)
**Depends on**: Tasks 1, 2, 3

### Commands

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)

# Fix PTI library path
NEW_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v pti | tr '\n' ':' | sed 's/:$//')

# 1. GPU-only correctness (no regression)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: "6, 7, 8, 9, 10"

# 2. GPU-only performance (wait 60s for thermal cooldown)
LD_LIBRARY_PATH="build/bin:$NEW_PATH" ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# Expect: PP512 >= 1200, TG128 >= 68

# 3. CPU offload 42% VRAM
GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_VRAM_BUDGET_PCT=42 \
  ONEAPI_DEVICE_SELECTOR='level_zero:0;opencl:0' ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: correct output, TG >= 2.5 tok/s (was 1.11)

# 4. CPU offload 30% VRAM (all CPU layers)
GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_VRAM_BUDGET_PCT=30 \
  ONEAPI_DEVICE_SELECTOR='level_zero:0;opencl:0' ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: correct output, TG >= 2.0 tok/s (was 0.96)

# 5. Thread count verification
GGML_SYCL_CPU_THREADS=1 GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_VRAM_BUDGET_PCT=42 \
  ONEAPI_DEVICE_SELECTOR='level_zero:0;opencl:0' ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: correct output, ~1.11 tok/s (single-thread baseline)
```

### Pass criteria
- Zero GPU-only regression (PP512 >= 1200, TG128 >= 68)
- Correct output at 42% and 30% budgets
- TG >= 2.5 tok/s at 42% budget (2.5x improvement from 1.11)
- Single-thread fallback works correctly

---

## Dependencies

```
Task 1 (multi-threaded vec_dot) ─┐
                                  ├──→ Task 4 (integration test)
Task 2 (persistent staging)  ────┤
                                  │
Task 3 (GPU island handling)  ───┘
```

Tasks 1+2 share `cpu-dispatch.cpp` → must be sequential (same implementer).
Task 3 modifies `ggml-sycl.cpp` → can run in parallel with Tasks 1+2.

## Team Structure

- **Track A** (implementer-1): Task 1 → Task 2 (sequential, same file)
- **Track B** (implementer-2): Task 3 (parallel with Track A)
- **After both**: Task 4 (integration test, either implementer)
- **Reviewers**: spec-reviewer + quality-reviewer

## Critical Files

| File | Tasks | Role |
|------|-------|------|
| `ggml/src/ggml-sycl/cpu-dispatch.cpp` | 1, 2 | Thread pool, vec_dot parallelism, staging cache |
| `ggml/src/ggml-sycl/cpu-dispatch.hpp` | 2 | Public API for staging cache clear |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | 3 | GPU island Phase 5, boundary detection |

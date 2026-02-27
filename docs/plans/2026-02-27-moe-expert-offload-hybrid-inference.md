# MoE Expert Offload & Hybrid Inference Plan

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.
> **For all agents:** Use Serena MCP tools (`mcp__plugin_serena_serena__*`) for ALL code exploration and editing. Do NOT use Read/Grep/Glob for code reading. Activate project first: `mcp__plugin_serena_serena__activate_project` with path `/Apps/llama.cpp`.

**Goal:** Enable efficient inference of 200GB+ MoE models (DeepSeek V3, Kimi, Qwen-MoE) on Intel Arc GPUs (24GB total VRAM) + 64GB host RAM by computing experts where their data lives — GPU for VRAM-cached experts, CPU for host-RAM-resident experts — eliminating PCIe as the bottleneck.

**Architecture:** KTransformers-inspired hybrid: dense layers (attention + shared experts) pinned in GPU VRAM and computed on GPU at full bandwidth. Routed expert weights live in host RAM with an LRU+frequency VRAM cache. Cache-hit experts run on GPU; cache-miss experts run on CPU from host RAM directly (38 GB/s DDR5 > 20 GB/s PCIe). Pre-attention expert prediction prefetches cache misses 1-2 layers ahead. Pipeline parallelism across B580+B50 for the dense portion.

**Tech Stack:** SYCL (oneAPI 2025.3), Level Zero, AVX2+VNNI CPU kernels, existing unified cache, existing cpu-dispatch infrastructure

**Why this approach:** Weight streaming (PCIe-bound at 20 GB/s) gives ~10 tok/s for DeepSeek V3. CPU expert compute from host RAM (38 GB/s DDR5) gives 1.9x more bandwidth than PCIe. With 80% VRAM cache hit rate, expected ~45 tok/s. Theoretical maximum: ~54 tok/s.

---

## Team Topology

**Recommended implementers:** 3 (based on 3 parallel tracks)
**Reviewers:** 1 spec-reviewer, 1 quality-reviewer

### Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A | 1, 4, 7 | Expert cache manager (VRAM cache with scoring + eviction) |
| B | 2, 5, 8 | CPU expert compute (MUL_MAT_ID CPU dispatch for cache-miss experts) |
| C | 3, 6 | Expert prefetch & prediction (async DMA + prediction model) |
| — | 9 | Integration + pipeline parallelism (depends on A+B+C) |

### Dependency Graph

```
T1 (Expert Cache Manager)          T2 (CPU MUL_MAT_ID Dispatch)       T3 (Expert Prefetch DMA)
         |                                    |                                |
         v                                    v                                v
T4 (Cache-Aware MUL_MAT_ID)       T5 (AVX2/VNNI Expert Kernels)     T6 (Pre-Attention Prediction)
         |                                    |                                |
         v                                    v                                |
T7 (Cache Scoring + Eviction)      T8 (Mixed-Precision Miss Load)           |
         |                                    |                                |
         +------------------------------------+--------------------------------+
                                              |
                                              v
                                    T9 (Integration + Pipeline)
```

### File Ownership Map

| File/Directory | Tasks | Conflict Risk |
|----------------|-------|---------------|
| `ggml/src/ggml-sycl/expert-cache.hpp` (NEW) | 1, 4, 7 | Sequential (Track A) |
| `ggml/src/ggml-sycl/expert-cache.cpp` (NEW) | 1, 4, 7 | Sequential (Track A) |
| `ggml/src/ggml-sycl/cpu-dispatch.cpp` | 2, 5 | Sequential (Track B) |
| `ggml/src/ggml-sycl/cpu-dispatch.hpp` | 2, 5 | Sequential (Track B) |
| `ggml/src/ggml-sycl/expert-prefetch.hpp` (NEW) | 3, 6 | Sequential (Track C) |
| `ggml/src/ggml-sycl/expert-prefetch.cpp` (NEW) | 3, 6 | Sequential (Track C) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | 4, 9 | T4 before T9 (sequential) |
| `ggml/src/ggml-sycl/mmvq.cpp` | 4 | Single task |
| `ggml/src/ggml-sycl/unified-kernel.cpp` | 9 | Single task |
| `ggml/src/ggml-sycl/CMakeLists.txt` | 1 | Single task (early) |

---

## Phase 1: Foundation (Tasks 1-3, all parallel)

### Task 1: Expert VRAM Cache Manager

**Track:** A
**Depends on:** None
**File scope:**
- Create: `ggml/src/ggml-sycl/expert-cache.hpp`
- Create: `ggml/src/ggml-sycl/expert-cache.cpp`
- Modify: `ggml/src/ggml-sycl/CMakeLists.txt` (add new source files)

**Goal & Intent:** Build the core data structure that tracks which MoE expert weights are cached in GPU VRAM vs residing in host RAM. This is the foundation for the entire hybrid inference strategy — every subsequent task depends on knowing "is this expert on GPU or CPU?" Without this cache, we'd stream ALL expert weights over PCIe (20 GB/s), hitting ~10 tok/s. With 80%+ cache hit rate, we target 45+ tok/s.

**Description:**

Create an expert-granular VRAM cache that sits alongside the existing `unified_cache`. The unified cache manages weight placement at tensor granularity; the expert cache manages placement at expert granularity within MoE tensors. Each "expert slot" is one expert's weight tensor for one layer (~4.2 MB at Q4_0 for DeepSeek V3).

Use Serena to explore:
- `unified-cache.hpp/cpp` — understand `unified_alloc()`, `runtime_category`, budget tracking via `unified_cache_add_runtime_bytes()`
- `ggml-sycl.cpp` around line 17694 — `is_moe_op` detection for `GGML_OP_MUL_MAT_ID`
- `mmvq.hpp` line 29-33 — `ggml_sycl_mul_mat_id_vec_q()` signature (GPU-side expert routing)

**Acceptance Criteria:**

- [ ] `ExpertCache` class with `init(device_id, vram_budget_bytes)` allocates a contiguous VRAM pool
- [ ] `lookup(layer_idx, expert_idx)` returns `{device_ptr, is_cached}` in O(1)
- [ ] `evict_and_load(layer_idx, expert_idx, host_src, bytes, queue)` evicts lowest-score slot, async H2D loads new expert
- [ ] `update_score(layer_idx, expert_idx)` bumps access frequency + recency
- [ ] Pool allocated via `sycl::malloc_device`, sub-allocated with 256-byte alignment per slot
- [ ] Budget tracked via `unified_cache_add_runtime_bytes()` with `runtime_category::EXPERT_CACHE` (new enum value)
- [ ] Thread-safe for concurrent GPU dispatch + CPU dispatch reads
- [ ] Env var `GGML_SYCL_EXPERT_CACHE_MB=N` overrides default budget (default: 50% of remaining VRAM after dense layers)
- [ ] Unit test: create cache with 100 slots, load/evict/lookup cycle, verify no leaks

**Implementation Guide:**

Core data structures:
```cpp
// expert-cache.hpp
namespace ggml_sycl {

struct ExpertSlot {
    int      layer_idx   = -1;    // -1 = empty slot
    int      expert_idx  = -1;
    void *   device_ptr  = nullptr; // Points into pool
    size_t   size_bytes  = 0;
    uint64_t frequency   = 0;      // Access count
    uint64_t last_access = 0;      // Token counter
    float    score       = 0.0f;   // Combined eviction score
};

struct ExpertLookup {
    void * device_ptr;  // Non-null if cached
    void * host_ptr;    // Always set (host RAM copy)
    bool   is_cached;
};

class ExpertCache {
public:
    void init(int device_id, size_t vram_budget_bytes, sycl::queue & q);
    void shutdown();

    // Register all experts at model load time (host pointers)
    void register_expert(int layer_idx, int expert_idx, const void * host_ptr, size_t bytes);

    // Fast lookup: is this expert in VRAM?
    ExpertLookup lookup(int layer_idx, int expert_idx) const;

    // Load expert into VRAM (evicting if necessary), returns device ptr
    void * ensure_cached(int layer_idx, int expert_idx, sycl::queue & q);

    // Async prefetch (non-blocking H2D)
    sycl::event prefetch_async(int layer_idx, int expert_idx, sycl::queue & q);

    // Update access statistics after use
    void update_score(int layer_idx, int expert_idx, uint64_t token_counter);

    // Stats
    size_t cached_count() const;
    size_t total_slots() const;
    float  hit_rate() const;  // Rolling window

private:
    void * pool_ = nullptr;           // sycl::malloc_device contiguous pool
    size_t pool_size_ = 0;
    size_t slot_size_ = 0;            // Max expert size (aligned to 256)
    int    n_slots_ = 0;
    std::vector<ExpertSlot> slots_;
    // Fast lookup: (layer_idx * max_experts + expert_idx) -> slot index or -1
    std::unordered_map<int64_t, int> lookup_map_;
    mutable std::shared_mutex mutex_;
    // Stats
    uint64_t hits_ = 0, misses_ = 0;
    int device_id_ = 0;

    int find_eviction_candidate() const;
    int64_t make_key(int layer, int expert) const;
};

} // namespace ggml_sycl
```

The eviction scoring: `score = alpha * frequency + beta * (token_counter - last_access)` with alpha=1.0, beta=-0.01 (penalize stale entries). Lowest score gets evicted.

**Commit:**
```bash
git add ggml/src/ggml-sycl/expert-cache.hpp ggml/src/ggml-sycl/expert-cache.cpp ggml/src/ggml-sycl/CMakeLists.txt
git commit -m "sycl: add MoE expert VRAM cache manager for hybrid inference"
```

**Notes for implementer:**
- Use Serena `get_symbols_overview` on `unified-cache.hpp` to understand `runtime_category` enum — you'll need to add `EXPERT_CACHE`
- The pool allocation must happen AFTER model loading (dense weights already placed) so remaining VRAM budget is known
- `sycl::malloc_device` for the pool, NOT `malloc_host` — experts in the cache must be at full GPU bandwidth (276 GB/s)
- Thread safety via `std::shared_mutex`: reads (lookup) take shared lock, writes (evict/load) take exclusive lock

---

### Task 2: CPU Expert MUL_MAT Dispatch for MoE

**Track:** B
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/cpu-dispatch.hpp` (add `ggml_sycl_cpu_expert_mul_mat()`)
- Modify: `ggml/src/ggml-sycl/cpu-dispatch.cpp` (implement CPU expert matmul)

**Goal & Intent:** Enable the CPU to compute individual expert MUL_MATs directly from host RAM, bypassing PCIe entirely. This is the KTransformers core insight: for batch=1, CPU DDR5 bandwidth (38 GB/s) beats PCIe (20 GB/s) for cache-miss experts. The CPU computes the expert matmul and returns only the 14KB activation output to GPU — 300x less PCIe traffic than streaming the 4.2MB expert weights.

**Description:**

Add a function that takes a single expert's weight pointer (in host RAM), an activation vector (host-accessible), and produces output in a host-pinned buffer. Uses the existing `ggml_sycl_cpu_vec_dot_rows()` infrastructure with the thread pool.

Use Serena to explore:
- `cpu-dispatch.cpp` — `ggml_sycl_cpu_vec_dot_rows()` and `ggml_sycl_cpu_vec_dot_batched()` for the existing CPU matmul infrastructure
- `cpu-dispatch.cpp` — `simd_mul_mat_q4_0_q8_0_4row/8row/16row` for SIMD kernel variants
- `cpu-dispatch.hpp` — `cpu_vec_dot_batch_item` struct for batched dispatch

**Acceptance Criteria:**

- [ ] `ggml_sycl_cpu_expert_mul_mat(weight_host, act_host, output_host, type, K, N)` computes output = weight * activation for one expert
- [ ] Uses existing thread pool (`ggml_sycl_cpu_vec_dot_rows`) for parallel row computation
- [ ] Supports Q4_0, Q8_0, Q6_K weight types (the common MoE quantizations)
- [ ] Output buffer is `sycl::malloc_host` (GPU-accessible for merge via PCIe zero-copy)
- [ ] Activation input auto-quantized to Q8_1 for vec_dot compatibility
- [ ] Batched variant: `ggml_sycl_cpu_expert_mul_mat_batched()` computes multiple experts concurrently
- [ ] No data copies of expert weights — reads directly from host pointer
- [ ] Benchmark: single expert Q4_0 [4096x4096] at batch=1 completes in <0.5ms on 20-core CPU

**Implementation Guide:**

```cpp
// In cpu-dispatch.hpp, add:
struct cpu_expert_task {
    const void * weight_host;  // Expert weight data in host RAM (quantized)
    const float * act_host;    // Activation vector (float, length K)
    float * output_host;       // Output buffer (float, length N)
    ggml_type type;            // Q4_0, Q8_0, Q6_K etc.
    int K;                     // Input dimension
    int N;                     // Output rows (expert output dim)
};

// Compute one expert's matmul on CPU, reading weights directly from host RAM
void ggml_sycl_cpu_expert_mul_mat(const cpu_expert_task & task);

// Compute multiple experts concurrently on CPU thread pool
void ggml_sycl_cpu_expert_mul_mat_batched(
    const std::vector<cpu_expert_task> & tasks,
    int n_threads = 0);  // 0 = auto
```

Implementation in `cpu-dispatch.cpp`: reuse `cpu_mul_mat()` logic but adapted for raw pointer inputs instead of ggml_tensor. The key difference: no tensor metadata needed, just raw weight pointer + dimensions.

**Commit:**
```bash
git add ggml/src/ggml-sycl/cpu-dispatch.hpp ggml/src/ggml-sycl/cpu-dispatch.cpp
git commit -m "sycl: add CPU expert MUL_MAT for MoE hybrid inference"
```

**Notes for implementer:**
- Use Serena `find_symbol` to read `cpu_mul_mat` body — it handles quantized weight types with `ggml_get_type_traits()->vec_dot`
- Our Arrow Lake CPU has NO AVX-512. Compile flags: `-mavx2 -mfma -mavxvnniint8`
- The activation must be quantized to Q8_1 before vec_dot — reuse `ggml_quantize_q8_1()` or the SYCL quantize path
- Output in `sycl::malloc_host` buffer so GPU can read it via PCIe zero-copy (no explicit H2D copy needed)
- Thread pool uses `std::hardware_concurrency() - 2` threads (see `ggml_sycl_cpu_threads_hint()`)

---

### Task 3: Async Expert Prefetch via DMA

**Track:** C
**Depends on:** None
**File scope:**
- Create: `ggml/src/ggml-sycl/expert-prefetch.hpp`
- Create: `ggml/src/ggml-sycl/expert-prefetch.cpp`
- Modify: `ggml/src/ggml-sycl/CMakeLists.txt` (add new source files)

**Goal & Intent:** Build the async DMA engine that can prefetch predicted-miss experts from host RAM to VRAM while the GPU is busy computing attention. This overlaps PCIe transfer with GPU compute, hiding latency. If we can prefetch 1-2 layers ahead with 93%+ prediction accuracy, most cache misses are resolved before they're needed.

**Description:**

Create a prefetch manager that accepts "prefetch hints" (layer_idx, expert_idx) and schedules non-blocking H2D DMA using an out-of-order SYCL queue. Tracks in-flight prefetch events. Integrates with the expert cache (Task 1) for slot allocation.

Use Serena to explore:
- `layer-streaming.hpp` — `layer_stream_manager::prefetch_next_layer()` and `prefetch_event_` for the existing double-buffer prefetch pattern
- `unified-kernel.cpp` — search for `ooo_q` or `out_of_order` to find OOQ usage patterns

**Acceptance Criteria:**

- [ ] `ExpertPrefetcher` class with `hint(layer_idx, expert_idx)` schedules non-blocking H2D
- [ ] Uses out-of-order queue for DMA (separate from compute queue)
- [ ] Double-buffered: can have 2 prefetches in flight simultaneously
- [ ] `await(layer_idx, expert_idx)` blocks until specific prefetch completes (returns device ptr)
- [ ] `cancel_all()` for graph shape changes
- [ ] Integrates with ExpertCache for slot allocation (doesn't manage memory itself)
- [ ] Env var `GGML_SYCL_EXPERT_PREFETCH_DEPTH=N` controls how many layers ahead to prefetch (default: 2)
- [ ] Thread-safe: hint() can be called from prediction thread while GPU thread calls await()

**Implementation Guide:**

```cpp
// expert-prefetch.hpp
namespace ggml_sycl {

struct PrefetchRequest {
    int          layer_idx;
    int          expert_idx;
    void *       device_dst;    // VRAM slot (from ExpertCache)
    const void * host_src;      // Host RAM source
    size_t       bytes;
    sycl::event  event;         // Completion event
    bool         completed = false;
};

class ExpertPrefetcher {
public:
    void init(sycl::queue & compute_q, ExpertCache * cache);
    void shutdown();

    // Schedule async prefetch (non-blocking)
    // Returns true if prefetch was scheduled, false if already cached or in-flight
    bool hint(int layer_idx, int expert_idx);

    // Batch hint: prefetch multiple experts for a layer
    void hint_batch(int layer_idx, const std::vector<int> & expert_indices);

    // Wait for specific expert to be available in VRAM
    // Returns device pointer (from cache)
    void * await(int layer_idx, int expert_idx);

    // Cancel all pending prefetches
    void cancel_all();

    // Stats
    int pending_count() const;
    int completed_count() const;

private:
    sycl::queue dma_queue_;     // OOQ for async H2D
    ExpertCache * cache_ = nullptr;
    std::vector<PrefetchRequest> in_flight_;
    mutable std::mutex mutex_;
};

} // namespace ggml_sycl
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/expert-prefetch.hpp ggml/src/ggml-sycl/expert-prefetch.cpp ggml/src/ggml-sycl/CMakeLists.txt
git commit -m "sycl: add async expert prefetch engine for MoE inference"
```

**Notes for implementer:**
- Use Serena to read `layer_stream_manager::prefetch_next_layer()` body — it shows the OOQ async memcpy + event tracking pattern
- The DMA queue MUST be out-of-order (NOT the compute queue). Create via `sycl::queue(ctx, device, {sycl::property::queue::enable_profiling{}})`
- Use `sycl::malloc_host` staging if needed, but prefer direct H2D from registered host pointers
- Key L2 coherency note from MEMORY.md: BCS H2D to `malloc_device` is NOT visible to normal kernel reads (stale L2). The expert cache uses `malloc_device` but the compute kernel reads via standard pointer — this works because the H2D completes BEFORE the kernel launches (in-order compute queue serializes after prefetch await)

---

## Phase 2: Integration (Tasks 4-6, parallel within tracks)

### Task 4: Cache-Aware MUL_MAT_ID Dispatch

**Track:** A
**Depends on:** Task 1 (ExpertCache)
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (MUL_MAT_ID dispatch path, around line 23378)
- Modify: `ggml/src/ggml-sycl/mmvq.cpp` (extend `ggml_sycl_mul_mat_id_vec_q` for split dispatch)

**Goal & Intent:** This is the critical integration point. Currently `ggml_sycl_mul_mat_id()` sends ALL experts to GPU. We modify it to check the expert cache: cache-hit experts go to GPU MMVQ, cache-miss experts go to CPU dispatch (Task 2). The GPU and CPU compute in parallel, then outputs are merged. This is where the 1.9x bandwidth advantage of CPU-over-PCIe materializes.

**Description:**

Modify the MUL_MAT_ID dispatch (line 23378) to:
1. Read the router's expert selection (ids tensor)
2. For each selected expert, check `ExpertCache::lookup()`
3. Partition into HIT set (GPU) and MISS set (CPU)
4. Dispatch HIT experts to `ggml_sycl_mul_mat_id_vec_q()` with cache device pointers
5. Dispatch MISS experts to `ggml_sycl_cpu_expert_mul_mat_batched()` concurrently
6. Merge CPU outputs into GPU output tensor (tiny H2D: 14KB per expert per layer)

Use Serena to explore:
- `ggml-sycl.cpp` line 23378 — `ggml_sycl_mul_mat_id()` full function body (670 lines!)
- `mmvq.cpp` line 3574 — `ggml_sycl_mul_mat_id_vec_q()` for GPU-side expert routing
- `mmvq.hpp` line 29-33 — signature showing `ids` tensor parameter

**Acceptance Criteria:**

- [ ] MUL_MAT_ID checks expert cache for each active expert
- [ ] Cache-hit experts dispatched to GPU (existing MMVQ path)
- [ ] Cache-miss experts dispatched to CPU (`ggml_sycl_cpu_expert_mul_mat_batched`)
- [ ] GPU and CPU compute run concurrently (CPU on thread pool, GPU on compute queue)
- [ ] CPU outputs merged into GPU output tensor via async memcpy
- [ ] Fallback: if ExpertCache not initialized, use existing full-GPU path (no regression)
- [ ] Env var `GGML_SYCL_MOE_HYBRID=1` enables hybrid dispatch (default: OFF during development)
- [ ] Correctness: output matches full-GPU baseline for same seed/temperature

**Implementation Guide:**

Key insertion point in `ggml_sycl_mul_mat_id()` around line 23430:
```cpp
// After reading expert IDs from router, before dispatching:
auto * expert_cache = ggml_sycl_get_expert_cache(ctx.device);
if (expert_cache && expert_cache->cached_count() > 0) {
    // Partition experts into GPU (cached) and CPU (not cached)
    std::vector<int> gpu_experts, cpu_experts;
    for (int e = 0; e < n_active_experts; e++) {
        int expert_id = expert_ids[e];
        auto lookup = expert_cache->lookup(layer_idx, expert_id);
        if (lookup.is_cached) {
            gpu_experts.push_back(expert_id);
        } else {
            cpu_experts.push_back(expert_id);
        }
    }
    // GPU: dispatch cached experts via MMVQ
    // CPU: dispatch miss experts via cpu_expert_mul_mat_batched
    // Merge: async memcpy CPU outputs to GPU output tensor
}
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp ggml/src/ggml-sycl/mmvq.cpp
git commit -m "sycl: cache-aware MUL_MAT_ID dispatch for MoE hybrid GPU+CPU inference"
```

**Notes for implementer:**
- Use Serena `find_symbol` with `include_body=true` on `ggml_sycl_mul_mat_id` to understand the full dispatch logic — it's 670 lines
- The `ids` tensor contains expert routing indices — use Serena to find how it's read (look for `ids->data` access patterns)
- Layer index extraction: use Serena to find `parse_layer_index` or similar in `unified-kernel.cpp` — we built this for the persistent kernel
- The merge step uses the in-order compute queue: CPU writes to `malloc_host` output, GPU reads via PCIe zero-copy. No explicit H2D needed if using `malloc_host` for CPU output buffers.

---

### Task 5: Optimized AVX2/VNNI Expert Kernels

**Track:** B
**Depends on:** Task 2 (CPU expert dispatch)
**File scope:**
- Modify: `ggml/src/ggml-sycl/cpu-dispatch.cpp` (add Q6_K and optimized batch dispatch)

**Goal & Intent:** The baseline CPU expert path from Task 2 uses generic vec_dot. For MoE models with hundreds of expert evaluations per token, CPU kernel speed matters. Optimized SIMD kernels (AVX2+VNNI) can give 2-3x speedup over generic vec_dot, directly translating to higher tok/s when CPU is the bottleneck (low cache hit rate scenarios).

**Description:**

Add Q6_K vec_dot SIMD kernel (most MoE models use Q6_K for expert layers). Optimize the batched dispatch to amortize thread pool overhead when computing 8 experts concurrently. Add quantization cache for activation Q8_1 to avoid re-quantizing the same activation for each expert.

Use Serena to explore:
- `cpu-dispatch.cpp` — `simd_mul_mat_q4_0_q8_0_4row` body for the AVX2 kernel pattern
- Search for `q6_k\|Q6_K` in `cpu-dispatch.cpp` to see if Q6_K is already supported
- `quants.hpp` — Q6_K SOA block structure

**Acceptance Criteria:**

- [ ] `simd_mul_mat_q6_k_q8_1_4row()` AVX2 kernel for Q6_K expert weights
- [ ] Activation Q8_1 quantization cached across experts in same layer (quantize once, reuse 8x)
- [ ] Batched expert dispatch amortizes thread pool overhead (single dispatch for 8 experts)
- [ ] Benchmark: 8 experts Q4_0 [4096x14336] at batch=1 completes in <4ms on 20-core CPU

**Commit:**
```bash
git add ggml/src/ggml-sycl/cpu-dispatch.cpp
git commit -m "sycl: optimized AVX2/VNNI CPU expert kernels for MoE inference"
```

---

### Task 6: Pre-Attention Expert Prediction

**Track:** C
**Depends on:** Task 3 (ExpertPrefetcher)
**File scope:**
- Modify: `ggml/src/ggml-sycl/expert-prefetch.hpp` (add prediction interface)
- Modify: `ggml/src/ggml-sycl/expert-prefetch.cpp` (implement prediction logic)

**Goal & Intent:** Pre-attention expert prediction uses the hidden state BEFORE attention to predict which experts will be needed AFTER attention. This gives a full attention computation's worth of time (~17ms) to prefetch cache-miss experts. Research shows 93-98% accuracy. Even without a trained predictor, a simple "use same experts as last token" heuristic gives ~70% accuracy due to expert access locality.

**Description:**

Implement a lightweight expert prediction mechanism. Start with the simple heuristic (last-token expert reuse + layer correlation), with hooks for a future learned predictor. The predictor runs on CPU (cheap, ~0.1ms) and feeds hints to the prefetcher.

Use Serena to explore:
- `ggml-sycl.cpp` around MUL_MAT_ID dispatch — find where router logits/ids are available
- Search for `gate\|router\|gating` in context of MoE to understand the routing mechanism

**Acceptance Criteria:**

- [ ] `ExpertPredictor` class with `predict(layer_idx, hidden_state)` returns predicted expert indices
- [ ] Simple heuristic mode: reuse last token's experts + top-K from global frequency table
- [ ] Integration: after each layer's attention, call `predict()` for current layer's MoE, feed results to `prefetcher.hint_batch()`
- [ ] Accuracy tracking: compare predictions vs actual router selections, log hit rate
- [ ] Env var `GGML_SYCL_EXPERT_PREDICT=1` enables prediction (default: ON when expert cache active)
- [ ] Prediction must complete in <0.5ms (no GPU involvement)

**Commit:**
```bash
git add ggml/src/ggml-sycl/expert-prefetch.hpp ggml/src/ggml-sycl/expert-prefetch.cpp
git commit -m "sycl: add pre-attention expert prediction for MoE prefetching"
```

**Notes for implementer:**
- Start with the simple "reuse last token's experts" heuristic — it's fast and gives ~70% accuracy
- The global frequency table tracks which experts are most common across all tokens — initialized during warmup
- DO NOT train a neural network predictor in this task. That's a future enhancement. Focus on the heuristic.
- The prediction MUST run on CPU to avoid occupying GPU compute during attention

---

## Phase 3: Polish (Tasks 7-8, parallel within tracks)

### Task 7: Adaptive Cache Scoring & Warm-Start

**Track:** A
**Depends on:** Task 4 (cache-aware dispatch)
**File scope:**
- Modify: `ggml/src/ggml-sycl/expert-cache.hpp` (scoring policy interface)
- Modify: `ggml/src/ggml-sycl/expert-cache.cpp` (implement adaptive scoring)

**Goal & Intent:** The initial LRU+frequency scoring from Task 1 is a starting point. Research (SpecMD, HOBBIT) shows expert access follows structured sequential patterns, not recency-based reuse. Adaptive scoring adjusts weights based on observed access patterns: if a model shows strong layer correlation (e.g., expert 5 in layer 10 always follows expert 5 in layer 9), boost correlation-aware scoring. This can push cache hit rate from 80% to 90%+, directly translating to 50+ tok/s.

**Description:**

Enhance eviction scoring with:
1. Layer-distance penalty (experts from distant layers evicted first)
2. Co-activation tracking (experts frequently activated together stay together)
3. Warm-start: profile first N tokens to pre-populate cache with high-frequency experts

**Acceptance Criteria:**

- [ ] Scoring formula: `score = α*frequency + β*recency + γ*layer_distance + δ*co_activation`
- [ ] Warm-start: first 32 tokens profile expert access, then bulk-load top-N experts to VRAM
- [ ] `GGML_SYCL_EXPERT_CACHE_WARMUP=N` controls warm-start token count (default: 32)
- [ ] Cache hit rate tracking with rolling window (last 100 tokens)
- [ ] Log message on stderr: `[EXPERT-CACHE] hit_rate=XX.X% cached=NNNN/TTTT pool=XX.XMB`

**Commit:**
```bash
git add ggml/src/ggml-sycl/expert-cache.hpp ggml/src/ggml-sycl/expert-cache.cpp
git commit -m "sycl: adaptive expert cache scoring with warm-start profiling"
```

---

### Task 8: Mixed-Precision Cache Miss Loading (HOBBIT-style)

**Track:** B
**Depends on:** Task 5 (CPU expert kernels)
**File scope:**
- Modify: `ggml/src/ggml-sycl/cpu-dispatch.cpp` (add INT4/MXFP4 dequant for miss experts)
- Modify: `ggml/src/ggml-sycl/expert-prefetch.cpp` (precision-aware prefetch)

**Goal & Intent:** HOBBIT showed that loading cache-miss experts at lower precision (INT4 instead of Q4_0) reduces PCIe transfer by 2x with <1% quality degradation. Since cache-miss experts are the bottleneck, halving their transfer time directly helps. This is a "graceful degradation" strategy: cached experts run at full precision on GPU, miss experts run at reduced precision on CPU.

**Description:**

When an expert misses the VRAM cache and must be computed on CPU or prefetched:
1. Option A (prefetch): load at full precision but only if time permits
2. Option B (CPU compute): compute at full precision from host (no transfer needed)
3. Option C (burst miss): if >3 misses per layer, prefetch remaining at INT4 (2x faster transfer)

**Acceptance Criteria:**

- [ ] Mixed precision only activates when miss count exceeds threshold (default: 3 per layer)
- [ ] INT4 dequant kernel for CPU expert path
- [ ] Env var `GGML_SYCL_EXPERT_MISS_PRECISION=full|mixed` (default: mixed)
- [ ] Quality validation: perplexity within 0.5% of full-precision baseline

**Commit:**
```bash
git add ggml/src/ggml-sycl/cpu-dispatch.cpp ggml/src/ggml-sycl/expert-prefetch.cpp
git commit -m "sycl: mixed-precision cache miss loading for MoE burst misses"
```

---

## Phase 4: System Integration (Task 9)

### Task 9: Full Integration + Pipeline Parallelism for Dense Layers

**Track:** — (convergence point)
**Depends on:** Tasks 4, 5, 6, 7, 8 (all Phase 2-3 tasks)
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (graph_compute integration, model load hooks)
- Modify: `ggml/src/ggml-sycl/unified-kernel.cpp` (pipeline parallelism for dense layers)
- Modify: `ggml/src/ggml-sycl/unified-cache.cpp` (expert registration at model load)

**Goal & Intent:** Wire everything together into a coherent system. At model load time: identify MoE layers, register experts in the cache, warm-start. During inference: dense layers run via pipeline parallelism across B580+B50 (each with its own micro-graph). MoE layers use hybrid dispatch. This is the capstone that delivers the full 25-45 tok/s target for DeepSeek V3 on our hardware.

**Description:**

Integration points:
1. **Model load**: detect MoE architecture (GGML_OP_MUL_MAT_ID in graph), register expert weights in ExpertCache
2. **Dense layer pipeline**: split dense layers across B580/B50 by bandwidth ratio (65%/35%), each device runs its own micro-graph
3. **MoE layer hybrid dispatch**: integrate ExpertCache + ExpertPrefetcher + CPU dispatch into the main `graph_compute` loop
4. **Activation transfer**: between dense pipeline and MoE dispatch, transfer activation vector (16KB) between devices as needed

Use Serena to explore:
- `ggml-sycl.cpp` around line 33161 — `PERSISTENT_SPLIT` and persistent TG entry point
- `unified-kernel.cpp` — `record_micro_graph()` and `launch_micro_graph_kernel()` for per-device graph setup
- `ggml-sycl.cpp` around line 34687 — main `graph_compute` switch statement for MUL_MAT_ID

**Acceptance Criteria:**

- [ ] `GGML_SYCL_MOE_HYBRID=1` activates full hybrid inference pipeline
- [ ] Model load auto-detects MoE architecture and initializes ExpertCache
- [ ] Dense layers pipeline across both GPUs (bandwidth-proportional split)
- [ ] MoE layers use hybrid GPU+CPU dispatch
- [ ] Expert prefetch runs during attention computation
- [ ] Correctness: output matches baseline for deterministic generation (seed 42, temp 0)
- [ ] Performance target: ≥25 tok/s for DeepSeek V3 Q4_0 on B580+B50+CPU

**Verification:**
```bash
source /opt/intel/oneapi/setvars.sh --force

# Test with a MoE model (Mixtral or similar available locally)
GGML_SYCL_MOE_HYBRID=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /path/to/moe-model.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# Benchmark
GGML_SYCL_MOE_HYBRID=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench \
  -m /path/to/moe-model.gguf -p 0 -n 128
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp ggml/src/ggml-sycl/unified-kernel.cpp ggml/src/ggml-sycl/unified-cache.cpp
git commit -m "sycl: full MoE hybrid inference integration with pipeline parallelism"
```

---

## Expected Performance Progression

| After Task | What Changes | Expected tok/s (DeepSeek V3 Q4_0) |
|-----------|-------------|-----------------------------------|
| Baseline (weight streaming) | All experts via PCIe | ~10 |
| Task 2 complete | CPU computes all experts from host RAM | ~19 |
| Task 1+4 complete | VRAM cache (30% experts), CPU handles misses | ~25-35 |
| Task 3+6 complete | Prefetch hides miss latency | ~35-40 |
| Task 7 complete | Adaptive scoring → 90%+ hit rate | ~40-45 |
| Task 9 complete | Dense pipeline + full integration | ~45-50 |

## Environment Variables Summary

| Variable | Default | Description |
|----------|---------|-------------|
| `GGML_SYCL_MOE_HYBRID` | 0 | Master switch for MoE hybrid inference |
| `GGML_SYCL_EXPERT_CACHE_MB` | auto | Expert VRAM cache budget in MB |
| `GGML_SYCL_EXPERT_PREFETCH_DEPTH` | 2 | Layers ahead to prefetch |
| `GGML_SYCL_EXPERT_PREDICT` | 1 | Enable expert prediction |
| `GGML_SYCL_EXPERT_CACHE_WARMUP` | 32 | Warm-start profiling tokens |
| `GGML_SYCL_EXPERT_MISS_PRECISION` | mixed | Cache miss precision (full/mixed) |

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|-----------|
| Expert cache thrashing (low hit rate) | MEDIUM | Adaptive scoring, warm-start, frequency profiling |
| CPU expert compute too slow | LOW | AVX2/VNNI kernels, 38 GB/s DDR5 is adequate for 1-2 experts/layer |
| Thread safety issues (concurrent GPU+CPU) | MEDIUM | shared_mutex on cache, malloc_host for CPU outputs |
| MoE model not available for testing | HIGH | Start with Mixtral 8x7B (smaller MoE), validate architecture before DeepSeek V3 |
| Memory pressure (cache + KV + compute) | MEDIUM | Budget-aware allocation via unified_cache_add_runtime_bytes() |
